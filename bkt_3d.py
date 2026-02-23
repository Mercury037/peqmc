# test_pemc_adaptive.py
# 兼容：
#   - 旧模型：2D训练（MSE），测试时 net(theta:[N,4], X:[N,d])
#   - 新模型：3D训练（RQMC group loss），测试时自动包装成 [1,N,*] 再过 net
#
# 依赖：
#   simulation_qmc.py 里应提供：
#     sample_theta, simulate_gbm_batch_qmc, inv_Phi_torch, arithmetic_payoff,
#     Cholesky, PCA, BB
#   train_model_qmc_pca_mse.py 里应提供：
#     normalize   （你已经改成兼容2D/3D广播版了）
#   model.py 里应提供：
#     PEMCNet     （你已经改成支持2D/3D forward 了）

import os
import math
import argparse
from types import SimpleNamespace

import yaml
import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from simulation_qmc import *   # 你的函数都在这里（sample_theta, simulate_gbm_batch_qmc, inv_Phi_torch, arithmetic_payoff, PCA/BB/Cholesky）
from model import PEMCNet
from train_model_qmc_pca_3d import normalize


# ============================================================
# 1) 加载训练结果
# ============================================================
def _load_cfg(cfg_path: str):
    if not os.path.exists(cfg_path):
        return SimpleNamespace()
    with open(cfg_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return SimpleNamespace(**d)


def load_all(results_dir: str, device: str | None = None):
    results_dir = os.path.abspath(results_dir)

    model_path = os.path.join(results_dir, "model.pth")
    norm_path = os.path.join(results_dir, "normalization.pth")
    cfg_path = os.path.join(results_dir, "config.yaml")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing normalization file: {norm_path}")

    cfg = _load_cfg(cfg_path)

    # fallback
    if not hasattr(cfg, "device"):
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(cfg, "dimX"):
        cfg.dimX = 16
    if not hasattr(cfg, "dropout"):
        cfg.dropout = 0.0
    if not hasattr(cfg, "seed"):
        cfg.seed = 42

    device = device or cfg.device

    # 如果你训练时保存了 use_bn，可以自动读；没有就默认 False
    use_bn = getattr(cfg, "use_bn", False)

    net = PEMCNet(
        dimX=cfg.dimX,
        dropout=cfg.dropout,
        use_bn=use_bn
    ).to(device)

    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    norm = torch.load(norm_path, map_location=device)

    # cfg.device 更新为实际测试设备，后面统一用 cfg.device
    cfg.device = device
    return net, norm, cfg, device


# ============================================================
# 2) 自适应 2D/3D 测试辅助函数
# ============================================================
@torch.no_grad()
def _infer_eval_mode(norm: dict, cfg=None):
    """
    根据 normalization 张量形状自动判断测试模式:
      - 2D训练常见: X_mean.shape == [1, d]
      - 3D训练常见: X_mean.shape == [1, 1, d]
    返回: "2d" 或 "3d"
    """
    if cfg is not None:
        if getattr(cfg, "eval_force_3d", False):
            return "3d"
        if getattr(cfg, "eval_force_2d", False):
            return "2d"

    x_mean = norm["X_mean"]
    return "3d" if x_mean.ndim >= 3 else "2d"


@torch.no_grad()
def predict_g_points_adaptive(
    net,
    norm: dict,
    cfg,
    theta_points: torch.Tensor,   # [N,4]
    X_points: torch.Tensor,       # [N,d]
):
    """
    统一点预测接口（自适应2D/3D模型）

    输入:
      theta_points: [N,4]
      X_points:     [N,d]
    输出:
      g: [N]
    """
    if theta_points.ndim != 2 or theta_points.shape[-1] != 4:
        raise ValueError(f"theta_points 应为 [N,4], got {theta_points.shape}")
    if X_points.ndim != 2:
        raise ValueError(f"X_points 应为 [N,d], got {X_points.shape}")

    mode = _infer_eval_mode(norm, cfg)

    if mode == "2d":
        theta_n, X_n = normalize(theta_points, X_points, norm)
        g = net(theta_n, X_n).squeeze(-1)  # [N]
        return g

    elif mode == "3d":
        # 包一层组维 -> [1,N,4], [1,N,d]
        theta_in = theta_points.unsqueeze(0)
        X_in = X_points.unsqueeze(0)

        theta_n, X_n = normalize(theta_in, X_in, norm)
        g = net(theta_n, X_n)              # [1,N,1]
        g = g.squeeze(0).squeeze(-1)       # [N]
        return g

    else:
        raise ValueError(f"未知模式: {mode}")


# ============================================================
# 3) PEMC 单次估计 / 重复估计方差
#    约定：
#      term1 用 N1=B 个点
#      term2 用 N2=(2^M)*B 个点
# ============================================================
@torch.no_grad()
def pemc_estimate(
    B: int,
    M: int,
    net,
    norm: dict,
    cfg,
    theta_1dim,          # 固定theta（4个标量tuple），给 term2 用
    theta_tuple,         # 每个参数 shape [B]，给 term1 用
    nD: int = 256,
    T: float = 1.0,
    method: str = "pca",
    rep_seed: int = 0,
):
    device = cfg.device
    net.eval()
    norm_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in norm.items()}

    # 方便比较：B和N2都要求2的幂（Sobol前缀）
    if B & (B - 1) != 0:
        raise ValueError(f"B must be a power of 2, got {B}")
    N2 = (2 ** M) * B
    if N2 & (N2 - 1) != 0:
        raise ValueError(f"N2=(2^M)*B must be a power of 2, got {N2}")

    r, S0, sigma, K = theta_tuple  # shape [B]

    # =========================
    # term1 = (1/B) sum_i [ f_i - g_i ]
    # =========================
    sobol_path = SobolEngine(
        dimension=nD,
        scramble=True,
        seed=int(cfg.seed) + 200000 + rep_seed
    )

    Z1, W1, S1 = simulate_gbm_batch_qmc(
        theta_tuple,
        nD=nD, T=T,
        device=device,
        method=method,
        sobol_engine=sobol_path
    )

    X1 = Z1[:, :cfg.dimX]              # [B, dimX]
    y1 = arithmetic_payoff(S1, K)      # [B]

    theta1_mat = torch.stack([r, S0, sigma, K], dim=-1)   # [B,4]
    g1 = predict_g_points_adaptive(net, norm_dev, cfg, theta1_mat, X1)  # [B]

    term1 = (y1 - g1).mean()

    # =========================
    # term2 = (1/N2) sum_j g(theta_fixed, X~_j)
    # =========================
    sobol_x = SobolEngine(
        dimension=cfg.dimX,
        scramble=True,
        seed=int(cfg.seed) + 300000 + rep_seed
    )
    U2 = sobol_x.draw(N2).to(device=device, dtype=torch.float32)  # [N2,dimX]
    X2 = inv_Phi_torch(U2)                                         # [N2,dimX]

    # theta_fixed -> [N2,4]
    theta2_row = torch.stack(list(theta_1dim), dim=0).to(device=device, dtype=torch.float32)  # [4]
    theta2_mat = theta2_row.unsqueeze(0).expand(N2, 4)  # [N2,4]

    g2 = predict_g_points_adaptive(net, norm_dev, cfg, theta2_mat, X2)  # [N2]
    term2 = g2.mean()

    return float((term1 + term2).item())


def pemc_mean_var(
    B: int,
    M: int,
    n_rep: int,
    net,
    norm,
    cfg,
    theta_1dim,
    theta_tuple,
    nD: int = 256,
    T: float = 1.0,
    method: str = "pca",
):
    est = []
    for rep in range(n_rep):
        est.append(
            pemc_estimate(
                B=B, M=M,
                net=net, norm=norm, cfg=cfg,
                theta_1dim=theta_1dim,
                theta_tuple=theta_tuple,
                nD=nD, T=T, method=method,
                rep_seed=rep
            )
        )
    est = np.array(est, dtype=np.float64)
    return {
        "algo": "pemc",
        "method": method,
        "N1": B,
        "N2": (2**M) * B,
        "n_rep": n_rep,
        "mean": float(est.mean()),
        "var": float(est.var(ddof=1)) if n_rep > 1 else float("nan"),
    }


# ============================================================
# 4) Baseline: MC / RQMC
# ============================================================
def _generator_matrix(method: str, nD: int, T: float, device: str, dtype=torch.float32) -> torch.Tensor:
    method = method.lower()
    if method == "cholesky":
        G = Cholesky(nD, T=T)
    elif method == "pca":
        G = PCA(nD, T=T)
    elif method == "bb":
        if nD <= 0 or (nD & (nD - 1)) != 0:
            raise ValueError("BB requires nD to be a positive power of 2 (e.g., 256).")
        n = nD.bit_length() - 1
        G = BB(n, T=T)
    else:
        raise ValueError("method must be one of {'cholesky','pca','bb'}")
    return torch.tensor(G, device=device, dtype=dtype)


@torch.no_grad()
def mc_estimate(B, cfg, theta, nD=256, T=1.0, method="pca", rep_seed=0):
    device = cfg.device
    seed0 = int(getattr(cfg, "seed", 0))

    r, S0, sigma, K = theta
    r = r.to(device); S0 = S0.to(device); sigma = sigma.to(device); K = K.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed0 + 400000 + rep_seed)

    Z = torch.randn((B, nD), device=device, generator=gen, dtype=torch.float32)  # [B,nD]

    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)
    W = Z @ G.T

    t = torch.linspace(T / nD, T, nD, device=device, dtype=torch.float32).view(1, nD)
    S = S0 * torch.exp((r - 0.5 * sigma * sigma) * t + sigma * W)

    y = arithmetic_payoff(S, K)
    return float(y.mean().item())


def mc_mean_var(B, n_rep, cfg, theta, nD=256, T=1.0, method="pca"):
    est = [mc_estimate(B, cfg, theta, nD=nD, T=T, method=method, rep_seed=rep) for rep in range(n_rep)]
    est = np.array(est, dtype=np.float64)
    return {
        "algo": "mc",
        "method": method,
        "N1": B,
        "N2": 0,
        "n_rep": n_rep,
        "mean": float(est.mean()),
        "var": float(est.var(ddof=1)) if n_rep > 1 else float("nan"),
    }


@torch.no_grad()
def qmc_estimate(B, cfg, theta, nD=256, T=1.0, method="pca", rep_seed=0, scramble=True):
    device = cfg.device
    seed0 = int(getattr(cfg, "seed", 0))

    r, S0, sigma, K = theta
    r = r.to(device); S0 = S0.to(device); sigma = sigma.to(device); K = K.to(device)

    if scramble:
        sob = SobolEngine(dimension=nD, scramble=True, seed=seed0 + 500000 + rep_seed)
    else:
        sob = SobolEngine(dimension=nD, scramble=False)

    U = sob.draw(B).to(device=device, dtype=torch.float32)  # [B,nD]
    Z = inv_Phi_torch(U)

    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)
    W = Z @ G.T

    t = torch.linspace(T / nD, T, nD, device=device, dtype=torch.float32).view(1, nD)
    S = S0 * torch.exp((r - 0.5 * sigma * sigma) * t + sigma * W)

    y = arithmetic_payoff(S, K)
    return float(y.mean().item())


def qmc_mean_var(B, n_rep, cfg, theta, nD=256, T=1.0, method="pca"):
    # RQMC: scramble=True + 改rep_seed 才能估方差
    est = [
        qmc_estimate(B, cfg, theta, nD=nD, T=T, method=method, rep_seed=rep, scramble=True)
        for rep in range(n_rep)
    ]
    est = np.array(est, dtype=np.float64)
    return {
        "algo": "qmc",   # 更准确可写 rqmc
        "method": method,
        "N1": B,
        "N2": 0,
        "n_rep": n_rep,
        "mean": float(est.mean()),
        "var": float(est.var(ddof=1)) if n_rep > 1 else float("nan"),
    }


# ============================================================
# 5) 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.getcwd(), "results_3d_N10_dim1"),
        help="训练结果目录（默认: CWD/results_qmc_pca_mse）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda / cuda:0 ... （可选）"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "cholesky", "bb"],
        help="路径生成矩阵"
    )
    parser.add_argument(
        "--nD",
        type=int,
        default=256,
        help="路径时间离散维度（如256）"
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="到期时间"
    )
    args = parser.parse_args()

    net, norm, cfg, device = load_all(args.results_dir, args.device)

    print(f"[OK] Loaded net  : {os.path.join(os.path.abspath(args.results_dir), 'model.pth')}")
    print(f"[OK] Loaded norm : {os.path.join(os.path.abspath(args.results_dir), 'normalization.pth')}")
    print(f"[OK] Device      : {device}")
    print(f"[OK] dimX        : {getattr(cfg, 'dimX', None)}")
    print(f"[OK] dropout     : {getattr(cfg, 'dropout', None)}")
    print(f"[OK] Eval mode   : {_infer_eval_mode(norm, cfg)}")
    print(f"[OK] X_mean shape: {tuple(norm['X_mean'].shape)}")
    print(f"[OK] th_mean shape: {tuple(norm['theta_mean'].shape)}")

    # 固定theta（单一合约）
    theta_fixed = (
        torch.tensor(0.02, device=device),    # r
        torch.tensor(100.0, device=device),   # S0
        torch.tensor(0.15, device=device),    # sigma
        torch.tensor(100.0, device=device),   # K
    )

    B_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    M = 4
    n_rep = 100

    # ---------- PEMC ----------
    rows_pemc = []
    for B in B_list:
        # term1 用 [B] 形状参数
        r, S0, sigma, K = sample_theta(
            mode=2,
            batch_size=B,
            is_same=True,
            theta_same=theta_fixed,
            device=device
        )

        out = pemc_mean_var(
            B=B, M=M, n_rep=n_rep,
            net=net, norm=norm, cfg=cfg,
            theta_1dim=theta_fixed,
            theta_tuple=(r, S0, sigma, K),
            nD=args.nD, T=args.T, method=args.method
        )
        rows_pemc.append({
            "algo": out["algo"],
            "method": out["method"],
            "N1": out["N1"],
            "N2": out["N2"],
            "n_rep": out["n_rep"],
            "mean": out["mean"],
            "var": out["var"],
            "r": float(theta_fixed[0].item()),
            "S0": float(theta_fixed[1].item()),
            "sigma": float(theta_fixed[2].item()),
            "K": float(theta_fixed[3].item()),
        })
        print(f"[PEMC] done B={B}")

    df_pemc = pd.DataFrame(rows_pemc)

    # ---------- Baselines ----------
    rows_base = []
    for B in B_list:
        rows_base.append(mc_mean_var(B, n_rep, cfg, theta_fixed, nD=args.nD, T=args.T, method=args.method))
        rows_base.append(qmc_mean_var(B, n_rep, cfg, theta_fixed, nD=args.nD, T=args.T, method=args.method))
        print(f"[BASE] done B={B}")

    df_base = pd.DataFrame(rows_base)
    df_base["r"] = float(theta_fixed[0].item())
    df_base["S0"] = float(theta_fixed[1].item())
    df_base["sigma"] = float(theta_fixed[2].item())
    df_base["K"] = float(theta_fixed[3].item())

    # ---------- 打印 ----------
    pd.set_option("display.float_format", lambda x: f"{x:.12e}")

    print("\n===== PEMC =====")
    print(
        df_pemc[["N1", "N2", "mean", "var"]].to_string(
            index=False,
            formatters={
                "mean": lambda x: f"{x:.8f}",
                "var":  lambda x: f"{x:.12e}",
            }
        )
    )

    print("\n===== MC / QMC =====")
    print(
        df_base[["algo", "N1", "N2", "mean", "var"]].to_string(
            index=False,
            formatters={
                "mean": lambda x: f"{x:.8f}",
                "var":  lambda x: f"{x:.12e}",
            }
        )
    )

    # 可选：保存测试结果
    out_dir = os.path.abspath(args.results_dir)
    df_pemc.to_csv(os.path.join(out_dir, "eval_pemc.csv"), index=False)
    df_base.to_csv(os.path.join(out_dir, "eval_baselines.csv"), index=False)
    print(f"\n[OK] Saved CSVs to {out_dir}")


if __name__ == "__main__":
    main()