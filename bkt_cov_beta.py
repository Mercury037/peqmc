# test_pemc_adaptive_decomp.py
# 兼容：
#   - 旧模型：2D训练（MSE），测试时 net(theta:[N,4], X:[N,d])
#   - 新模型：3D训练（RQMC group loss），测试时自动包装成 [1,N,*] 再过 net
#
# 功能：
#   1) 测 PEMC / MC / QMC 的 mean / var
#   2) 对 PEMC 做方差分解：
#      Var(total) = Var(term1) + Var(term2) + 2Cov(term1, term2)
#   3) 对 term1 再分解（带 beta）：
#      term1 = ybar - beta * g1bar
#      Var(term1) = Var(ybar) + beta^2 Var(g1bar) - 2 beta Cov(ybar, g1bar)
#   4) 从训练目录读取 beta_map.json 或 beta_rows.json（按 B 取对应 beta）
#
# 依赖：
#   simulation_qmc.py: sample_theta, simulate_gbm_batch_qmc, inv_Phi_torch, arithmetic_payoff, Cholesky/PCA/BB
#   train_model_qmc_pca_3d.py: normalize（已兼容2D/3D广播）
#   model.py: PEMCNet（已支持2D/3D forward；若你是旧版2D forward也能测2D模型）

import os
import math
import json
import argparse
from types import SimpleNamespace

import yaml
import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from simulation_qmc import *
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
    if not hasattr(cfg, "method"):
        cfg.method = "pca"

    device = device or cfg.device
    use_bn = getattr(cfg, "use_bn", False)

    # 兼容不同版本 PEMCNet（有的有 use_bn，有的没有）
    try:
        net = PEMCNet(dimX=cfg.dimX, dropout=cfg.dropout, use_bn=use_bn).to(device)
    except TypeError:
        net = PEMCNet(dimX=cfg.dimX, dropout=cfg.dropout).to(device)

    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    norm = torch.load(norm_path, map_location=device)
    cfg.device = device
    return net, norm, cfg, device


# ============================================================
# 1.5) 加载 beta_rows / beta_map
# ============================================================
def load_beta_map(results_dir: str):
    """
    优先读取 beta_map.json；若不存在则从 beta_rows.json 构建 {N: beta_cv}.
    返回:
      beta_map: dict[int, float]
    """
    results_dir = os.path.abspath(results_dir)
    beta_map_path = os.path.join(results_dir, "beta_map.json")
    beta_rows_path = os.path.join(results_dir, "beta_rows.json")

    beta_map = {}

    if os.path.exists(beta_map_path):
        with open(beta_map_path, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        for k, v in d.items():
            try:
                beta_map[int(k)] = float(v)
            except Exception:
                pass
        return beta_map

    if os.path.exists(beta_rows_path):
        with open(beta_rows_path, "r", encoding="utf-8") as f:
            rows = json.load(f) or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            N = row.get("N", None)
            beta = row.get("beta_cv", None)
            try:
                N = int(N)
                beta = float(beta)
                if np.isfinite(beta):
                    beta_map[N] = beta
            except Exception:
                continue
        return beta_map

    return beta_map


def get_beta_for_B(beta_map: dict, B: int, default: float = 1.0):
    """
    你的设定：beta_rows 与 B_list 一一对应，所以这里按 B 取 beta。
    若缺失则回退到 default=1.0
    """
    beta = beta_map.get(int(B), default)
    try:
        beta = float(beta)
    except Exception:
        beta = float(default)
    if not np.isfinite(beta):
        beta = float(default)
    return beta


# ============================================================
# 2) 自适应 2D/3D 测试辅助函数
# ============================================================
def _infer_eval_mode(norm: dict, cfg=None):
    """
    根据 normalization 形状判断：
      2D训练常见: X_mean.shape == [1, d]
      3D训练常见: X_mean.shape == [1, 1, d]
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
        theta_in = theta_points.unsqueeze(0)  # [1,N,4]
        X_in = X_points.unsqueeze(0)          # [1,N,d]
        theta_n, X_n = normalize(theta_in, X_in, norm)
        g = net(theta_n, X_n)                 # [1,N,1]
        g = g.squeeze(0).squeeze(-1)          # [N]
        return g

    else:
        raise ValueError(f"未知模式: {mode}")


# ============================================================
# 3) 统计工具（方差 / 协方差 / 相关）
# ============================================================
def _sample_var(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 1:
        return float("nan")
    return float(x.var(ddof=1))


def _sample_cov(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"长度不一致: len(x)={x.size}, len(y)={y.size}")
    n = x.size
    if n <= 1:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    return float((xm * ym).sum() / (n - 1))


def _sample_corr(x, y, eps=1e-30):
    vx = _sample_var(x)
    vy = _sample_var(y)
    if not np.isfinite(vx) or not np.isfinite(vy):
        return float("nan")
    den = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if den < eps:
        return float("nan")
    return float(_sample_cov(x, y) / den)


# ============================================================
# 4) PEMC 单次估计 / 重复估计方差（含分解，带beta）
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
    beta: float = 1.0,   # <<< 新增：对应当前B的beta
    return_components: bool = False,
):
    device = cfg.device
    net.eval()
    norm_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in norm.items()}

    if B & (B - 1) != 0:
        raise ValueError(f"B must be a power of 2, got {B}")
    N2 = (2 ** M) * B
    if N2 & (N2 - 1) != 0:
        raise ValueError(f"N2=(2^M)*B must be a power of 2, got {N2}")

    beta = float(beta)

    r, S0, sigma, K = theta_tuple  # [B]

    # -------------------------
    # term1 = (1/B) sum_i [ f_i - beta * g_i ]
    # -------------------------
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

    ybar = y1.mean()
    g1bar = g1.mean()
    term1 = ybar - beta * g1bar

    # -------------------------
    # term2 = beta * (1/N2) sum_j g(theta_fixed, X~_j)
    # -------------------------
    sobol_x = SobolEngine(
        dimension=cfg.dimX,
        scramble=True,
        seed=int(cfg.seed) + 300000 + rep_seed
    )
    U2 = sobol_x.draw(N2).to(device=device, dtype=torch.float32)  # [N2, dimX]
    X2 = inv_Phi_torch(U2)                                         # [N2, dimX]

    theta2_row = torch.stack(list(theta_1dim), dim=0).to(device=device, dtype=torch.float32)  # [4]
    theta2_mat = theta2_row.unsqueeze(0).expand(N2, 4)  # [N2,4]

    g2 = predict_g_points_adaptive(net, norm_dev, cfg, theta2_mat, X2)  # [N2]
    g2bar = g2.mean()
    term2 = beta * g2bar

    total = term1 + term2

    if not return_components:
        return float(total.item())

    return {
        "total": float(total.item()),
        "term1": float(term1.item()),
        "term2": float(term2.item()),
        "beta": float(beta),
        "ybar": float(ybar.item()),
        "g1bar": float(g1bar.item()),
        "g2bar": float(g2bar.item()),
    }


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
    beta: float = 1.0,
    return_rep_df: bool = False,
):
    rep_rows = []
    for rep in range(n_rep):
        out = pemc_estimate(
            B=B, M=M,
            net=net, norm=norm, cfg=cfg,
            theta_1dim=theta_1dim,
            theta_tuple=theta_tuple,
            nD=nD, T=T, method=method,
            rep_seed=rep,
            beta=beta,
            return_components=True
        )
        out["rep"] = rep
        rep_rows.append(out)

    rep_df = pd.DataFrame(rep_rows)

    total = rep_df["total"].to_numpy(dtype=np.float64)
    t1 = rep_df["term1"].to_numpy(dtype=np.float64)
    t2 = rep_df["term2"].to_numpy(dtype=np.float64)
    ybar = rep_df["ybar"].to_numpy(dtype=np.float64)
    g1bar = rep_df["g1bar"].to_numpy(dtype=np.float64)
    g2bar = rep_df["g2bar"].to_numpy(dtype=np.float64)

    beta = float(beta)

    # 总分解：Var(total) = Var(t1)+Var(t2)+2Cov(t1,t2)
    var_total = _sample_var(total)
    var_t1 = _sample_var(t1)
    var_t2 = _sample_var(t2)
    cov_t1_t2 = _sample_cov(t1, t2)
    corr_t1_t2 = _sample_corr(t1, t2)
    two_cov_t1_t2 = 2.0 * cov_t1_t2 if np.isfinite(cov_t1_t2) else float("nan")
    var_recon = (
        var_t1 + var_t2 + two_cov_t1_t2
        if np.isfinite(var_t1) and np.isfinite(var_t2) and np.isfinite(two_cov_t1_t2)
        else float("nan")
    )
    var_recon_err = (var_total - var_recon) if np.isfinite(var_total) and np.isfinite(var_recon) else float("nan")

    # term1 内部分解（带 beta）
    # t1 = ybar - beta * g1bar
    # Var(t1) = Var(ybar) + beta^2 Var(g1bar) - 2 beta Cov(ybar, g1bar)
    var_ybar = _sample_var(ybar)
    var_g1bar = _sample_var(g1bar)
    cov_ybar_g1bar = _sample_cov(ybar, g1bar)
    corr_ybar_g1bar = _sample_corr(ybar, g1bar)

    beta2_var_g1bar = (beta ** 2) * var_g1bar if np.isfinite(var_g1bar) else float("nan")
    minus2beta_cov_ybar_g1bar = (-2.0 * beta * cov_ybar_g1bar) if np.isfinite(cov_ybar_g1bar) else float("nan")

    var_t1_recon = (
        var_ybar + beta2_var_g1bar + minus2beta_cov_ybar_g1bar
        if np.isfinite(var_ybar) and np.isfinite(beta2_var_g1bar) and np.isfinite(minus2beta_cov_ybar_g1bar)
        else float("nan")
    )
    var_t1_recon_err = (var_t1 - var_t1_recon) if np.isfinite(var_t1) and np.isfinite(var_t1_recon) else float("nan")

    if np.isfinite(var_total) and abs(var_total) > 0:
        share_t1 = var_t1 / var_total
        share_t2 = var_t2 / var_total
        share_2cov = two_cov_t1_t2 / var_total
    else:
        share_t1 = share_t2 = share_2cov = float("nan")

    summary = {
        "algo": "pemc",
        "method": method,
        "N1": B,
        "N2": (2 ** M) * B,
        "n_rep": n_rep,
        "beta": float(beta),

        "mean": float(total.mean()),
        "var": float(var_total),

        # PEMC总分解
        "term1_mean": float(t1.mean()),
        "term2_mean": float(t2.mean()),
        "var_term1": float(var_t1),
        "var_term2": float(var_t2),
        "cov_term1_term2": float(cov_t1_t2),
        "corr_term1_term2": float(corr_t1_t2),
        "two_cov_term1_term2": float(two_cov_t1_t2),
        "var_recon": float(var_recon),
        "var_recon_err": float(var_recon_err),

        "share_var_term1": float(share_t1),
        "share_var_term2": float(share_t2),
        "share_2cov": float(share_2cov),

        # term1 内部分解（带beta）
        "ybar_mean": float(ybar.mean()),
        "g1bar_mean": float(g1bar.mean()),
        "g2bar_mean": float(g2bar.mean()),
        "var_ybar": float(var_ybar),
        "var_g1bar": float(var_g1bar),
        "cov_ybar_g1bar": float(cov_ybar_g1bar),
        "corr_ybar_g1bar": float(corr_ybar_g1bar),

        "beta2_var_g1bar": float(beta2_var_g1bar),
        "minus2beta_cov_ybar_g1bar": float(minus2beta_cov_ybar_g1bar),
        "var_term1_recon": float(var_t1_recon),
        "var_term1_recon_err": float(var_t1_recon_err),
    }

    if return_rep_df:
        rep_df = rep_df.copy()
        rep_df["N1"] = B
        rep_df["N2"] = (2 ** M) * B
        rep_df["method"] = method
        rep_df["beta"] = float(beta)
        return summary, rep_df

    return summary


# ============================================================
# 5) Baseline: MC / RQMC
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
    r = r.to(device)
    S0 = S0.to(device)
    sigma = sigma.to(device)
    K = K.to(device)

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
    r = r.to(device)
    S0 = S0.to(device)
    sigma = sigma.to(device)
    K = K.to(device)

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
    est = [
        qmc_estimate(B, cfg, theta, nD=nD, T=T, method=method, rep_seed=rep, scramble=True)
        for rep in range(n_rep)
    ]
    est = np.array(est, dtype=np.float64)
    return {
        "algo": "qmc",   # 更准确其实是 rqmc
        "method": method,
        "N1": B,
        "N2": 0,
        "n_rep": n_rep,
        "mean": float(est.mean()),
        "var": float(est.var(ddof=1)) if n_rep > 1 else float("nan"),
    }


# ============================================================
# 6) 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.getcwd(), "results_lookback_Xdim1_N6_loss2"),
        help="训练结果目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda / cuda:0 ..."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "cholesky", "bb"],
        help="路径生成矩阵（若不想用训练配置里的 method，可在这里覆盖）"
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
    parser.add_argument(
        "--M",
        type=int,
        default=4,
        help="PEMC term2 样本倍数指数，N2=(2^M)*B"
    )
    parser.add_argument(
        "--n_rep",
        type=int,
        default=100,
        help="重复次数（估方差）"
    )
    parser.add_argument(
        "--use_cli_method",
        action="store_true",
        help="若提供，则使用 --method；否则默认使用训练cfg.method"
    )
    args = parser.parse_args()

    net, norm, cfg, device = load_all(args.results_dir, args.device)

    # beta map
    beta_map = load_beta_map(args.results_dir)

    # method选择：默认跟训练配置走；若显式给 --use_cli_method 则用命令行
    eval_method = args.method if args.use_cli_method else getattr(cfg, "method", args.method)

    print(f"[OK] Loaded net        : {os.path.join(os.path.abspath(args.results_dir), 'model.pth')}")
    print(f"[OK] Loaded norm       : {os.path.join(os.path.abspath(args.results_dir), 'normalization.pth')}")
    print(f"[OK] Device            : {device}")
    print(f"[OK] dimX              : {getattr(cfg, 'dimX', None)}")
    print(f"[OK] dropout           : {getattr(cfg, 'dropout', None)}")
    print(f"[OK] train method      : {getattr(cfg, 'method', None)}")
    print(f"[OK] eval method       : {eval_method}")
    print(f"[OK] Eval mode         : {_infer_eval_mode(norm, cfg)}")
    print(f"[OK] X_mean shape      : {tuple(norm['X_mean'].shape)}")
    print(f"[OK] theta_mean shape  : {tuple(norm['theta_mean'].shape)}")

    if len(beta_map) == 0:
        print("[WARN] beta_map.json / beta_rows.json 未找到，PEMC将回退为 beta=1.0")
    else:
        print(f"[OK] Loaded beta map    : {len(beta_map)} entries")
        print(f"[OK] beta keys          : {sorted(beta_map.keys())}")

    # 固定theta（单一合约）
    # theta_fixed = (
    #     torch.tensor(0.03, device=device),    # r
    #     torch.tensor(102.0, device=device),   # S0
    #     torch.tensor(0.20, device=device),    # sigma
    #     torch.tensor(100.0, device=device),   # K
    # )

    theta_fixed = (
        torch.tensor(0.02, device=device),  # r
        torch.tensor(100.0, device=device),  # S0
        torch.tensor(0.15, device=device),  # sigma
        torch.tensor(100, device=device),  # K
    )

    B_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    M = args.M
    n_rep = args.n_rep

    # ---------- PEMC（含方差分解，带beta） ----------
    rows_pemc = []
    rep_parts_all = []

    for B in B_list:
        beta_B = get_beta_for_B(beta_map, B, default=1.0)

        # term1 用 [B] 形状参数（固定合约，广播成长度B）
        r, S0, sigma, K = sample_theta(
            mode=2,
            batch_size=B,
            is_same=True,
            theta_same=theta_fixed,
            device=device
        )

        out, rep_df = pemc_mean_var(
            B=B, M=M, n_rep=n_rep,
            net=net, norm=norm, cfg=cfg,
            theta_1dim=theta_fixed,
            theta_tuple=(r, S0, sigma, K),
            nD=args.nD, T=args.T, method=eval_method,
            beta=beta_B,
            return_rep_df=True
        )

        rows_pemc.append({
            "algo": out["algo"],
            "method": out["method"],
            "N1": out["N1"],
            "N2": out["N2"],
            "n_rep": out["n_rep"],
            "beta": out["beta"],

            "mean": out["mean"],
            "var": out["var"],

            # PEMC总分解
            "term1_mean": out["term1_mean"],
            "term2_mean": out["term2_mean"],
            "var_term1": out["var_term1"],
            "var_term2": out["var_term2"],
            "cov_term1_term2": out["cov_term1_term2"],
            "corr_term1_term2": out["corr_term1_term2"],
            "two_cov_term1_term2": out["two_cov_term1_term2"],
            "var_recon": out["var_recon"],
            "var_recon_err": out["var_recon_err"],

            "share_var_term1": out["share_var_term1"],
            "share_var_term2": out["share_var_term2"],
            "share_2cov": out["share_2cov"],

            # term1 内部分解（带beta版本）
            "ybar_mean": out["ybar_mean"],
            "g1bar_mean": out["g1bar_mean"],
            "g2bar_mean": out["g2bar_mean"],
            "var_ybar": out["var_ybar"],
            "var_g1bar": out["var_g1bar"],
            "cov_ybar_g1bar": out["cov_ybar_g1bar"],
            "corr_ybar_g1bar": out["corr_ybar_g1bar"],
            "beta2_var_g1bar": out["beta2_var_g1bar"],
            "minus2beta_cov_ybar_g1bar": out["minus2beta_cov_ybar_g1bar"],
            "var_term1_recon": out["var_term1_recon"],
            "var_term1_recon_err": out["var_term1_recon_err"],

            # 合约信息
            "r": float(theta_fixed[0].item()),
            "S0": float(theta_fixed[1].item()),
            "sigma": float(theta_fixed[2].item()),
            "K": float(theta_fixed[3].item()),
        })

        rep_parts_all.append(rep_df)
        print(f"[PEMC] done B={B}, beta={beta_B:.8f}")

    df_pemc = pd.DataFrame(rows_pemc)
    df_pemc_rep = pd.concat(rep_parts_all, axis=0, ignore_index=True) if len(rep_parts_all) > 0 else pd.DataFrame()

    # ---------- Baselines ----------
    rows_base = []
    for B in B_list:
        rows_base.append(mc_mean_var(B, n_rep, cfg, theta_fixed, nD=args.nD, T=args.T, method=eval_method))
        rows_base.append(qmc_mean_var(B, n_rep, cfg, theta_fixed, nD=args.nD, T=args.T, method=eval_method))
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
        df_pemc[["N1", "N2", "beta", "mean", "var"]].to_string(
            index=False,
            formatters={
                "beta": lambda x: f"{x:.8f}",
                "mean": lambda x: f"{x:.8f}",
                "var": lambda x: f"{x:.12e}",
            }
        )
    )

    print("\n===== PEMC 方差分解（total = term1 + term2）=====")
    print(
        df_pemc[[
            "N1", "N2", "beta", "var",
            "var_term1", "var_term2", "two_cov_term1_term2",
            "var_recon", "var_recon_err",
            "corr_term1_term2",
            "share_var_term1", "share_var_term2", "share_2cov"
        ]].to_string(
            index=False,
            formatters={
                "beta": lambda x: f"{x:.8f}",
                "var": lambda x: f"{x:.12e}",
                "var_term1": lambda x: f"{x:.12e}",
                "var_term2": lambda x: f"{x:.12e}",
                "two_cov_term1_term2": lambda x: f"{x:.12e}",
                "var_recon": lambda x: f"{x:.12e}",
                "var_recon_err": lambda x: f"{x:.3e}",
                "corr_term1_term2": lambda x: f"{x:.6f}",
                "share_var_term1": lambda x: f"{x:.4f}",
                "share_var_term2": lambda x: f"{x:.4f}",
                "share_2cov": lambda x: f"{x:.4f}",
            }
        )
    )

    print("\n===== term1 内部分解（term1 = ybar - beta*g1bar）=====")
    print(
        df_pemc[[
            "N1", "beta",
            "var_term1", "var_ybar", "beta2_var_g1bar", "minus2beta_cov_ybar_g1bar",
            "var_term1_recon", "var_term1_recon_err",
            "corr_ybar_g1bar"
        ]].to_string(
            index=False,
            formatters={
                "beta": lambda x: f"{x:.8f}",
                "var_term1": lambda x: f"{x:.12e}",
                "var_ybar": lambda x: f"{x:.12e}",
                "beta2_var_g1bar": lambda x: f"{x:.12e}",
                "minus2beta_cov_ybar_g1bar": lambda x: f"{x:.12e}",
                "var_term1_recon": lambda x: f"{x:.12e}",
                "var_term1_recon_err": lambda x: f"{x:.3e}",
                "corr_ybar_g1bar": lambda x: f"{x:.6f}",
            }
        )
    )

    print("\n===== MC / QMC =====")
    print(
        df_base[["algo", "N1", "N2", "mean", "var"]].to_string(
            index=False,
            formatters={
                "mean": lambda x: f"{x:.8f}",
                "var": lambda x: f"{x:.12e}",
            }
        )
    )

    # ---------- 保存 ----------
    out_dir = os.path.abspath(args.results_dir)
    df_pemc.to_csv(os.path.join(out_dir, "eval_pemc_with_decomp.csv"), index=False)
    df_base.to_csv(os.path.join(out_dir, "eval_baselines.csv"), index=False)
    if not df_pemc_rep.empty:
        df_pemc_rep.to_csv(os.path.join(out_dir, "eval_pemc_rep_components.csv"), index=False)

    print(f"\n[OK] Saved CSVs to {out_dir}")
    print("[OK] - eval_pemc_with_decomp.csv")
    print("[OK] - eval_baselines.csv")
    if not df_pemc_rep.empty:
        print("[OK] - eval_pemc_rep_components.csv")


if __name__ == "__main__":
    main()