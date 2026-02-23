import os
import argparse
from types import SimpleNamespace
from simulation_qmc import *
import yaml
from model import PEMCNet
import os, math
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from train_model_qmc_pca_mse import normalize
import pandas as pd
def _load_cfg(cfg_path: str):
    if not os.path.exists(cfg_path):
        return SimpleNamespace()
    with open(cfg_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    # dict -> namespace，方便 cfg.xxx
    return SimpleNamespace(**d)


def load_all(results_dir: str, device: str | None = None):
    results_dir = os.path.abspath(results_dir)

    model_path = os.path.join(results_dir, "model.pth")
    norm_path  = os.path.join(results_dir, "normalization.pth")
    cfg_path   = os.path.join(results_dir, "config.yaml")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing normalization file: {norm_path}")

    cfg = _load_cfg(cfg_path)

    device = device or cfg.device
    dimX = cfg.dimX
    dropout = cfg.dropout

    net = PEMCNet(dimX=dimX, dropout=dropout).to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    norm = torch.load(norm_path, map_location=device)

    return net, norm, cfg, device

"""
B是样本量，M是 2^M倍样本量辅助估计，net是函数g，norm是归一化常数，
cfg是各类参数，nD是路径上时间点个数，T是总时间，method是生成矩阵，rep_seed是重复pemc估计用的随机种子

返回：一个标量 PEMC 估计
约定：第二项样本量 N2 = (2^M)*B
"""

# ====== 一次 PEMC 估计：只用一个 theta ======
@torch.no_grad()
def pemc_estimate(
    B: int, M: int,
    net, norm: dict, cfg,
    theta_1dim,theta_tuple,                     # <-- 外面传进来，固定不变
    nD: int = 256, T: float = 1.0,
    method: str = "pca",
    rep_seed: int = 0,
):
    device = cfg.device
    net.eval()
    norm_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in norm.items()}

    if B & (B - 1) != 0:
        raise ValueError(f"B must be a power of 2, got {B}")
    N2 = (2 ** M) * B
    if N2 & (N2 - 1) != 0:
        raise ValueError(f"N2=(2^M)*B must be a power of 2, got {N2}")

    r, S0, sigma, K = theta_tuple

    # =========================
    # term1：1/B sum (f(Y)-g)
    # =========================
    sobol_path = SobolEngine(dimension=nD, scramble=True,
                             seed=int(cfg.seed) + 200000 + rep_seed)

    # 关键：如果 simulate_gbm_batch_qmc 已经接收 sobol_engine，
    #      尽量别再传 seed/scramble，避免内部覆盖你的 engine
    Z1, W1, S1 = simulate_gbm_batch_qmc(
        theta_tuple, nD=nD, T=T, device=device, method=method,
        sobol_engine=sobol_path
    )

    X1 = Z1[:, :cfg.dimX]
    y1 = arithmetic_payoff(S1, K)

    theta1_mat = torch.stack([r,S0,sigma,K],dim= 1)   # [B,4]
    theta1_norm, X1_norm = normalize(theta1_mat, X1, norm_dev)
    g1 = net(theta1_norm, X1_norm).squeeze(-1)  # [8192]
    term1 = (y1 - g1).mean()


    # =========================
    # term2：1/N2 sum g(theta, X~)
    # =========================
    sobol_x = SobolEngine(dimension=cfg.dimX, scramble=True,
                          seed=int(cfg.seed) + 300000 + rep_seed)

    U2 = sobol_x.draw(N2).to(device=device, dtype=torch.float32)
    X2 = inv_Phi_torch(U2)

    theta2_mat = torch.stack(theta_1dim).unsqueeze(0).repeat(N2, 1)   # [B, 4]  # ✅ 同一个 theta（也修掉你之前写成 theta1 的坑）
    theta2_norm, X2_norm = normalize(theta2_mat, X2, norm_dev)
    g2 = net(theta2_norm, X2_norm)
    term2 = g2.mean()

    return (term1 + term2).item()
    # return term1

def pemc_mean_var(B, M, n_rep, net, norm, cfg, theta_1dim,theta_tuple, nD=256, T=1.0, method="pca"):
    est = []
    for rep in range(n_rep):
        est.append(pemc_estimate(B, M, net, norm, cfg, theta_1dim=theta_1dim ,theta_tuple=theta_tuple,
                                 nD=nD, T=T, method=method, rep_seed=rep))
    est = np.array(est, dtype=np.float64)
    return {
        "B": B, "M": M, "N2": (2**M)*B, "n_rep": n_rep,
        "mean": float(est.mean()),
        "var": float(est.var(ddof=1)) if n_rep > 1 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.getcwd(), "results_qmc_pca_mse"),
        help="path to the results directory (default: CWD/results_qmc_pca_mse)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda / cuda:0 ... (optional)"
    )
    args = parser.parse_args()

    net, norm, cfg, device = load_all(args.results_dir, args.device)

    print(f"[OK] Loaded net from: {os.path.join(os.path.abspath(args.results_dir), 'model.pth')}")
    print(f"[OK] Loaded norm from: {os.path.join(os.path.abspath(args.results_dir), 'normalization.pth')}")
    print(f"[OK] Device: {device}")
    print(f"[OK] dimX={getattr(cfg, 'dimX', getattr(cfg, 'dim_x', 16))}, dropout={getattr(cfg, 'dropout', 0.0)}")

    # main 里：最外层固定 theta


    B_list = [128, 256, 512, 1024,2048,4096,8192]
    M = 4
    n_rep = 100
    theta_fixed = (
        torch.tensor(0.02, device=device),  # r
        torch.tensor(100.0, device=device),  # S0
        torch.tensor(0.15, device=device),  # sigma
        torch.tensor(100.0, device=device),  # K
    )


    rows = []
    for B in B_list:
        r, S0, sigma, K = sample_theta(
            mode=2,
            batch_size=B,
            is_same=True,
            theta_same=theta_fixed,
            device=device
        )
        out = pemc_mean_var(B, M, n_rep, net, norm, cfg,theta_1dim=theta_fixed, theta_tuple = (r,S0,sigma,K), nD=256, T=1.0, method="pca")
        rows.append({
            "method": "pca",
            "N1": out["B"],
            "N2": out["N2"],
            "n_rep": out["n_rep"],
            "mean": out["mean"],
            "var": out["var"],
            # 可选：把 theta 也存进去（方便复现实验）
            "r": float(theta_fixed[0].item()),
            "S0": float(theta_fixed[1].item()),
            "sigma": float(theta_fixed[2].item()),
            "K": float(theta_fixed[3].item()),
        })
        print("B:",B)

    df = pd.DataFrame(rows)

    n_rep = 100

    rows = []
    for B in B_list:
        rows.append(mc_mean_var(B, n_rep, cfg, theta_fixed, nD=256, T=1.0, method="pca"))
        rows.append(qmc_mean_var(B, n_rep, cfg, theta_fixed, nD=256, T=1.0, method="pca"))

    df_baseline = pd.DataFrame(rows)
    df_baseline["r"] = float(theta_fixed[0].item())
    df_baseline["S0"] = float(theta_fixed[1].item())
    df_baseline["sigma"] = float(theta_fixed[2].item())
    df_baseline["K"] = float(theta_fixed[3].item())

    pd.set_option("display.float_format", lambda x: f"{x:.12e}")
    print(df[["N1", "N2", "mean", "var"]].to_string(
        index=False,
        formatters={
            "mean": lambda x: f"{x:.8f}",
            "var": lambda x: f"{x:.12e}",  # var用科学计数法最稳
        }
    ))
    print(df_baseline[["N1", "N2", "mean", "var"]].to_string(
        index=False,
        formatters={
            "mean": lambda x: f"{x:.8f}",
            "var": lambda x: f"{x:.12e}",  # var用科学计数法最稳
        }
    ))
import numpy as np




























import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
def _generator_matrix(method: str, nD: int, T: float, device: str, dtype=torch.float32) -> torch.Tensor:
    """
    选生成矩阵
    """
    method = method.lower()
    if method == "cholesky":
        G = Cholesky(nD, T=T)
    elif method == "pca":
        G = PCA(nD, T=T)
    elif method == "bb":
        if nD <= 0 or (nD & (nD - 1)) != 0:
            raise ValueError("BB requires nD to be a positive power of 2 (e.g., 256).")
        n = nD.bit_length() - 1  # 因为 nD=2^n 时，二进制长度刚好是 n+1
        G = BB(n, T=T)
    else:
        raise ValueError("method must be one of {'cholesky','pca','bb'}")
    return torch.tensor(G, device=device, dtype=dtype)
# ===== MC：一次估计（固定 theta，只变 rep_seed）=====
@torch.no_grad()
def mc_estimate(B, cfg, theta, nD=256, T=1.0, method="pca", rep_seed=0):
    device = cfg.device
    seed0 = int(getattr(cfg, "seed", 0))

    r, S0, sigma, K = theta
    r = r.to(device); S0 = S0.to(device); sigma = sigma.to(device); K = K.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed0 + 400000 + rep_seed)

    Z = torch.randn((B, nD), device=device, generator=gen, dtype=torch.float32)  # i.i.d. N(0,1)

    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)  # [nD,nD]
    W = Z @ G.T  # [B,nD]  Brownian motion at t_1..t_nD

    t = torch.linspace(T / nD, T, nD, device=device, dtype=torch.float32).view(1, nD)
    S = S0 * torch.exp((r - 0.5 * sigma * sigma) * t + sigma * W)  # [B,nD]

    y = arithmetic_payoff(S, K)  # [B]
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


# ===== QMC：一次估计（RQMC 用 scramble=True + rep_seed）=====
@torch.no_grad()
def qmc_estimate(B, cfg, theta, nD=256, T=1.0, method="pca", rep_seed=0, scramble=True):
    device = cfg.device
    seed0 = int(getattr(cfg, "seed", 0))

    r, S0, sigma, K = theta
    r = r.to(device); S0 = S0.to(device); sigma = sigma.to(device); K = K.to(device)

    # Sobol -> U -> Z = Phi^{-1}(U)
    if scramble:
        sob = SobolEngine(dimension=nD, scramble=True, seed=seed0 + 500000 + rep_seed)
    else:
        sob = SobolEngine(dimension=nD, scramble=False)

    U = sob.draw(B).to(device=device, dtype=torch.float32)  # [B,nD] in [0,1)
    Z = inv_Phi_torch(U)                                  # [B,nD] ~ N(0,1)

    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)
    W = Z @ G.T

    t = torch.linspace(T / nD, T, nD, device=device, dtype=torch.float32).view(1, nD)
    S = S0 * torch.exp((r - 0.5 * sigma * sigma) * t + sigma * W)

    y = arithmetic_payoff(S, K)
    return float(y.mean().item())


def qmc_mean_var(B, n_rep, cfg, theta, nD=256, T=1.0, method="pca"):
    # 这里用 RQMC：scramble=True，rep_seed 变化 => 才能估方差
    est = [qmc_estimate(B, cfg, theta, nD=nD, T=T, method=method, rep_seed=rep, scramble=True)
           for rep in range(n_rep)]
    est = np.array(est, dtype=np.float64)
    return {
        "algo": "qmc",   # 更准确叫 rqmc，但你想写 qmc 也行
        "method": method,
        "N1": B,
        "N2": 0,
        "n_rep": n_rep,
        "mean": float(est.mean()),
        "var": float(est.var(ddof=1)) if n_rep > 1 else float("nan"),
    }

if __name__ == "__main__":
    main()

