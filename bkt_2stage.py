# test_pemc_adaptive_decomp.py
# 兼容：
#   - 旧模型：2D训练（MSE），测试时 net(theta:[N,4], X:[N,d])
#   - 新模型：3D训练（RQMC group loss），测试时自动包装成 [1,N,*] 再过 net
#   - 二阶段训练版：支持加载 stage1 / stage2 / final 模型与 normalization
#
# 功能：
#   1) 测 PEMC / MC / QMC 的 mean / var
#   2) 对 PEMC 做方差分解：
#      Var(total) = Var(term1) + Var(term2) + 2Cov(term1, term2)
#   3) 对 term1 再分解（带 beta）：
#      term1 = ybar - beta * g1bar
#      Var(term1) = Var(ybar) + beta^2 Var(g1bar) - 2 beta Cov(ybar, g1bar)
#   4) beta 支持：
#      - file: 从 beta_map / beta_rows 读
#      - rep_term1: 用当前B的rep循环估计 beta*=Cov(ybar,g1bar)/Var(g1bar)  （推荐）
#      - rep_total: 用当前B的rep循环估计 total最优beta（对 total 方差最优）
#      - one: beta=1
#
# 依赖：
#   simulation_qmc.py:
#       sample_theta, simulate_gbm_batch_qmc, inv_Phi_torch,
#       arithmetic_payoff, Cholesky/PCA/BB, features_from_Z
#   model.py: PEMCNet

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


# ============================================================
# 0.5) 本地 normalize（与训练脚本一致，避免依赖训练文件名）
# ============================================================
def normalize(theta, X, norm):
    """
    与训练脚本保持一致：
      theta: [N,4] 或 [1,N,4] 或 [B,N,4]
      X:     [N,d] 或 [1,N,d] 或 [B,N,d]
    """
    theta = (theta - norm["theta_mean"]) / norm["theta_std"]
    X = (X - norm["X_mean"]) / norm["X_std"]
    return theta, X


# ============================================================
# 1) 加载训练结果（支持二阶段训练产物）
# ============================================================
def _load_cfg(cfg_path: str):
    if not os.path.exists(cfg_path):
        return SimpleNamespace()
    with open(cfg_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return SimpleNamespace(**d)


def _pick_model_and_norm_paths(
    results_dir: str,
    eval_stage: str = "final",
    model_name: str | None = None,
    norm_name: str | None = None,
):
    """
    二阶段训练产物兼容：
      - final  : model.pth + normalization.pth
      - stage1 : model_stage1_best.pth + normalization_stage1.pth
      - stage2 : 优先 model.pth / normalization_stage2.pth；不存在再回退 normalization.pth

    也支持手动覆盖：
      --model_name xxx.pth
      --norm_name  xxx.pth
    """
    results_dir = os.path.abspath(results_dir)

    # model path
    if model_name is not None:
        model_path = os.path.join(results_dir, model_name)
    else:
        if eval_stage == "stage1":
            model_path = os.path.join(results_dir, "model_stage1_best.pth")
        elif eval_stage in ("stage2", "final"):
            # 二阶段训练脚本通常把 stage2 best 存为 model.pth
            model_path = os.path.join(results_dir, "model.pth")
        else:
            raise ValueError(f"Unknown eval_stage: {eval_stage}")

    # norm path
    if norm_name is not None:
        norm_path = os.path.join(results_dir, norm_name)
    else:
        if eval_stage == "stage1":
            norm_path = os.path.join(results_dir, "normalization_stage1.pth")
        elif eval_stage == "stage2":
            norm_stage2 = os.path.join(results_dir, "normalization_stage2.pth")
            norm_final = os.path.join(results_dir, "normalization.pth")
            norm_path = norm_stage2 if os.path.exists(norm_stage2) else norm_final
        elif eval_stage == "final":
            norm_path = os.path.join(results_dir, "normalization.pth")
        else:
            raise ValueError(f"Unknown eval_stage: {eval_stage}")

    return model_path, norm_path


def load_all(
    results_dir: str,
    device: str | None = None,
    eval_stage: str = "final",
    model_name: str | None = None,
    norm_name: str | None = None,
):
    results_dir = os.path.abspath(results_dir)

    model_path, norm_path = _pick_model_and_norm_paths(
        results_dir=results_dir,
        eval_stage=eval_stage,
        model_name=model_name,
        norm_name=norm_name,
    )
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
    return net, norm, cfg, device, model_path, norm_path


# ============================================================
# 1.5) 加载 beta_rows / beta_map（用于对照或 file 模式）
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
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n <= 1:
        return float("nan")
    return float(x.var(ddof=1))


def _sample_cov(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
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


def _safe_beta_from_cov_var(cov_xz, var_z, default=np.nan):
    if (not np.isfinite(cov_xz)) or (not np.isfinite(var_z)) or abs(var_z) < 1e-30:
        return float(default)
    return float(cov_xz / var_z)


# ============================================================
# 4) payoff兼容包装（有的版本 arithmetic_payoff 需要 S0，有的不需要）
# ============================================================
def _call_payoff(S, K, S0=None):
    try:
        if S0 is None:
            return arithmetic_payoff(S, K)
        return arithmetic_payoff(S, K, S0)
    except TypeError:
        # 回退到二参数版本
        return arithmetic_payoff(S, K)


# ============================================================
# 5) PEMC 单次rep：先产出“原始分量”（与 beta 无关）
#    约定：
#      term1 用 N1=B 个点
#      term2 用 N2=(2^M)*B 个点
# ============================================================
@torch.no_grad()
def pemc_rep_raw_components(
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
    """
    返回一个rep下与 beta 无关的原始量:
      ybar, g1bar, g2bar
    后续任意 beta 都能重建:
      term1 = ybar - beta*g1bar
      term2 = beta*g2bar
      total = term1 + term2 = ybar + beta*(g2bar-g1bar)
    """
    device = cfg.device
    net.eval()
    norm_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in norm.items()}

    if B & (B - 1) != 0:
        raise ValueError(f"B must be a power of 2, got {B}")
    N2 = (2 ** M) * B
    if N2 & (N2 - 1) != 0:
        raise ValueError(f"N2=(2^M)*B must be a power of 2, got {N2}")

    r, S0, sigma, K = theta_tuple  # [B]

    # -------------------------
    # term1 部分需要 ybar, g1bar
    # -------------------------
    sobol_path1 = SobolEngine(
        dimension=nD,
        scramble=True,
        seed=int(cfg.seed) + 200000 + rep_seed
    )

    # 新接口：返回 Z, W, S, G
    Z1, W1, S1, G1 = simulate_gbm_batch_qmc(
        theta_tuple,
        nD=nD, T=T,
        device=device,
        method=method,
        sobol_engine=sobol_path1
    )

    # 新接口：features_from_Z 使用 (Z, dimX, theta, G, T)
    X1 = features_from_Z(
        Z1,
        dimX=cfg.dimX,
        theta=(r, S0, sigma, K),
        G=G1,
        T=T,
    )

    y1 = _call_payoff(S1, K, S0)  # [B]

    theta1_mat = torch.stack([r, S0, sigma, K], dim=-1)   # [B,4]
    g1 = predict_g_points_adaptive(net, norm_dev, cfg, theta1_mat, X1)  # [B]

    ybar = y1.mean()
    g1bar = g1.mean()

    # -------------------------
    # term2 部分需要 g2bar（固定theta，N2个点）
    # -------------------------
    r_fix, S0_fix, sigma_fix, K_fix = theta_1dim
    theta2_tuple = (
        r_fix.expand(N2),
        S0_fix.expand(N2),
        sigma_fix.expand(N2),
        K_fix.expand(N2),
    )

    sobol_path2 = SobolEngine(
        dimension=nD,
        scramble=True,
        seed=int(cfg.seed) + 300000 + rep_seed
    )

    Z2, W2, S2, G2 = simulate_gbm_batch_qmc(
        theta2_tuple,
        nD=nD, T=T,
        device=device,
        method=method,
        sobol_engine=sobol_path2
    )

    X2 = features_from_Z(
        Z2,
        dimX=cfg.dimX,
        theta=theta2_tuple,   # 关键：这里必须是 theta2_tuple
        G=G2,
        T=T,
    )

    theta2_row = torch.stack(list(theta_1dim), dim=0).to(device=device, dtype=torch.float32)  # [4]
    theta2_mat = theta2_row.unsqueeze(0).expand(N2, 4)  # [N2,4]

    g2 = predict_g_points_adaptive(net, norm_dev, cfg, theta2_mat, X2)  # [N2]
    g2bar = g2.mean()

    return {
        "ybar": float(ybar.item()),
        "g1bar": float(g1bar.item()),
        "g2bar": float(g2bar.item()),
    }


def _choose_beta_from_rep(
    ybar_arr: np.ndarray,
    g1bar_arr: np.ndarray,
    g2bar_arr: np.ndarray,
    beta_mode: str = "rep_term1",
    beta_file: float | None = None,
):
    """
    返回:
      beta_used, beta_rep_term1_star, beta_rep_total_star
    """
    ybar_arr = np.asarray(ybar_arr, dtype=np.float64)
    g1bar_arr = np.asarray(g1bar_arr, dtype=np.float64)
    g2bar_arr = np.asarray(g2bar_arr, dtype=np.float64)

    # term1最优：最小化 Var(ybar - beta*g1bar)
    var_g1 = _sample_var(g1bar_arr)
    cov_y_g1 = _sample_cov(ybar_arr, g1bar_arr)
    beta_rep_term1_star = _safe_beta_from_cov_var(cov_y_g1, var_g1, default=np.nan)

    # total最优：total = ybar + beta*(g2bar - g1bar)
    dbar = g2bar_arr - g1bar_arr
    var_d = _sample_var(dbar)
    cov_y_d = _sample_cov(ybar_arr, dbar)
    beta_rep_total_star = float("nan")
    if np.isfinite(var_d) and abs(var_d) > 1e-30 and np.isfinite(cov_y_d):
        beta_rep_total_star = float(- cov_y_d / var_d)

    if beta_mode == "rep_term1":
        beta_used = beta_rep_term1_star
    elif beta_mode == "rep_total":
        beta_used = beta_rep_total_star
    elif beta_mode == "file":
        beta_used = float(beta_file) if beta_file is not None else 1.0
    elif beta_mode == "one":
        beta_used = 1.0
    else:
        raise ValueError(f"未知 beta_mode: {beta_mode}")

    if (not np.isfinite(beta_used)):
        beta_used = 1.0

    return float(beta_used), float(beta_rep_term1_star), float(beta_rep_total_star)


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
    beta_mode: str = "rep_term1",   # 默认用rep-level term1最优beta
    beta_file: float | None = None, # file模式或对照用
    return_rep_df: bool = False,
):
    """
    核心改动：
      1) 先跑rep循环，收集原始分量(ybar,g1bar,g2bar)
      2) 用当前B的rep数据估计正确beta（默认rep_term1）
      3) 再构造 term1/term2/total 并做方差分解
    """
    raw_rows = []
    for rep in range(n_rep):
        out_raw = pemc_rep_raw_components(
            B=B, M=M,
            net=net, norm=norm, cfg=cfg,
            theta_1dim=theta_1dim,
            theta_tuple=theta_tuple,
            nD=nD, T=T, method=method,
            rep_seed=rep,
        )
        out_raw["rep"] = rep
        raw_rows.append(out_raw)

    rep_df = pd.DataFrame(raw_rows)

    ybar = rep_df["ybar"].to_numpy(dtype=np.float64)
    g1bar = rep_df["g1bar"].to_numpy(dtype=np.float64)
    g2bar = rep_df["g2bar"].to_numpy(dtype=np.float64)

    beta_used, beta_rep_term1_star, beta_rep_total_star = _choose_beta_from_rep(
        ybar_arr=ybar,
        g1bar_arr=g1bar,
        g2bar_arr=g2bar,
        beta_mode=beta_mode,
        beta_file=beta_file,
    )
    beta = float(beta_used)

    # 用选定 beta 重建各量
    term1 = ybar - beta * g1bar
    term2 = beta * g2bar
    total = term1 + term2

    rep_df["beta_used"] = beta
    rep_df["term1"] = term1
    rep_df["term2"] = term2
    rep_df["total"] = total

    # =====================================================
    # 总分解：Var(total) = Var(term1)+Var(term2)+2Cov(term1,term2)
    # =====================================================
    var_total = _sample_var(total)
    var_t1 = _sample_var(term1)
    var_t2 = _sample_var(term2)
    cov_t1_t2 = _sample_cov(term1, term2)
    corr_t1_t2 = _sample_corr(term1, term2)
    two_cov_t1_t2 = 2.0 * cov_t1_t2 if np.isfinite(cov_t1_t2) else float("nan")
    var_recon = (
        var_t1 + var_t2 + two_cov_t1_t2
        if np.isfinite(var_t1) and np.isfinite(var_t2) and np.isfinite(two_cov_t1_t2)
        else float("nan")
    )
    var_recon_err = (var_total - var_recon) if np.isfinite(var_total) and np.isfinite(var_recon) else float("nan")

    # =====================================================
    # term1 内部分解（带 beta）
    # t1 = ybar - beta*g1bar
    # =====================================================
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

    # 诊断：若 beta = Cov/Var(g1bar)，则 ratio 应接近 1-rho^2
    vr_term1_vs_ybar = float(var_t1 / var_ybar) if np.isfinite(var_t1) and np.isfinite(var_ybar) and abs(var_ybar) > 0 else float("nan")
    one_minus_rho2 = float(1.0 - corr_ybar_g1bar ** 2) if np.isfinite(corr_ybar_g1bar) else float("nan")
    vr_gap = float(vr_term1_vs_ybar - one_minus_rho2) if np.isfinite(vr_term1_vs_ybar) and np.isfinite(one_minus_rho2) else float("nan")

    # total分解占比
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

        # beta信息（核心）
        "beta_mode": beta_mode,
        "beta_file": float(beta_file) if (beta_file is not None and np.isfinite(beta_file)) else float("nan"),
        "beta": float(beta),  # 实际使用的beta
        "beta_rep_term1_star": float(beta_rep_term1_star),
        "beta_rep_total_star": float(beta_rep_total_star),

        "mean": float(total.mean()),
        "var": float(var_total),

        # PEMC总分解
        "term1_mean": float(term1.mean()),
        "term2_mean": float(term2.mean()),
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

        # term1 内部分解（带beta版本）
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

        # 关键诊断
        "vr_term1_vs_ybar": float(vr_term1_vs_ybar),
        "one_minus_rho2": float(one_minus_rho2),
        "vr_gap_term1_vs_1mrho2": float(vr_gap),
    }

    if return_rep_df:
        rep_df = rep_df.copy()
        rep_df["N1"] = B
        rep_df["N2"] = (2 ** M) * B
        rep_df["method"] = method
        rep_df["beta_mode"] = beta_mode
        rep_df["beta_file"] = float(beta_file) if beta_file is not None else np.nan
        rep_df["beta_rep_term1_star"] = float(beta_rep_term1_star)
        rep_df["beta_rep_total_star"] = float(beta_rep_total_star)
        rep_df["beta"] = float(beta)
        return summary, rep_df

    return summary


# ============================================================
# 6) Baseline: MC / RQMC
#    注意：QMC baseline 要与 PEMC 的 ybar 用同样seed规则，才会 var 对齐
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

    y = _call_payoff(S, K, S0)
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
    """
    为了和 PEMC 里的 ybar 完全对齐：
      - 使用同样的 SobolEngine seed: cfg.seed + 200000 + rep_seed
      - 使用同样的 simulate_gbm_batch_qmc 路径生成
    这样 qmc 的跨rep方差应与 PEMC分解里的 var_ybar 对齐。
    """
    device = cfg.device
    seed0 = int(getattr(cfg, "seed", 0))

    r, S0, sigma, K = theta
    r = r.to(device)
    S0 = S0.to(device)
    sigma = sigma.to(device)
    K = K.to(device)

    # 扩成 [B]，与 PEMC term1 的 ybar 完全同型
    theta_tuple_B = (
        r.expand(B),
        S0.expand(B),
        sigma.expand(B),
        K.expand(B),
    )

    if scramble:
        sob = SobolEngine(
            dimension=nD,
            scramble=True,
            seed=seed0 + 200000 + rep_seed   # 与 pemc term1 同 seed 规则
        )
    else:
        sob = SobolEngine(dimension=nD, scramble=False)

    Z, W, S, G = simulate_gbm_batch_qmc(
        theta_tuple_B,
        nD=nD,
        T=T,
        device=device,
        method=method,
        sobol_engine=sob
    )

    y = _call_payoff(S, K, S0)
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
# 7) 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.getcwd(), "results_lookback_two_stage"),
        help="训练结果目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda / cuda:0 ..."
    )

    # 新增：二阶段模型选择
    parser.add_argument(
        "--eval_stage",
        type=str,
        default="final",
        choices=["final", "stage1", "stage2"],
        help="加载哪个阶段的模型/归一化：final / stage1 / stage2"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="手动指定模型文件名（优先级高于 --eval_stage），例如 model_stage1_best.pth"
    )
    parser.add_argument(
        "--norm_name",
        type=str,
        default=None,
        help="手动指定归一化文件名（优先级高于 --eval_stage），例如 normalization_stage1.pth"
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
        default=200,
        help="重复次数（估方差）"
    )
    parser.add_argument(
        "--use_cli_method",
        action="store_true",
        help="若提供，则使用 --method；否则默认使用训练cfg.method"
    )
    parser.add_argument(
        "--beta_mode",
        type=str,
        default="rep_term1",
        choices=["rep_term1", "rep_total", "file", "one"],
        help="beta使用方式：默认按每个B的rep循环估计 term1 最优beta"
    )
    parser.add_argument(
        "--B_list",
        type=int,
        nargs="*",
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
        help="测试的B列表（空格分隔）"
    )
    args = parser.parse_args()

    net, norm, cfg, device, loaded_model_path, loaded_norm_path = load_all(
        args.results_dir,
        device=args.device,
        eval_stage=args.eval_stage,
        model_name=args.model_name,
        norm_name=args.norm_name,
    )

    # beta map（用于 file 模式，或作为对照打印）
    beta_map = load_beta_map(args.results_dir)

    # method选择：默认跟训练配置走；若显式给 --use_cli_method 则用命令行
    eval_method = args.method if args.use_cli_method else getattr(cfg, "method", args.method)

    print(f"[OK] Loaded net         : {loaded_model_path}")
    print(f"[OK] Loaded norm        : {loaded_norm_path}")
    print(f"[OK] eval_stage         : {args.eval_stage}")
    print(f"[OK] Device             : {device}")
    print(f"[OK] dimX               : {getattr(cfg, 'dimX', None)}")
    print(f"[OK] dropout            : {getattr(cfg, 'dropout', None)}")
    print(f"[OK] train method       : {getattr(cfg, 'method', None)}")
    print(f"[OK] eval method        : {eval_method}")
    print(f"[OK] Eval mode          : {_infer_eval_mode(norm, cfg)}")
    print(f"[OK] X_mean shape       : {tuple(norm['X_mean'].shape)}")
    print(f"[OK] theta_mean shape   : {tuple(norm['theta_mean'].shape)}")
    print(f"[OK] beta_mode          : {args.beta_mode}")
    print(f"[OK] n_rep              : {args.n_rep}")
    print(f"[OK] B_list             : {args.B_list}")

    if len(beta_map) == 0:
        print("[WARN] beta_map.json / beta_rows.json 未找到（file模式会回退 beta=1.0）")
    else:
        print(f"[OK] Loaded beta map    : {len(beta_map)} entries")
        print(f"[OK] beta keys          : {sorted(beta_map.keys())}")

    # 固定theta（单一合约）
    theta_fixed = (
        torch.tensor(0.02, device=device),    # r
        torch.tensor(100.0, device=device),   # S0
        torch.tensor(0.15, device=device),    # sigma
        torch.tensor(100.0, device=device),   # K
    )

    B_list = [int(x) for x in args.B_list]
    M = int(args.M)
    n_rep = int(args.n_rep)

    # ---------- PEMC（含方差分解，带beta） ----------
    rows_pemc = []
    rep_parts_all = []

    for B in B_list:
        beta_file_B = get_beta_for_B(beta_map, B, default=1.0)

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
            beta_mode=args.beta_mode,
            beta_file=beta_file_B,
            return_rep_df=True
        )

        rows_pemc.append({
            "algo": out["algo"],
            "method": out["method"],
            "N1": out["N1"],
            "N2": out["N2"],
            "n_rep": out["n_rep"],

            # beta信息
            "beta_mode": out["beta_mode"],
            "beta": out["beta"],  # 实际使用beta
            "beta_file": out["beta_file"],
            "beta_rep_term1_star": out["beta_rep_term1_star"],
            "beta_rep_total_star": out["beta_rep_total_star"],

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

            # 诊断：你关心的关系
            "vr_term1_vs_ybar": out["vr_term1_vs_ybar"],
            "one_minus_rho2": out["one_minus_rho2"],
            "vr_gap_term1_vs_1mrho2": out["vr_gap_term1_vs_1mrho2"],

            # 合约信息
            "r": float(theta_fixed[0].item()),
            "S0": float(theta_fixed[1].item()),
            "sigma": float(theta_fixed[2].item()),
            "K": float(theta_fixed[3].item()),
        })

        rep_parts_all.append(rep_df)
        beta_file_val = out["beta_file"]
        beta_file_str = f"{beta_file_val:.8f}" if np.isfinite(beta_file_val) else "nan"

        print(
            f"[PEMC] done B={B:5d} | "
            f"beta_mode={args.beta_mode} | "
            f"beta_used={out['beta']:.8f} | "
            f"beta_file={beta_file_str} | "
            f"beta_rep_term1*={out['beta_rep_term1_star']:.8f} | "
            f"corr(ybar,g1bar)={out['corr_ybar_g1bar']:.6f}"
        )

    df_pemc = pd.DataFrame(rows_pemc)
    df_pemc_rep = pd.concat(rep_parts_all, axis=0, ignore_index=True) if len(rep_parts_all) > 0 else pd.DataFrame()

    # ---------- Baselines ----------
    # QMC baseline 的方差应与 PEMC 里的 var_ybar 对齐（同seed规则、同路径生成）
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
        df_pemc[["N1", "N2", "beta_mode", "beta", "beta_file", "beta_rep_term1_star", "mean", "var"]].to_string(
            index=False,
            formatters={
                "beta": lambda x: f"{x:.8f}",
                "beta_file": lambda x: f"{x:.8f}" if np.isfinite(x) else "nan",
                "beta_rep_term1_star": lambda x: f"{x:.8f}" if np.isfinite(x) else "nan",
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
            "corr_ybar_g1bar",
            "vr_term1_vs_ybar", "one_minus_rho2", "vr_gap_term1_vs_1mrho2"
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
                "vr_term1_vs_ybar": lambda x: f"{x:.6f}",
                "one_minus_rho2": lambda x: f"{x:.6f}",
                "vr_gap_term1_vs_1mrho2": lambda x: f"{x:.3e}",
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

    # ---------- 额外核对：QMC var vs PEMC var_ybar ----------
    if not df_pemc.empty and not df_base.empty:
        df_qmc = df_base[df_base["algo"] == "qmc"][["N1", "var"]].rename(columns={"var": "var_qmc"})
        df_chk = df_pemc[["N1", "var_ybar"]].merge(df_qmc, on="N1", how="left")
        if not df_chk.empty:
            df_chk["diff_qmc_minus_var_ybar"] = df_chk["var_qmc"] - df_chk["var_ybar"]
            df_chk["ratio_qmc_over_var_ybar"] = df_chk["var_qmc"] / df_chk["var_ybar"]
            print("\n===== Check: qmc var vs var_ybar（应接近1）=====")
            print(
                df_chk.to_string(
                    index=False,
                    formatters={
                        "var_ybar": lambda x: f"{x:.12e}",
                        "var_qmc": lambda x: f"{x:.12e}",
                        "diff_qmc_minus_var_ybar": lambda x: f"{x:.3e}",
                        "ratio_qmc_over_var_ybar": lambda x: f"{x:.6f}" if np.isfinite(x) else "nan",
                    }
                )
            )

    # ---------- 保存 ----------
    out_dir = os.path.abspath(args.results_dir)

    # 文件名带 stage 标记，避免你跑 stage1/stage2 时互相覆盖
    suffix = f"_{args.eval_stage}"
    if args.model_name is not None or args.norm_name is not None:
        suffix += "_custom"

    df_pemc.to_csv(os.path.join(out_dir, f"eval_pemc_with_decomp{suffix}.csv"), index=False)
    df_base.to_csv(os.path.join(out_dir, f"eval_baselines{suffix}.csv"), index=False)
    if not df_pemc_rep.empty:
        df_pemc_rep.to_csv(os.path.join(out_dir, f"eval_pemc_rep_components{suffix}.csv"), index=False)

    print(f"\n[OK] Saved CSVs to {out_dir}")
    print(f"[OK] - eval_pemc_with_decomp{suffix}.csv")
    print(f"[OK] - eval_baselines{suffix}.csv")
    if not df_pemc_rep.empty:
        print(f"[OK] - eval_pemc_rep_components{suffix}.csv")


if __name__ == "__main__":
    main()