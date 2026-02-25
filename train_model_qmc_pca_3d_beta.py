import os
import json
import random
import shutil
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import yaml

from model import PEMCNet
from simulation_qmc import *


# =============================
# 0) Config
# =============================
class Config:
    def __init__(self):
        # ===== Reproducibility / Device =====
        self.results_dir_name = "results_lookback_Xdim1_N6_loss2"

        self.seed = 44
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== Simulation / Feature =====
        self.method = "pca"      # 路径构造方法（示例：pca / bb / cholesky）
        self.dimX = 4      # 特征维度（从Z截断）
        self.thetadim = 2       # 2 或 3（传给 sample_theta 的 mode）
        self.N = 128           # 训练数据中每组QMC点数（仅训练/数据生成用，不等于beta曲线中的N）

        # ===== Dataset =====
        self.dataset_size = 2 ** 16  # 3D模式下表示“组数B_total”
        self.train_ratio = 0.7
        self.val_ratio = 0.15

        # ===== Training =====
        self.batch_size = 512   # 3D模式下单位是“组数”
        self.epochs = 200
        self.lr = 1e-3
        self.dropout = 0.3

        # ===== Loss (RQMC group loss) =====
        self.rqmc_loss_center = True
        self.rqmc_loss_unbiased = False

        # ===== Beta estimation (post-training) =====
        # 注意：这里的 N_list 才是“beta曲线”的 N，和 cfg.N 没直接绑定关系
        self.beta_N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
        self.B_beta = 2 ** 7  # 每个N用于估计beta的组数（rep数）

        # ===== Save =====

# =============================
# 1) Paths / Globals
# =============================
cfg = Config()
current_working_directory = os.getcwd()
print("CWD =", current_working_directory)

results_dir = os.path.join(current_working_directory, cfg.results_dir_name)
print("Results directory:", results_dir)


# =============================
# 2) Utility
# =============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_plain_python(obj):
    """把 numpy / torch 类型转成可 json/yaml 序列化的原生 Python 类型。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [to_plain_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain_python(v) for k, v in obj.items()}
    return obj


def cfg_to_dict(cfg_obj):
    out = {}
    for k, v in vars(cfg_obj).items():
        out[k] = to_plain_python(v)
    out["device"] = str(out.get("device", "cpu"))
    return out


def generate_dataset(cfg, theta):
    """
    theta: (r, S0, sigma, K)
      - 2D模式时每个参数 shape [M]
      - 3D模式时每个参数 shape [B,N]
    返回 TensorDataset(theta, X, PA)
      - 2D: theta [M,4],   X [M,d],   y [M,1]
      - 3D: theta [B,N,4], X [B,N,d], y [B,N,1]
    """
    r, S0, sigma, K = theta

    Z, W, S = simulate_gbm_batch_qmc(
        (r, S0, sigma, K),
        method=cfg.method,
        device=cfg.device,
        seed=cfg.seed
    )
    PA = arithmetic_payoff(S, K).unsqueeze(-1)   # 2D->[M,1], 3D->[B,N,1]
    X = features_from_Z(Z, dimX=cfg.dimX)

    # 关键：dim=-1 才能同时兼容 2D/3D
    theta_tensor = torch.stack([r, S0, sigma, K], dim=-1)

    dataset = TensorDataset(theta_tensor, X, PA)
    return dataset


def split_dataset(dataset, cfg):
    N_total = len(dataset)
    train_size = int(cfg.train_ratio * N_total)
    val_size = int(cfg.val_ratio * N_total)
    test_size = N_total - train_size - val_size
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )


def _reduce_dims_except_last(x: torch.Tensor):
    """
    对最后一维视作特征维，其余维都做统计。
    2D: [M,d]    -> reduce dims=(0,)
    3D: [B,N,d]  -> reduce dims=(0,1)
    """
    if x.ndim < 2:
        raise ValueError(f"期望至少2维张量（最后一维是特征维），got shape={x.shape}")
    return tuple(range(x.ndim - 1))


def compute_normalization(train_set, cfg):
    """
    兼容:
      - 2D数据: theta [M,4],   X [M,d],   y [M,1]
      - 3D数据: theta [B,N,4], X [B,N,d], y [B,N,1]
    只对 theta / X 做标准化
    """
    theta_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    X_train = torch.stack([train_set[i][1] for i in range(len(train_set))])

    theta_dims = _reduce_dims_except_last(theta_train)
    X_dims = _reduce_dims_except_last(X_train)

    norm = {
        "theta_mean": theta_train.mean(dim=theta_dims, keepdim=True),
        "theta_std": theta_train.std(dim=theta_dims, keepdim=True, unbiased=False).clamp_min(1e-6),
        "X_mean": X_train.mean(dim=X_dims, keepdim=True),
        "X_std": X_train.std(dim=X_dims, keepdim=True, unbiased=False).clamp_min(1e-6),
    }

    for k in norm:
        norm[k] = norm[k].to(cfg.device)

    return norm


def normalize(theta, X, norm):
    """
    兼容:
      theta: [M,4] 或 [B,N,4] 或 [B,4]
      X:     [M,d] 或 [B,N,d]
    """
    theta = (theta - norm["theta_mean"]) / norm["theta_std"]
    X = (X - norm["X_mean"]) / norm["X_std"]
    return theta, X


# =============================
# 3) Loss helpers
# =============================
def rqmc_group_var_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    center: bool = True,
    unbiased: bool = False,
    keepdim_last: bool = True,
    return_stats: bool = False,
):
    """
    pred: [B, N, 1] 或 [B, N]
    y:    [B, N, 1] 或 [B, N]

    center=True:
        loss = 方差版本（推荐）
             ~= Var_b( mean_i (y_{b,i} - pred_{b,i}) )

    center=False:
        loss = 二阶矩版本
             = E_b[hbar_b^2]
             = Var(hbar) + (E[hbar])^2
    """
    # 统一成 [B, N, 1]
    if pred.ndim == 2:
        pred = pred.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)

    assert pred.shape == y.shape, f"pred/y shape mismatch: {pred.shape} vs {y.shape}"
    assert pred.ndim == 3, f"pred/y 应为 [B,N,1] 或 [B,N]，got {pred.shape}"

    B, N, C = pred.shape
    assert C == 1, f"目前按标量输出写的，最后一维应为1，got {C}"

    h = y - pred          # [B,N,1]
    hbar = h.mean(dim=1)  # [B,1]

    if center:
        if unbiased:
            loss = hbar.var(dim=0, unbiased=True).mean()
        else:
            hbar_centered = hbar - hbar.mean(dim=0, keepdim=True)
            loss = (hbar_centered ** 2).mean()
    else:
        loss = (hbar ** 2).mean()

    if return_stats:
        with torch.no_grad():
            stats = {
                "B": int(B),
                "N": int(N),
                "hbar_mean": float(hbar.mean().item()),
                "hbar_var_pop": float(((hbar - hbar.mean()) ** 2).mean().item()),
                "hbar_var_sample": float(hbar.var(unbiased=True).item()) if B > 1 else float("nan"),
                "h_mean": float(h.mean().item()),
            }
        return loss, stats

    return loss


def _batch_loss_auto(pred, y, X, cfg, for_eval=False):
    """
    pred, y:
      - 2D模式: [M,1] / [M]
      - 3D模式: [B,N,1] / [B,N]
    X:
      - 2D模式: [M,d]
      - 3D模式: [B,N,d]
    """
    if X.ndim == 2:
        # 点级 MSE
        if for_eval:
            loss = F.mse_loss(pred, y, reduction="sum")
            weight = y.size(0)
            return loss, weight
        else:
            return F.mse_loss(pred, y)

    elif X.ndim == 3:
        # RQMC 分组方差损失
        center = getattr(cfg, "rqmc_loss_center", True)
        unbiased = getattr(cfg, "rqmc_loss_unbiased", False)

        loss = rqmc_group_var_loss(
            pred=pred,
            y=y,
            center=center,
            unbiased=unbiased,
            return_stats=False,
        )

        if for_eval:
            weight = X.size(0)  # 用组数B做权重
            return loss, weight
        else:
            return loss

    else:
        raise ValueError(f"不支持的 X 维度: {X.ndim}，期望 2 或 3")


# =============================
# 4) Train / Eval
# =============================
def train_model(train_loader, val_loader, norm, cfg):
    net = PEMCNet(dimX=cfg.dimX, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    first_step_loss = None

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    for epoch in range(cfg.epochs):
        # ===== Train =====
        net.train()
        for step, (theta, X, y) in enumerate(train_loader):
            theta = theta.to(cfg.device)
            X = X.to(cfg.device)
            y = y.to(cfg.device)

            theta, X = normalize(theta, X, norm)

            pred = net(theta, X)
            loss = _batch_loss_auto(pred, y, X, cfg, for_eval=False)

            if epoch == 0 and step == 0:
                first_step_loss = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # ===== Val =====
        net.eval()
        val_sum, val_weight = 0.0, 0
        mode_name = None

        with torch.no_grad():
            for theta, X, y in val_loader:
                theta = theta.to(cfg.device)
                X = X.to(cfg.device)
                y = y.to(cfg.device)

                theta, X = normalize(theta, X, norm)
                pred = net(theta, X)

                loss, weight = _batch_loss_auto(pred, y, X, cfg, for_eval=True)
                val_sum += loss.item()
                val_weight += weight

                mode_name = "RQMC-group-var" if X.ndim == 3 else "MSE"

        val_loss = val_sum / max(val_weight, 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        print(f"Epoch {epoch+1} | Val {mode_name}: {val_loss:.6f} | Best: {best_val_loss:.6f} (epoch {best_epoch})")

    return net, first_step_loss, train_losses, val_losses, best_state_dict, best_val_loss, best_epoch


def evaluate(net, loader, norm, cfg):
    net.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for theta, X, y in loader:
            theta = theta.to(cfg.device)
            X = X.to(cfg.device)
            y = y.to(cfg.device)

            theta, X = normalize(theta, X, norm)

            pred = net(theta, X)
            loss, weight = _batch_loss_auto(pred, y, X, cfg, for_eval=True)

            total += loss.item()
            count += weight

    return total / max(count, 1)


# =============================
# 5) Beta statistics
# =============================
def _np_sample_var(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        return float("nan")
    return float(x.var(ddof=1))


def _np_sample_cov(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"len mismatch: {x.size} vs {y.size}")
    n = x.size
    if n <= 1:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    return float((xm * ym).sum() / (n - 1))


def _np_sample_corr(x, y, eps=1e-30):
    vx = _np_sample_var(x)
    vy = _np_sample_var(y)
    if (not np.isfinite(vx)) or (not np.isfinite(vy)):
        return float("nan")
    den = float(np.sqrt(max(vx, 0.0) * max(vy, 0.0)))
    if den < eps:
        return float("nan")
    return float(_np_sample_cov(x, y) / den)


@torch.no_grad()
def estimate_beta_on_3d_dataset(net, dataset, norm, cfg, loader_batch_size=None):
    """
    dataset: TensorDataset(theta, X, y), 且应为3D样本:
      theta [B,N,4], X [B,N,d], y [B,N,1]
    """
    if loader_batch_size is None:
        loader_batch_size = min(256, len(dataset)) if len(dataset) > 0 else 1

    loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False)
    net.eval()

    ybar_all = []
    gbar_all = []

    for theta, X, y in loader:
        theta = theta.to(cfg.device)
        X = X.to(cfg.device)
        y = y.to(cfg.device)

        if X.ndim != 3:
            raise ValueError(f"这里要求3D数据 [B,N,d]，但拿到 X.shape={X.shape}")

        theta_n, X_n = normalize(theta, X, norm)
        pred = net(theta_n, X_n)

        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)

        assert pred.shape == y.shape, f"pred/y shape mismatch: {pred.shape} vs {y.shape}"

        ybar = y.mean(dim=1).squeeze(-1)    # [B]
        gbar = pred.mean(dim=1).squeeze(-1) # [B]

        ybar_all.append(ybar.detach().cpu().numpy())
        gbar_all.append(gbar.detach().cpu().numpy())

    ybar_all = np.concatenate(ybar_all, axis=0).astype(np.float64)
    gbar_all = np.concatenate(gbar_all, axis=0).astype(np.float64)

    var_y = _np_sample_var(ybar_all)
    var_g = _np_sample_var(gbar_all)
    cov_yg = _np_sample_cov(ybar_all, gbar_all)
    corr_yg = _np_sample_corr(ybar_all, gbar_all)

    if (not np.isfinite(var_g)) or abs(var_g) < 1e-30:
        beta = float("nan")
    else:
        beta = float(cov_yg / var_g)

    alpha = float(ybar_all.mean() - beta * gbar_all.mean()) if np.isfinite(beta) else float("nan")

    if np.isfinite(beta):
        resid = ybar_all - beta * gbar_all
        var_resid = _np_sample_var(resid)
    else:
        var_resid = float("nan")

    return {
        "n_groups": int(ybar_all.size),
        "mean_ybar": float(ybar_all.mean()),
        "mean_gbar": float(gbar_all.mean()),
        "var_ybar": float(var_y),
        "var_gbar": float(var_g),
        "cov_ybar_gbar": float(cov_yg),
        "corr_ybar_gbar": float(corr_yg),
        "beta_cv": float(beta),
        "alpha_ols": float(alpha),
        "var_ybar_minus_beta_gbar": float(var_resid),
        "vr_ratio_vs_ybar": float(var_resid / var_y) if np.isfinite(var_resid) and np.isfinite(var_y) and abs(var_y) > 0 else float("nan"),
    }


@torch.no_grad()
def estimate_beta_curve_by_N(net, norm, cfg, theta_fixed, N_list, B_beta=None, loader_batch_size=None):
    """
    对多个 N 估计 beta_N。每个 N 都重新生成一套 3D 数据（固定 theta）。
    注意：这里的 N_list 与 cfg.N（训练数据组内点数）可以完全无关。
    """
    rows = []

    if B_beta is None:
        B_beta = int(cfg.dataset_size)

    for N_group in N_list:
        # 生成固定theta的 3D 参数 [B_beta, N_group]
        r, S0, sigma, K = sample_theta(
            mode=3,
            batch_size=B_beta,   # 兼容你原接口
            B=B_beta,
            N=int(N_group),
            is_same=True,        # 固定同一个theta，广播到所有组/点
            theta_same=theta_fixed,
            device=cfg.device
        )

        # 临时改 cfg.N，让 generate_dataset 内部使用当前 N_group
        old_N = getattr(cfg, "N", None)
        cfg.N = int(N_group)

        dataset_beta = generate_dataset(cfg, theta=(r, S0, sigma, K))
        stats = estimate_beta_on_3d_dataset(
            net=net,
            dataset=dataset_beta,
            norm=norm,
            cfg=cfg,
            loader_batch_size=loader_batch_size
        )

        # 恢复 cfg.N
        if old_N is not None:
            cfg.N = old_N

        row = {
            "N": int(N_group),
            "B_beta": int(B_beta),
            **stats
        }
        rows.append(row)

        print(
            f"[Beta] N={N_group:5d} | "
            f"beta={row['beta_cv']:.8f} | "
            f"corr={row['corr_ybar_gbar']:.6f} | "
            f"vr_ratio={row['vr_ratio_vs_ybar']:.6f}"
        )

    return rows


# =============================
# 6) Main
# =============================
def main():
    cfg = Config()
    set_seed(cfg.seed)

    # ===== 固定一个测试合约 theta（用于 beta 曲线估计）=====
    theta_fixed = (
        torch.tensor(0.02, device=cfg.device),    # r
        torch.tensor(100.0, device=cfg.device),   # S0
        torch.tensor(0.15, device=cfg.device),    # sigma
        torch.tensor(100, device=cfg.device),   # K
    )

    # ===== 训练数据生成（这里用 cfg.N，仅用于训练/验证/测试数据）=====
    N_group_train = cfg.N

    r, S0, sigma, K = sample_theta(
        mode=cfg.thetadim,
        batch_size=cfg.dataset_size,
        B=cfg.dataset_size,
        N=N_group_train,
        is_same=False,          # 这里你原来写的是 False；保留
        theta_same=theta_fixed, # 当 is_same=False 时通常不会用到
        device=cfg.device
    )

    dataset = generate_dataset(cfg, theta=(r, S0, sigma, K))

    # 划分：len(dataset) 在3D模式下是组数 B_total
    train_set, val_set, test_set = split_dataset(dataset, cfg)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    norm = compute_normalization(train_set, cfg)

    # ===== 训练 =====
    net, first_step_loss, train_losses, val_losses, best_state_dict, best_val_loss, best_epoch = train_model(
        train_loader, val_loader, norm, cfg
    )

    # 使用 best-val checkpoint 做测试
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    test_loss = evaluate(net, test_loader, norm, cfg)

    print("First step loss:", first_step_loss)
    print(f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print("Final Test Loss (using best-val checkpoint):", test_loss)

    # ===== 估计 beta(N) 曲线（N 与 cfg.N 可无关）=====
    print("\n===== Estimate beta(N) on fresh 3D datasets (fixed theta) =====")
    beta_rows = estimate_beta_curve_by_N(
        net=net,
        norm=norm,
        cfg=cfg,
        theta_fixed=theta_fixed,
        N_list=cfg.beta_N_list,
        B_beta=cfg.B_beta,
        loader_batch_size=min(256, cfg.B_beta)
    )

    # 可选：构建一个 {N: beta} 映射，方便后续测试脚本直接读
    beta_map = {
        int(row["N"]): float(row["beta_cv"])
        for row in beta_rows
        if np.isfinite(row.get("beta_cv", np.nan))
    }

    print("[Beta map keys]", sorted(beta_map.keys()))

    # =============================
    # 7) Save
    # =============================
    os.makedirs(results_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 这里 net 已经 load 了 best_state_dict，所以保存的是 best-val 模型
    model_state_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    norm_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm.items()}

    # ---- model / norm / losses ----
    print("[Save] model.pth ...")
    torch.save(model_state_cpu, os.path.join(results_dir, "model.pth"))

    print("[Save] normalization.pth ...")
    torch.save(norm_cpu, os.path.join(results_dir, "normalization.pth"))

    print("[Save] train_losses.npy / val_losses.npy ...")
    np.save(os.path.join(results_dir, "train_losses.npy"), np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(results_dir, "val_losses.npy"), np.array(val_losses, dtype=np.float32))

    # ---- config ----
    # 注意：cfg.N 是训练N；beta曲线的N在 beta_rows / beta_N_list 里
    print("[Save] config.yaml ...")
    cfg_dict = cfg_to_dict(cfg)
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ---- summary ----
    print("[Save] summary.json ...")
    summary = {
        "first_step_loss": float(first_step_loss) if first_step_loss is not None else None,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "num_train_steps": int(len(train_losses)),
        "num_epochs": int(cfg.epochs),

        # 记录beta估计任务配置（不是beta值本身）
        "beta_num_points": int(len(beta_rows)),
        "beta_N_list": [int(n) for n in cfg.beta_N_list],
        "B_beta": int(cfg.B_beta),
    }
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- beta rows（重点：存完整beta_rows）----
    # 这是你说“一会都要用”的核心文件
    print("[Save] beta_rows.json ...")
    beta_rows_plain = [to_plain_python(row) for row in beta_rows]
    with open(os.path.join(results_dir, "beta_rows.json"), "w", encoding="utf-8") as f:
        json.dump(beta_rows_plain, f, ensure_ascii=False, indent=2)

    # 可选：再存一份 {N: beta} 的轻量映射（后续测试脚本读取更方便）
    # 如果你只想存 beta_rows.json，这段可以删掉
    print("[Save] beta_map.json ...")
    with open(os.path.join(results_dir, "beta_map.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in beta_map.items()}, f, ensure_ascii=False, indent=2)

    print("All results saved.")


if __name__ == "__main__":
    main()