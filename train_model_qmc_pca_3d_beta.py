import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import yaml

from model import PEMCNet
from simulation_qmc import *  # 依赖你精简后的: sample_theta / simulate_gbm_batch_qmc / features_from_Z / arithmetic_payoff


# =============================
# 0) Config
# =============================
class Config:
    def __init__(self):
        # ===== Reproducibility / Device =====
        self.results_dir_name = "results_lookback_float"

        self.seed = 44
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== Simulation / Feature =====
        self.method = "pca"       # pca / bb / cholesky
        self.dimX = 6  # 特征维度
        self.dimZ = 4
        self.dimProxy = 2

        self.thetadim = 2      # 2 或 3（训练数据模式）
        self.N = 2048              # 训练数据中每组QMC点数（仅训练/数据生成用）

        self.nD = 256
        self.T = 1.0

        # ===== Dataset =====
        # 注意：thetadim=2 时表示样本数 M；thetadim=3 时表示组数 B_total
        self.dataset_size = 2 ** 16
        self.train_ratio = 0.7
        self.val_ratio = 0.15

        # ===== Training =====
        self.batch_size = 512   # thetadim=3 时单位是“组数”
        self.epochs = 50
        self.lr = 1e-3
        self.dropout = 0.3

        # ===== Loss (仅当3D训练时会用到 RQMC group loss) =====
        self.rqmc_loss_center = True
        self.rqmc_loss_unbiased = False

        # ===== Beta estimation (rep-loop mode, post-training) =====
        # 每个 N：做 beta_num_reps 次独立 rep；每个 rep 用 1 组 [1,N,*] 数据
        self.beta_N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
        self.beta_num_reps = 2 ** 7
        self.beta_rep_seed_base = 100000  # beta评估用的基础seed（避免和训练共用）
        self.nrep_beta = 100

# =============================
# 1) Utility
# =============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 追求可复现（会牺牲一点速度）
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


# =============================
# 2) Dataset / Normalization
# =============================
def generate_dataset(cfg, theta, seed=None):
    """
    theta: (r, S0, sigma, K)
      - 2D模式时每个参数 shape [M]
      - 3D模式时每个参数 shape [B,N]
    返回 TensorDataset(theta, X, y)
      - 2D: theta [M,4],   X [M,d],   y [M,1]
      - 3D: theta [B,N,4], X [B,N,d], y [B,N,1]
    """
    r, S0, sigma, K = theta
    sim_seed = cfg.seed if seed is None else int(seed)

    # 新接口：返回 Z, W, S, G
    Z, W, S, G = simulate_gbm_batch_qmc(
        (r, S0, sigma, K),
        nD=cfg.nD,
        T=cfg.T,
        method=cfg.method,
        device=cfg.device,
        seed=sim_seed
    )

    # lookback payoff（精简版 arithmetic_payoff 不需要 S0）
    y = arithmetic_payoff(S, K).unsqueeze(-1)  # 2D->[M,1], 3D->[B,N,1]

    X = features_from_Z_lookback(
        Z=Z,
        dimX=cfg.dimX,
        dimZ=cfg.dimZ,
        dimProxy=cfg.dimProxy,
        theta = (r, S0, sigma, K),         # (r, S0, sigma, K)
        G=G,
        k_proxy=64,
        )
    # dim=-1 同时兼容 2D/3D
    theta_tensor = torch.stack([r, S0, sigma, K], dim=-1)

    return TensorDataset(theta_tensor, X, y)


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
    return_stats: bool = False,
):
    """
    pred: [B, N, 1] 或 [B, N]
    y:    [B, N, 1] 或 [B, N]

    center=True:
        loss ~= Var_b( mean_i (y_{b,i} - pred_{b,i}) )

    center=False:
        loss = E_b[hbar_b^2] = Var(hbar) + (E[hbar])^2
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
            return F.mse_loss(pred, y, reduction="sum"), y.size(0)
        return F.mse_loss(pred, y)

    if X.ndim == 3:
        # RQMC 分组方差损失
        loss = rqmc_group_var_loss(
            pred=pred,
            y=y,
            center=cfg.rqmc_loss_center,
            unbiased=cfg.rqmc_loss_unbiased,
            return_stats=False,
        )
        if for_eval:
            return loss, X.size(0)  # 用组数B做权重
        return loss

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

    # 只判一次模式，别每轮都猜
    sample_X_ndim = train_loader.dataset[0][1].ndim
    val_mode_name = "RQMC-group-var" if sample_X_ndim == 3 else "MSE"

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

            if first_step_loss is None:
                first_step_loss = float(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(float(loss.item()))

        # ===== Val =====
        net.eval()
        val_sum, val_weight = 0.0, 0

        with torch.no_grad():
            for theta, X, y in val_loader:
                theta = theta.to(cfg.device)
                X = X.to(cfg.device)
                y = y.to(cfg.device)

                theta, X = normalize(theta, X, norm)
                pred = net(theta, X)

                loss, weight = _batch_loss_auto(pred, y, X, cfg, for_eval=True)
                val_sum += float(loss.item())
                val_weight += int(weight)

        val_loss = val_sum / max(val_weight, 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        print(f"Epoch {epoch+1} | Val {val_mode_name}: {val_loss:.6f} | Best: {best_val_loss:.6f} (epoch {best_epoch})")

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
            total += float(loss.item())
            count += int(weight)

    return total / max(count, 1)



import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# =============================
# 5) Beta statistics (numpy)
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


# =============================
# helper: collect (ybar, gbar) pairs from a 3D dataset
# =============================
@torch.no_grad()
def collect_ybar_gbar_pairs(net, dataset: TensorDataset, norm: dict, cfg, loader_batch_size=None):
    """
    dataset: TensorDataset(theta, X, y), 且应为3D样本:
      theta [B,N,4], X [B,N,d], y [B,N,1]   (或 y [B,N])
    返回:
      ybar_all: np.ndarray [B_total]
      gbar_all: np.ndarray [B_total]
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

        theta_n, X_n = normalize(theta, X, norm)   # <<< 依赖你已有的 normalize
        pred = net(theta_n, X_n)

        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)

        assert pred.shape == y.shape, f"pred/y shape mismatch: {pred.shape} vs {y.shape}"

        # 每个 group 一个标量
        ybar = y.mean(dim=1).squeeze(-1)     # [B]
        gbar = pred.mean(dim=1).squeeze(-1)  # [B]

        ybar_all.append(ybar.detach().cpu().numpy())
        gbar_all.append(gbar.detach().cpu().numpy())

    ybar_all = np.concatenate(ybar_all, axis=0).astype(np.float64)
    gbar_all = np.concatenate(gbar_all, axis=0).astype(np.float64)
    return ybar_all, gbar_all


# =============================
# helper: compute beta/corr/vars from pairs
# =============================
def beta_stats_from_pairs(ybar_all: np.ndarray, gbar_all: np.ndarray):
    """
    给定 nrep 个 ybar/gbar：
      beta = cov(ybar,gbar)/var(gbar)
      corr(ybar,gbar)
      var(ybar)
      var(ybar - beta*gbar)
    """
    ybar_all = np.asarray(ybar_all, dtype=np.float64).reshape(-1)
    gbar_all = np.asarray(gbar_all, dtype=np.float64).reshape(-1)

    var_y = _np_sample_var(ybar_all)
    var_g = _np_sample_var(gbar_all)
    cov_yg = _np_sample_cov(ybar_all, gbar_all)
    corr_yg = _np_sample_corr(ybar_all, gbar_all)

    if (not np.isfinite(var_g)) or abs(var_g) < 1e-30:
        beta = float("nan")
        var_resid = float("nan")
    else:
        beta = float(cov_yg / var_g)
        resid = ybar_all - beta * gbar_all
        var_resid = _np_sample_var(resid)

    return {
        "nrep": int(ybar_all.size),
        "beta": float(beta),
        "corr": float(corr_yg),
        "var_ybar": float(var_y),
        "var_resid": float(var_resid),  # Var(ybar - beta*gbar)
    }


# =============================
# main: beta curve by N with nrep reps
# =============================
@torch.no_grad()
def estimate_beta_curve_by_N_nrep(
    net,
    norm: dict,
    cfg,
    theta_fixed,          # 用你现有的 theta_fixed 结构（传给 sample_theta 的 theta_same）
    N_list,
    nrep: int = 64,
    loader_batch_size=None,
    base_seed: int | None = None,
):
    """
    对每个 N：
      - 重复 nrep 次（每次生成 1 个 group: B=1, N=N_group），seed 每次变
      - 得到 nrep 个 ybar/gbar
      - 计算并打印：beta, corr, var(ybar), var(ybar - beta*gbar)
    依赖你已有函数：
      - sample_theta(mode=3, ...)
      - generate_dataset(cfg, theta=(r,S0,sigma,K,H))
      - normalize(theta, X, norm)
    """
    rows = []
    if base_seed is None:
        base_seed = int(getattr(cfg, "seed", 0))+10000

    for N_group in N_list:
        ybars = np.empty((nrep,), dtype=np.float64)
        gbars = np.empty((nrep,), dtype=np.float64)

        for rep in range(nrep):
            cfg_rep = copy.copy(cfg)
            cfg_rep.seed = base_seed + rep  # 关键：让每次 RQMC scramble/shift 不同（前提是你模拟器用到了 seed）

            # 生成固定theta的 3D 参数 [B=1, N_group]
            r, S0, sigma, K= sample_theta(
                mode=3,
                batch_size=1,
                B=1,
                N=int(N_group),
                is_same=True,
                theta_same=theta_fixed,
                device=cfg_rep.device
            )

            dataset_beta = generate_dataset(cfg_rep, theta=(r, S0, sigma, K))
            ybar_rep, gbar_rep = collect_ybar_gbar_pairs(
                net=net,
                dataset=dataset_beta,
                norm=norm,
                cfg=cfg_rep,
                loader_batch_size=loader_batch_size
            )

            if ybar_rep.size != 1 or gbar_rep.size != 1:
                raise ValueError(f"期望每次rep得到1个group，但 got ybar={ybar_rep.shape}, gbar={gbar_rep.shape}")

            ybars[rep] = float(ybar_rep[0])
            gbars[rep] = float(gbar_rep[0])

        stats = beta_stats_from_pairs(ybars, gbars)
        row = {"N": int(N_group), **stats}
        rows.append(row)

        # 按你要的 4 个量打印：beta, corr, var(ybar), var(ybar-beta*gbar)
        print(
            f"[Beta-nrep] N={row['N']:5d} | nrep={row['nrep']:4d} | "
            f"beta={row['beta']:.8f} | "
            f"corr={row['corr']:.6f} | "
            f"var_ybar={row['var_ybar']:.6e} | "
            f"var_resid={row['var_resid']:.6e}"
        )

    return rows

# =============================
# 6) Main
# =============================
def main():
    cfg = Config()
    set_seed(cfg.seed)

    current_working_directory = os.getcwd()
    print("CWD =", current_working_directory)

    results_dir = os.path.join(current_working_directory, cfg.results_dir_name)
    print("Results directory:", results_dir)
    print("Device =", cfg.device)

    # ===== 固定一个测试合约 theta（用于 beta 曲线估计）=====
    theta_fixed = (
        torch.tensor(0.02, device=cfg.device),     # r
        torch.tensor(100.0, device=cfg.device),    # S0
        torch.tensor(0.15, device=cfg.device),     # sigma
        torch.tensor(100.0, device=cfg.device),    # K
    )

    # ===== 训练数据生成（这里用 cfg.N，仅用于训练/验证/测试数据）=====
    if cfg.thetadim == 2:
        r, S0, sigma, K = sample_theta(
            mode=2,
            batch_size=cfg.dataset_size,
            is_same=False,
            device=cfg.device
        )
    elif cfg.thetadim == 3:
        r, S0, sigma, K = sample_theta(
            mode=3,
            B=cfg.dataset_size,
            N=cfg.N,
            is_same=False,
            device=cfg.device
        )
    else:
        raise ValueError(f"cfg.thetadim must be 2 or 3, got {cfg.thetadim}")

    dataset = generate_dataset(cfg, theta=(r, S0, sigma, K), seed=cfg.seed)

    # 划分：len(dataset) 在3D模式下是组数 B_total；在2D模式下是样本数 M
    train_set, val_set, test_set = split_dataset(dataset, cfg)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

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

    # ===== 估计 beta(N) 曲线（rep-loop mode）=====
    print("\n===== Estimate beta(N) by rep-loop on fresh 3D datasets (fixed theta) =====")
    beta_rows = estimate_beta_curve_by_N_nrep(
        net=net,
        norm=norm,
        cfg=cfg,
        theta_fixed=theta_fixed,
        N_list=cfg.beta_N_list,
        nrep=cfg.nrep_beta,  # 原来的 B_beta 现在当作 nrep
        loader_batch_size=1,  # 每次rep只生成1个group，1最稳
        base_seed=cfg.seed  # 可选：控制可复现
    )


    # 轻量映射：N -> beta
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

        # beta rep-loop 配置（不是beta值本身）
        "beta_num_points": int(len(beta_rows)),
        "beta_N_list": [int(n) for n in cfg.beta_N_list],
        "beta_num_reps": int(cfg.beta_num_reps),
        "beta_rep_seed_base": int(cfg.beta_rep_seed_base),
    }
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- beta rows（聚合统计）----
    print("[Save] beta_rows.json ...")
    beta_rows_plain = [to_plain_python(row) for row in beta_rows]
    with open(os.path.join(results_dir, "beta_rows.json"), "w", encoding="utf-8") as f:
        json.dump(beta_rows_plain, f, ensure_ascii=False, indent=2)

    # ---- beta rep pairs（每个N的rep级 ybar/gbar 序列）----
    # 这个文件很有用：后面你想重算 corr/beta/robust统计都能直接读它，不用重跑路径
    print("[Save] beta_rep_pairs.json ...")
    with open(os.path.join(results_dir, "beta_rep_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(to_plain_python(beta_rep_pairs), f, ensure_ascii=False, indent=2)

    # ---- 轻量 beta map ----
    print("[Save] beta_map.json ...")
    with open(os.path.join(results_dir, "beta_map.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in beta_map.items()}, f, ensure_ascii=False, indent=2)

    print("All results saved.")


if __name__ == "__main__":
    main()