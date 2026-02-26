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
        self.results_dir_name = "results_lookback_two_stage"

        self.seed = 44
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== Simulation / Feature (shared by both stages) =====
        self.method = "pca"       # pca / bb / cholesky
        self.dimX = 26      # 特征维度
        self.nD = 256
        self.T = 1.0

        # ===== Dataset split ratio (shared) =====
        self.train_ratio = 0.7
        self.val_ratio = 0.15

        # ===== Model / Optim =====
        self.dropout = 0.3

        # ===== 3D loss config (used when X is [B,N,d]) =====
        self.rqmc_loss_center = True
        self.rqmc_loss_unbiased = False

        # =============================
        # Stage 1: 2D pretrain (MSE)
        # =============================
        self.stage1_enable = True
        self.stage1_thetadim = 2                 # 固定 2D
        self.stage1_dataset_size = 2 ** 16       # 点数 M
        self.stage1_batch_size = 512             # 点级 batch
        self.stage1_epochs = 50
        self.stage1_lr = 1e-3
        self.stage1_seed_offset = 0

        # =============================
        # Stage 2: 3D finetune (RQMC-group loss)
        # =============================
        self.stage2_enable = True
        self.stage2_thetadim = 3                 # 固定 3D
        self.stage2_dataset_size = 2 ** 6        # 组数 B_total
        self.stage2_N = 2048                     # 每组点数 N
        self.stage2_batch_size = 8               # “组”为单位
        self.stage2_epochs = 30
        self.stage2_lr = 3e-4
        self.stage2_seed_offset = 12345

        # Stage2 是否重新计算归一化（建议先 False，稳一点）
        self.recompute_norm_in_stage2 = False


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

    # payoff（你的 simulation_qmc.py 里当前 arithmetic_payoff 是啥就用啥）
    y = arithmetic_payoff(S, K).unsqueeze(-1)  # 2D->[M,1], 3D->[B,N,1]

    # 特征（你当前是 min/max 版本）
    X = features_from_Z(
        Z,
        dimX=cfg.dimX,
        theta=(r, S0, sigma, K),
        G=G,
        T=cfg.T,
        # k_proxy=8,   # 可选
    )

    theta_tensor = torch.stack([r, S0, sigma, K], dim=-1)  # 兼容2D/3D

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


def build_loaders_for_mode(cfg, thetadim, dataset_size, batch_size, N_group=None, seed=None):
    """
    根据模式(2D/3D)生成 fresh 数据并返回 DataLoader + 子集
    """
    if thetadim == 2:
        r, S0, sigma, K = sample_theta(
            mode=2,
            batch_size=int(dataset_size),
            is_same=False,
            device=cfg.device
        )
    elif thetadim == 3:
        if N_group is None:
            raise ValueError("3D模式需要传 N_group")
        r, S0, sigma, K = sample_theta(
            mode=3,
            B=int(dataset_size),
            N=int(N_group),
            is_same=False,
            device=cfg.device
        )
    else:
        raise ValueError(f"thetadim must be 2 or 3, got {thetadim}")

    dataset = generate_dataset(cfg, theta=(r, S0, sigma, K), seed=(cfg.seed if seed is None else int(seed)))
    train_set, val_set, test_set = split_dataset(dataset, cfg)

    train_loader = DataLoader(train_set, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(batch_size), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=int(batch_size), shuffle=False)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set


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
      - 2D数据: theta [M,4],   X [M,d], y [M,1]
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
def train_model(train_loader, val_loader, norm, cfg, net=None, lr=None, epochs=None, stage_name="Train"):
    """
    支持:
      - net=None: 从头训练
      - net=已有模型: 继续训练（finetune）
    """
    if net is None:
        net = PEMCNet(dimX=cfg.dimX, dropout=cfg.dropout).to(cfg.device)
    else:
        net = net.to(cfg.device)

    use_lr = cfg.lr if (lr is None and hasattr(cfg, "lr")) else (1e-3 if lr is None else float(lr))
    use_epochs = cfg.epochs if (epochs is None and hasattr(cfg, "epochs")) else (100 if epochs is None else int(epochs))

    opt = torch.optim.AdamW(net.parameters(), lr=use_lr)

    train_losses = []
    val_losses = []
    first_step_loss = None

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    # 只判一次模式，别每轮都猜
    sample_X_ndim = train_loader.dataset[0][1].ndim
    val_mode_name = "RQMC-group-var" if sample_X_ndim == 3 else "MSE"

    for epoch in range(use_epochs):
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

        print(f"[{stage_name}] Epoch {epoch+1} | Val {val_mode_name}: {val_loss:.6f} | Best: {best_val_loss:.6f} (epoch {best_epoch})")

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


# =============================
# 5) Main (Two-stage training)
# =============================
def main():
    cfg = Config()
    set_seed(cfg.seed)

    current_working_directory = os.getcwd()
    print("CWD =", current_working_directory)

    results_dir = os.path.join(current_working_directory, cfg.results_dir_name)
    print("Results directory:", results_dir)
    print("Device =", cfg.device)

    os.makedirs(results_dir, exist_ok=True)

    # 用于最终保存 / 汇总
    net = None
    norm_stage1 = None
    norm_stage2 = None
    norm_final = None

    stage1_record = None
    stage2_record = None

    # =============================
    # Stage 1: 2D pretrain (MSE)
    # =============================
    if cfg.stage1_enable:
        print("\n===== Stage 1: 2D pretrain =====")

        train_loader_1, val_loader_1, test_loader_1, train_set_1, val_set_1, test_set_1 = build_loaders_for_mode(
            cfg,
            thetadim=cfg.stage1_thetadim,
            dataset_size=cfg.stage1_dataset_size,
            batch_size=cfg.stage1_batch_size,
            N_group=None,
            seed=cfg.seed + cfg.stage1_seed_offset
        )

        norm_stage1 = compute_normalization(train_set_1, cfg)

        net, first_step_loss_1, train_losses_1, val_losses_1, best_state_dict_1, best_val_loss_1, best_epoch_1 = train_model(
            train_loader_1,
            val_loader_1,
            norm_stage1,
            cfg,
            net=None,
            lr=cfg.stage1_lr,
            epochs=cfg.stage1_epochs,
            stage_name="Stage1-2D"
        )

        if best_state_dict_1 is not None:
            net.load_state_dict(best_state_dict_1)

        test_loss_1 = evaluate(net, test_loader_1, norm_stage1, cfg)

        print(f"[Stage1] First step loss: {first_step_loss_1}")
        print(f"[Stage1] Best Val Loss: {best_val_loss_1:.6f} (epoch {best_epoch_1})")
        print(f"[Stage1] Test Loss: {test_loss_1:.6f}")

        # 保存一份 stage1 checkpoint（可选但很有用）
        model_state_stage1_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save(model_state_stage1_cpu, os.path.join(results_dir, "model_stage1_best.pth"))
        torch.save({k: v.detach().cpu() for k, v in norm_stage1.items()}, os.path.join(results_dir, "normalization_stage1.pth"))

        np.save(os.path.join(results_dir, "train_losses_stage1.npy"), np.array(train_losses_1, dtype=np.float32))
        np.save(os.path.join(results_dir, "val_losses_stage1.npy"), np.array(val_losses_1, dtype=np.float32))

        stage1_record = {
            "first_step_loss": float(first_step_loss_1) if first_step_loss_1 is not None else None,
            "best_val_loss": float(best_val_loss_1),
            "best_epoch": int(best_epoch_1),
            "test_loss": float(test_loss_1),
            "num_train_steps": int(len(train_losses_1)),
            "num_epochs": int(cfg.stage1_epochs),
            "dataset_mode": "2D",
            "dataset_size": int(cfg.stage1_dataset_size),
            "batch_size": int(cfg.stage1_batch_size),
            "lr": float(cfg.stage1_lr),
        }
    else:
        train_losses_1, val_losses_1 = [], []

    # =============================
    # Stage 2: 3D finetune (RQMC group loss)
    # =============================
    if cfg.stage2_enable:
        print("\n===== Stage 2: 3D finetune =====")

        train_loader_2, val_loader_2, test_loader_2, train_set_2, val_set_2, test_set_2 = build_loaders_for_mode(
            cfg,
            thetadim=cfg.stage2_thetadim,
            dataset_size=cfg.stage2_dataset_size,
            batch_size=cfg.stage2_batch_size,
            N_group=cfg.stage2_N,
            seed=cfg.seed + cfg.stage2_seed_offset
        )

        # 如果没做stage1，也允许直接stage2从头训练
        if net is None:
            print("[Stage2] No pretrained model found. Train from scratch on 3D data.")
            net_for_stage2 = None
        else:
            net_for_stage2 = net

        # Stage2 归一化策略
        if cfg.recompute_norm_in_stage2 or (norm_stage1 is None):
            norm_stage2 = compute_normalization(train_set_2, cfg)
            print("[Stage2] Use recomputed normalization from 3D train set.")
        else:
            norm_stage2 = norm_stage1
            print("[Stage2] Reuse Stage1 normalization.")

        # 先看一下 Stage1 模型在 Stage2 test 上的起点（很有用）
        if net_for_stage2 is not None:
            pre_stage2_test_loss = evaluate(net_for_stage2, test_loader_2, norm_stage2, cfg)
            print(f"[Before Stage2 finetune] Stage2 test loss = {pre_stage2_test_loss:.6f}")
        else:
            pre_stage2_test_loss = None

        net, first_step_loss_2, train_losses_2, val_losses_2, best_state_dict_2, best_val_loss_2, best_epoch_2 = train_model(
            train_loader_2,
            val_loader_2,
            norm_stage2,
            cfg,
            net=net_for_stage2,          # 关键：继续训练
            lr=cfg.stage2_lr,            # 通常更小学习率
            epochs=cfg.stage2_epochs,
            stage_name="Stage2-3D"
        )

        if best_state_dict_2 is not None:
            net.load_state_dict(best_state_dict_2)

        test_loss_2 = evaluate(net, test_loader_2, norm_stage2, cfg)

        print(f"[Stage2] First step loss: {first_step_loss_2}")
        print(f"[Stage2] Best Val Loss: {best_val_loss_2:.6f} (epoch {best_epoch_2})")
        print(f"[Stage2] Test Loss: {test_loss_2:.6f}")

        np.save(os.path.join(results_dir, "train_losses_stage2.npy"), np.array(train_losses_2, dtype=np.float32))
        np.save(os.path.join(results_dir, "val_losses_stage2.npy"), np.array(val_losses_2, dtype=np.float32))

        stage2_record = {
            "first_step_loss": float(first_step_loss_2) if first_step_loss_2 is not None else None,
            "best_val_loss": float(best_val_loss_2),
            "best_epoch": int(best_epoch_2),
            "test_loss": float(test_loss_2),
            "pre_stage2_test_loss": float(pre_stage2_test_loss) if pre_stage2_test_loss is not None else None,
            "num_train_steps": int(len(train_losses_2)),
            "num_epochs": int(cfg.stage2_epochs),
            "dataset_mode": "3D",
            "dataset_size_B": int(cfg.stage2_dataset_size),
            "N_group": int(cfg.stage2_N),
            "batch_size_groups": int(cfg.stage2_batch_size),
            "lr": float(cfg.stage2_lr),
            "recompute_norm_in_stage2": bool(cfg.recompute_norm_in_stage2),
            "rqmc_loss_center": bool(cfg.rqmc_loss_center),
            "rqmc_loss_unbiased": bool(cfg.rqmc_loss_unbiased),
        }

        norm_final = norm_stage2
    else:
        train_losses_2, val_losses_2 = [], []
        norm_final = norm_stage1 if norm_stage1 is not None else None

    # 如果两个stage都没开，直接报错
    if net is None:
        raise RuntimeError("stage1_enable 和 stage2_enable 不能同时为 False。")

    # =============================
    # 6) Save final artifacts
    # =============================
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 最终保存的是当前 net（若有stage2则是stage2 best；否则是stage1 best）
    model_state_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(model_state_cpu, os.path.join(results_dir, "model.pth"))
    print("[Save] model.pth ...")

    if norm_final is not None:
        norm_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm_final.items()}
        torch.save(norm_cpu, os.path.join(results_dir, "normalization.pth"))
        print("[Save] normalization.pth ...")
    else:
        print("[Save] Skip normalization.pth (norm_final is None)")

    # 也把stage2的norm单独存一下（如果有）
    if norm_stage2 is not None:
        torch.save({k: v.detach().cpu() for k, v in norm_stage2.items()},
                   os.path.join(results_dir, "normalization_stage2.pth"))

    # config
    cfg_dict = cfg_to_dict(cfg)
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print("[Save] config.yaml ...")

    # summary
    summary = {
        "device": str(cfg.device),
        "seed": int(cfg.seed),
        "method": str(cfg.method),
        "dimX": int(cfg.dimX),
        "nD": int(cfg.nD),
        "T": float(cfg.T),
        "stage1": stage1_record,
        "stage2": stage2_record,
        "final_model_source": "stage2_best" if cfg.stage2_enable else "stage1_best",
    }

    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_plain_python(summary), f, ensure_ascii=False, indent=2)
    print("[Save] summary.json ...")

    print("All results saved.")


if __name__ == "__main__":
    main()