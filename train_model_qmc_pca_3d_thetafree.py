import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import yaml

from model_thetafree import PEMCNet
from simulation_qmc import *   # 你自己的函数：simulate_gbm_batch_qmc / arithmetic_payoff / features_from_Z / sample_theta 等


# 获取当前工作目录（CWD）
current_working_directory = os.getcwd()
print("CWD =", current_working_directory)

# 保存路径
results_dir = os.path.join(current_working_directory, "results_qmc_pca_mse")
print("Results directory:", results_dir)


class Config:
    def __init__(self):
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dimX = 4
        self.dataset_size = 2 ** 16   # “组数”B_total（3D模式时）

        self.train_ratio = 0.7
        self.val_ratio = 0.15

        self.batch_size = 256        # DataLoader 的 batch 是“组”
        self.epochs = 300

        self.lr = 1e-3
        self.dropout = 0.1
        self.use_bn = False

        # ===== 新增：是否使用 theta 作为网络输入 =====
        self.use_theta = False       # <<< 改这里：False=只输入X；True=用(theta,X)
        self.same_theta = True

        # RQMC 组损失设置
        self.rqmc_loss_center = True
        self.rqmc_loss_unbiased = False


# =============================
# 2️⃣ 工具函数
# =============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_dataset(cfg, theta):
    """
    theta: (r, S0, sigma, K)
      - 2D模式时每个参数 shape [M]
      - 3D模式时每个参数 shape [B,N]
    返回 TensorDataset(theta, X, PA)
      - 2D: theta [M,4],   X [M,d],   y [M,1]
      - 3D: theta [B,N,4], X [B,N,d], y [B,N,1]

    注意：
      即使 cfg.use_theta=False，这里仍然保留 theta 在数据集中，
      训练时会忽略它（便于兼容你现有代码）。
    """
    r, S0, sigma, K = theta

    Z, W, S = simulate_gbm_batch_qmc((r, S0, sigma, K), method="pca", device=cfg.device)
    PA = arithmetic_payoff(S, K).unsqueeze(-1)   # 2D->[M,1], 3D->[B,N,1]
    X = features_from_Z(Z, dimX=cfg.dimX)

    # dim=-1 同时兼容 2D/3D
    theta_tensor = torch.stack([r, S0, sigma, K], dim=-1)

    dataset = TensorDataset(theta_tensor, X, PA)
    return dataset


def split_dataset(dataset, cfg):
    N = len(dataset)
    train_size = int(cfg.train_ratio * N)
    val_size = int(cfg.val_ratio * N)
    test_size = N - train_size - val_size
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

    当 cfg.use_theta=False 时，不计算 theta 的标准化参数（节省一点事）。
    """
    X_train = torch.stack([train_set[i][1] for i in range(len(train_set))])  # [M,d] or [B,N,d]
    X_dims = _reduce_dims_except_last(X_train)

    norm = {
        "X_mean": X_train.mean(dim=X_dims, keepdim=True),
        "X_std": X_train.std(dim=X_dims, keepdim=True, unbiased=False).clamp_min(1e-6),
    }

    if cfg.use_theta:
        theta_train = torch.stack([train_set[i][0] for i in range(len(train_set))])  # [M,4] or [B,N,4]
        theta_dims = _reduce_dims_except_last(theta_train)
        norm["theta_mean"] = theta_train.mean(dim=theta_dims, keepdim=True)
        norm["theta_std"] = theta_train.std(dim=theta_dims, keepdim=True, unbiased=False).clamp_min(1e-6)

    for k in norm:
        norm[k] = norm[k].to(cfg.device)

    return norm


def normalize(theta, X, norm, use_theta=True):
    """
    兼容:
      theta: [M,4] / [B,N,4] / [B,4]
      X:     [M,d] / [B,N,d]

    use_theta=False 时，theta 原样返回（不做标准化），只标准化 X。
    """
    X = (X - norm["X_mean"]) / norm["X_std"]

    if use_theta:
        theta = (theta - norm["theta_mean"]) / norm["theta_std"]

    return theta, X


# =============================
# RQMC 分组方差损失
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
    pred: [B,N,1] 或 [B,N]
    y:    [B,N,1] 或 [B,N]

    center=True:
        loss = 方差版本（推荐）
              ~= Var_b( mean_i (y_{b,i}-pred_{b,i}) )

    center=False:
        loss = 二阶矩版本
              = E_b[ hbar_b^2 ] = Var(hbar) + (E[hbar])^2
    """
    # 统一成 [B,N,1]
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


# =============================
# 辅助：根据输入维度选择损失
# =============================
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
            weight = X.size(0)  # 用组数 B 做权重
            return loss, weight
        else:
            return loss

    else:
        raise ValueError(f"不支持的 X 维度: {X.ndim}，期望 2 或 3")


# =============================
# 前向调用（根据 cfg.use_theta 自动选择）
# =============================
def model_forward_auto(net, theta, X, cfg):
    """
    cfg.use_theta=True  -> net(theta, X)
    cfg.use_theta=False -> net(X)
    """
    if cfg.use_theta:
        return net(theta, X)
    else:
        return net(X)


# =============================
# 3️⃣ 训练函数（兼容 2D / 3D；兼容 use_theta=True/False）
# =============================
def train_model(train_loader, val_loader, norm, cfg):
    net = PEMCNet(
        dimX=cfg.dimX,
        dropout=cfg.dropout,
        use_bn=getattr(cfg, "use_bn", False),
        use_theta=getattr(cfg, "use_theta", True),   # <<< 关键
    ).to(cfg.device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    first_step_loss = None

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    for epoch in range(cfg.epochs):
        # ===== 训练 =====
        net.train()
        for step, (theta, X, y) in enumerate(train_loader):
            theta = theta.to(cfg.device)
            X = X.to(cfg.device)
            y = y.to(cfg.device)

            theta, X = normalize(theta, X, norm, use_theta=cfg.use_theta)

            pred = model_forward_auto(net, theta, X, cfg)
            loss = _batch_loss_auto(pred, y, X, cfg, for_eval=False)

            if epoch == 0 and step == 0:
                first_step_loss = float(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(float(loss.item()))

        # ===== 验证 =====
        net.eval()
        val_sum, val_weight = 0.0, 0
        mode_name = None

        with torch.no_grad():
            for theta, X, y in val_loader:
                theta = theta.to(cfg.device)
                X = X.to(cfg.device)
                y = y.to(cfg.device)

                theta, X = normalize(theta, X, norm, use_theta=cfg.use_theta)
                pred = model_forward_auto(net, theta, X, cfg)

                loss, weight = _batch_loss_auto(pred, y, X, cfg, for_eval=True)
                val_sum += float(loss.item())
                val_weight += int(weight)

                mode_name = "RQMC-group-var" if X.ndim == 3 else "MSE"

        val_loss = val_sum / max(val_weight, 1)
        val_losses.append(float(val_loss))

        # 保存最优参数（CPU副本）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        print(
            f"Epoch {epoch+1} | Val {mode_name}: {val_loss:.6f} | "
            f"Best: {best_val_loss:.6f} (epoch {best_epoch})"
        )

    return net, first_step_loss, train_losses, val_losses, best_state_dict, best_val_loss, best_epoch


# =============================
# 4️⃣ 测试函数（兼容 2D / 3D；兼容 use_theta=True/False）
# =============================
def evaluate(net, loader, norm, cfg):
    net.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for theta, X, y in loader:
            theta = theta.to(cfg.device)
            X = X.to(cfg.device)
            y = y.to(cfg.device)

            theta, X = normalize(theta, X, norm, use_theta=cfg.use_theta)
            pred = model_forward_auto(net, theta, X, cfg)

            loss, weight = _batch_loss_auto(pred, y, X, cfg, for_eval=True)
            total += float(loss.item())
            count += int(weight)

    return total / max(count, 1)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    print(f"[Config] device={cfg.device}, use_theta={cfg.use_theta}, dimX={cfg.dimX}")

    # ===== 3D 训练配置 =====
    N_group = 4096  # 每组QMC点数 N

    # ===== 固定一个 theta =====
    theta_fixed = (
        torch.tensor(0.02, device=cfg.device),    # r
        torch.tensor(100.0, device=cfg.device),   # S0
        torch.tensor(0.15, device=cfg.device),    # sigma
        torch.tensor(100.0, device=cfg.device),   # K
    )

    # 生成 theta（3D模式）
    # 这里假设你的 sample_theta 支持 theta_same 参数；若你本地函数名/参数不同，按你的版本改一下
    r, S0, sigma, K = sample_theta(
        mode=2,
        batch_size=cfg.dataset_size,
        is_same=cfg.same_theta,  # 用 same_theta
        theta_same=theta_fixed,
        device=cfg.device
    )

    # 数据集: theta [B,N,4], X [B,N,d], y [B,N,1]
    dataset = generate_dataset(cfg, theta=(r, S0, sigma, K))

    # 划分（按组划分，因为 len(dataset)=B_total）
    train_set, val_set, test_set = split_dataset(dataset, cfg)

    # DataLoader 的 batch 单位是“组”
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    # 标准化（会根据 cfg.use_theta 自动决定是否计算 theta norm）
    norm = compute_normalization(train_set, cfg)

    # 训练
    net, first_step_loss, train_losses, val_losses, best_state_dict, best_val_loss, best_epoch = train_model(
        train_loader, val_loader, norm, cfg
    )

    # 用 best val 参数做测试
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    test_loss = evaluate(net, test_loader, norm, cfg)

    print("First step loss:", first_step_loss)
    print(f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print("Final Test Loss (using best-val checkpoint):", test_loss)

    # ===== 保存 =====
    os.makedirs(results_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model_state_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    norm_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm.items()}

    print("[Save] model.pth ...")
    torch.save(model_state_cpu, f"{results_dir}/model.pth")

    print("[Save] normalization.pth ...")
    torch.save(norm_cpu, f"{results_dir}/normalization.pth")

    print("[Save] train_losses.npy / val_losses.npy ...")
    np.save(f"{results_dir}/train_losses.npy", np.array(train_losses, dtype=np.float32))
    np.save(f"{results_dir}/val_losses.npy", np.array(val_losses, dtype=np.float32))

    print("[Save] config.yaml ...")
    cfg_dict = vars(cfg).copy()
    cfg_dict["device"] = str(cfg_dict.get("device", "cpu"))
    with open(f"{results_dir}/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    summary = {
        "first_step_loss": float(first_step_loss) if first_step_loss is not None else None,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "num_train_steps": int(len(train_losses)),
        "num_epochs": int(cfg.epochs),
        "use_theta": bool(cfg.use_theta),
    }
    print("[Save] summary.json ...")
    with open(f"{results_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("All results saved.")


if __name__ == "__main__":
    main()