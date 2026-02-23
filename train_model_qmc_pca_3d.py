import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import json
import os
import random
import shutil
import torch
import torch.nn.functional as F
from torch.xpu import device

from model import PEMCNet
# from simulation_mc import *
from simulation_qmc import *
import yaml


# 获取当前工作目录（CWD）
current_working_directory = os.getcwd()
print("CWD =", current_working_directory)
#保存路径
results_dir = os.path.join(current_working_directory, "results_qmc_pca_mse")
print("Results directory:", results_dir)



class Config:
    def __init__(self):
        self.seed = 42 #随机种子
        self.device = "cuda" if torch.cuda.is_available() else "cpu"#设备

        self.dimX = 1#特征维度
        self.dataset_size = 2 ** 16#样本量

        self.train_ratio = 0.7  #训练集比例
        self.val_ratio = 0.15  #验证集比例

        self.batch_size = 512  #sgd的batch
        self.epochs = 150#进行轮数

        self.lr = 1e-3  #初始学习率
        self.dropout = 0.3


        self.rqmc_loss_center = False
        self.rqmc_loss_unbiased = False

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




s = 1

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
    """
    r, S0, sigma, K = theta

    Z, W, S = simulate_gbm_batch_qmc((r, S0, sigma, K), method="pca", device=cfg.device)
    PA = arithmetic_payoff(S, K).unsqueeze(-1)   # 2D->[M,1], 3D->[B,N,1]
    X = features_from_Z(Z, dimX=cfg.dimX)

    # 关键：dim=-1 才能同时兼容 2D/3D
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
    只对 theta / X 做标准化（和你之前一致）
    """
    theta_train = torch.stack([train_set[i][0] for i in range(len(train_set))])  # [M,4] or [B,N,4]
    X_train = torch.stack([train_set[i][1] for i in range(len(train_set))])      # [M,d] or [B,N,d]

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
      theta: [M,4] 或 [B,N,4] 或 [B,4]（广播会自动处理）
      X:     [M,d] 或 [B,N,d]
    """
    theta = (theta - norm["theta_mean"]) / norm["theta_std"]
    X = (X - norm["X_mean"]) / norm["X_std"]
    return theta, X



# 这里假设你已经把我前面给你的函数放进来了：
# - PEMCNet
# - rqmc_group_var_loss
# - normalize(theta, X, norm)

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

    for_eval:
      - False: 返回训练时用的标量loss（通常mean）
      - True : 返回 (loss_scalar, weight) 方便外部做加权平均
    """
    if X.ndim == 2:
        # ===== 原来的点级 MSE =====
        if for_eval:
            # sum 再除样本数，和你原逻辑一致
            loss = F.mse_loss(pred, y, reduction="sum")
            weight = y.size(0)
            return loss, weight
        else:
            loss = F.mse_loss(pred, y)  # mean
            return loss

    elif X.ndim == 3:
        # ===== RQMC 分组方差损失 =====
        center = getattr(cfg, "rqmc_loss_center", True)      # True=方差, False=二阶矩
        unbiased = getattr(cfg, "rqmc_loss_unbiased", False) # 训练建议 False

        loss = rqmc_group_var_loss(
            pred=pred,
            y=y,
            center=center,
            unbiased=unbiased,
            return_stats=False,
        )

        if for_eval:
            # 用 B 做权重，把不同batch的组损失做平均
            weight = X.size(0)  # B
            return loss, weight
        else:
            return loss

    else:
        raise ValueError(f"不支持的 X 维度: {X.ndim}，期望 2 或 3")

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

    unbiased:
        仅在 center=True 且想用样本方差(B-1)时生效。
        一般训练里用 unbiased=False 更稳（除以B）。
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

    # h = y - g
    h = y - pred                 # [B,N,1]

    # 每组沿 N 求均值 -> [B,1]
    hbar = h.mean(dim=1)         # [B,1]

    if center:
        if unbiased:
            # 样本方差（除 B-1）
            # hbar.var(dim=0) -> [1]
            loss = hbar.var(dim=0, unbiased=True).mean()
        else:
            # 总体方差（除 B）
            hbar_centered = hbar - hbar.mean(dim=0, keepdim=True)
            loss = (hbar_centered ** 2).mean()
    else:
        # 二阶矩
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
# 3️⃣ 训练函数（兼容 2D / 3D）
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
        # ===== 训练 =====
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

        # ===== 验证 =====
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

        # 保存最优参数（存 CPU 副本，稳）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        print(f"Epoch {epoch+1} | Val {mode_name}: {val_loss:.6f} | Best: {best_val_loss:.6f} (epoch {best_epoch})")

    return net, first_step_loss, train_losses, val_losses, best_state_dict, best_val_loss, best_epoch

# =============================
# 4️⃣ 测试函数（兼容 2D / 3D）
# =============================
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

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 3D 训练配置（你要的版本）=====
    N_group = 1024  # 每组QMC点数（你指定的 N）

    # ===== 固定一个 theta（你也可以改成 is_same=False 做每组不同theta）=====
    theta_fixed = (
        torch.tensor(0.02, device=cfg.device),    # r
        torch.tensor(100.0, device=cfg.device),   # S0
        torch.tensor(0.15, device=cfg.device),    # sigma
        torch.tensor(100.0, device=cfg.device),   # K
    )

    # sample_theta(mode=3): 返回每个参数 shape [B_total, N_group]
    r, S0, sigma, K = sample_theta(
        mode=2,
        batch_size=cfg.dataset_size,
        B=cfg.dataset_size,
        N=N_group,
        is_same=False,          # 固定同一个theta广播到所有组/点
        theta_same=theta_fixed,
        device=cfg.device
    )

    # 生成数据:
    # theta [B,N,4], X [B,N,d], y [B,N,1]
    dataset = generate_dataset(cfg, theta=(r, S0, sigma, K))

    # 划分：按“组”划分（因为 len(dataset)=B_total）
    train_set, val_set, test_set = split_dataset(dataset, cfg)

    # DataLoader 现在 batch 的单位是“组”
    # 每个 batch:
    #   theta [b,N,4], X [b,N,d], y [b,N,1]
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    # 计算标准化（会自动兼容3D）
    norm = compute_normalization(train_set, cfg)

    # 训练
    # 训练
    # 注意：这里对应 train_model 的返回值数量（你前面已经改成包含 best 信息了）
    net, first_step_loss, train_losses, val_losses, best_state_dict, best_val_loss, best_epoch = train_model(
        train_loader, val_loader, norm, cfg
    )

    # 用 best val 参数做测试（关键！）
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    # 测试（这里返回的是 group-var loss 的平均值，不是普通MSE）
    test_loss = evaluate(net, test_loader, norm, cfg)

    print("First step loss:", first_step_loss)
    print(f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print("Final Test Loss (using best-val checkpoint):", test_loss)

    # ===== 保存 =====
    # 不要先 rmtree；避免后续保存失败时目录被删空
    os.makedirs(results_dir, exist_ok=True)

    # 如果你在 CUDA 上训练，先同步一下，避免看起来像“卡住”
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 保存前转 CPU，减少保存卡顿/兼容问题
    model_state_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    norm_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm.items()}

    # 额外也保存一份 final（最后一轮）参数（可选）
    # 如果你想保存“最后一轮”而不是“best”，就把 net.load_state_dict(best_state_dict) 那行删掉
    # 这里 net 已经被加载成 best_state_dict，所以 model.pth 保存的是 best-val 模型
    print("[Save] model.pth ...")
    torch.save(model_state_cpu, f"{results_dir}/model.pth")

    # 如果想同时保存 best / final 两份，可以在 train_model 中也返回 final_state_dict，再单独存
    # 这里先按你当前需求：model.pth = best-val checkpoint

    print("[Save] normalization.pth ...")
    torch.save(norm_cpu, f"{results_dir}/normalization.pth")

    print("[Save] train_losses.npy / val_losses.npy ...")
    np.save(f"{results_dir}/train_losses.npy", np.array(train_losses, dtype=np.float32))
    np.save(f"{results_dir}/val_losses.npy", np.array(val_losses, dtype=np.float32))

    # Config 保存
    print("[Save] config.yaml ...")
    cfg_dict = vars(cfg).copy()
    cfg_dict["device"] = str(cfg_dict.get("device", "cpu"))  # 保险一点
    with open(f"{results_dir}/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # 训练摘要（建议存一下）
    summary = {
        "first_step_loss": float(first_step_loss) if first_step_loss is not None else None,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "num_train_steps": int(len(train_losses)),
        "num_epochs": int(cfg.epochs),
    }
    print("[Save] summary.json ...")
    with open(f"{results_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("All results saved.")
if __name__ == "__main__":
    main()
