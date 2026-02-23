import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import json
import os
import random
import shutil
import copy  # 新增：用于保存最优参数的深拷贝

from torch.xpu import device

from model import PEMCNet
# from simulation_mc import *
from simulation_qmc import *
import yaml


# 获取当前工作目录（CWD）
current_working_directory = os.getcwd()
print("CWD =", current_working_directory)
# 保存路径
results_dir = os.path.join(current_working_directory, "results_qmc_pca_mse")
print("Results directory:", results_dir)


class Config:
    def __init__(self):
        self.seed = 42  # 随机种子
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 设备

        self.dimX = 4  # 特征维度
        self.dataset_size = 2 ** 20  # 样本量

        self.train_ratio = 0.7  # 训练集比例
        self.val_ratio = 0.15  # 验证集比例

        self.batch_size = 512  # sgd 的 batch
        self.epochs = 150  # 进行轮数

        self.lr = 1e-3  # 初始学习率
        self.dropout = 0.3


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


# def generate_dataset(cfg):
#     r, S0, sigma, K = sample_theta(cfg.dataset_size, device=cfg.device)
#     S, dW = simulate_gbm_batch((r, S0, sigma, K), cfg.dataset_size, device=cfg.device)
#
#     PA = arithmetic_payoff(S, K).unsqueeze(1)
#     X = features_from_dW(dW, dimX=cfg.dimX)
#     theta = torch.stack([r, S0, sigma, K], dim=1)
#
#     return TensorDataset(theta, X, PA)


def generate_dataset(cfg, theta):
    r, S0, sigma, K = theta

    Z, W, S = simulate_gbm_batch_qmc((r, S0, sigma, K), method="pca", device=cfg.device)
    PA = arithmetic_payoff(S, K).unsqueeze(1)
    X = features_from_Z(Z, dimX=cfg.dimX)
    theta = torch.stack([r, S0, sigma, K], dim=1)
    dataset = TensorDataset(theta, X, PA)
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


def compute_normalization(train_set, cfg):
    theta_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    X_train = torch.stack([train_set[i][1] for i in range(len(train_set))])
    y_train = torch.stack([train_set[i][2] for i in range(len(train_set))])

    norm = {
        "theta_mean": theta_train.mean(0, keepdim=True),
        "theta_std": theta_train.std(0, keepdim=True).clamp_min(1e-6),
        "X_mean": X_train.mean(0, keepdim=True),
        "X_std": X_train.std(0, keepdim=True).clamp_min(1e-6),
    }

    for k in norm:
        norm[k] = norm[k].to(cfg.device)

    return norm


def normalize(theta, X, norm):
    theta = (theta - norm["theta_mean"]) / norm["theta_std"]
    X = (X - norm["X_mean"]) / norm["X_std"]
    return theta, X


# =============================
# 3️⃣ 训练函数
# =============================

def train_model(train_loader, val_loader, norm, cfg):

    net = PEMCNet(dimX=cfg.dimX, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    first_step_loss = None

    # ===== 新增：记录最优验证集模型 =====
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    for epoch in range(cfg.epochs):

        net.train()

        for step, (theta, X, y) in enumerate(train_loader):

            theta, X, y = theta.to(cfg.device), X.to(cfg.device), y.to(cfg.device)
            theta, X = normalize(theta, X, norm)

            pred = net(theta, X)
            loss = F.mse_loss(pred, y)

            if epoch == 0 and step == 0:
                first_step_loss = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # ===== 验证 =====
        net.eval()
        val_sum, val_n = 0.0, 0

        with torch.no_grad():
            for theta, X, y in val_loader:
                theta, X, y = theta.to(cfg.device), X.to(cfg.device), y.to(cfg.device)
                theta, X = normalize(theta, X, norm)

                pred = net(theta, X)
                loss = F.mse_loss(pred, y, reduction="sum")

                val_sum += loss.item()
                val_n += y.size(0)

        val_loss = val_sum / val_n
        val_losses.append(val_loss)

        # ===== 新增：保存最优参数（按验证集 loss）=====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1  # 记录为 1-based epoch
            # 用深拷贝保存参数，避免后续训练覆盖
            best_state_dict = copy.deepcopy(net.state_dict())

        print(
            f"Epoch {epoch+1} | Val Loss: {val_loss:.6f}"
            + ("  <-- best" if (epoch + 1) == best_epoch else "")
        )

    # ===== 新增：训练结束后恢复到最优参数 =====
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
        print(f"\nBest model restored from epoch {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
    else:
        print("\nWarning: best_state_dict is None, using last epoch weights.")

    return net, first_step_loss, train_losses, val_losses


# =============================
# 4️⃣ 测试函数
# =============================

def evaluate(net, loader, norm, cfg):

    net.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for theta, X, y in loader:
            theta, X, y = theta.to(cfg.device), X.to(cfg.device), y.to(cfg.device)
            theta, X = normalize(theta, X, norm)

            pred = net(theta, X)
            loss = F.mse_loss(pred, y, reduction="sum")

            total += loss.item()
            count += y.size(0)

    return total / count


def main():
    set_seed(cfg.seed)
    theta = (
        torch.tensor(0.02, device=cfg.device),   # r
        torch.tensor(100.0, device=cfg.device),  # S0
        torch.tensor(0.15, device=cfg.device),   # sigma
        torch.tensor(100.0, device=cfg.device),  # K
    )
    r, S0, sigma, K = sample_theta(
        mode=2,
        batch_size=cfg.dataset_size,
        is_same=False,
        theta_same=theta,
        device=device
    )

    dataset = generate_dataset(cfg, theta=(r, S0, sigma, K))

    train_set, val_set, test_set = split_dataset(dataset, cfg)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    norm = compute_normalization(train_set, cfg)

    net, first_step_loss, train_losses, val_losses = train_model(
        train_loader, val_loader, norm, cfg
    )

    # 注意：这里 evaluate 的 net 已经是“最优验证集参数”恢复后的 net
    test_loss = evaluate(net, test_loader, norm, cfg)

    print("Final Test Loss:", test_loss)

    # ===== 保存 =====
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 保存模型参数 和 norm用训练集估计的均值
    # 此时保存的是“最优网络”
    torch.save(net.state_dict(), f"{results_dir}/model.pth")
    torch.save(norm, f"{results_dir}/normalization.pth")

    # 训练损失
    np.save(f"{results_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{results_dir}/val_losses.npy", np.array(val_losses))

    with open(f"{results_dir}/config.yaml", "w") as f:
        yaml.dump(vars(cfg), f, default_flow_style=False, allow_unicode=True)

    print("All results saved.")


if __name__ == "__main__":
    main()