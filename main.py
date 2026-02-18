import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import json
import numpy as np
import os
import random

from model import *
from data_process import *

# ========================
# 0️⃣ 固定随机种子
# ========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# 1️⃣ 生成数据
# ========================

dimX = 16
N = 10000

r, S0, sigma, K = sample_theta(N, device=device)
S, dW = simulate_gbm_batch((r, S0, sigma, K), N, device=device)

PA = arithmetic_payoff(S, K)

y = PA.unsqueeze(1)
X = features_from_dW(dW, dimX=dimX)
theta = torch.stack([r, S0, sigma, K], dim=1)

dataset = TensorDataset(theta, X, y)

# ========================
# 2️⃣ 划分数据
# ========================

train_size = int(0.7 * N)
val_size   = int(0.15 * N)
test_size  = N - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size)
test_loader  = DataLoader(test_set, batch_size=batch_size)

# ========================
# 3️⃣ 计算标准化参数（仅用 train）
# ========================

theta_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
X_train     = torch.stack([train_set[i][1] for i in range(len(train_set))])
y_train     = torch.stack([train_set[i][2] for i in range(len(train_set))])

theta_mean = theta_train.mean(0, keepdim=True)
theta_std  = theta_train.std(0, keepdim=True).clamp_min(1e-6)

X_mean = X_train.mean(0, keepdim=True)
X_std  = X_train.std(0, keepdim=True).clamp_min(1e-6)

y_mean = y_train.mean(0, keepdim=True)
y_std  = y_train.std(0, keepdim=True).clamp_min(1e-6)

theta_mean, theta_std = theta_mean.to(device), theta_std.to(device)
X_mean, X_std = X_mean.to(device), X_std.to(device)
y_mean, y_std = y_mean.to(device), y_std.to(device)

def normalize(theta, X, y):
    theta = (theta - theta_mean) / theta_std
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    return theta, X, y

# ========================
# 4️⃣ 初始化模型
# ========================

net = PEMCNet(dimX=dimX, dropout=0.1).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

epochs = 50

train_step_losses = []
val_epoch_losses = []

first_step_loss = None

# ========================
# 5️⃣ 训练
# ========================

for epoch in range(1, epochs + 1):

    net.train()

    for step, (theta_batch, X_batch, y_batch) in enumerate(train_loader):

        theta_batch = theta_batch.to(device)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        theta_batch, X_batch, y_batch = normalize(theta_batch, X_batch, y_batch)

        pred = net(theta_batch, X_batch)
        loss = F.mse_loss(pred, y_batch)

        if epoch == 1 and step == 0:
            first_step_loss = loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_step_losses.append(loss.item())

    # ===== 验证 =====

    net.eval()
    val_sum, val_n = 0.0, 0

    with torch.no_grad():
        for theta_batch, X_batch, y_batch in val_loader:

            theta_batch = theta_batch.to(device)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            theta_batch, X_batch, y_batch = normalize(theta_batch, X_batch, y_batch)

            pred = net(theta_batch, X_batch)
            loss = F.mse_loss(pred, y_batch, reduction="sum")

            val_sum += loss.item()
            val_n += y_batch.size(0)

    val_loss = val_sum / val_n
    val_epoch_losses.append(val_loss)

    print(f"Epoch {epoch} | Val Loss: {val_loss:.6f}")

# ========================
# 6️⃣ 测试
# ========================

net.eval()
test_sum, test_n = 0.0, 0

with torch.no_grad():
    for theta_batch, X_batch, y_batch in test_loader:

        theta_batch = theta_batch.to(device)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        theta_batch, X_batch, y_batch = normalize(theta_batch, X_batch, y_batch)

        pred = net(theta_batch, X_batch)
        loss = F.mse_loss(pred, y_batch, reduction="sum")

        test_sum += loss.item()
        test_n += y_batch.size(0)

test_loss = test_sum / test_n

print("First Step Loss:", first_step_loss)
print("Final Val Loss:", val_epoch_losses[-1])
print("Final Test Loss:", test_loss)

# ========================
# 7️⃣ 保存结果
# ========================

os.makedirs("results", exist_ok=True)

torch.save(net.state_dict(), "results/model.pth")

torch.save({
    "theta_mean": theta_mean,
    "theta_std": theta_std,
    "X_mean": X_mean,
    "X_std": X_std,
    "y_mean": y_mean,
    "y_std": y_std
}, "results/normalization.pth")

np.save("results/train_step_losses.npy", np.array(train_step_losses))
np.save("results/val_epoch_losses.npy", np.array(val_epoch_losses))

with open("results/test_result.json", "w") as f:
    json.dump({
        "seed": 42,
        "N": N,
        "first_step_loss": first_step_loss,
        "final_val_loss": val_epoch_losses[-1],
        "test_loss": test_loss,
        "y_mean": y_mean.item(),
        "y_std": y_std.item()
    }, f)

print("All results saved.")
