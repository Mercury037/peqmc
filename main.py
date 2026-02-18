import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import json
import os
import random

from model import PEMCNet
from data_process import *


class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dimX = 16
    dataset_size = 10000

    train_ratio = 0.7
    val_ratio = 0.15

    batch_size = 128
    epochs = 100

    lr = 1e-3
    dropout = 0.1

    results_dir = "results"


cfg = Config()


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


def generate_dataset(cfg):
    r, S0, sigma, K = sample_theta(cfg.dataset_size, device=cfg.device)
    S, dW = simulate_gbm_batch((r, S0, sigma, K), cfg.dataset_size, device=cfg.device)

    PA = arithmetic_payoff(S, K).unsqueeze(1)
    X = features_from_dW(dW, dimX=cfg.dimX)
    theta = torch.stack([r, S0, sigma, K], dim=1)

    return TensorDataset(theta, X, PA)


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
        "y_mean": y_train.mean(0, keepdim=True),
        "y_std": y_train.std(0, keepdim=True).clamp_min(1e-6),
    }

    for k in norm:
        norm[k] = norm[k].to(cfg.device)

    return norm


def normalize(theta, X, y, norm):
    theta = (theta - norm["theta_mean"]) / norm["theta_std"]
    X = (X - norm["X_mean"]) / norm["X_std"]
    y = (y - norm["y_mean"]) / norm["y_std"]
    return theta, X, y


# =============================
# 3️⃣ 训练函数
# =============================

def train_model(train_loader, val_loader, norm, cfg):

    net = PEMCNet(dimX=cfg.dimX, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)

    train_losses = []
    val_losses = []
    first_step_loss = None

    for epoch in range(cfg.epochs):

        net.train()

        for step, (theta, X, y) in enumerate(train_loader):

            theta, X, y = theta.to(cfg.device), X.to(cfg.device), y.to(cfg.device)
            theta, X, y = normalize(theta, X, y, norm)

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
                theta, X, y = normalize(theta, X, y, norm)

                pred = net(theta, X)
                loss = F.mse_loss(pred, y, reduction="sum")

                val_sum += loss.item()
                val_n += y.size(0)

        val_loss = val_sum / val_n
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.6f}")

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
            theta, X, y = normalize(theta, X, y, norm)

            pred = net(theta, X)
            loss = F.mse_loss(pred, y, reduction="sum")

            total += loss.item()
            count += y.size(0)

    return total / count




def main():

    set_seed(cfg.seed)

    dataset = generate_dataset(cfg)
    train_set, val_set, test_set = split_dataset(dataset, cfg)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    norm = compute_normalization(train_set, cfg)

    net, first_step_loss, train_losses, val_losses = train_model(
        train_loader, val_loader, norm, cfg
    )

    test_loss = evaluate(net, test_loader, norm, cfg)

    print("Final Test Loss:", test_loss)

    # ===== 保存 =====
    os.makedirs(cfg.results_dir, exist_ok=True)

    torch.save(net.state_dict(), f"{cfg.results_dir}/model.pth")
    torch.save(norm, f"{cfg.results_dir}/normalization.pth")

    np.save(f"{cfg.results_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{cfg.results_dir}/val_losses.npy", np.array(val_losses))

    with open(f"{cfg.results_dir}/results.json", "w") as f:
        json.dump({
            "config": vars(cfg),
            "first_step_loss": first_step_loss,
            "final_val_loss": val_losses[-1],
            "test_loss": test_loss
        }, f, indent=4)

    print("All results saved.")


if __name__ == "__main__":
    main()
