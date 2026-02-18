import torch
import torch.nn.functional as F
import numpy as np
import json

from model import PEMCNet
from data_process import *

# =============================
# 1️⃣ 读取配置
# =============================

with open("results/results.json", "r") as f:
    result_dict = json.load(f)

cfg = result_dict["config"]

device = "cuda" if torch.cuda.is_available() else "cpu"
dimX = 16

# =============================
# 2️⃣ 加载模型
# =============================

net = PEMCNet(dimX=dimX, dropout=0.0).to(device)
net.load_state_dict(torch.load("results/model.pth", map_location=device))
net.eval()

print("Model loaded.")

# =============================
# 3️⃣ 加载标准化参数
# =============================

norm = torch.load("results/normalization.pth", map_location=device)

theta_mean = norm["theta_mean"]
theta_std  = norm["theta_std"]
X_mean     = norm["X_mean"]
X_std      = norm["X_std"]
y_mean     = norm["y_mean"]
y_std      = norm["y_std"]

print("Normalization loaded.")

# =============================
# 4️⃣ 生成测试数据
# =============================

B = 10000   # 用大样本更稳定

r, S0, sigma, K = sample_theta(B, device=device)
S, dW = simulate_gbm_batch((r, S0, sigma, K), B, device=device)

PA = arithmetic_payoff(S, K).unsqueeze(1)
X = features_from_dW(dW, dimX=dimX)
theta = torch.stack([r, S0, sigma, K], dim=1)

# =============================
# 5️⃣ 标准化
# =============================

theta_norm = (theta - theta_mean) / theta_std
X_norm = (X - X_mean) / X_std

# =============================
# 6️⃣ 预测
# =============================

with torch.no_grad():
    pred_norm = net(theta_norm, X_norm)

pred = pred_norm * y_std + y_mean

# =============================
# 7️⃣ 评估
# =============================

mse = F.mse_loss(pred, PA)
rmse = torch.sqrt(mse)

bias = (pred - PA).mean()

# R²
ss_tot = ((PA - PA.mean()) ** 2).sum()
ss_res = ((PA - pred) ** 2).sum()
r2 = 1 - ss_res / ss_tot

print("========== Evaluation ==========")
print(f"Prediction mean : {pred.mean().item():.6f}")
print(f"True mean       : {PA.mean().item():.6f}")
print(f"Bias            : {bias.item():.6f}")
print(f"MSE             : {mse.item():.6f}")
print(f"RMSE            : {rmse.item():.6f}")
print(f"R^2             : {r2.item():.6f}")
