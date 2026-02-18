import torch
import torch.nn.functional as F
import numpy as np

from model import PEMCNet
from data_process import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 1️⃣ 加载模型
# =====================

dimX = 16   # 必须和训练时一致

net = PEMCNet(dimX=dimX, dropout=0.0).to(device)

state_dict = torch.load("results/model.pth", map_location=device)
net.load_state_dict(state_dict)

net.eval()

print("Model loaded.")

# =====================
# 2️⃣ 加载标准化参数
# =====================

norm_dict = torch.load("results/normalization.pth", map_location=device)


theta_mean = norm_dict["theta_mean"]
theta_std  = norm_dict["theta_std"]
X_mean = norm_dict["X_mean"]
X_std  = norm_dict["X_std"]
y_mean = norm_dict["y_mean"]
y_std  = norm_dict["y_std"]


print("Normalization loaded.")

# =====================
# 3️⃣ 生成新测试数据
# =====================

B = 128

r, S0, sigma, K = sample_theta(B, device=device)
S, dW = simulate_gbm_batch((r, S0, sigma, K), B, device=device)

PA = arithmetic_payoff(S, K).unsqueeze(1)

X = features_from_dW(dW, dimX=dimX)
theta = torch.stack([r, S0, sigma, K], dim=1)

# =====================
# 4️⃣ 标准化输入
# =====================

theta_norm = (theta - theta_mean) / theta_std
X_norm = (X - X_mean) / X_std

# =====================
# 5️⃣ 预测
# =====================

with torch.no_grad():
    pred_norm = net(theta_norm, X_norm)

# 反标准化输出
pred = pred_norm * y_std + y_mean

# =====================
# 6️⃣ 输出结果
# =====================

print("Prediction mean:", pred.mean().item())
print("True mean:", PA.mean().item())

mse = F.mse_loss(pred, PA)
print("MSE on new batch:", mse.item())
