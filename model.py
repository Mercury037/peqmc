# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class MLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, out_dim)
#         self.bn2 = nn.BatchNorm1d(out_dim)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.drop(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop(F.relu(self.bn2(self.fc2(x))))
#         return x


# class PEMCNet(nn.Module):
#     def __init__(self, dimX, dropout=0.1):
#         super().__init__()

#         # ===== theta branch =====
#         self.theta_branch = MLP(4, 64, 64, dropout=dropout)

#         # ===== X branch: Conv1D =====
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)

#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)

#         self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(64)

#         self.pool = nn.AdaptiveAvgPool1d(1)

#         self.drop = nn.Dropout(dropout)

#         # ===== FiLM layers =====
#         # 生成 gamma 和 beta
#         self.gamma_fc = nn.Linear(64, 64)
#         self.beta_fc  = nn.Linear(64, 64)

#         # ===== regression head =====
#         self.fc1 = nn.Linear(64, 128)
#         self.bn_fc1 = nn.BatchNorm1d(128)

#         self.fc2 = nn.Linear(128, 128)
#         self.bn_fc2 = nn.BatchNorm1d(128)

#         self.out = nn.Linear(128, 1)

#     def forward(self, theta, X):

#         # ===== theta encoding =====
#         t = self.theta_branch(theta)  # [B,64]

#         # ===== X encoding =====
#         x = X.unsqueeze(1)            # [B,1,dimX]

#         x = self.drop(F.relu(self.bn1(self.conv1(x))))
#         x = self.drop(F.relu(self.bn2(self.conv2(x))))
#         x = self.drop(F.relu(self.bn3(self.conv3(x))))

#         x = self.pool(x)              # [B,64,1]
#         x = x.squeeze(-1)             # [B,64]

#         # ===== FiLM modulation =====
#         gamma = self.gamma_fc(t)      # [B,64]
#         beta  = self.beta_fc(t)       # [B,64]

#         x = gamma * x + beta          

#         # ===== regression =====
#         y = self.drop(F.relu(self.bn_fc1(self.fc1(x))))
#         y = self.drop(F.relu(self.bn_fc2(self.fc2(y))))

#         return self.out(y)            # [B,1]



import torch
import torch.nn as nn
import torch.functional as F
from simulation_mc import *
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return x


class PEMCNet(nn.Module):
    def __init__(self, dimX, dropout=0.1):
        super().__init__()
        # theta branch: 4 -> (hidden) -> 10  (paper: two FC + BN + ReLU, output 10)
        self.theta_branch = MLP(4, 64, 10, dropout=dropout)

        # X branch: width max(32, 2*dimX), two FC
        w = max(32, 2 * dimX)
        self.x_fc1 = nn.Linear(dimX, w)
        self.x_bn1 = nn.BatchNorm1d(w)
        self.x_fc2 = nn.Linear(w, w)
        self.x_bn2 = nn.BatchNorm1d(w)
        self.drop = nn.Dropout(dropout)

        # combined with skip (ResNet-ish): (10+w) -> h -> h -> 1
        h = 128
        self.c_fc1 = nn.Linear(10 + w, h)
        self.c_bn1 = nn.BatchNorm1d(h)
        self.c_fc2 = nn.Linear(h, h)
        self.c_bn2 = nn.BatchNorm1d(h)
        self.skip = nn.Linear(10 + w, h)
        self.out = nn.Linear(h, 1)

    def forward(self, theta, X):
        t = self.theta_branch(theta)

        x = self.drop(F.relu(self.x_bn1(self.x_fc1(X))))
        x = self.drop(F.relu(self.x_bn2(self.x_fc2(x))))

        z = torch.cat([t, x], dim=1)

        y = self.drop(F.relu(self.c_bn1(self.c_fc1(z))))
        y = self.drop(F.relu(self.c_bn2(self.c_fc2(y))))
        y = y + self.skip(z)

        return self.out(F.relu(y))
