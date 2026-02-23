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
















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from simulation_mc import *
# class MLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, out_dim)
#         self.bn2 = nn.BatchNorm1d(out_dim)
#         self.drop = nn.Dropout(dropout)
#
#     def forward(self, x):
#         x = self.drop(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop(F.relu(self.bn2(self.fc2(x))))
#         return x
#
#
# class PEMCNet(nn.Module):
#     def __init__(self, dimX, dropout=0.1):
#         super().__init__()
#         # theta branch: 4 -> (hidden) -> 10  (paper: two FC + BN + ReLU, output 10)
#         self.theta_branch = MLP(4, 64, 10, dropout=dropout)
#
#         # X branch: width max(32, 2*dimX), two FC
#         w = max(32, 2 * dimX)
#         self.x_fc1 = nn.Linear(dimX, w)
#         self.x_bn1 = nn.BatchNorm1d(w)
#         self.x_fc2 = nn.Linear(w, w)
#         self.x_bn2 = nn.BatchNorm1d(w)
#         self.drop = nn.Dropout(dropout)
#
#         # combined with skip (ResNet-ish): (10+w) -> h -> h -> 1
#         h = 128
#         self.c_fc1 = nn.Linear(10 + w, h)
#         self.c_bn1 = nn.BatchNorm1d(h)
#         self.c_fc2 = nn.Linear(h, h)
#         self.c_bn2 = nn.BatchNorm1d(h)
#         self.skip = nn.Linear(10 + w, h)
#         self.out = nn.Linear(h, 1)
#
#     def forward(self, theta, X):
#         t = self.theta_branch(theta)
#
#         x = self.drop(F.relu(self.x_bn1(self.x_fc1(X))))
#         x = self.drop(F.relu(self.x_bn2(self.x_fc2(x))))
#
#         z = torch.cat([t, x], dim=1)
#
#         y = self.drop(F.relu(self.c_bn1(self.c_fc1(z))))
#         y = self.drop(F.relu(self.c_bn2(self.c_fc2(y))))
#         y = y + self.skip(z)
#
#         return self.out(F.relu(y))
#










import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) 基础 MLP 模块（可选 BN / Dropout）
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0, use_bn=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.use_bn = use_bn
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [M, in_dim]
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop(x)
        return x


# ============================================================
# 2) PEMC 网络
#    - 支持 X:[M,d] -> out:[M,1]
#    - 支持 X:[B,N,d] -> out:[B,N,1]
# ============================================================
class PEMCNet(nn.Module):
    def __init__(
        self,
        dimX: int,
        dropout: float = 0.0,
        use_bn: bool = False,
        theta_hidden: int = 64,
        theta_out: int = 10,
        combined_hidden: int = 128,
        x_width_min: int = 32,
        x_width_mul: int = 2,
    ):
        super().__init__()
        self.dimX = dimX
        self.theta_out = theta_out

        # theta branch: 4 -> hidden -> theta_out
        self.theta_branch = MLP(
            in_dim=4,
            hidden_dim=theta_hidden,
            out_dim=theta_out,
            dropout=dropout,
            use_bn=use_bn,
        )

        # X branch
        w = max(x_width_min, x_width_mul * dimX)
        self.w = w

        self.x_fc1 = nn.Linear(dimX, w)
        self.x_fc2 = nn.Linear(w, w)
        self.x_bn1 = nn.BatchNorm1d(w) if use_bn else nn.Identity()
        self.x_bn2 = nn.BatchNorm1d(w) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout)

        # combined + skip
        h = combined_hidden
        self.c_fc1 = nn.Linear(theta_out + w, h)
        self.c_fc2 = nn.Linear(h, h)
        self.c_bn1 = nn.BatchNorm1d(h) if use_bn else nn.Identity()
        self.c_bn2 = nn.BatchNorm1d(h) if use_bn else nn.Identity()
        self.skip = nn.Linear(theta_out + w, h)
        self.out = nn.Linear(h, 1)

    # ---------- 内部：二维输入前向 ----------
    def _forward_2d(self, theta_2d: torch.Tensor, X_2d: torch.Tensor) -> torch.Tensor:
        """
        theta_2d: [M,4]
        X_2d:     [M,dimX]
        return:   [M,1]
        """
        # theta branch
        t = self.theta_branch(theta_2d)  # [M,theta_out]

        # x branch
        x = self.x_fc1(X_2d)
        x = self.x_bn1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.x_fc2(x)
        x = self.x_bn2(x)
        x = F.relu(x)
        x = self.drop(x)

        # combine
        z = torch.cat([t, x], dim=1)  # [M, theta_out+w]

        y = self.c_fc1(z)
        y = self.c_bn1(y)
        y = F.relu(y)
        y = self.drop(y)

        y = self.c_fc2(y)
        y = self.c_bn2(y)
        y = F.relu(y)
        y = self.drop(y)

        y = y + self.skip(z)
        y = F.relu(y)

        return self.out(y)  # [M,1]

    # ---------- theta 形状处理 ----------
    def _expand_theta_for_3d(self, theta: torch.Tensor, B: int, N: int, device, dtype) -> torch.Tensor:
        """
        输入 theta 支持：
          [4]       -> broadcast 到 [B,N,4]
          [B,4]     -> 每组一个 theta，broadcast 到 [B,N,4]
          [B,N,4]   -> 直接使用
        返回:
          theta_3d: [B,N,4]
        """
        if theta.ndim == 1:
            # [4]
            assert theta.shape[0] == 4, f"theta若为1维，应是[4]，got {theta.shape}"
            theta = theta.to(device=device, dtype=dtype)
            theta = theta.view(1, 1, 4).expand(B, N, 4)

        elif theta.ndim == 2:
            # [B,4]
            assert theta.shape == (B, 4), f"theta若为2维，应是[B,4]，got {theta.shape}, B={B}"
            theta = theta.to(device=device, dtype=dtype)
            theta = theta.unsqueeze(1).expand(B, N, 4)

        elif theta.ndim == 3:
            # [B,N,4]
            assert theta.shape == (B, N, 4), f"theta若为3维，应是[B,N,4]，got {theta.shape}"
            theta = theta.to(device=device, dtype=dtype)

        else:
            raise ValueError(f"theta维度不支持: {theta.ndim}")

        return theta

    # ---------- 对外 forward ----------
    def forward(self, theta: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        支持两种模式：

        (A) 点模式（兼容旧代码）
            X:     [M, dimX]
            theta: [M, 4] 或 [4]
            返回:  [M, 1]

        (B) RQMC组模式（你现在要的）
            X:     [B, N, dimX]
            theta: [4] / [B,4] / [B,N,4]
            返回:  [B, N, 1]
        """
        if X.ndim == 2:
            # 点模式
            M, d = X.shape
            assert d == self.dimX, f"X.shape={X.shape}, 最后一维应为 dimX={self.dimX}"

            if theta.ndim == 1:
                assert theta.shape[0] == 4, f"theta若为1维，应是[4]，got {theta.shape}"
                theta = theta.view(1, 4).to(device=X.device, dtype=X.dtype).expand(M, 4)
            elif theta.ndim == 2:
                assert theta.shape == (M, 4), f"theta若为2维，应是[M,4]，got {theta.shape}, M={M}"
                theta = theta.to(device=X.device, dtype=X.dtype)
            else:
                raise ValueError(f"点模式下 theta 维度不支持: {theta.ndim}")

            return self._forward_2d(theta, X)

        elif X.ndim == 3:
            # RQMC组模式
            B, N, d = X.shape
            assert d == self.dimX, f"X.shape={X.shape}, 最后一维应为 dimX={self.dimX}"

            theta_3d = self._expand_theta_for_3d(
                theta=theta, B=B, N=N, device=X.device, dtype=X.dtype
            )

            # 展平做点函数前向，再 reshape 回来
            X_flat = X.reshape(B * N, d)              # [BN,d]
            theta_flat = theta_3d.reshape(B * N, 4)   # [BN,4]
            out_flat = self._forward_2d(theta_flat, X_flat)  # [BN,1]
            out = out_flat.reshape(B, N, 1)                 # [B,N,1]
            return out

        else:
            raise ValueError(f"X 维度不支持: {X.ndim}, 期望2维或3维")