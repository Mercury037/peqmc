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
# 2) PEMC 网络（可选是否使用 theta）
#    - use_theta=True : 支持 (theta, X)
#    - use_theta=False: 支持 (X) 或 (theta, X)【theta会被忽略】
#
# 支持:
#   X:[M,d]   -> out:[M,1]
#   X:[B,N,d] -> out:[B,N,1]
# ============================================================
class PEMCNet(nn.Module):
    def __init__(
        self,
        dimX: int,
        dropout: float = 0.0,
        use_bn: bool = False,
        use_theta: bool = True,      # <<< 新增开关
        theta_hidden: int = 64,
        theta_out: int = 10,
        combined_hidden: int = 128,
        x_width_min: int = 32,
        x_width_mul: int = 2,
    ):
        super().__init__()
        self.dimX = dimX
        self.use_theta = bool(use_theta)
        self.theta_out = theta_out if self.use_theta else 0

        # ---------------- theta branch（可选） ----------------
        if self.use_theta:
            self.theta_branch = MLP(
                in_dim=4,
                hidden_dim=theta_hidden,
                out_dim=theta_out,
                dropout=dropout,
                use_bn=use_bn,
            )
        else:
            self.theta_branch = None

        # ---------------- X branch ----------------
        w = max(x_width_min, x_width_mul * dimX)
        self.w = w

        self.x_fc1 = nn.Linear(dimX, w)
        self.x_fc2 = nn.Linear(w, w)
        self.x_bn1 = nn.BatchNorm1d(w) if use_bn else nn.Identity()
        self.x_bn2 = nn.BatchNorm1d(w) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout)

        # ---------------- combined + skip ----------------
        h = combined_hidden
        in_fuse = w + (theta_out if self.use_theta else 0)

        self.c_fc1 = nn.Linear(in_fuse, h)
        self.c_fc2 = nn.Linear(h, h)
        self.c_bn1 = nn.BatchNorm1d(h) if use_bn else nn.Identity()
        self.c_bn2 = nn.BatchNorm1d(h) if use_bn else nn.Identity()
        self.skip = nn.Linear(in_fuse, h)
        self.out = nn.Linear(h, 1)

    # ---------- X branch 前向 ----------
    def _x_forward_2d(self, X_2d: torch.Tensor) -> torch.Tensor:
        """
        X_2d: [M, dimX]
        return: [M, w]
        """
        x = self.x_fc1(X_2d)
        x = self.x_bn1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.x_fc2(x)
        x = self.x_bn2(x)
        x = F.relu(x)
        x = self.drop(x)
        return x

    # ---------- 内部：二维输入前向 ----------
    def _forward_2d(self, X_2d: torch.Tensor, theta_2d: torch.Tensor = None) -> torch.Tensor:
        """
        X_2d:     [M, dimX]
        theta_2d: [M, 4] 或 None
        return:   [M, 1]
        """
        # x branch
        x = self._x_forward_2d(X_2d)  # [M, w]

        # theta branch（可选）
        if self.use_theta:
            if theta_2d is None:
                raise ValueError("当前模型 use_theta=True，必须提供 theta。")
            t = self.theta_branch(theta_2d)  # [M, theta_out]
            z = torch.cat([t, x], dim=1)     # [M, theta_out+w]
        else:
            # 不用 theta，直接用 x
            z = x                             # [M, w]

        # combine
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
            assert theta.shape[0] == 4, f"theta若为1维，应是[4]，got {theta.shape}"
            theta = theta.to(device=device, dtype=dtype)
            theta = theta.view(1, 1, 4).expand(B, N, 4)

        elif theta.ndim == 2:
            assert theta.shape == (B, 4), f"theta若为2维，应是[B,4]，got {theta.shape}, B={B}"
            theta = theta.to(device=device, dtype=dtype)
            theta = theta.unsqueeze(1).expand(B, N, 4)

        elif theta.ndim == 3:
            assert theta.shape == (B, N, 4), f"theta若为3维，应是[B,N,4]，got {theta.shape}"
            theta = theta.to(device=device, dtype=dtype)

        else:
            raise ValueError(f"theta维度不支持: {theta.ndim}")

        return theta

    # ---------- 对外 forward ----------
    def forward(self, *args) -> torch.Tensor:
        """
        支持两种调用方式：

        1) net(theta, X)
           - 兼容旧代码
           - 如果 use_theta=False，会忽略 theta，只用 X

        2) net(X)
           - 仅当 use_theta=False 时建议使用
           - 如果 use_theta=True 会报错（因为缺 theta）

        X 支持：
          [M, dimX]   -> 输出 [M, 1]
          [B, N, dimX] -> 输出 [B, N, 1]
        """
        if len(args) == 1:
            theta = None
            X = args[0]
        elif len(args) == 2:
            theta, X = args
        else:
            raise ValueError(f"forward 参数数量不支持: {len(args)}，应为 net(X) 或 net(theta, X)")

        if not torch.is_tensor(X):
            raise TypeError("X 必须是 torch.Tensor")

        # ===== 点模式 =====
        if X.ndim == 2:
            M, d = X.shape
            assert d == self.dimX, f"X.shape={X.shape}, 最后一维应为 dimX={self.dimX}"

            if self.use_theta:
                if theta is None:
                    raise ValueError("use_theta=True 时，点模式必须提供 theta。")
                if theta.ndim == 1:
                    assert theta.shape[0] == 4, f"theta若为1维，应是[4]，got {theta.shape}"
                    theta = theta.view(1, 4).to(device=X.device, dtype=X.dtype).expand(M, 4)
                elif theta.ndim == 2:
                    assert theta.shape == (M, 4), f"theta若为2维，应是[M,4]，got {theta.shape}, M={M}"
                    theta = theta.to(device=X.device, dtype=X.dtype)
                else:
                    raise ValueError(f"点模式下 theta 维度不支持: {theta.ndim}")
                return self._forward_2d(X, theta)
            else:
                # use_theta=False：忽略 theta（若传了也无所谓）
                return self._forward_2d(X, None)

        # ===== RQMC组模式 =====
        elif X.ndim == 3:
            B, N, d = X.shape
            assert d == self.dimX, f"X.shape={X.shape}, 最后一维应为 dimX={self.dimX}"

            X_flat = X.reshape(B * N, d)  # [BN,d]

            if self.use_theta:
                if theta is None:
                    raise ValueError("use_theta=True 时，3D模式必须提供 theta。")
                theta_3d = self._expand_theta_for_3d(
                    theta=theta, B=B, N=N, device=X.device, dtype=X.dtype
                )
                theta_flat = theta_3d.reshape(B * N, 4)        # [BN,4]
                out_flat = self._forward_2d(X_flat, theta_flat)  # [BN,1]
            else:
                out_flat = self._forward_2d(X_flat, None)        # [BN,1]

            out = out_flat.reshape(B, N, 1)  # [B,N,1]
            return out

        else:
            raise ValueError(f"X 维度不支持: {X.ndim}, 期望2维或3维")