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
from data_process import *
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

# ----------------------------
# 3) Training (streaming, no dataset stored)
# ----------------------------
def train_pemc(dimX=14, label_mode="PA", steps=20000, batch_size=4096, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    label_mode:
      "PA"        -> train g to predict arithmetic payoff P_A
      "PA_minus_PG" -> train g to predict (P_A - P_G) for Boost PEMC
    """
    net = PEMCNet(dimX=dimX, dropout=0.5).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    for step in range(1, steps + 1):
        r, S0, sigma, K = sample_theta(batch_size, device=device)
        S, dW = simulate_gbm_batch((r, S0, sigma, K), batch_size, device=device)
        PA = arithmetic_payoff(S, K)
        PG = geometric_payoff(S, K)

        if label_mode == "PA":
            y = PA
        elif label_mode == "PA_minus_PG":
            y = PA - PG
        else:
            raise ValueError("label_mode must be 'PA' or 'PA_minus_PG'")

        X = features_from_dW(dW, dimX=dimX)
        theta = torch.stack([r, S0, sigma, K], dim=1)
        # print("theta", theta.shape, theta.dtype, theta.device)
        # print("X    ", X.shape, X.dtype, X.device)
        # print("net  ", next(net.parameters()).dtype, next(net.parameters()).device)
        print("round: ", step)
        pred = net(theta, X)
        loss = F.mse_loss(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    return net

# ----------------------------
# 4) Evaluation: MC / PEMC / Geometric CV / Boost PEMC
# ----------------------------
@torch.no_grad()
def eval_estimators(net_pa, net_boost, dimX, n=4000, N_ratio=10, reps=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    reps: repeat count (paper uses 300)
    """
    net_pa.eval()
    if net_boost is not None:
        net_boost.eval()

    # fixed theta for eval (paper)
    r0, S00, sig0, K0 = 0.02, 100.0, 0.2, 100.0
    r = torch.full((n,), r0, device=device)
    S0 = torch.full((n,), S00, device=device)
    sigma = torch.full((n,), sig0, device=device)
    K = torch.full((n,), K0, device=device)

    # exact geometric expectation (undiscounted)
    PG_exact = geometric_asian_price_undiscounted(r0, S00, sig0, K0, device=device).item()

    mc_list, pemc_list, cv_list, boost_list = [], [], [], []

    for _ in range(reps):
        # coupled samples for f(Y)-g(X)
        S, dW = simulate_gbm_batch((r, S0, sigma, K), n, device=device)
        PA = arithmetic_payoff(S, K)
        PG = geometric_payoff(S, K)
        X = features_from_dW(dW, dimX=dimX)
        theta = torch.stack([r, S0, sigma, K], dim=1)

        mc = PA.mean().item()
        g = net_pa(theta, X)
        part1 = (PA - g).mean().item()

        # independent X~ samples (cheap marginal)
        N = int(N_ratio * n)
        if dimX == 1:
            # W_T ~ N(0,T). Here T=1 in simulation code.
            Xtilde = torch.randn(N, 1, device=device)
        else:
            # 14 independent blocks, each sum of 18 increments: Var = 18*dt = 18/252 = 1/14
            Xtilde = math.sqrt(1.0/14.0) * torch.randn(N, 14, device=device)

        theta_tilde = torch.tensor([r0, S00, sig0, K0], device=device).view(1,4).repeat(N,1)
        gtilde = net_pa(theta_tilde, Xtilde).mean().item()
        pemc = part1 + gtilde

        # geometric CV
        cv = (PA - PG).mean().item() + PG_exact

        mc_list.append(mc)
        pemc_list.append(pemc)
        cv_list.append(cv)

        # Boost PEMC (optional)
        if net_boost is not None:
            # train on (PA-PG), apply PEMC to that, then + PG_exact
            diff = (PA - PG)
            g_diff = net_boost(theta, X)
            part1b = (diff - g_diff).mean().item()

            gtilde_b = net_boost(theta_tilde, Xtilde).mean().item()
            boost = part1b + gtilde_b + PG_exact
            boost_list.append(boost)

    return {
        "PG_exact": PG_exact,
        "MC": mc_list,
        "PEMC": pemc_list,
        "GeoCV": cv_list,
        "BoostPEMC": boost_list if net_boost is not None else None
    }



if __name__ == "__main__":
    dimX = 14
    net_pa = train_pemc(dimX=dimX, label_mode="PA", steps=2000, batch_size=2048)
    net_boost = train_pemc(dimX=dimX, label_mode="PA_minus_PG", steps=2000, batch_size=2048)

    out = eval_estimators(net_pa=net_pa, net_boost=net_boost, dimX=dimX, n=4000, N_ratio=10, reps=30)
    print("done:", out.keys())