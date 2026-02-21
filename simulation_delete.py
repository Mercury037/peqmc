import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 0) Utils
# ----------------------------
def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

@torch.no_grad()
def geometric_asian_price_undiscounted(r, S0, sigma, K, nD=252, T=1.0, device="cpu"):
    """
    E[(G - K)^+] where G = exp( (1/nD) sum log S_{t_i} ), t_i = iT/nD under GBM.
    Undiscounted expectation (paper omits discount in payoff).
    Vectorized in torch.
    """
    r = torch.as_tensor(r, device=device)
    S0 = torch.as_tensor(S0, device=device)
    sigma = torch.as_tensor(sigma, device=device)
    K = torch.as_tensor(K, device=device)

    dt = T / nD
    m = torch.log(S0) + (r - 0.5 * sigma**2) * T * (nD + 1.0) / (2.0 * nD)  # mean of log G
    s2 = (sigma ** 2) * T * (nD + 1.0) * (2.0 * nD + 1.0) / (6.0 * nD * nD)
    s = torch.sqrt(s2 + 1e-12)
    d2 = (m - torch.log(K)) / s
    d1 = d2 + s
    price = torch.exp(m + 0.5 * s2)* normal_cdf(d1) - K * normal_cdf(d2)
    return price

# ----------------------------
# 1) GBM simulation + payoff + features
# ----------------------------
def simulate_gbm_batch(theta, batch_size, nD=256, T=1.0, device="cpu"):
    """
    theta: (r,S0,sigma,K) each is tensor scalar or shape [batch]
    returns:
      S: [B, nD] sampled at t_1..t_nD
      dW: [B, nD] Brownian increments
    """
    r, S0, sigma, K = theta
    dt = T / nD
    # Brownian increments
    Z = torch.randn(batch_size, nD, device=device)
    dW = math.sqrt(dt) * Z
    W = torch.cumsum(dW, dim=1)

    t = torch.arange(1, nD + 1, device=device).float() * dt  # [nD]
    t = t.unsqueeze(0)  # [1,nD]

    logS = torch.log(S0).unsqueeze(1) + (r - 0.5 * sigma**2).unsqueeze(1) * t + sigma.unsqueeze(1) * W
    S = torch.exp(logS)
    return S, dW

def arithmetic_payoff(S, K):
    A = S.mean(dim=1)
    return torch.clamp(A - K, min=0.0) #min下界

def geometric_payoff(S, K):
    G = torch.exp(torch.log(S).mean(dim=1))
    return torch.clamp(G - K, min=0.0)

def features_from_dW(dW, dimX):
    # dW: [B,nD]
    if dimX == 1:
        return dW.sum(dim=1, keepdim=True)  # [B,1]
    elif dimX == 16:
        # 14 blocks * 18 increments
        B, nD = dW.shape
        x = dW.reshape(B, 16, 16).sum(dim=2)  # [B,14]
        return x
    else:
        raise ValueError("dimX must be 1 or 14")

def sample_theta(batch_size, device="cpu"):
    # r ∈ [0.01,0.03], S0 ∈ [80,120], sigma ∈ [0.05,0.25], K ∈ [90,110]
    r = 0.01 + (0.03 - 0.01) * torch.rand(batch_size, device=device)
    S0 = 80.0 + (120.0 - 80.0) * torch.rand(batch_size, device=device)
    sigma = 0.05 + (0.25 - 0.05) * torch.rand(batch_size, device=device)
    K = 90.0 + (110.0 - 90.0) * torch.rand(batch_size, device=device)
    return r, S0, sigma, K
