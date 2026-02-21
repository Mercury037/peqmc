import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.quasirandom import SobolEngine


def Cholesky(n: int, T: float = 1) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    dt_sqrt = math.sqrt(T / n)
    L = np.tril(np.ones((n, n), dtype=float)) * dt_sqrt
    return L

def PCA(n: int, T:float =1) -> np.ndarray:
    """
    PCA/谱分解法生成 Brownian Motion 的生成矩阵 A（不含0点，共 n 个点）：
      C_{ij} = min(t_i, t_j)
      C = V diag(lam) V^T
      A = V diag(sqrt(lam))
    并且按特征值从大到小排序，使第1列对应最大特征值（第一主成分）。

    返回 A，shape = (n, n)
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    t = np.linspace(1.0 / n, 1.0, n)

    # 协方差矩阵 C_{ij} = min(t_i, t_j)
    C = np.minimum.outer(t, t)

    # 对称特征分解：eigh 返回升序特征值
    evals, evecs = np.linalg.eigh(C)

    # 截断负特征值（数值误差）
    evals = np.maximum(evals, 0.0)

    # 关键改动：降序排列，使 evals[0] 为最大（第一主成分）
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # 生成矩阵 A = V D^{1/2}
    A = evecs @ np.diag(np.sqrt(evals))
    return A*np.sqrt(T)

def BB(n: int, T:float =1) -> np.ndarray:
    """
    2^n 是生成的样本点个数
    """
    size = 2 ** n
    A = np.zeros((size, size), dtype=float)
    A[size - 1, 0] = 1.0  # R: A[size, 1] <- 1

    k_list = np.arange(1, size, dtype=int)  # R: 1:(size-1)

    # i_k = 最右侧1的位置（从0开始）
    i_k = np.array([(int(k) & -int(k)).bit_length() - 1 for k in k_list], dtype=int)
    rounds = n - i_k

    order = np.argsort(rounds, kind="stable")
    k_sorted, i_sorted, r_sorted = k_list[order], i_k[order], rounds[order]

    for k, ik, rr in zip(k_sorted, i_sorted, r_sorted):
        k, ik, rr = int(k), int(ik), int(rr)
        power = 2 ** ik
        j = (k - power) // (2 ** (ik + 1))

        k_prev, k_next = k - power, k + power
        a_prev = A[k_prev - 1, :] if k_prev >= 1 else np.zeros(size)
        a_next = A[k_next - 1, :] if k_next <= size else np.zeros(size)

        a_k = (a_prev + a_next) / 2.0

        # R: col_pos = 2^(round-1) + j + 1 (1-based)
        # Py: col_idx = col_pos - 1 = 2^(round-1) + j
        col_idx = (2 ** (rr - 1)) + j
        a_k[col_idx] = math.sqrt(1.0 / (2 ** (rr + 1)))

        A[k - 1, :] = a_k

    return np.round(A, 4)*np.sqrt(T)

# # 1) 4x4
# A4 = BB(4)
# print("A (n=2, 4x4):\n", A4)
#
# # 2) 8x8
# A8 = BB(8,8)
# print("\nA (n=3, 8x8):\n", A8)
#
# # 3) 验证协方差：R 里 as.integer 是向0截断
# cov_int = np.trunc( (A8 @ A8.T)).astype(int)
# target = np.fromfunction(lambda i, j: np.minimum(i + 1, j + 1), (8, 8), dtype=int)
#
# print("\ntrunc(8 * A A^T):\n", cov_int)
# print("\nTarget min(i,j):\n", target)
# print("\nMatch?", np.array_equal(cov_int, target))


def _to_batch_param(x, batch_size, device, dtype=torch.float32):
    """
    可以给单个的去广播，也可以传B个
    """
    if torch.is_tensor(x):
        x = x.to(device=device, dtype=dtype)
        if x.ndim == 0:
            return x.expand(batch_size)
        if x.ndim == 1 and x.shape[0] == batch_size:
            return x
        raise ValueError(f"param must be scalar or [B], got {tuple(x.shape)}")
    return torch.full((batch_size,), float(x), device=device, dtype=dtype)


def _generator_matrix(method: str, nD: int, T: float, device: str, dtype=torch.float32) -> torch.Tensor:
    """
    选生成矩阵
    """
    method = method.lower()
    if method == "cholesky":
        G = Cholesky(nD, T=T)
    elif method == "pca":
        G = PCA(nD, T=T)
    elif method == "bb":
        if nD <= 0 or (nD & (nD - 1)) != 0:
            raise ValueError("BB requires nD to be a positive power of 2 (e.g., 256).")
        n = nD.bit_length() - 1  # 因为 nD=2^n 时，二进制长度刚好是 n+1
        G = BB(n, T=T)
    else:
        raise ValueError("method must be one of {'cholesky','pca','bb'}")
    return torch.tensor(G, device=device, dtype=dtype)

def simulate_gbm_batch_mc(theta, batch_size, nD=256, T=1.0, device="cpu", method="cholesky"):
    """
    theta: (r,S0,sigma,K) each is torch scalar or shape [B]
    returns:
      W: [B, nD] Brownian motion at t_1..t_nD
      S: [B, nD] GBM price at t_1..t_nD
    """
    r, S0, sigma, K = theta  # K 保留接口一致，但路径生成不需要它

    r = _to_batch_param(r, batch_size, device)          # [B,]
    S0 = _to_batch_param(S0, batch_size, device)        # [B,]
    sigma = _to_batch_param(sigma, batch_size, device)  # [B,]
    K = _to_batch_param(K, batch_size, device)          # [B,]

    # W = Z @ G^T
    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)  # [nD,nD]
    Z = torch.randn(batch_size, nD, device=device)                                 # [B,nD]
    W = Z @ G.T                                                                    # [B,nD]

    dt = T / nD
    t = (torch.arange(1, nD + 1, device=device, dtype=torch.float32) * dt).unsqueeze(0)  # [1,nD]

    logS = torch.log(S0).unsqueeze(1) + (r - 0.5 * sigma**2).unsqueeze(1) * t + sigma.unsqueeze(1) * W
    S = torch.exp(logS)

    return Z,W,S


def sample_theta(batch_size, device="cpu", gen=None):
    r = 0.01 + (0.03 - 0.01) * torch.rand(batch_size, device=device, generator=gen)
    S0 = 80.0 + (120.0 - 80.0) * torch.rand(batch_size, device=device, generator=gen)
    sigma = 0.05 + (0.25 - 0.05) * torch.rand(batch_size, device=device, generator=gen)
    K = 90.0 + (110.0 - 90.0) * torch.rand(batch_size, device=device, generator=gen)
    return r, S0, sigma, K


## qmc的simulation

def _norm_ppf_torch(U: torch.Tensor) -> torch.Tensor:
    """
    标准正态的 Phi^{-1}(U)，纯 torch 实现
    Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    """
    eps = torch.finfo(U.dtype).eps
    U = U.clamp(min=eps, max=1.0 - eps)   # 避免 ppf(0/1) 变成 ±inf
    return math.sqrt(2.0) * torch.erfinv(2.0 * U - 1.0)


def simulate_gbm_batch_qmc(theta, batch_size, nD=256, T=1.0, device="cpu", method="cholesky",
                           scramble=True, seed=42, sobol_engine=None):
    """
    theta: (r,S0,sigma,K) each is torch scalar or shape [B]
    returns:
      W: [B, nD] Brownian motion at t_1..t_nD
      S: [B, nD] GBM price at t_1..t_nD
      如果不传sobol_engine qmc生成器，那么每次都是根据seed重新来一个，
      如果传的话，就是跟着外面的走，循环调用 每次都是不同的rqmc点
    """
    r, S0, sigma, K = theta  # K 保留接口一致，但路径生成不需要它

    r = _to_batch_param(r, batch_size, device)          # [B,]
    S0 = _to_batch_param(S0, batch_size, device)        # [B,]
    sigma = _to_batch_param(sigma, batch_size, device)  # [B,]
    K = _to_batch_param(K, batch_size, device)          # [B,]

    # ---- QMC 生成 Z ~ N(0,1)^{nD} ----
    # 默认强制 batch_size=2^m（最规整）
    if batch_size & (batch_size - 1) != 0:
        raise ValueError(f"[qmc] batch_size must be a power of 2 (2^m), got {batch_size}")

    if sobol_engine is None:
        # 注意：如果你每次都在函数里新建 engine + 固定 seed，那么每次都会从头开始，Z 会重复
        sobol_engine = SobolEngine(dimension=nD, scramble=scramble, seed=seed)

    U = sobol_engine.draw(batch_size).to(device=device, dtype=torch.float32)  # [B,nD] in [0,1)
    Z = _norm_ppf_torch(U)                                                    # [B,nD] ~ N(0,1)

    # ---- W = Z @ G^T ----
    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)  # [nD,nD]
    W = Z @ G.T                                                                    # [B,nD]

    dt = T / nD
    t = (torch.arange(1, nD + 1, device=device, dtype=torch.float32) * dt).unsqueeze(0)  # [1,nD]

    logS = torch.log(S0).unsqueeze(1) + (r - 0.5 * sigma**2).unsqueeze(1) * t + sigma.unsqueeze(1) * W
    S = torch.exp(logS)

    return Z,W, S


def features_from_Z(Z, dimX):
    # Z: [B, nD]
    if dimX <= 0 or dimX > Z.shape[1]:
        raise ValueError(f"dimX must be in [1, {Z.shape[1]}], got {dimX}")
    return Z[:, :dimX]






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


def arithmetic_payoff(S, K):
    A = S.mean(dim=1)
    return torch.clamp(A - K, min=0.0) #min下界

def geometric_payoff(S, K):
    G = torch.exp(torch.log(S).mean(dim=1))
    return torch.clamp(G - K, min=0.0)