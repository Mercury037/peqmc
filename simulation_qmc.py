import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.quasirandom import SobolEngine
from utils import *

"""
生成矩阵
    cholesky
    bb
    pca
    gpca
    
模拟theta
    合并: [B,N,4],[B,4] 
        if_same
模拟轨道
    mc
    qmc
"""
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

def generator_matrix(method: str, nD: int, T: float, device: str, dtype=torch.float32) -> torch.Tensor:
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




import torch

def sample_theta(
    mode,                    # 2 or 3
    batch_size=None,         # mode=2 时用
    B=None, N=None,          # mode=3 时用
    is_same=False,           # 是否所有元素都共享同一个值
    theta_same=None,         # is_same=True 时必须传: (r, S0, sigma, K)
    device="cpu",
    gen=None
):
    """
    返回:
      r, S0, sigma, K

    mode=2:
      每个参数 shape = [batch_size]
      - is_same=False: 每个样本独立采样
      - is_same=True : 使用 theta_same 中给定的标量，广播到 [batch_size]

    mode=3:
      每个参数 shape = [B, N]
      - is_same=False: 先采 [B,1]，再沿 N 维复制到 [B,N]
                      （即每个 i 的 [i,:] 全相同）
      - is_same=True : 使用 theta_same 中给定的标量，广播到 [B,N]
    """
    if mode not in (2, 3):
        raise ValueError(f"mode 必须是 2 或 3，收到: {mode}")

    def _u(low, high, shape):
        return low + (high - low) * torch.rand(*shape, device=device, generator=gen)

    def _to_scalar_tensor(x, name):
        """
        把 python 数 / 0维tensor / 单元素tensor 转成 device 上的 0维 tensor
        """
        t = torch.as_tensor(x, device=device, dtype=torch.float32)
        if t.numel() != 1:
            raise ValueError(f"theta_same 中的 {name} 必须是标量或单元素张量，收到 shape={tuple(t.shape)}")
        return t.reshape(())

    # 参数范围（仅 is_same=False 时使用）
    ranges = {
        "r":     (0.01, 0.03),
        "S0":    (80.0, 120.0),
        "sigma": (0.05, 0.25),
        "K":     (90.0, 110.0),
        # "K": (0.01, 0.02),
    }

    # ---------- 固定参数模式：不随机 ----------
    if is_same:
        if theta_same is None:
            raise ValueError("is_same=True 时必须传 theta_same=(r, S0, sigma, K)")
        if len(theta_same) != 4:
            raise ValueError(f"theta_same 长度必须是4，收到长度={len(theta_same)}")

        r0, S00, sigma0, K0 = theta_same
        r0     = _to_scalar_tensor(r0, "r")
        S00    = _to_scalar_tensor(S00, "S0")
        sigma0 = _to_scalar_tensor(sigma0, "sigma")
        K0     = _to_scalar_tensor(K0, "K")

        if mode == 2:
            if batch_size is None:
                raise ValueError("mode=2 时必须传 batch_size")

            r     = r0.expand(batch_size)
            S0    = S00.expand(batch_size)
            sigma = sigma0.expand(batch_size)
            K     = K0.expand(batch_size)

        else:  # mode == 3
            if B is None or N is None:
                raise ValueError("mode=3 时必须传 B 和 N")

            r     = r0.expand(B, N)
            S0    = S00.expand(B, N)
            sigma = sigma0.expand(B, N)
            K     = K0.expand(B, N)

        return r, S0, sigma, K

    # ---------- 随机采样模式 ----------
    if mode == 2:
        if batch_size is None:
            raise ValueError("mode=2 时必须传 batch_size")

        r     = _u(*ranges["r"],     (batch_size,))
        S0    = _u(*ranges["S0"],    (batch_size,))
        sigma = _u(*ranges["sigma"], (batch_size,))
        K     = _u(*ranges["K"],     (batch_size,))

    else:  # mode == 3
        if B is None or N is None:
            raise ValueError("mode=3 时必须传 B 和 N")

        # 先采 [B,1]，再沿 N 维复制到 [B,N]
        r     = _u(*ranges["r"],     (B, 1)).expand(B, N)
        S0    = _u(*ranges["S0"],    (B, 1)).expand(B, N)
        sigma = _u(*ranges["sigma"], (B, 1)).expand(B, N)
        K     = _u(*ranges["K"],     (B, 1)).expand(B, N)

    return r, S0, sigma, K


def draw_Z_block_qmc(num_points, nD, device="cpu", *,
                      scramble=True, seed=42, block_idx=0,
                      sobol_engine=None, dtype=torch.float32):
    """
    返回 [num_points, nD] 的标准正态QMC样本 Z

    - 若 sobol_engine is None: 用 seed+block_idx 新建 engine（适合 RQMC 独立随机化块）
    - 若传入 sobol_engine: 连续 draw（同一 digital sequence，理论上一直不用传）
    """
    check_power_of_two(num_points, name="num_points")

    if sobol_engine is None:
        eng = SobolEngine(dimension=nD, scramble=scramble, seed=seed + block_idx)
        U = eng.draw(num_points).to(device=device, dtype=dtype)
    else:
        U = sobol_engine.draw(num_points).to(device=device, dtype=dtype)

    Z = inv_Phi_torch(U)  # [num_points, nD]
    return Z


def simulate_gbm_batch_qmc(theta, nD=256, T=1.0, device="cpu", method="pca",
                           scramble=True, seed=37, sobol_engine=None):
    """
    QMC / RQMC 生成 GBM 路径
    不传轨道数和块数，和theta保持一致

    Parameters
    ----------
    theta : tuple (r, S0, sigma, K)
        - 情况1：每个参数 shape=[B]
        - 情况2：每个参数 shape=[B, N]
    nD : int
        时间离散维度（也是 QMC 维度）
    T : float
        到期时间
    device : str
        "cpu" / "cuda"
    method : str
        生成矩阵方法（如 "cholesky", "bb", "pca"）
    scramble : bool
        Sobol scramble 开关
    seed : int
        基础随机种子
    sobol_engine : SobolEngine or None
        - None: 内部创建 engine
        - 非 None: 使用外部 engine 连续 draw

    Returns
    -------
    Z, W, S
        - 若 theta 为 [B]，则返回 shape=[B, nD]
        - 若 theta 为 [B,N]，则返回 shape=[B, N, nD]

    Notes
    -----
    - dim==2 时，如果 sobol_engine is None，会对每个 b 用 seed+b 创建一个独立 scramble 的 SobolEngine，即不同块是独立的
    """
    r, S0, sigma, K = theta
    dtype = torch.float32
    r = r.to(device=device, dtype=dtype)
    S0 = S0.to(device=device, dtype=dtype)
    sigma = sigma.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)  # noqa: F841  # 保留接口一致，路径里不使用

    # 预计算：生成矩阵 + 时间网格
    G = generator_matrix(method, nD=nD, T=T, device=device, dtype=dtype)  # [nD, nD]
    dt = T / nD
    t = torch.arange(1, nD + 1, device=device, dtype=dtype) * dt           # [nD]

    # ============================================================
    # Case 1: theta 参数是 [B]
    # 返回 [B, nD]
    # ============================================================
    if S0.dim() == 1:
        B = S0.shape[0]
        check_power_of_two(B, name="batch_size")

        # QMC -> Z: [B, nD]
        Z = draw_Z_block_qmc(num_points=B, nD=nD, device=device, scramble=scramble, seed=seed, block_idx=0,
                              sobol_engine=sobol_engine, dtype=dtype)

        # W = Z @ G^T
        W = Z @ G.T  # [B, nD]

        # GBM closed-form on grid t_j
        # log S_t = logS0 + (r - 0.5 sigma^2)t + sigma W_t
        logS = (
            torch.log(S0).unsqueeze(1) +
            (r - 0.5 * sigma**2).unsqueeze(1) * t.unsqueeze(0) +
            sigma.unsqueeze(1) * W
        )  # [B, nD]

        S = torch.exp(logS)  # [B, nD]
        return Z, W, S

    # ============================================================
    # Case 2: theta 参数是 [B, N]
    # 返回 [B, N, nD]
    # ============================================================
    elif S0.dim() == 2:
        B, N = S0.shape
        check_power_of_two(N, name="N")

        # 逐个 block 生成 Z_b: [N, nD]
        # 若 sobol_engine is None，则 block_idx=b -> seed+b，不同 scramble，生成独立的块
        Z_list = []
        for b in range(B):
            Z_b = draw_Z_block_qmc(num_points=N, nD=nD, device=device, scramble=scramble, seed=seed, block_idx=b,
                                   sobol_engine=sobol_engine, dtype=dtype)
            Z_list.append(Z_b)

        Z = torch.stack(Z_list, dim=0)  # [B, N, nD]

        # W = Z @ G^T
        # [B,N,nD] @ [nD,nD] -> [B,N,nD]
        W = torch.matmul(Z, G.T)

        # broadcast 时间维
        t3 = t.view(1, 1, nD)  # [1,1,nD]

        logS = (
            torch.log(S0).unsqueeze(-1) +
            (r - 0.5 * sigma**2).unsqueeze(-1) * t3 +
            sigma.unsqueeze(-1) * W
        )  # [B, N, nD]

        S = torch.exp(logS)  # [B, N, nD]
        return Z, W, S

    else:
        raise ValueError(f"[qmc] theta.dim() must be 1 or 2, got {S0.dim()}")



def features_from_Z(Z, dimX):
    """
    从 Z 的最后一维截取前 dimX 个特征。

    支持：
      - Z: [B, nD]      -> 返回 [B, dimX]
      - Z: [B, N, nD]   -> 返回 [B, N, dimX]
      - 更一般地：[..., nD] -> [..., dimX]
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
    nD = Z.shape[-1]
    if dimX <= 0 or dimX > nD:
        raise ValueError(f"dimX must be in [1, {nD}], got {dimX}")

    return Z[..., :dimX]






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


# def arithmetic_payoff(S, K):
#     A = S.mean(dim=-1)
#     return torch.clamp(A - K, min=0.0) #min下界

#障碍
# def arithmetic_payoff(S, K, H=105):
#     S_T = S[..., -1]
#     vanilla = torch.clamp(S_T - K, min=0.0)
#     knocked_out = (S >= H).any(dim=-1)
#     return torch.where(knocked_out, torch.zeros_like(vanilla), vanilla)

#回望
def arithmetic_payoff(S, K):
    S_max = S.max(dim=-1).values
    return torch.clamp(S_max - K, min=0.0)
#
# #方差呼唤
# def arithmetic_payoff(S, K, eps=1e-12):
#     """
#     Variance call payoff (unannualized):
#       ( sum_i (log S_i - log S_{i-1})^2 - K )^+
#
#     S: [..., m+1]
#     return: [...]
#     """
#     logS = torch.log(S.clamp_min(eps))
#     dlogS = logS[..., 1:] - logS[..., :-1]
#     rv = (dlogS ** 2).sum(dim=-1)
#     return torch.clamp(rv - K, min=0.0)
#     # return rv

def geometric_payoff(S, K):
    G = torch.exp(torch.log(S).mean(dim=1))
    return torch.clamp(G - K, min=0.0)