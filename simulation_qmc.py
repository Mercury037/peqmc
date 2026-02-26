import math
import numpy as np
import torch
from torch.quasirandom import SobolEngine

# 依赖你 utils.py 里的函数
# - inv_Phi_torch
# - check_power_of_two
from utils import *


# =========================================================
# 1) 生成矩阵：Cholesky / PCA / BB
# =========================================================
def Cholesky(n: int, T: float = 1.0) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    dt_sqrt = math.sqrt(T / n)
    return np.tril(np.ones((n, n), dtype=float)) * dt_sqrt


def PCA(n: int, T: float = 1.0) -> np.ndarray:
    """
    Brownian motion grid covariance C_ij = min(t_i, t_j), t_i=i/n
    A = V diag(sqrt(lambda))，并按特征值降序排序（第一列=第一主成分）
    返回 A.shape = (n, n)
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    t = np.linspace(1.0 / n, 1.0, n)
    C = np.minimum.outer(t, t)

    evals, evecs = np.linalg.eigh(C)  # evals升序
    evals = np.maximum(evals, 0.0)

    idx = np.argsort(evals)[::-1]     # 降序
    evals = evals[idx]
    evecs = evecs[:, idx]

    A = evecs @ np.diag(np.sqrt(evals))
    return A * np.sqrt(T)


def BB(n: int, T: float = 1.0) -> np.ndarray:
    """
    Brownian Bridge 生成矩阵（size = 2^n）
    返回 A.shape = (2^n, 2^n)
    """
    size = 2 ** n
    A = np.zeros((size, size), dtype=float)
    A[size - 1, 0] = 1.0

    k_list = np.arange(1, size, dtype=int)

    # i_k = k 的二进制最右侧1的位置（从0开始）
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

        col_idx = (2 ** (rr - 1)) + j
        a_k[col_idx] = math.sqrt(1.0 / (2 ** (rr + 1)))

        A[k - 1, :] = a_k

    # 不 round，避免人为截断数值精度
    return A * np.sqrt(T)


def generator_matrix(method: str, nD: int, T: float, device: str, dtype=torch.float32) -> torch.Tensor:
    method = method.lower()
    if method == "cholesky":
        G = Cholesky(nD, T=T)
    elif method == "pca":
        G = PCA(nD, T=T)
    elif method == "bb":
        if nD <= 0 or (nD & (nD - 1)) != 0:
            raise ValueError("BB requires nD to be a positive power of 2 (e.g., 256).")
        n = nD.bit_length() - 1
        G = BB(n, T=T)
    else:
        raise ValueError("method must be one of {'cholesky','pca','bb'}")

    return torch.tensor(G, device=device, dtype=dtype)


# =========================================================
# 2) theta 采样（保留你原来的 2D / 3D 逻辑）
# =========================================================
def sample_theta(
    mode,                    # 2 or 3
    batch_size=None,         # mode=2 时用
    B=None, N=None,          # mode=3 时用
    is_same=False,           # 是否固定同一个 theta
    theta_same=None,         # (r, S0, sigma, K)
    device="cpu",
    gen=None
):
    """
    返回 r, S0, sigma, K

    mode=2:
      shape = [batch_size]

    mode=3:
      shape = [B, N]
      is_same=False 时：先采 [B,1] 再 expand 到 [B,N]
    """
    if mode not in (2, 3):
        raise ValueError(f"mode 必须是 2 或 3，收到: {mode}")

    def _u(low, high, shape):
        return low + (high - low) * torch.rand(*shape, device=device, generator=gen)

    def _to_scalar_tensor(x, name):
        t = torch.as_tensor(x, device=device, dtype=torch.float32)
        if t.numel() != 1:
            raise ValueError(f"theta_same 中的 {name} 必须是标量或单元素张量，收到 shape={tuple(t.shape)}")
        return t.reshape(())

    ranges = {
        "r":     (0.01, 0.03),
        "S0":    (80.0, 120.0),
        "sigma": (0.05, 0.25),
        "K":     (90.0, 110.0),
    }

    if is_same:
        if theta_same is None or len(theta_same) != 4:
            raise ValueError("is_same=True 时必须传 theta_same=(r, S0, sigma, K)")
        r0, S00, sigma0, K0 = theta_same
        r0     = _to_scalar_tensor(r0, "r")
        S00    = _to_scalar_tensor(S00, "S0")
        sigma0 = _to_scalar_tensor(sigma0, "sigma")
        K0     = _to_scalar_tensor(K0, "K")

        if mode == 2:
            if batch_size is None:
                raise ValueError("mode=2 时必须传 batch_size")
            return (
                r0.expand(batch_size),
                S00.expand(batch_size),
                sigma0.expand(batch_size),
                K0.expand(batch_size),
            )
        else:
            if B is None or N is None:
                raise ValueError("mode=3 时必须传 B 和 N")
            return (
                r0.expand(B, N),
                S00.expand(B, N),
                sigma0.expand(B, N),
                K0.expand(B, N),
            )

    # 随机采样
    if mode == 2:
        if batch_size is None:
            raise ValueError("mode=2 时必须传 batch_size")
        r     = _u(*ranges["r"],     (batch_size,))
        S0    = _u(*ranges["S0"],    (batch_size,))
        sigma = _u(*ranges["sigma"], (batch_size,))
        K     = _u(*ranges["K"],     (batch_size,))
        return r, S0, sigma, K

    else:
        if B is None or N is None:
            raise ValueError("mode=3 时必须传 B 和 N")
        r     = _u(*ranges["r"],     (B, 1)).expand(B, N)
        S0    = _u(*ranges["S0"],    (B, 1)).expand(B, N)
        sigma = _u(*ranges["sigma"], (B, 1)).expand(B, N)
        K     = _u(*ranges["K"],     (B, 1)).expand(B, N)
        return r, S0, sigma, K


# =========================================================
# 3) QMC 标准正态块
# =========================================================
def draw_Z_block_qmc(
    num_points,
    nD,
    device="cpu",
    *,
    scramble=True,
    seed=42,
    block_idx=0,
    sobol_engine=None,
    dtype=torch.float32
):
    """
    返回 [num_points, nD] 的标准正态 QMC 样本 Z
    """
    check_power_of_two(num_points, name="num_points")

    if sobol_engine is None:
        eng = SobolEngine(dimension=nD, scramble=scramble, seed=seed + block_idx)
        U = eng.draw(num_points).to(device=device, dtype=dtype)
    else:
        U = sobol_engine.draw(num_points).to(device=device, dtype=dtype)

    Z = inv_Phi_torch(U)
    return Z


# =========================================================
# 4) GBM 路径模拟（QMC/RQMC）
#    现在多返回一个 G
# =========================================================
def simulate_gbm_batch_qmc(
    theta,
    nD=256,
    T=1.0,
    device="cpu",
    method="pca",
    scramble=True,
    seed=37,
    sobol_engine=None
):
    """
    输入 theta=(r,S0,sigma,K)
      - 若参数 shape=[B]   -> 返回 Z,W,S,G 其中 Z/W/S shape=[B,nD]
      - 若参数 shape=[B,N] -> 返回 Z,W,S,G 其中 Z/W/S shape=[B,N,nD]
    """
    r, S0, sigma, K = theta
    dtype = torch.float32

    r = r.to(device=device, dtype=dtype)
    S0 = S0.to(device=device, dtype=dtype)
    sigma = sigma.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)  # 保留接口一致性（路径里不用）

    # 预计算生成矩阵 + 时间网格
    G = generator_matrix(method, nD=nD, T=T, device=device, dtype=dtype)  # [nD,nD]
    dt = T / nD
    t = torch.arange(1, nD + 1, device=device, dtype=dtype) * dt          # [nD]

    # ---------- Case 1: theta shape=[B] ----------
    if S0.dim() == 1:
        B = S0.shape[0]
        check_power_of_two(B, name="batch_size")

        Z = draw_Z_block_qmc(
            num_points=B, nD=nD, device=device,
            scramble=scramble, seed=seed, block_idx=0,
            sobol_engine=sobol_engine, dtype=dtype
        )  # [B,nD]

        W = Z @ G.T  # [B,nD]

        logS = (
            torch.log(S0).unsqueeze(1)
            + (r - 0.5 * sigma**2).unsqueeze(1) * t.unsqueeze(0)
            + sigma.unsqueeze(1) * W
        )
        S = torch.exp(logS)
        return Z, W, S, G

    # ---------- Case 2: theta shape=[B,N] ----------
    elif S0.dim() == 2:
        B, N = S0.shape
        check_power_of_two(N, name="N")

        Z_list = []
        for b in range(B):
            Z_b = draw_Z_block_qmc(
                num_points=N, nD=nD, device=device,
                scramble=scramble, seed=seed, block_idx=b,
                sobol_engine=sobol_engine, dtype=dtype
            )
            Z_list.append(Z_b)

        Z = torch.stack(Z_list, dim=0)       # [B,N,nD]
        W = torch.matmul(Z, G.T)             # [B,N,nD]

        t3 = t.view(1, 1, nD)
        logS = (
            torch.log(S0).unsqueeze(-1)
            + (r - 0.5 * sigma**2).unsqueeze(-1) * t3
            + sigma.unsqueeze(-1) * W
        )
        S = torch.exp(logS)
        return Z, W, S, G

    else:
        raise ValueError(f"[qmc] theta.dim() must be 1 or 2, got {S0.dim()}")


# =========================================================
# 5) 粗路径 + min/max proxy（只保留你要的 minmax 特征链路）
# =========================================================
def _to_like(x, ref: torch.Tensor):
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def build_coarse_gbm_path_from_Z(
    Z: torch.Tensor,
    r,
    S0,
    sigma,
    *,
    G: torch.Tensor,
    T: float = 1.0,
    k_proxy: int = 1,
):
    """
    用前 k_proxy 个 Z + G 前 k_proxy 列重建粗糙 GBM 路径
    Z: [..., nD]
    G: [nD, nD]
    返回 S_coarse: [..., nD]
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
    if G.dim() != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square [nD,nD], got shape={tuple(G.shape)}")

    nD = Z.shape[-1]
    if G.shape[0] != nD:
        raise ValueError(f"G.shape[0] must equal Z.shape[-1]; got G={tuple(G.shape)}, Z={tuple(Z.shape)}")

    k = min(max(int(k_proxy), 1), nD)

    Zk = Z[..., :k]            # [..., k]
    Gk = G[:, :k]              # [nD, k]
    Wc = torch.matmul(Zk, Gk.transpose(0, 1))  # [..., nD]

    r = _to_like(r, Z)
    S0 = _to_like(S0, Z)
    sigma = _to_like(sigma, Z)

    dt = T / nD
    t = torch.arange(1, nD + 1, device=Z.device, dtype=Z.dtype) * dt

    drift = (r - 0.5 * sigma**2).unsqueeze(-1) * t
    diffusion = sigma.unsqueeze(-1) * Wc
    logS = torch.log(S0).unsqueeze(-1) + drift + diffusion
    S_coarse = torch.exp(logS)
    return S_coarse


def coarse_extrema_from_Z_gbm(
    Z: torch.Tensor,
    k_proxy: int,
    *,
    r,
    S0,
    sigma,
    G: torch.Tensor,
    T: float = 1.0,
):
    """
    返回 (proxy_max, proxy_min)，shape 都是 [...,1]
    """
    S_coarse = build_coarse_gbm_path_from_Z(
        Z, r=r, S0=S0, sigma=sigma, G=G, T=T, k_proxy=k_proxy
    )
    # proxy_max = S_coarse.max(dim=-1, keepdim=True).values
    proxy_max= S_coarse[..., -1:]
    proxy_min = S_coarse.min(dim=-1, keepdim=True).values
    return proxy_max, proxy_min


def features_from_Z(
    Z: torch.Tensor,
    dimX: int,
    *,
    theta,                 # (r, S0, sigma, K) 或至少前3个
    G: torch.Tensor,       # simulate_gbm_batch_qmc 返回的生成矩阵
    T: float = 1.0,
    k_proxy: int = None,
):
    """
    只保留 minmax 特征版本：
      X = [Z前(dimX-2)维, coarse_max_proxy, coarse_min_proxy]

    支持:
      - Z: [B, nD] -> X: [B, dimX]
      - Z: [B,N,nD] -> X: [B,N,dimX]
      - 更一般 [...,nD] -> [...,dimX]

    注:
      - 若 dimX < 2，则退化为只返回 [max,min] 的前 dimX 维
      - 粗路径重建用前 k_proxy 个因子；若不传，默认 k_proxy = max(dimX-2, 1)
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
    if dimX <= 0:
        raise ValueError(f"dimX must be >= 1, got {dimX}")

    nD = Z.shape[-1]
    if G is None:
        raise ValueError("features_from_Z 需要传入 G（生成矩阵）")
    if theta is None or len(theta) < 3:
        raise ValueError("features_from_Z 需要 theta，且至少包含 (r, S0, sigma)")

    r, S0, sigma = theta[0], theta[1], theta[2]

    # dimX 太小时：只返回结构特征的前 dimX 维
    if dimX < 2:
        pmax, pmin = coarse_extrema_from_Z_gbm(
            Z, k_proxy=1, r=r, S0=S0, sigma=sigma, G=G, T=T
        )
        struct = torch.cat([pmax, pmin], dim=-1)  # [...,2]
        return struct[..., :dimX]

    z_keep = dimX - 2
    if z_keep > nD:
        raise ValueError(f"Need z_keep <= nD, got z_keep={z_keep}, nD={nD}")

    k = max(z_keep, 1) if k_proxy is None else int(k_proxy)

    pmax, pmin = coarse_extrema_from_Z_gbm(
        Z, k_proxy=k, r=r, S0=S0, sigma=sigma, G=G, T=T
    )
    struct = torch.cat([pmax, pmin], dim=-1)      # [...,2]

    if z_keep == 0:
        return struct

    z_feat = Z[..., :z_keep]
    return torch.cat([z_feat, struct], dim=-1)    # [..., dimX]


# def features_from_Z(
#     Z: torch.Tensor,
#     dimX: int,
#     *,
#     theta,                 # (r, S0, sigma, K) 或至少前3个
#     G: torch.Tensor,       # simulate_gbm_batch_qmc 返回的生成矩阵
#     T: float = 1.0,
#     k_proxy: int = None,
# ):
#     """
#     只保留 max 特征版本：
#       X = [Z前(dimX-1)维, coarse_max_proxy]
#
#     支持:
#       - Z: [B, nD] -> X: [B, dimX]
#       - Z: [B,N,nD] -> X: [B,N,dimX]
#       - 更一般 [...,nD] -> [...,dimX]
#
#     注:
#       - 若 dimX == 1，则只返回 [coarse_max_proxy]
#       - 粗路径重建用前 k_proxy 个因子；若不传，默认 k_proxy = max(dimX-1, 1)
#     """
#     if Z.dim() < 2:
#         raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
#     if dimX <= 0:
#         raise ValueError(f"dimX must be >= 1, got {dimX}")
#
#     nD = Z.shape[-1]
#     if G is None:
#         raise ValueError("features_from_Z 需要传入 G（生成矩阵）")
#     if theta is None or len(theta) < 3:
#         raise ValueError("features_from_Z 需要 theta，且至少包含 (r, S0, sigma)")
#
#     r, S0, sigma = theta[0], theta[1], theta[2]
#
#     # dimX == 1：只返回 coarse max
#     if dimX == 1:
#         pmax, _ = coarse_extrema_from_Z_gbm(
#             Z, k_proxy=1, r=r, S0=S0, sigma=sigma, G=G, T=T
#         )
#         return pmax  # [...,1]
#
#     z_keep = dimX - 1
#     if z_keep > nD:
#         raise ValueError(f"Need z_keep <= nD, got z_keep={z_keep}, nD={nD}")
#
#     k = max(z_keep, 1) if k_proxy is None else int(k_proxy)
#
#     _, pmin = coarse_extrema_from_Z_gbm(
#         Z, k_proxy=k, r=r, S0=S0, sigma=sigma, G=G, T=T
#     )  # pmax: [...,1]
#
#     z_feat = Z[..., :z_keep]
#     return torch.cat([z_feat, pmin], dim=-1)  # [..., dimX]

# =========================================================
# 6) Lookback payoff（只保留这一种）
# =========================================================
# def arithmetic_payoff(S, K=None, S0=None):
#     """
#     Fixed-strike lookback call:
#       payoff = (max_t S_t - K)^+
#     """
#     if K is None:
#         raise ValueError("Lookback (fixed-strike) payoff requires K.")
#     S_max = S.max(dim=-1).values
#     return torch.clamp(S_max - K, min=0.0)


# （如果你之后想切换成浮动回望看涨，就改成下面这个）
def arithmetic_payoff(S, K=None, S0=None):
    S_min = S.min(dim=-1).values
    S_T = S[..., -1]
    return S_T - S_min