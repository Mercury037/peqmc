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


import torch
from typing import Optional

# =========================================================
# 0) 小工具：把 theta 转成和 Z 同设备同 dtype
# =========================================================
def _to_like(x, ref: torch.Tensor) -> torch.Tensor:
    """把 x 变成和 ref 相同的 device 和 dtype"""
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def _broadcast_to(x, ref: torch.Tensor) -> torch.Tensor:
    """
    x: 标量 / [B] / [B,N] / ...
    输出: 维度补到 ref.dim()，只在最后 unsqueeze，便于广播到 ref[..., nD]
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
    else:
        x = x.to(device=ref.device, dtype=ref.dtype)
    while x.dim() < ref.dim():
        x = x.unsqueeze(-1)
    return x


# =========================================================
# 1) 用“前 k 个 Z + 生成矩阵前 k 列”重建粗糙 GBM 路径
#    （你原来的函数可直接复用，这里原样放着）
# =========================================================
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
    用前 k_proxy 个因子重建粗糙 GBM 路径（与 simulate_gbm_batch_qmc 同一离散网格）

    输入:
      Z: [..., nD]
      r, S0, sigma: 可广播到 Z.shape[:-1]
      G: [nD, nD] 生成矩阵（PCA/BB/Cholesky）
      T: 到期时间
      k_proxy: 使用前 k_proxy 个因子（以及 G 的前 k_proxy 列）

    返回:
      S_coarse: [..., nD]
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
    if G.dim() != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square [nD,nD], got shape={tuple(G.shape)}")

    nD = Z.shape[-1]
    if G.shape[0] != nD:
        raise ValueError(f"G.shape[0] must equal Z.shape[-1]; got G={tuple(G.shape)}, Z={tuple(Z.shape)}")

    k = min(max(int(k_proxy), 1), nD)

    Zk = Z[..., :k]          # [..., k]
    Gk = G[:, :k]            # [nD, k]

    # 粗糙 Brownian 路径值：[..., k] @ [k, nD] -> [..., nD]
    Wc = torch.matmul(Zk, Gk.transpose(0, 1))

    r = _to_like(r, Z)
    S0 = _to_like(S0, Z)
    sigma = _to_like(sigma, Z)

    dt = T / nD
    t = torch.arange(1, nD + 1, device=Z.device, dtype=Z.dtype) * dt   # [nD]

    drift = (r - 0.5 * sigma**2).unsqueeze(-1) * t
    diffusion = sigma.unsqueeze(-1) * Wc
    logS = torch.log(S0).unsqueeze(-1) + drift + diffusion
    S_coarse = torch.exp(logS)

    return S_coarse


def coarse_lookback_struct2_from_Z_gbm(
    Z: torch.Tensor,
    k_proxy: int,
    *,
    r,
    S0,
    sigma,
    K,          # 保留在签名里以兼容旧调用（此版本不再使用）
    G: torch.Tensor,
    T: float = 1.0,
):
    """
    返回:
      struct2: [..., 2]
        [s_max, s_T]
    """
    S_coarse = build_coarse_gbm_path_from_Z(
        Z, r=r, S0=S0, sigma=sigma, G=G, T=T, k_proxy=k_proxy
    )  # [..., nD]

    s_max = S_coarse.max(dim=-1, keepdim=True).values  # [..., 1]
    s_0   = S_coarse[..., 0:1]                         # [..., 1]
    s_mean = S_coarse.mean(dim=-1, keepdim=True)
    struct2 = torch.cat([s_mean, s_0], dim=-1)          # [..., 2]
    return struct2


# =========================================================
# 3) 回望期权专用 features_from_Z：总维度严格 = dimX
#    结构： [Z前(dimX-2)维, s_max, s_max-K]
# =========================================================
from typing import Optional
import torch

def features_from_Z_lookback(
    Z: torch.Tensor,
    dimX: int,
    dimZ: int,
    dimProxy: int,
    *,
    theta,                 # (r, S0, sigma, K)
    G: torch.Tensor,
    T: float = 1.0,
    k_proxy: Optional[int] = None,
):
    """
    输出:
      X: [..., dimX]

    约束：
      dimX = dimZ + dimProxy

    规则：
      - Z 部分取前 dimZ 维：Z[..., :dimZ]
      - proxy 部分来自 coarse_lookback_struct2_from_Z_gbm 的前 dimProxy 维
        （该 struct2 目前只有 2 维：[s_max, s_max-K]）
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
    if dimX <= 0:
        raise ValueError(f"dimX must be >= 1, got {dimX}")
    if dimZ < 0 or dimProxy < 0:
        raise ValueError(f"dimZ and dimProxy must be >= 0, got dimZ={dimZ}, dimProxy={dimProxy}")
    if dimX != dimZ + dimProxy:
        raise ValueError(f"Require dimX = dimZ + dimProxy, got dimX={dimX}, dimZ={dimZ}, dimProxy={dimProxy}")
    if theta is None or len(theta) < 4:
        raise ValueError("theta 至少需要 (r, S0, sigma, K)")
    if G is None:
        raise ValueError("G cannot be None")

    r, S0, sigma, K = theta[0], theta[1], theta[2], theta[3]
    nD = Z.shape[-1]

    if dimZ > nD:
        raise ValueError(f"Need dimZ <= nD, got dimZ={dimZ}, nD={nD}")

    # k_proxy 默认：跟随你用作 Z 特征的前缀维度 dimZ（至少 1）
    if k_proxy is None:
        k_proxy_eff = max(dimZ, 1)
    else:
        k_proxy_eff = int(k_proxy)
    k_proxy_eff = min(max(k_proxy_eff, 1), nD)

    # struct2: [..., 2] = [s_max, s_max-K]
    struct2 = coarse_lookback_struct2_from_Z_gbm(
        Z,
        k_proxy=k_proxy_eff,
        r=r, S0=S0, sigma=sigma, K=K,
        G=G, T=T
    )

    if dimProxy > struct2.shape[-1]:
        raise ValueError(
            f"dimProxy={dimProxy} exceeds available proxy dims={struct2.shape[-1]} "
            f"(currently struct2 provides [s_max, s_max-K])"
        )

    z_raw = Z[..., :dimZ]          # [..., dimZ]（dimZ=0 时是空张量，OK）
    proxy = struct2[..., :dimProxy]  # [..., dimProxy]（dimProxy=0 时是空张量，OK）

    X = torch.cat([z_raw, proxy], dim=-1)  # [..., dimZ+dimProxy] == [..., dimX]

    if X.shape[-1] != dimX:
        raise RuntimeError(f"Output feature dim mismatch: got {X.shape[-1]}, expected dimX={dimX}")
    return X

# （如果你之后想切换成浮动回望看涨，就改成下面这个）
# def arithmetic_payoff(S, K=None, S0=None):
#     S_min = S.min(dim=-1).values
#     S_T = S[..., -1]
#     return S_T - S_min



# def arithmetic_payoff(S, K=None, S0=None):
#     """
#     Lookback call (fixed-strike, max-fix):
#       payoff = (S_max - K)^+
#     S: [..., m+1]
#     K: broadcastable to S[..., -1]
#     """
#     S_max = S.max(dim=-1).values
#     return torch.clamp(S_max - K, min=0.0)


# def arithmetic_payoff(S, K, S0=None):
#
#     A = S[..., 1:].mean(dim=-1)
#     return torch.clamp(A - K, min=0.0)
#


#
def arithmetic_payoff(S, K, S0=None):
    """
    算术平均亚式 call 的 Delta 被积函数（pathwise）
    S: [..., m+1]  (包含 t0)
    K: 可广播到 [...] 的形状
    返回: [...]
      - 不贴现: 1_{A>K} * A / S0
      - 贴现  : exp(-rT) * 1_{A>K} * A / S0
    """
    S0 = S[..., 0]                      # [...]
    A  = S[..., 1:].mean(dim=-1)        # [...]
    Kt = torch.as_tensor(K, device=S.device, dtype=S.dtype)

    g = (A > Kt).to(S.dtype) * (A / S0)

    return g