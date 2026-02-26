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


import torch
from torch.quasirandom import SobolEngine

@torch.no_grad()
def fit_gpca_rotation_weighted(
    *,
    nD: int,
    T: float,
    theta_fixed,            # (r, S0, sigma, K) 标量
    H: float = 105.0,
    base_method: str = "pca",
    M: int = 8192,          # 用多少样本拟合 R
    scramble: bool = True,
    seed: int = 123,
    weight_mode: str = "var",   # "var" | "abs" | "sq"
    tau_barrier: float = 1.5,   # smooth barrier 的温度（越小越接近硬障碍）
    device: str = "cpu",
):
    """
    返回:
      R: [nD, nD] 正交矩阵（列向量为“重要方向”，已按重要性降序）
    """
    dtype_work = torch.float64  # 拟合时用64更稳
    # 1) 采样 Z0 ~ N(0,I)（QMC）
    eng = SobolEngine(dimension=nD, scramble=scramble, seed=seed)
    U = eng.draw(M).to(device=device, dtype=torch.float32)
    Z0 = inv_Phi_torch(U).to(device=device, dtype=dtype_work)   # [M,nD]

    # 2) base 生成矩阵（例如 PCA）
    G_base = generator_matrix(base_method, nD=nD, T=T, device=device, dtype=dtype_work)  # [nD,nD]

    # 3) 用 base 路径算一个“平滑 barrier KO call” proxy payoff
    r, S0, sigma, K = theta_fixed
    r     = torch.as_tensor(r, device=device, dtype=dtype_work)
    S0    = torch.as_tensor(S0, device=device, dtype=dtype_work)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype_work)
    K     = torch.as_tensor(K, device=device, dtype=dtype_work)
    Ht    = torch.as_tensor(H, device=device, dtype=dtype_work)

    dt = T / nD
    t = torch.arange(1, nD + 1, device=device, dtype=dtype_work) * dt  # [nD]

    W = Z0 @ G_base.T  # [M,nD]
    logS = torch.log(S0) + (r - 0.5 * sigma**2) * t + sigma * W
    S = torch.exp(logS)  # [M,nD]

    S_T = S[:, -1]
    vanilla = torch.clamp(S_T - K, min=0.0)

    # barrier: not_hit ≈ 1{max<S<H} 用 sigmoid 平滑一下
    S_max = S.max(dim=-1).values
    not_hit = torch.sigmoid((Ht - S_max) / tau_barrier)   # [M]
    y = vanilla * not_hit                                  # [M]

    # 4) 权重 w
    if weight_mode == "var":
        yc = y - y.mean()
        w = yc**2
    elif weight_mode == "abs":
        w = y.abs()
    elif weight_mode == "sq":
        w = y**2
    else:
        raise ValueError("weight_mode must be in {'var','abs','sq'}")

    w = w / (w.mean() + 1e-12)

    # 5) 加权协方差 Σ = E[w Z Z^T]
    # （Z0 均值本来就≈0，这里不强制中心化也行；中心化更稳）
    Zc = Z0 - Z0.mean(dim=0, keepdim=True)  # [M,nD]
    Sigma = (Zc.T * w.unsqueeze(0)) @ Zc / Zc.shape[0]     # [nD,nD]
    Sigma = 0.5 * (Sigma + Sigma.T)

    # 6) 特征分解 -> R（列向量按重要性降序）
    evals, evecs = torch.linalg.eigh(Sigma)  # 升序
    idx = torch.argsort(evals, descending=True)
    R = evecs[:, idx]                        # [nD,nD] 正交
    return R.to(dtype=torch.float32)



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



# def features_from_Z(Z,S, dimX):
#     """
#     从 Z 的最后一维截取前 dimX 个特征。
#
#     支持：
#       - Z: [B, nD]      -> 返回 [B, dimX]
#       - Z: [B, N, nD]   -> 返回 [B, N, dimX]
#       - 更一般地：[..., nD] -> [..., dimX]
#     """
#     if Z.dim() < 2:
#         raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
#     nD = Z.shape[-1]
#     if dimX <= 0 or dimX > nD:
#         raise ValueError(f"dimX must be in [1, {nD}], got {dimX}")
#     return Z[..., :dimX]


import torch

# =========================================================
# 0) 小工具：把 theta 转成和 Z 同设备同 dtype
# =========================================================
def _to_like(x, ref: torch.Tensor):
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

import math
from typing import Optional, Tuple, List

import torch


# =========================================================
# 0) 小工具：把标量/张量转成和 Z 可广播、同 device/dtype 的张量
# =========================================================
def _to_like(x, ref: torch.Tensor) -> torch.Tensor:
    """
    把 x 转成和 ref 同 device/dtype 的张量，并尽量保持可广播。
    x 可以是 python 标量 / 0-d tensor / shape 与 ref 前缀兼容的 tensor
    """
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


# =========================================================
# 1) 用“前 k 个 Z + 生成矩阵前 k 列”重建粗糙 GBM 路径
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

    # 粗糙 Brownian 路径值
    # [..., k] @ [k, nD] -> [..., nD]
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


# =========================================================
# 2) 粗轨道 barrier-aware 特征（固定 10 维，不超过10）
#    前4维保留你原来的核心，再加6维“时间/持续性/波动”信息
# =========================================================
def coarse_barrier_struct10_from_Z_gbm(
    Z: torch.Tensor,
    k_proxy: int,
    *,
    r,
    S0,
    sigma,
    G: torch.Tensor,
    T: float = 1.0,
    H: float = 105.0,
    near_ratio: float = 0.98,   # 预警线 alpha*H
    near_band: float = 2.0,     # 近障碍带宽（价格单位）
    eps: float = 1e-8,
):
    """
    返回:
      struct10: [..., 10]

    10维定义（建议先用这版做消融）:
      1  s_max                粗路径最大值
      2  s_T                  粗路径终点
      3  H - s_max            距障碍边际（<=0 表示粗路径越障）
      4  s_max - s_T          回撤程度（从峰值到终点）
      5  t_max                达到粗最大值的时间（归一化到 [0,1]）
      6  near_frac            进入近障碍带(H-S<=near_band)的时间比例
      7  hit98_frac           超过 near_ratio*H 的时间比例
      8  first_hit98_t        首次超过 near_ratio*H 的时间（未超过则记1）
      9  rv_coarse            粗路径对数收益 realized variance
      10 s_max_over_H         无量纲最大值（尺度稳定）
    """
    S_coarse = build_coarse_gbm_path_from_Z(
        Z, r=r, S0=S0, sigma=sigma, G=G, T=T, k_proxy=k_proxy
    )  # [..., nD]

    *prefix, nD = S_coarse.shape
    dtype = S_coarse.dtype
    device = S_coarse.device
    Ht = torch.as_tensor(float(H), dtype=dtype, device=device)

    # 基础量
    s_max, idx_max = S_coarse.max(dim=-1, keepdim=True)
    s_T = S_coarse[..., -1:].clone()

    margin_H = Ht - s_max
    retrace = s_max - s_T

    denom_t = max(nD - 1, 1)
    t_max = idx_max.to(dtype) / denom_t

    # 近障碍带持续时间
    margin_path = Ht - S_coarse                         # [..., nD]
    near_mask = (margin_path <= float(near_band))       # 包括越障也算 near
    near_frac = near_mask.to(dtype).mean(dim=-1, keepdim=True)

    # 预警线 above alpha*H 的持续性 + 首次触达时间
    thr = float(near_ratio) * Ht
    hit_mask = (S_coarse >= thr)
    hit98_frac = hit_mask.to(dtype).mean(dim=-1, keepdim=True)

    any_hit = hit_mask.any(dim=-1, keepdim=True)
    first_idx = hit_mask.to(torch.int64).argmax(dim=-1, keepdim=True)  # all-false时为0
    first_hit98_t = first_idx.to(dtype) / denom_t
    first_hit98_t = torch.where(any_hit, first_hit98_t, torch.ones_like(first_hit98_t))

    # 粗路径波动（log-return RV）
    if nD >= 2:
        logS = torch.log(S_coarse.clamp_min(eps))
        dlogS = logS[..., 1:] - logS[..., :-1]
        rv_coarse = (dlogS ** 2).sum(dim=-1, keepdim=True)
    else:
        rv_coarse = torch.zeros_like(s_max)

    # 无量纲版本（只保留一个，防止太多）
    s_max_over_H = s_max / (Ht + eps)

    struct10 = torch.cat([
        s_max,          # 1
        s_T,            # 2
        margin_H,       # 3
        retrace,        # 4
        t_max,          # 5
        near_frac,      # 6
        hit98_frac,     # 7
        first_hit98_t,  # 8
        rv_coarse,      # 9
        s_max_over_H,   # 10
    ], dim=-1)

    return struct10


# =========================================================
# 3) 从 Z 构造“少量组合特征”（可选）
#    目标：让网络更容易学一些非线性，不要太多
# =========================================================
def z_combo_features(
    Z: torch.Tensor,
    max_num: int,
):
    """
    返回最多 max_num 个 Z 组合特征（shape [..., m], m<=max_num）

    候选顺序（按优先级）:
      z1^2, z2^2, z1*z2, |z1|, z3^2, z1*z3, |z2|, z2*z3
    会根据 Z 的维度自动裁剪。

    说明：
      - 这里只从前几维取组合，避免维度爆炸
      - 你后续可做消融看哪些真有用
    """
    if max_num <= 0:
        shape = list(Z.shape[:-1]) + [0]
        return Z.new_empty(shape)

    nD = Z.shape[-1]
    feats: List[torch.Tensor] = []

    z1 = Z[..., 0:1] if nD >= 1 else None
    z2 = Z[..., 1:2] if nD >= 2 else None
    z3 = Z[..., 2:3] if nD >= 3 else None

    candidates: List[Optional[torch.Tensor]] = []

    if z1 is not None:
        candidates.append(z1 ** 2)
    if z2 is not None:
        candidates.append(z2 ** 2)
    if z1 is not None and z2 is not None:
        candidates.append(z1 * z2)
    if z1 is not None:
        candidates.append(torch.abs(z1))
    if z3 is not None:
        candidates.append(z3 ** 2)
    if z1 is not None and z3 is not None:
        candidates.append(z1 * z3)
    if z2 is not None:
        candidates.append(torch.abs(z2))
    if z2 is not None and z3 is not None:
        candidates.append(z2 * z3)

    for c in candidates:
        if c is not None:
            feats.append(c)
        if len(feats) >= max_num:
            break

    if len(feats) == 0:
        shape = list(Z.shape[:-1]) + [0]
        return Z.new_empty(shape)

    return torch.cat(feats, dim=-1)


# =========================================================
# 4) 统一版 features_from_Z（仅 barrier 模式）
#    总维度严格 = dimX
#
#    结构:
#      [Z_raw_prefix, Z_combo(optional), barrier_struct10]
#
#    barrier_struct10 固定占 10 维（dimX < 10 时会自动截断）
# =========================================================
def features_from_Z_barrier(
    Z: torch.Tensor,
    dimX: int,
    *,
    theta,                    # 至少 (r, S0, sigma)，可传 (r,S0,sigma,K)
    G: torch.Tensor,          # [nD,nD]
    T: float = 1.0,
    H: float = 105.0,
    k_proxy: Optional[int] = None,   # 不传时默认按 z_raw_used + z_combo_used 决定
    use_z_combos: bool = True,
    z_combo_max: int = 3,            # 最多加几个 Z 组合特征（建议 2~4）
    near_ratio: float = 0.98,
    near_band: float = 2.0,
):
    """
    只做 barrier 模式，输出总维度严格为 dimX。

    逻辑：
      - barrier struct 固定 10 维（若 dimX<10 则截断）
      - 剩余维度给 Z 特征块：
          Z 特征块 = [raw Z 前缀 + 少量 Z 组合]
      - 为了保证总维度，raw 和 combo 会自动分配数量

    返回:
      X: [..., dimX]
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must have at least 2 dims, got shape={tuple(Z.shape)}")
    if dimX <= 0:
        raise ValueError(f"dimX must be >= 1, got {dimX}")
    if theta is None or len(theta) < 3:
        raise ValueError("theta 至少需要 (r, S0, sigma)")
    if G is None:
        raise ValueError("G cannot be None")

    nD = Z.shape[-1]
    r, S0, sigma = theta[0], theta[1], theta[2]

    # barrier 结构特征目标维数（最多10）
    struct_dim_full = 10
    struct_dim_use = min(struct_dim_full, dimX)

    # 给 Z 块留下的维度
    z_budget = dimX - struct_dim_use

    # Z 组合特征数量（自动裁剪）
    z_combo_use = 0
    if use_z_combos and z_budget > 0:
        z_combo_use = min(int(z_combo_max), z_budget)

    # 先实际生成组合特征，再按实际维度回填（因为 Z 维度可能不够）
    z_combo = z_combo_features(Z, max_num=z_combo_use) if z_combo_use > 0 else Z.new_empty((*Z.shape[:-1], 0))
    z_combo_actual = z_combo.shape[-1]

    # 剩余给 raw Z 前缀
    z_raw_use = z_budget - z_combo_actual
    z_raw_use = max(z_raw_use, 0)

    if z_raw_use > nD:
        raise ValueError(f"Need z_raw_use <= nD, got z_raw_use={z_raw_use}, nD={nD}")

    # 粗路径重建时用多少维（默认至少覆盖你喂给模型的 Z 信息）
    if k_proxy is None:
        k_proxy_eff = max(z_raw_use + z_combo_actual, 1)
    else:
        k_proxy_eff = int(k_proxy)
    k_proxy_eff = min(max(k_proxy_eff, 1), nD)

    # barrier struct10
    struct10 = coarse_barrier_struct10_from_Z_gbm(
        Z,
        k_proxy=k_proxy_eff,
        r=r,
        S0=S0,
        sigma=sigma,
        G=G,
        T=T,
        H=H,
        near_ratio=near_ratio,
        near_band=near_band,
    )

    struct = struct10[..., :struct_dim_use]

    # raw Z
    z_raw = Z[..., :z_raw_use] if z_raw_use > 0 else Z.new_empty((*Z.shape[:-1], 0))

    # 拼接：raw Z + combo Z + barrier struct
    X = torch.cat([z_raw, z_combo, struct], dim=-1)

    # 最终严格校验总维度
    if X.shape[-1] != dimX:
        raise RuntimeError(f"Output feature dim mismatch: got {X.shape[-1]}, expected dimX={dimX}")

    return X


# =========================================================
# 5) （可选）打印一下当前 dimX 下特征构成，方便你做消融
# =========================================================
def describe_barrier_feature_layout(
    dimX: int,
    *,
    use_z_combos: bool = True,
    z_combo_max: int = 3,
):
    """
    只是描述“设计意图”的维度分配（不考虑 Z 维度不足导致组合特征减少的情况）
    真正分配以 features_from_Z_barrier 的实际输出为准。
    """
    struct_dim_use = min(10, dimX)
    z_budget = dimX - struct_dim_use
    z_combo_use = min(z_combo_max, z_budget) if (use_z_combos and z_budget > 0) else 0
    z_raw_use = max(z_budget - z_combo_use, 0)

    print(f"[layout] dimX={dimX}")
    print(f"  raw Z (planned):   {z_raw_use}")
    print(f"  Z combos (planned):{z_combo_use}")
    print(f"  barrier struct:    {struct_dim_use} (from struct10)")
    print(f"  total:             {z_raw_use + z_combo_use + struct_dim_use}")


# def arithmetic_payoff(S, K,S0):
#     A = S.mean(dim=-1)
#     return torch.clamp(A - K, min=0.0) #min下界
#     # return A

# def arithmetic_payoff(S, K, S0):
#     """
#     亚式看涨 payoff = max(A-K, 0) 的 pathwise Delta payoff（对 S0 求导）
#
#     参数
#     ----
#     S  : 路径价格张量, shape [..., n_steps]
#     K  : 行权价（标量或可广播）
#     S0 : 初始价格（标量或可广播到 A 的shape）
#
#     返回
#     ----
#     delta_payoff : shape [...]
#         未贴现的 Delta payoff
#     """
#     A = S.mean(dim=-1)
#     indicator = (A > K).to(S.dtype)
#     return indicator * (A / S0)


#障碍
def arithmetic_payoff(S, K, S0=None,H=105):
    S_T = S[..., -1]
    vanilla = torch.clamp(S_T - K, min=0.0)
    knocked_out = (S >= H).any(dim=-1)
    return torch.where(knocked_out, torch.zeros_like(vanilla), vanilla)

#回望
# def arithmetic_payoff(S, K=None,S0=None):
#     S_max = S.max(dim=-1).values
#     return torch.clamp(S_max - K, min=0.0)

# def arithmetic_payoff(S, K=None, S0 =None):
#     S_min = S.min(dim=-1).values
#     S_T = S[..., -1]
#     return S_T - S_min   # 浮动回望看涨


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
#     # return torch.clamp(rv - K, min=0.0)
#     return rv

# def geometric_payoff(S, K):
#     G = torch.exp(torch.log(S).mean(dim=1))
#     return torch.clamp(G - K, min=0.0)


if __name__ == "__main__":
    # 假数据自测
    B, N, nD = 4, 8, 16
    Z = torch.randn(B, N, nD)
    G = torch.eye(nD)

    r = torch.full((B, N), 0.02)
    S0 = torch.full((B, N), 100.0)
    sigma = torch.full((B, N), 0.2)
    K = torch.full((B, N), 100.0)

    describe_barrier_feature_layout(dimX=36, use_z_combos=True, z_combo_max=3)

    X = features_from_Z_barrier(
        Z,
        dimX=20,
        theta=(r, S0, sigma, K),
        G=G,
        T=1.0,
        H=105.0,
        k_proxy=8,
        use_z_combos=True,
        z_combo_max=3,
        near_ratio=0.98,
        near_band=2.0,
    )
    print("X.shape =", X.shape)  # 期望 [B, N, 20]