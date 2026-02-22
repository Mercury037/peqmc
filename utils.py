import torch
import math
def to_batch_param(x, batch_size, device, dtype=torch.float32):
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


def inv_Phi_torch(U: torch.Tensor) -> torch.Tensor:
    """
    标准正态的 Phi^{-1}(U)，纯 torch 实现
    Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    """
    eps = torch.finfo(U.dtype).eps
    U = U.clamp(min=eps, max=1.0 - eps)   # 避免 ppf(0/1) 变成 ±inf
    return math.sqrt(2.0) * torch.erfinv(2.0 * U - 1.0)



def check_power_of_two(n, name="n"):
    if n & (n - 1) != 0:
        raise ValueError(f"[qmc] {name} must be a power of 2 (2^m), got {n}")



def simulate_gbm_batch_mc(theta, batch_size, nD=256, T=1.0, device="cpu", method="cholesky"):
    """
    theta: (r,S0,sigma,K) each is torch scalar or shape [B]
    returns:
      W: [B, nD] Brownian motion at t_1..t_nD
      S: [B, nD] GBM price at t_1..t_nD
    """
    r, S0, sigma, K = theta  # K 保留接口一致，但路径生成不需要它

    r = to_batch_param(r, batch_size, device)          # [B,]
    S0 = to_batch_param(S0, batch_size, device)        # [B,]
    sigma = to_batch_param(sigma, batch_size, device)  # [B,]
    K = to_batch_param(K, batch_size, device)          # [B,]

    # W = Z @ G^T
    G = _generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)  # [nD,nD]
    Z = torch.randn(batch_size, nD, device=device)                                 # [B,nD]
    W = Z @ G.T                                                                    # [B,nD]

    dt = T / nD
    t = (torch.arange(1, nD + 1, device=device, dtype=torch.float32) * dt).unsqueeze(0)  # [1,nD]

    logS = torch.log(S0).unsqueeze(1) + (r - 0.5 * sigma**2).unsqueeze(1) * t + sigma.unsqueeze(1) * W
    S = torch.exp(logS)

    return Z,W,S


#
# def simulate_gbm_batch_qmc(theta, batch_size, nD=256, T=1.0, device="cpu", method="pca",
#                            scramble=True, seed=42, sobol_engine=None):
#     """
#     theta: (r,S0,sigma,K) each is [B] or [B,N]
#       W: [B, nD] Brownian motion at t_1..t_nD
#       S: [B, nD] GBM price at t_1..t_nD
#       如果不传sobol_engine qmc生成器，那么每次都是根据seed重新来一个，
#       如果传的话，就是跟着外面的走，循环调用 每次都是不同的rqmc点
#     """
#     r, S0, sigma, K = theta
#     if S0.dim() == 1:
#         # ---- QMC 生成 Z ~ N(0,1)^{nD} ----
#         # 默认强制 batch_size=2^m（最规整）
#         if batch_size & (batch_size - 1) != 0:
#             raise ValueError(f"[qmc] batch_size must be a power of 2 (2^m), got {batch_size}")
#
#         if sobol_engine is None:
#             # 注意：如果你每次都在函数里新建 engine + 固定 seed，那么每次都会从头开始，Z 会重复
#             sobol_engine = SobolEngine(dimension=nD, scramble=scramble, seed=seed)
#
#         U = sobol_engine.draw(batch_size).to(device=device, dtype=torch.float32)  # [B,nD] in [0,1)
#         Z = inv_Phi_torch(U)                                                    # [B,nD] ~ N(0,1)
#
#         # ---- W = Z @ G^T ----
#         G = generator_matrix(method, nD=nD, T=T, device=device, dtype=torch.float32)  # [nD,nD]
#         W = Z @ G.T                                                                    # [B,nD]
#
#         dt = T / nD
#         t = (torch.arange(1, nD + 1, device=device, dtype=torch.float32) * dt).unsqueeze(0)  # [1,nD]
#
#         logS = torch.log(S0).unsqueeze(1) + (r - 0.5 * sigma**2).unsqueeze(1) * t + sigma.unsqueeze(1) * W
#         S = torch.exp(logS)
#
#     return Z,W, S


# def sample_theta_2dim(batch_size, device="cpu", gen=None):
#     r = 0.01 + (0.03 - 0.01) * torch.rand(batch_size, device=device, generator=gen)
#     S0 = 80.0 + (120.0 - 80.0) * torch.rand(batch_size, device=device, generator=gen)
#     sigma = 0.05 + (0.25 - 0.05) * torch.rand(batch_size, device=device, generator=gen)
#     K = 90.0 + (110.0 - 90.0) * torch.rand(batch_size, device=device, generator=gen)
#     return r, S0, sigma, K
#
# def sample_theta_2dim_same(batch_size, device="cpu", gen=None):
#     """
#     [B,4] 所有样本都一样
#     """
#     r1 = 0.01 + (0.03 - 0.01) * torch.rand(1, device=device, generator=gen)
#     S01 = 80.0 + (120.0 - 80.0) * torch.rand(1, device=device, generator=gen)
#     sigma1 = 0.05 + (0.25 - 0.05) * torch.rand(1, device=device, generator=gen)
#     K1 = 90.0 + (110.0 - 90.0) * torch.rand(1, device=device, generator=gen)
#     # 扩展成 batch_size 个完全一样的值
#     r = r1.expand(batch_size)
#     S0 = S01.expand(batch_size)
#     sigma = sigma1.expand(batch_size)
#     K = K1.expand(batch_size)
#
#     return r, S0, sigma, K
#
# def sample_theta_3dim(B, N, device="cpu", gen=None):
#     """
#     返回:
#       r, S0, sigma, K: 都是 [B, N]
#     且对每个 i，固定 i 后的 [i, :] 中所有行都相同
#     （即沿 N 维复制）
#     """
#     # 先采 [B, 1]
#     r_base = 0.01 + (0.03 - 0.01) * torch.rand(B, 1, device=device, generator=gen)
#     S0_base = 80.0 + (120.0 - 80.0) * torch.rand(B, 1, device=device, generator=gen)
#     sigma_base = 0.05 + (0.25 - 0.05) * torch.rand(B, 1, device=device, generator=gen)
#     K_base = 90.0 + (110.0 - 90.0) * torch.rand(B, 1, device=device, generator=gen)
#
#     # 扩展到 [B, N]，每个 [i, :] 的 N 行都一样
#     r = r_base.expand(B, N)
#     S0 = S0_base.expand(B, N)
#     sigma = sigma_base.expand(B, N)
#     K = K_base.expand(B, N)
#
#     return r, S0, sigma, K
#
# import torch
#
# def sample_theta_3dim_same(B, N, d, device="cpu", gen=None):
#     """
#     返回:
#       r, S0, sigma, K: 都是 [B, N]
#     且每个张量内部所有元素都相同（全局共享一个值）
#     """
#     # 每个参数只采一个值（shape=[1,1,1]）
#     r1 = 0.01 + (0.03 - 0.01) * torch.rand(1, 1, device=device, generator=gen)
#     S01 = 80.0 + (120.0 - 80.0) * torch.rand(1, 1, device=device, generator=gen)
#     sigma1 = 0.05 + (0.25 - 0.05) * torch.rand(1, 1, device=device, generator=gen)
#     K1 = 90.0 + (110.0 - 90.0) * torch.rand(1, 1, device=device, generator=gen)
#
#     # 广播到 [B,N,d]
#     r = r1.expand(B, N)
#     S0 = S01.expand(B, N)
#     sigma = sigma1.expand(B, N)
#     K = K1.expand(B, N)
#
#     return r, S0, sigma, K
#