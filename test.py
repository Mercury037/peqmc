# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Iterable, Optional, Dict
#
#
# def PCA(n: int, T: float = 1.0) -> np.ndarray:
#     """
#     PCA/谱分解法生成 Brownian Motion 的生成矩阵 A（不含0点，共 n 个点）：
#       C_{ij} = min(t_i, t_j)
#       C = V diag(lam) V^T
#       A = V diag(sqrt(lam))
#     并且按特征值从大到小排序，使第1列对应最大特征值（第一主成分）。
#
#     返回:
#       A: shape = (n, n)
#     """
#     if n <= 0:
#         raise ValueError("n must be positive.")
#     if T < 0:
#         raise ValueError("T must be non-negative.")
#
#     t = np.linspace(1.0 / n, 1.0, n)
#
#     # 协方差矩阵 C_{ij} = min(t_i, t_j)
#     C = np.minimum.outer(t, t)
#
#     # 对称特征分解：eigh 返回升序特征值
#     evals, evecs = np.linalg.eigh(C)
#
#     # 截断负特征值（数值误差）
#     evals = np.maximum(evals, 0.0)
#
#     # 降序排列，使 evals[0] 为最大（第一主成分）
#     idx = np.argsort(evals)[::-1]
#     evals = evals[idx]
#     evecs = evecs[:, idx]
#
#     # 生成矩阵 A = V D^{1/2}
#     A = evecs @ np.diag(np.sqrt(evals))
#
#     # 时间尺度 T：BM 协方差整体乘 T，因此 A 乘 sqrt(T)
#     return A * np.sqrt(T)
#
#
# def bm_covariance(n: int, T: float = 1.0) -> np.ndarray:
#     """
#     Brownian Motion 在时刻 t_i=i/n (i=1,...,n) 的协方差矩阵:
#       C_{ij} = T * min(i/n, j/n)
#     """
#     if n <= 0:
#         raise ValueError("n must be positive.")
#     if T < 0:
#         raise ValueError("T must be non-negative.")
#
#     t = np.linspace(1.0 / n, 1.0, n)
#     return T * np.minimum.outer(t, t)
#
#
# def pca_eigens_bm(n: int, T: float = 1.0):
#     """
#     返回 BM 协方差矩阵的 PCA 特征值/特征向量（按降序）
#     """
#     C = bm_covariance(n, T)
#     evals, evecs = np.linalg.eigh(C)
#     evals = np.maximum(evals, 0.0)
#     idx = np.argsort(evals)[::-1]
#     evals = evals[idx]
#     evecs = evecs[:, idx]
#     return evals, evecs
#
#
# def cumulative_ratio_from_evals(evals: np.ndarray) -> np.ndarray:
#     """
#     累计贡献率:
#       cum[k-1] = (sum_{i=1}^k lambda_i) / (sum_{i=1}^n lambda_i)
#     """
#     evals = np.asarray(evals, dtype=np.float64).reshape(-1)
#     total = evals.sum()
#     if total <= 0:
#         raise ValueError("Sum of eigenvalues is non-positive.")
#     return np.cumsum(evals) / total
#
#
# def cumulative_ratio_from_A(A: np.ndarray) -> np.ndarray:
#     """
#     用 A 的列能量来算累计贡献率:
#       col_energy_i = ||a_i||^2
#       cum[k-1] = (sum_{i=1}^k ||a_i||^2) / (sum_{i=1}^n ||a_i||^2)
#
#     理论上应与特征值法一致（数值误差内）。
#     """
#     A = np.asarray(A, dtype=np.float64)
#     if A.ndim != 2 or A.shape[0] != A.shape[1]:
#         raise ValueError(f"A must be square matrix, got shape={A.shape}")
#     col_energy = np.sum(A ** 2, axis=0)
#     total = col_energy.sum()
#     if total <= 0:
#         raise ValueError("Total column energy is non-positive.")
#     return np.cumsum(col_energy) / total
#
#
# def ratios_at_k(cum_ratio: np.ndarray, k_list: Iterable[int]) -> Dict[int, float]:
#     """
#     提取指定 k 的累计贡献率
#     """
#     cum_ratio = np.asarray(cum_ratio, dtype=np.float64).reshape(-1)
#     n = cum_ratio.size
#     out = {}
#     for k in k_list:
#         if not (1 <= k <= n):
#             raise ValueError(f"k must be in [1,{n}], got {k}")
#         out[int(k)] = float(cum_ratio[k - 1])
#     return out
#
#
# def first_k_for_threshold(cum_ratio: np.ndarray, threshold: float) -> int:
#     """
#     返回达到累计贡献率阈值（如 0.95/0.99）所需的最小 k
#     """
#     if not (0.0 < threshold <= 1.0):
#         raise ValueError("threshold must be in (0,1].")
#     cum_ratio = np.asarray(cum_ratio, dtype=np.float64).reshape(-1)
#     k = int(np.searchsorted(cum_ratio, threshold, side="left") + 1)
#     return min(k, cum_ratio.size)
#
#
# def plot_cumulative_ratio(cum_ratio: np.ndarray, title: str = "BM PCA cumulative explained variance ratio"):
#     """
#     画累计贡献率曲线
#     """
#     cum_ratio = np.asarray(cum_ratio, dtype=np.float64).reshape(-1)
#     n = cum_ratio.size
#     x = np.arange(1, n + 1)
#
#     plt.figure(figsize=(7, 4.5))
#     plt.plot(x, cum_ratio, linewidth=2)
#     plt.xlabel("k (number of PCA components)")
#     plt.ylabel("Cumulative explained variance ratio")
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_individual_ratio(evals: np.ndarray, max_k: Optional[int] = 50, title: str = "BM PCA individual explained variance ratio"):
#     """
#     画前 max_k 个主成分的单独贡献率（柱状图）
#     """
#     evals = np.asarray(evals, dtype=np.float64).reshape(-1)
#     total = evals.sum()
#     if total <= 0:
#         raise ValueError("Sum of eigenvalues is non-positive.")
#     ratio = evals / total
#
#     n = ratio.size
#     if max_k is None:
#         max_k = n
#     max_k = max(1, min(int(max_k), n))
#
#     x = np.arange(1, max_k + 1)
#     y = ratio[:max_k]
#
#     plt.figure(figsize=(7, 4.5))
#     plt.bar(x, y)
#     plt.xlabel("PCA component index")
#     plt.ylabel("Individual explained variance ratio")
#     plt.title(title + f" (top {max_k})")
#     plt.grid(True, axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # ===== 你可以改这里 =====
#     n = 256          # 维度（时间步数，不含 t=0）
#     T = 1.0          # 到期时间
#     k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
#     thresholds = [0.90, 0.95, 0.99]
#
#     # ===== 1) 生成 A =====
#     A = PCA(n=n, T=T)
#
#     # ===== 2) 用特征值法算累计贡献率（推荐）=====
#     evals, _ = pca_eigens_bm(n=n, T=T)
#     cum_eig = cumulative_ratio_from_evals(evals)
#
#     # ===== 3) 用 A 的列能量算累计贡献率（交叉验证）=====
#     cum_A = cumulative_ratio_from_A(A)
#
#     # ===== 4) 检查两种算法是否一致 =====
#     max_abs_diff = float(np.max(np.abs(cum_eig - cum_A)))
#     print("=" * 70)
#     print(f"n = {n}, T = {T}")
#     print(f"max |cum_ratio_from_evals - cum_ratio_from_A| = {max_abs_diff:.3e}")
#     print("=" * 70)
#
#     # ===== 5) 打印指定 k 的累计贡献率 =====
#     selected = ratios_at_k(cum_eig, [k for k in k_list if k <= n])
#     print("Cumulative explained variance ratio at selected k:")
#     for k, v in selected.items():
#         print(f"  k={k:>4d} : {v:.8f}")
#
#     # ===== 6) 打印达到阈值所需的最小 k =====
#     print("\nMinimal k for target cumulative ratio:")
#     for th in thresholds:
#         k_need = first_k_for_threshold(cum_eig, th)
#         print(f"  threshold={th:.2%} -> k={k_need}, ratio={cum_eig[k_need-1]:.8f}")
#
#     # ===== 7) 画图 =====
#     plot_cumulative_ratio(cum_eig, title=f"BM PCA cumulative explained variance ratio (n={n}, T={T})")
#     plot_individual_ratio(evals, max_k=min(50, n), title=f"BM PCA individual explained variance ratio (n={n}, T={T})")
#
#
# if __name__ == "__main__":
#     main()


import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Dict


def BB(level: int, T: float = 1.0, round_digits: int | None = None) -> np.ndarray:
    """
    Brownian Bridge 生成矩阵（离散时点，不含 t=0，含终点 t=T）。

    参数
    ----
    level : int
        层数。输出矩阵大小为 size x size，其中 size = 2^level。
        例如 level=8 -> size=256
    T : float
        到期时间（协方差整体乘 T，因此生成矩阵乘 sqrt(T)）
    round_digits : int | None
        若给定整数，则最后对 A 做 np.round(A, round_digits)。
        默认 None（推荐，不做舍入，保留精度）

    返回
    ----
    A : np.ndarray, shape = (2^level, 2^level)
    """
    if level <= 0:
        raise ValueError(f"level must be positive, got {level}")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")

    size = 2 ** level
    A = np.zeros((size, size), dtype=np.float64)

    # 终点 W(T) 放在第1列（最重要维度）
    A[size - 1, 0] = 1.0

    # k 使用 1-based 内部编号：1,2,...,size-1
    k_list = np.arange(1, size, dtype=int)

    # i_k = 最右侧1的位置（0-based）
    i_k = np.array([(int(k) & -int(k)).bit_length() - 1 for k in k_list], dtype=int)

    # round 数
    rounds = level - i_k

    # 按 round 从小到大（稳定排序）
    order = np.argsort(rounds, kind="stable")
    k_sorted = k_list[order]
    i_sorted = i_k[order]
    r_sorted = rounds[order]

    for k, ik, rr in zip(k_sorted, i_sorted, r_sorted):
        k = int(k)
        ik = int(ik)
        rr = int(rr)

        power = 2 ** ik
        j = (k - power) // (2 ** (ik + 1))

        k_prev = k - power
        k_next = k + power

        # 邻居越界视为边界（零向量）
        a_prev = A[k_prev - 1, :] if (1 <= k_prev <= size - 1) else np.zeros(size, dtype=np.float64)
        a_next = A[k_next - 1, :] if (1 <= k_next <= size - 1) else np.zeros(size, dtype=np.float64)

        a_k = (a_prev + a_next) / 2.0

        # 0-based 列索引
        col_idx = (2 ** (rr - 1)) + j
        a_k[col_idx] = math.sqrt(1.0 / (2 ** (rr + 1)))

        # 写入第 k 个时点（0-based 行索引 k-1）
        A[k - 1, :] = a_k

    A = A * math.sqrt(T)

    if round_digits is not None:
        A = np.round(A, round_digits)

    return A


def bb_cumulative_energy_ratio(A: np.ndarray) -> np.ndarray:
    """
    计算 BB 生成矩阵 A 的列能量累计占比：
      cum[k-1] = (sum_{i=1}^k ||a_i||^2) / (sum_{i=1}^N ||a_i||^2)

    其中 a_i 是 A 的第 i 列。
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square matrix, got shape={A.shape}")

    col_energy = np.sum(A ** 2, axis=0)   # 每列 ||a_i||^2
    total_energy = col_energy.sum()
    if total_energy <= 0:
        raise ValueError("Total column energy must be positive.")

    cum_ratio = np.cumsum(col_energy) / total_energy
    return cum_ratio


def bb_energy_ratio_at_k(A: np.ndarray, k: int) -> float:
    """
    返回前 k 列能量占比
    """
    n = A.shape[1]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}], got {k}")
    cum_ratio = bb_cumulative_energy_ratio(A)
    return float(cum_ratio[k - 1])


def ratios_at_k(cum_ratio: np.ndarray, k_list: Iterable[int]) -> Dict[int, float]:
    """
    提取指定 k 的累计能量占比
    """
    cum_ratio = np.asarray(cum_ratio, dtype=np.float64).reshape(-1)
    n = cum_ratio.size
    out = {}
    for k in k_list:
        if 1 <= int(k) <= n:
            out[int(k)] = float(cum_ratio[int(k) - 1])
    return out


def first_k_for_threshold(cum_ratio: np.ndarray, threshold: float) -> int:
    """
    达到累计比例阈值（如 0.90/0.95/0.99）所需最小 k
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in (0,1].")
    cum_ratio = np.asarray(cum_ratio, dtype=np.float64).reshape(-1)
    k = int(np.searchsorted(cum_ratio, threshold, side="left") + 1)
    return min(k, cum_ratio.size)


def plot_bb_energy(cum_ratio: np.ndarray, level: int, T: float):
    """
    画 BB 前k列累计能量占比曲线
    """
    cum_ratio = np.asarray(cum_ratio, dtype=np.float64).reshape(-1)
    n = cum_ratio.size
    x = np.arange(1, n + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, cum_ratio, linewidth=2)
    plt.xlabel("k (number of BB dimensions)")
    plt.ylabel("Cumulative column-energy ratio")
    plt.title(f"BB cumulative energy ratio (level={level}, size={n}, T={T})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bb_individual_energy(A: np.ndarray, max_k: Optional[int] = 50):
    """
    画前 max_k 列的单列能量占比柱状图
    """
    A = np.asarray(A, dtype=np.float64)
    col_energy = np.sum(A ** 2, axis=0)
    ratio = col_energy / col_energy.sum()

    n = ratio.size
    if max_k is None:
        max_k = n
    max_k = max(1, min(int(max_k), n))

    x = np.arange(1, max_k + 1)
    y = ratio[:max_k]

    plt.figure(figsize=(7, 4.5))
    plt.bar(x, y)
    plt.xlabel("BB dimension index")
    plt.ylabel("Individual column-energy ratio")
    plt.title(f"BB individual energy ratio (top {max_k})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # ===== 你改这里 =====
    level = 8                 # size = 2^level, 比如 8 -> 256
    T = 1.0
    k_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    thresholds = [0.90, 0.95, 0.99]

    # ===== 生成 BB 矩阵 =====
    A = BB(level=level, T=T, round_digits=None)
    size = A.shape[0]

    # ===== 计算累计能量占比 =====
    cum_ratio = bb_cumulative_energy_ratio(A)

    print("=" * 72)
    print(f"BB matrix generated: level={level}, size={size}, T={T}")
    print("=" * 72)

    # 打印指定 k 的结果
    selected = ratios_at_k(cum_ratio, [k for k in k_list if k <= size])
    print("Cumulative BB column-energy ratio at selected k:")
    for k, v in selected.items():
        print(f"  k={k:>4d} : {v:.8f}")

    # 打印达到阈值所需最小 k
    print("\nMinimal k for target cumulative ratio:")
    for th in thresholds:
        k_need = first_k_for_threshold(cum_ratio, th)
        print(f"  threshold={th:.2%} -> k={k_need}, ratio={cum_ratio[k_need-1]:.8f}")

    # （可选）检查 A A^T 是否接近 BM 协方差
    C_emp = A @ A.T
    t = np.linspace(1 / size, 1.0, size)
    C_theory = T * np.minimum.outer(t, t)
    err = np.max(np.abs(C_emp - C_theory))
    print(f"\nSanity check: max|A A^T - C_BM| = {err:.3e}")

    # 画图
    plot_bb_energy(cum_ratio, level=level, T=T)
    plot_bb_individual_energy(A, max_k=min(50, size))


if __name__ == "__main__":
    main()