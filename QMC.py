import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ====== 参数 ======
d = 256
m = 13
n = 2**m
seed_qmc = 42
seed_mc  = 42

# ====== QMC: Sobol (scrambled) ======
sampler = qmc.Sobol(d=d, scramble=True, seed=seed_qmc)
U_qmc = sampler.random_base2(m=m)          # (n, d)

# ====== MC: i.i.d. Uniform ======
rng = np.random.default_rng(seed_mc)
U_mc = rng.random((n, d))                  # (n, d)

# （可选）算个 discrepancy 做个量化对比（越小越“均匀”）
disc_qmc = qmc.discrepancy(U_qmc)
disc_mc  = qmc.discrepancy(U_mc)

# ====== 投影对比： (1,2), (1,3), (4,5) ======
pairs = [(1, 211), (1, 124), (1, 250)]  # 1-based
for a, b in pairs:
    i, j = a - 1, b - 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)

    axes[0].scatter(U_qmc[:, i], U_qmc[:, j], s=6, alpha=0.35)
    axes[0].set_title(f"Sobol QMC (scramble, n={n})")
    axes[0].set_xlabel(f"U[{a}]")
    axes[0].set_ylabel(f"U[{b}]")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.4)

    axes[1].scatter(U_mc[:, i], U_mc[:, j], s=6, alpha=0.35)
    axes[1].set_title(f"i.i.d. Uniform (n={n})")
    axes[1].set_xlabel(f"U[{a}]")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.4)

    fig.suptitle(
        f"Projection ({a},{b})  |  discrepancy: QMC={disc_qmc:.4g}, MC={disc_mc:.4g}",
        y=1.03
    )
    plt.tight_layout()
    plt.show()
