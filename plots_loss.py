import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = "results_qmc_pca_mse"

train_losses = np.load(os.path.join(results_dir, "train_losses.npy"))
val_losses = np.load(os.path.join(results_dir, "val_losses.npy"))

# =========================
# 平滑函数
# =========================
def moving_average(x, window=200):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

train_smooth = moving_average(train_losses, window=200)

# =========================
# 1️⃣ Train Loss (Smoothed)
# =========================
plt.figure(figsize=(8,5))
plt.plot(train_smooth, label="Train (smoothed)")
plt.title("Train Loss (Smoothed)")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "train_loss.png"))
plt.show()

# =========================
# 2️⃣ Validation Loss
# =========================
best_epoch = np.argmin(val_losses)
best_val = val_losses[best_epoch]

plt.figure(figsize=(8,5))
plt.plot(val_losses, marker="o", label="Validation")
plt.scatter(best_epoch, best_val, color="red", zorder=3)
plt.title("Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "val_loss.png"))
plt.show()

print("Best epoch:", best_epoch + 1)
print("Best validation loss:", best_val)

# =========================
# 3️⃣ Train vs Val 对比
# =========================
plt.figure(figsize=(8,5))
plt.plot(train_smooth, label="Train (smoothed)")
plt.plot(
    np.linspace(0, len(train_smooth), len(val_losses)),
    val_losses,
    marker="o",
    label="Validation"
)
plt.title("Train vs Validation Loss")
plt.xlabel("Training Progress")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "train_val_compare.png"))
plt.show()
