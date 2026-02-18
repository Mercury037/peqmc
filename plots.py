import numpy as np
import matplotlib.pyplot as plt

# 读取数据
train_losses = np.load("results/train_step_losses.npy")
val_losses = np.load("results/val_epoch_losses.npy")

# =========================
# 1️⃣ 画 Train step loss
# =========================
plt.figure()
plt.plot(train_losses)
plt.title("Train Loss (per step)")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("results/train_loss.png")
plt.show()

# =========================
# 2️⃣ 画 Validation epoch loss
# =========================
plt.figure()
plt.plot(val_losses)
plt.title("Validation Loss (per epoch)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("results/val_loss.png")
plt.show()
