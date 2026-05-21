import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOG_FILE = "training_log.txt"

val_acc1, val_acc5 = [], []
train_loss, train_acc1 = [], []
lr_per_epoch = []

# Patterns
pat_val = re.compile(r"^\s*\* Acc@1 ([\d.]+) Acc@5 ([\d.]+)")
pat_train_last = re.compile(r"Epoch: \[(\d+)/50\]\[700/749\]\s+LR: ([\d.e+\-]+)\s+Loss [\d.]+ \(([\d.]+)\)\s+Acc@1 [\d.]+ \(([\d.]+)\)")

with open(LOG_FILE, encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Collect val results (first 50 only — skip final test eval at line ~5072)
val_collected = 0
train_collected = 0

for line in lines:
    m_val = pat_val.match(line)
    if m_val and val_collected < 50:
        val_acc1.append(float(m_val.group(1)))
        val_acc5.append(float(m_val.group(2)))
        val_collected += 1

    m_train = pat_train_last.search(line)
    if m_train and train_collected < 50:
        lr_per_epoch.append(float(m_train.group(2)))
        train_loss.append(float(m_train.group(3)))
        train_acc1.append(float(m_train.group(4)))
        train_collected += 1

epochs = list(range(1, len(val_acc1) + 1))
best_epoch = int(np.argmax(val_acc1)) + 1
best_acc = max(val_acc1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("GLSim — CUB-200-2011 Training (50 epochs)", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# 1. Validation Top-1 Accuracy
ax1.plot(epochs, val_acc1, color="#2196F3", linewidth=2, label="Val Acc@1")
ax1.axvline(best_epoch, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.scatter([best_epoch], [best_acc], color="red", zorder=5)
ax1.annotate(f"Best: {best_acc:.2f}%\n(epoch {best_epoch})",
             xy=(best_epoch, best_acc), xytext=(best_epoch + 2, best_acc - 4),
             fontsize=8, color="red",
             arrowprops=dict(arrowstyle="->", color="red", lw=1))
ax1.set_title("Validation Accuracy — Top-1")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Acc@1 (%)")
ax1.set_xlim(1, 50)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# 2. Validation Top-1 vs Top-5
ax2.plot(epochs, val_acc1, color="#2196F3", linewidth=2, label="Val Acc@1")
ax2.plot(epochs, val_acc5, color="#4CAF50", linewidth=2, linestyle="--", label="Val Acc@5")
ax2.set_title("Validation Accuracy — Top-1 vs Top-5")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_xlim(1, 50)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# 3. Training Loss
ax3.plot(epochs, train_loss, color="#FF5722", linewidth=2, label="Train Loss (avg)")
ax3.set_title("Training Loss")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Cross-Entropy Loss")
ax3.set_xlim(1, 50)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# 4. Learning Rate Schedule
ax4.plot(epochs, lr_per_epoch, color="#9C27B0", linewidth=2, label="Learning Rate")
ax4.set_title("Learning Rate Schedule (cosine)")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("LR")
ax4.set_xlim(1, 50)
ax4.set_yscale("log")
ax4.grid(True, alpha=0.3, which="both")
ax4.legend(fontsize=9)

# Summary box
summary = (
    f"Best Val Acc@1: {best_acc:.2f}%  (epoch {best_epoch})\n"
    f"Final Val Acc@1: {val_acc1[-1]:.2f}%  |  Acc@5: {val_acc5[-1]:.2f}%\n"
    f"Total training time: 285.46 min  |  Params: 93.04 M"
)
fig.text(0.5, 0.01, summary, ha="center", va="bottom", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

out_path = "training_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
