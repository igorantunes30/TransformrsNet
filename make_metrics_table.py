"""Gera um PNG com a tabela de métricas do test split."""
import os
import matplotlib.pyplot as plt

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "eval_test_output", "metrics_table.png")

rows = [
    ("Top-1 accuracy",    "90.85%"),
    ("Top-5 accuracy",    "98.34%"),
    ("F1 (macro)",        "0.9084"),
    ("F1 (weighted)",     "0.9076"),
    ("Precision (macro)", "0.9111"),
    ("Recall (macro)",    "0.9095"),
]

fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=["Métrica", "Valor"],
    cellLoc="center",
    colLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.7)

header_color = "#1565C0"
alt_color    = "#F2F6FA"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#9AA0A6")
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor(alt_color if r % 2 == 0 else "white")
        if c == 1:
            cell.set_text_props(weight="bold", color="#1565C0")

ax.set_title("GLSim ViT-B/16 — CUB-200-2011 (test split, N=5794, K=200)",
             fontsize=12, fontweight="bold", pad=14)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Salvo: {OUT_PATH}")
