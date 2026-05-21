"""
Copia uma imagem representativa de pares de espécies do MESMO grupo
(top confusões) para eval_test_output/grupos/ e gera uma figura comparativa.
"""
import os
import shutil
import random
import matplotlib.pyplot as plt
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_ROOT = r"D:\mestrado\RNA\SEMINARIO\data\cub\CUB_200_2011\images"
OUT_DIR = os.path.join(BASE, "eval_test_output", "grupos")
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
    ("Jaeger", "071.Long_tailed_Jaeger", "072.Pomarine_Jaeger"),
    ("Crow",   "029.American_Crow",     "030.Fish_Crow"),
    ("Tern",   "144.Common_Tern",       "146.Forsters_Tern"),
]

random.seed(0)

def pick_image(folder):
    files = sorted(os.listdir(folder))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return random.choice(files) if files else None

selected = []
for group, sp1, sp2 in PAIRS:
    for sp in (sp1, sp2):
        src_dir = os.path.join(IMG_ROOT, sp)
        fname   = pick_image(src_dir)
        src     = os.path.join(src_dir, fname)
        dst_name = f"{group}__{sp}__{fname}"
        dst     = os.path.join(OUT_DIR, dst_name)
        shutil.copy2(src, dst)
        selected.append((group, sp, dst))
        print(f"copiado: {dst_name}")

# figura comparativa
fig, axes = plt.subplots(len(PAIRS), 2, figsize=(8, 11))
for i, (group, sp1, sp2) in enumerate(PAIRS):
    for j, sp in enumerate((sp1, sp2)):
        # encontra arquivo copiado
        path = next(p for g, s, p in selected if g == group and s == sp)
        img = Image.open(path).convert("RGB")
        ax = axes[i, j]
        ax.imshow(img)
        ax.axis("off")
        nice = sp.split(".", 1)[-1].replace("_", " ")
        ax.set_title(nice, fontsize=11, fontweight="bold")
    axes[i, 0].text(-0.08, 0.5, f"Grupo: {group}",
                    transform=axes[i, 0].transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=12, fontweight="bold", color="#1565C0")

plt.suptitle("Pares de espécies do mesmo grupo — top confusões do modelo",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "pares_comparacao.png")
plt.savefig(out_fig, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nFigura salva: {out_fig}")
print(f"Pasta: {OUT_DIR}")
