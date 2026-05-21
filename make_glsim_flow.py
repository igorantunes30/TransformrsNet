"""Gera um fluxograma do pipeline GLSim em PNG."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "eval_test_output", "glsim_flow.png")

fig, ax = plt.subplots(figsize=(15, 11))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# paleta
C_IN     = "#FFE0B2"   # entrada
C_VIT    = "#90CAF9"   # ViT encoder
C_TOK    = "#CE93D8"   # tokens
C_GLS    = "#FFCC80"   # GLSCM
C_CROP   = "#A5D6A7"   # crop
C_AGG    = "#F48FB1"   # aggregator
C_OUT    = "#EF9A9A"   # classificador
EDGE     = "#37474F"


def box(x, y, w, h, color, text, fontsize=10, weight="bold", subtitle=None):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.4,rounding_size=1.2",
                          facecolor=color, edgecolor=EDGE, linewidth=1.6)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#102027")
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center",
                fontsize=fontsize-2, color="#37474F", style="italic")
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#102027")


def arrow(x1, y1, x2, y2, label=None, color=EDGE, style="->", curve=0.0,
          label_offset=(0, 0), fs=8):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=18,
                          color=color, linewidth=1.7,
                          connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.5))


ax.text(50, 97.5, "Pipeline GLSim — Global-Local Similarity for Fine-Grained Recognition",
        ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(50, 94, "Rios, Hu & Lai (2024) — ISCAS 2025",
        ha="center", va="center", fontsize=10, style="italic", color="#455A64")

box(3, 80, 16, 9, C_IN, "1. Imagem", subtitle="224 × 224 × 3", fontsize=11)

box(25, 80, 21, 9, C_VIT, "2. ViT Encoder",
    subtitle="ViT-B/16 — 12 camadas", fontsize=11)

box(52, 80, 23, 9, C_TOK, "3. Tokens de saída",
    subtitle="CLS  ⊕  196 patches  (768d)", fontsize=11)

box(81, 80, 16, 9, C_VIT, "CLS global",
    subtitle="vetor 768d", fontsize=10)

arrow(19, 84.5, 25, 84.5)
arrow(46, 84.5, 52, 84.5)
arrow(75, 84.5, 81, 84.5)

# ── linha 2: GLSCM ────────────────────────────────────────────────────────────
ax.text(50, 73, "GLSCM — Global-Local Similarity Crop Module",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#E65100")
ax.add_patch(Rectangle((2, 48), 96, 22, facecolor="#FFF8E1",
                       edgecolor="#FB8C00", linewidth=1.2, linestyle="--",
                       alpha=0.5))

box(4, 56, 19, 10, C_GLS, "4. Similaridade",
    subtitle="cos(CLS, patch_i)\npara i = 1..196", fontsize=10)
box(27, 56, 19, 10, C_GLS, "5. Top-K patches",
    subtitle="K patches mais\nsimilares ao CLS", fontsize=10)
box(50, 56, 19, 10, C_GLS, "6. Bounding box",
    subtitle="min/max em (x, y)\ndos patches top-K", fontsize=10)
box(73, 56, 22, 10, C_GLS, "7. Crop + Resize",
    subtitle="recorta imagem original\nupsample → 224 × 224", fontsize=10)

arrow(23, 61, 27, 61)
arrow(46, 61, 50, 61)
arrow(69, 61, 73, 61)

# entrada vinda dos tokens
arrow(63, 80, 13.5, 66, curve=-0.25, color="#FB8C00",
      label="tokens", label_offset=(-3, -3))

# ── linha 3: segunda passada ─────────────────────────────────────────────────
box(3, 35, 17, 9, C_CROP, "8. Crop",
    subtitle="224 × 224 × 3", fontsize=11)

box(25, 35, 22, 9, C_VIT, "9. Mesmo ViT",
    subtitle="encoder compartilhado", fontsize=11)

box(53, 35, 20, 9, C_TOK, "10. CLS local",
    subtitle="vetor 768d (crop)", fontsize=10)

arrow(84, 56, 11.5, 44, curve=-0.3, color="#43A047",
      label="imagem recortada")
arrow(20, 39.5, 25, 39.5)
arrow(47, 39.5, 53, 39.5)

# ── linha 4: aggregator + classificador ──────────────────────────────────────
box(15, 17, 25, 11, C_AGG, "11. Aggregator",
    subtitle="mini-transformer\n(1 camada)\nconcat(CLS_global, CLS_local)",
    fontsize=10)

box(48, 17, 22, 11, C_OUT, "12. Classifier head",
    subtitle="Linear(768 → 200)", fontsize=11)

box(78, 17, 18, 11, C_OUT, "13. Predição",
    subtitle="200 classes\n(softmax)", fontsize=11)

# CLS global → aggregator
arrow(89, 80, 36, 28, curve=0.35, color="#1565C0",
      label="CLS global")
# CLS local → aggregator
arrow(63, 35, 28, 28, curve=-0.25, color="#2E7D32",
      label="CLS local")
arrow(40, 22.5, 48, 22.5)
arrow(70, 22.5, 78, 22.5)

# legenda inferior
ax.text(50, 7, "Saída final do agregador → cabeça linear → logits sobre as 200 espécies",
        ha="center", va="center", fontsize=10, style="italic", color="#455A64")
ax.text(50, 3, "Custo extra do GLSim ≈ top-K + 1 forward extra do encoder (compartilhado, sem parâmetros novos no ViT)",
        ha="center", va="center", fontsize=9, color="#607D8B")

plt.tight_layout()
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Salvo: {OUT}")
