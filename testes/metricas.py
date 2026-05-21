"""
Métricas completas: F1, Precision, Recall, Confusion Matrix
Dataset: especie_passaros (20 classes)
Modelo : GLSim ViT-B/16 treinado em CUB-200-2011 (200 classes)

Estratégia de avaliação (majority-vote mapping):
  Como os espaços de rótulos são diferentes (20 vs 200 classes), para cada
  classe verdadeira determina-se qual classe CUB é predita com maior frequência
  (voto majoritário). A partir desse mapeamento computam-se todas as métricas.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, accuracy_score
)

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "resultados.csv")
OUT_DIR  = os.path.join(BASE_DIR, "metricas_output")

# Apenas espécies com correspondência taxonômica no CUB-200-2011
ESPECIES_FILTRO = {
    "AMERICAN GOLDFINCH",     # exata
    "ALBATROSS",              # exata (nome)
    "ALTAMIRA YELLOWTHROAT",  # mesmo gênero (Geothlypis)
    "AFRICAN EMERALD CUCKOO", # mesma família (Cuculidae)
    "AFRICAN PYGMY GOOSE",    # mesma família (Anatidae)
    "ALPINE CHOUGH",          # mesma família (Corvidae)
    "ALBERTS TOWHEE",         # mesma família (Passerellidae)
}
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Carregar resultados e extrair rótulo verdadeiro ────────────────────────
df = pd.read_csv(CSV_PATH)

def extract_true_label(filename):
    # CLASSNAME__split__file.jpg  →  CLASSNAME
    if filename.startswith("predict__"):
        return None
    m = re.match(r"^(.+?)__(?:train|test|valid)__", filename)
    return m.group(1).replace("_", " ") if m else None

df["true_class"] = df["filename"].apply(extract_true_label)
df_eval = df[df["true_class"].notna()].copy()
df_eval = df_eval[df_eval["true_class"].isin(ESPECIES_FILTRO)].copy()
print(f"Após filtro taxonômico     : {len(df_eval)} imagens")
df_eval["pred_cub"] = df_eval["pred_top1_name"].str.replace(r"^\d+\.", "", regex=True).str.replace("_", " ")

print(f"Total imagens com rótulo: {len(df_eval)}")
print(f"Classes verdadeiras     : {df_eval['true_class'].nunique()}")
print(f"Classes CUB preditas    : {df_eval['pred_cub'].nunique()}")

true_classes = sorted(df_eval["true_class"].unique())

# ── 2. Majority-vote mapping: true_class → CUB_class ─────────────────────────
mapping = {}   # true_class → CUB class com maior frequência
for tc in true_classes:
    sub = df_eval[df_eval["true_class"] == tc]["pred_cub"]
    mapping[tc] = sub.value_counts().idxmax()

print("\n=== Majority-vote mapping (classe verdadeira → CUB mais predita) ===")
for tc, cub in mapping.items():
    cnt  = (df_eval[df_eval["true_class"] == tc]["pred_cub"] == cub).sum()
    total = (df_eval["true_class"] == tc).sum()
    print(f"  {tc:<35} → {cub:<35} ({cnt}/{total} = {cnt/total*100:.1f}%)")

# Rótulo mapeado para cada imagem
df_eval["pred_mapped"] = df_eval["pred_cub"].map(
    {v: k for k, v in mapping.items()}
).fillna("OTHER")

# ── 3. Métricas globais ───────────────────────────────────────────────────────
y_true = df_eval["true_class"].values
y_pred = df_eval["pred_mapped"].values

acc = accuracy_score(y_true, y_pred)
f1_macro   = f1_score(y_true, y_pred, average="macro",    zero_division=0, labels=true_classes)
f1_weighted= f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=true_classes)
prec_macro = precision_score(y_true, y_pred, average="macro",    zero_division=0, labels=true_classes)
rec_macro  = recall_score(y_true, y_pred,    average="macro",    zero_division=0, labels=true_classes)

print(f"\n{'='*55}")
print(f"  Acurácia        : {acc*100:.2f}%")
print(f"  F1  (macro)     : {f1_macro:.4f}")
print(f"  F1  (weighted)  : {f1_weighted:.4f}")
print(f"  Precision (macro): {prec_macro:.4f}")
print(f"  Recall    (macro): {rec_macro:.4f}")
print(f"{'='*55}")

# ── 4. Relatório por classe ───────────────────────────────────────────────────
report = classification_report(y_true, y_pred, labels=true_classes,
                                zero_division=0, output_dict=True)
df_report = pd.DataFrame(report).T.loc[true_classes, ["precision","recall","f1-score","support"]]
df_report = df_report.round(4).sort_values("f1-score", ascending=False)
print("\n=== Relatório por classe ===")
print(df_report.to_string())
df_report.to_csv(os.path.join(OUT_DIR, "relatorio_por_classe.csv"))

# ── 5. Intra-class consistency (sem mapeamento) ───────────────────────────────
print("\n=== Consistência interna (top-1 CUB por classe verdadeira) ===")
consistency = {}
for tc in true_classes:
    sub  = df_eval[df_eval["true_class"] == tc]["pred_cub"]
    top1 = sub.value_counts()
    cons = top1.iloc[0] / len(sub)
    consistency[tc] = {"top_cub": top1.index[0], "consistency": cons, "n": len(sub)}
    print(f"  {tc:<35} {cons*100:5.1f}%  [{top1.index[0]}]")

# ── 6. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=true_classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# Absoluta
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=true_classes, yticklabels=true_classes,
            linewidths=0.4, linecolor="white", ax=axes[0],
            annot_kws={"size": 7})
axes[0].set_title("Confusion Matrix — valores absolutos", fontweight="bold", pad=10)
axes[0].set_xlabel("Predito")
axes[0].set_ylabel("Verdadeiro")
axes[0].tick_params(axis="x", rotation=45, labelsize=7)
axes[0].tick_params(axis="y", rotation=0,  labelsize=7)

# Normalizada
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=true_classes, yticklabels=true_classes,
            linewidths=0.4, linecolor="white", vmin=0, vmax=1, ax=axes[1],
            annot_kws={"size": 7})
axes[1].set_title("Confusion Matrix — normalizada (recall por classe)", fontweight="bold", pad=10)
axes[1].set_xlabel("Predito")
axes[1].set_ylabel("Verdadeiro")
axes[1].tick_params(axis="x", rotation=45, labelsize=7)
axes[1].tick_params(axis="y", rotation=0,  labelsize=7)

plt.suptitle(f"GLSim CUB-200-2011 × especie_passaros (7 espécies c/ parentesco taxonômico)  |  Acc={acc*100:.1f}%  F1={f1_macro:.3f} (macro)",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSalvo: {cm_path}")

# ── 7. F1 / Precision / Recall por classe (barras) ───────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
x   = np.arange(len(true_classes))
w   = 0.26
labels_short = [c.replace("_", " ") for c in df_report.index]

ax.bar(x - w,   df_report["precision"], w, label="Precision", color="#2196F3")
ax.bar(x,       df_report["recall"],    w, label="Recall",    color="#4CAF50")
ax.bar(x + w,   df_report["f1-score"],  w, label="F1-score",  color="#FF5722")

ax.set_xticks(x)
ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 por classe (majority-vote mapping)", fontweight="bold")
ax.axhline(f1_macro,    color="#FF5722", linestyle="--", alpha=0.5, linewidth=1, label=f"F1 macro={f1_macro:.3f}")
ax.axhline(prec_macro,  color="#2196F3", linestyle="--", alpha=0.5, linewidth=1, label=f"Prec macro={prec_macro:.3f}")
ax.axhline(rec_macro,   color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1, label=f"Rec macro={rec_macro:.3f}")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "f1_precision_recall.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Salvo: {bar_path}")

# ── 8. Heatmap cross-dataset (true × top CUB preditas) ───────────────────────
top_cub_classes = (df_eval["pred_cub"].value_counts().head(25).index.tolist())
cross = pd.crosstab(df_eval["true_class"], df_eval["pred_cub"])[top_cub_classes]
cross_norm = cross.div(cross.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(18, 8))
sns.heatmap(cross_norm, cmap="YlOrRd", linewidths=0.3, linecolor="white",
            annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax,
            cbar_kws={"label": "Proporção de predições"})
ax.set_title("Mapeamento cross-dataset: especie_passaros → CUB-200-2011 (top-25 predições)\n"
             "Cada linha = proporção de imagens da classe verdadeira preditas para cada classe CUB",
             fontweight="bold", pad=10)
ax.set_xlabel("Classe CUB predita (top-25 mais frequentes)")
ax.set_ylabel("Classe verdadeira (especie_passaros)")
ax.tick_params(axis="x", rotation=60, labelsize=7)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
cross_path = os.path.join(OUT_DIR, "cross_dataset_heatmap.png")
plt.savefig(cross_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Salvo: {cross_path}")

# ── 9. Consistência por classe (barras) ──────────────────────────────────────
df_cons = pd.DataFrame(consistency).T.sort_values("consistency", ascending=False)
fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336"
          for v in df_cons["consistency"]]
bars = ax.bar(df_cons.index, df_cons["consistency"] * 100, color=colors, edgecolor="white")
for bar, val in zip(bars, df_cons["consistency"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val*100:.0f}%", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Consistência interna (%)")
ax.set_title("Consistência das predições top-1 por classe (% de imagens com mesmo top-1 CUB)",
             fontweight="bold")
ax.set_xticklabels([c.replace("_", " ") for c in df_cons.index], rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 115)
ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
cons_path = os.path.join(OUT_DIR, "consistencia_por_classe.png")
plt.savefig(cons_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Salvo: {cons_path}")

print("\n=== Arquivos gerados ===")
for f in os.listdir(OUT_DIR):
    print(f"  metricas_output/{f}")
