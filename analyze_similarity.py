"""
Análise: espécies visualmente parecidas (mesmo "grupo" — sufixo do nome)
concentram a maior parte dos erros na matriz de confusão?

Estratégia:
  - Para cada classe CUB, extrai o grupo = último token do nome
    (Albatross, Auklet, Sparrow, Warbler, Wren ...). É um proxy do gênero/família.
  - Compara taxa de erro intra-grupo (predição cai em espécie do mesmo grupo)
    vs inter-grupo (espécie de grupo diferente).
  - Reordena a matriz de confusão por grupo → padrão de bloco-diagonal visível.
  - Lista os pares (verdadeiro, predito) mais confundidos e marca se são do
    mesmo grupo.
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "eval_test_output")
PRED_CSV = os.path.join(OUT, "predictions.csv")
CLASS_CSV = os.path.join(BASE, "data", "cub", "classid_classname.csv")  # fallback
if not os.path.exists(CLASS_CSV):
    CLASS_CSV = r"D:\mestrado\RNA\SEMINARIO\data\cub\CUB_200_2011\classid_classname.csv"

df_pred = pd.read_csv(PRED_CSV)
df_cls  = pd.read_csv(CLASS_CSV)
id2name = dict(zip(df_cls["class_id"], df_cls["class_name"]))
class_ids = sorted(id2name.keys())
class_names = [id2name[i] for i in class_ids]
K = len(class_ids)

# ── 1) extrai "grupo" = último token significativo do nome ────────────────────
def get_group(name):
    raw = re.sub(r"^\d+\.", "", name)
    parts = raw.split("_")
    return parts[-1]

groups = [get_group(n) for n in class_names]
group_of = dict(zip(class_ids, groups))

group_to_classes = {}
for cid, g in group_of.items():
    group_to_classes.setdefault(g, []).append(cid)

group_sizes = Counter(groups)
multi_groups = [g for g, n in group_sizes.items() if n > 1]
print(f"Total de grupos      : {len(group_sizes)}")
print(f"Grupos com ≥2 espécies: {len(multi_groups)}")
print(f"Top-10 grupos por nº de espécies: "
      f"{group_sizes.most_common(10)}")

# ── 2) anota cada predição com grupo verdadeiro/predito ──────────────────────
df_pred["true_group"] = df_pred["y_true"].map(group_of)
df_pred["pred_group"] = df_pred["y_pred"].map(group_of)

df_err = df_pred[~df_pred["correct"]].copy()
n_total = len(df_pred)
n_err   = len(df_err)
n_intra = (df_err["true_group"] == df_err["pred_group"]).sum()
n_inter = n_err - n_intra

# baseline aleatório dado o grupo verdadeiro
def expected_intra_random(df, group_sizes):
    """Se um erro fosse uniformemente aleatório entre as outras 199 classes,
    qual fração cairia no mesmo grupo?"""
    e = 0.0
    for _, row in df.iterrows():
        g = row["true_group"]
        same_group_others = group_sizes[g] - 1
        e += same_group_others / (K - 1)
    return e / len(df)

p_intra_obs = n_intra / max(n_err, 1)
p_intra_exp = expected_intra_random(df_err, group_sizes)
lift = p_intra_obs / max(p_intra_exp, 1e-9)

# ── 3) print resumo ──────────────────────────────────────────────────────────
print("\n================= Confusão intra-grupo =================")
print(f"  Total predições        : {n_total}")
print(f"  Erros totais           : {n_err}  ({n_err/n_total*100:.2f}%)")
print(f"  Erros intra-grupo      : {n_intra}  ({p_intra_obs*100:.2f}% dos erros)")
print(f"  Erros inter-grupo      : {n_inter}  ({n_inter/n_err*100:.2f}% dos erros)")
print(f"  Baseline (aleatório)   : {p_intra_exp*100:.2f}% intra-grupo esperado")
print(f"  Lift sobre o aleatório : {lift:.1f}×")
print("=========================================================")

# ── 4) top confusões + flag mesmo grupo ──────────────────────────────────────
pair_counts = (df_err.groupby(["true_name", "pred_name"]).size()
               .reset_index(name="count").sort_values("count", ascending=False))
pair_counts["true_group"] = pair_counts["true_name"].map(lambda n: get_group(n))
pair_counts["pred_group"] = pair_counts["pred_name"].map(lambda n: get_group(n))
pair_counts["same_group"] = pair_counts["true_group"] == pair_counts["pred_group"]
top20 = pair_counts.head(20)
top20.to_csv(os.path.join(OUT, "top20_confusion_pairs.csv"), index=False)
print("\n=== Top-20 pares mais confundidos ===")
for _, r in top20.iterrows():
    flag = "✓ MESMO GRUPO" if r["same_group"] else "   "
    print(f"  {r['count']:>3}× | {r['true_name']:<42} → {r['pred_name']:<42} {flag}")

# ── 5) matriz de confusão reordenada por grupo ───────────────────────────────
# ordena classes para que mesmo grupo fique adjacente
order = sorted(class_ids, key=lambda c: (group_of[c], id2name[c]))
ord_pos = {c: i for i, c in enumerate(order)}

y_true = df_pred["y_true"].values
y_pred = df_pred["y_pred"].values
cm = np.zeros((K, K), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[ord_pos[t], ord_pos[p]] += 1
cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

# linhas separando blocos de grupo
boundaries = []
last_g = None
for i, c in enumerate(order):
    g = group_of[c]
    if g != last_g and i > 0:
        boundaries.append(i)
    last_g = g

fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(cm_norm, cmap="Blues", square=True, vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "recall por classe", "shrink": 0.55},
            xticklabels=False, yticklabels=False)
for b in boundaries:
    ax.axhline(b, color="orange", linewidth=0.35, alpha=0.6)
    ax.axvline(b, color="orange", linewidth=0.35, alpha=0.6)
ax.set_title("Matriz de confusão reordenada por GRUPO (sufixo do nome).\n"
             "Linhas/colunas laranja marcam fronteiras entre grupos — "
             "blocos pequenos junto à diagonal = confusões intra-grupo.\n"
             f"Erros intra-grupo: {p_intra_obs*100:.1f}% (esperado aleatório: {p_intra_exp*100:.1f}%, "
             f"lift = {lift:.1f}×)",
             fontweight="bold", fontsize=11, pad=12)
ax.set_xlabel("Predito  (classes agrupadas por sufixo)")
ax.set_ylabel("Verdadeiro  (classes agrupadas por sufixo)")
plt.tight_layout()
fp = os.path.join(OUT, "confusion_matrix_by_group.png")
plt.savefig(fp, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSalvo: {fp}")

# ── 6) zoom em grupos grandes (>=5 espécies) ─────────────────────────────────
big_groups = [g for g, n in group_sizes.most_common() if n >= 5]
n_panels = len(big_groups)
cols = 3
rows = (n_panels + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5))
axes = np.array(axes).reshape(rows, cols)
for idx, g in enumerate(big_groups):
    cids = sorted(group_to_classes[g])
    labels = [id2name[c].split(".")[-1].replace("_", " ") for c in cids]
    sub_pos = {c: i for i, c in enumerate(cids)}
    sub_cm = np.zeros((len(cids), len(cids)), dtype=int)
    mask = df_pred["y_true"].isin(cids)
    for _, row in df_pred[mask].iterrows():
        if row["y_pred"] in sub_pos:
            sub_cm[sub_pos[row["y_true"]], sub_pos[row["y_pred"]]] += 1
    # normaliza por linha mas mantém valores absolutos como anotação
    sub_norm = sub_cm.astype(float) / np.clip(sub_cm.sum(axis=1, keepdims=True), 1, None)
    r, c = divmod(idx, cols)
    ax = axes[r, c]
    sns.heatmap(sub_norm, annot=sub_cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar=False, square=True, linewidths=0.4, linecolor="white",
                annot_kws={"size": 8})
    diag = sub_cm.trace()
    total_in = sub_cm[:, :].sum()  # com base nas linhas (verdadeiros do grupo)
    rows_total = sub_cm.sum()
    ax.set_title(f"{g}  ({len(cids)} espécies)\n"
                 f"intra-grupo: {(sub_cm.sum()-sub_cm.trace())} erros entre si",
                 fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
for j in range(n_panels, rows * cols):
    r, c = divmod(j, cols)
    axes[r, c].axis("off")

plt.suptitle("Confusão dentro dos grupos com 5+ espécies (intensidade = recall; número = contagem)",
             fontweight="bold", fontsize=12, y=1.0)
plt.tight_layout()
fp = os.path.join(OUT, "confusion_within_big_groups.png")
plt.savefig(fp, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {fp}")

# ── 7) accuracy por tamanho de grupo ─────────────────────────────────────────
df_pred["group_size"] = df_pred["true_group"].map(group_sizes)
acc_by_size = (df_pred.groupby("group_size")["correct"]
               .agg(["mean", "count"]).reset_index()
               .rename(columns={"mean": "accuracy", "count": "n_images"}))
print("\n=== Accuracy vs nº de espécies no grupo ===")
print(acc_by_size.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(acc_by_size["group_size"], acc_by_size["accuracy"] * 100,
       color="#1565C0", edgecolor="white")
for x, y, n in zip(acc_by_size["group_size"], acc_by_size["accuracy"],
                   acc_by_size["n_images"]):
    ax.text(x, y*100 + 0.6, f"{y*100:.1f}%\n(n={n})",
            ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Nº de espécies no grupo (sufixo do nome)")
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Accuracy cai quando o grupo tem mais espécies parecidas",
             fontweight="bold")
ax.set_ylim(80, 102)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fp = os.path.join(OUT, "accuracy_vs_group_size.png")
plt.savefig(fp, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {fp}")

# ── 8) resumo final txt ──────────────────────────────────────────────────────
summary = f"""================= Confusão x similaridade =================
Critério de similaridade: classes que compartilham o último token do
nome (ex.: Albatross, Sparrow, Warbler, Wren) — proxy de gênero/família.

Total grupos                    : {len(group_sizes)}
Grupos com ≥2 espécies          : {len(multi_groups)}
Erros totais                    : {n_err} / {n_total}  ({n_err/n_total*100:.2f}%)
Erros intra-grupo               : {n_intra}  ({p_intra_obs*100:.2f}% dos erros)
Erros inter-grupo               : {n_inter}  ({n_inter/n_err*100:.2f}% dos erros)
Esperado por chance (aleatório) : {p_intra_exp*100:.2f}%
Lift (observado / aleatório)    : {lift:.1f}×

Interpretação:
  Se os erros fossem aleatórios entre as 199 classes restantes,
  apenas ~{p_intra_exp*100:.1f}% deles cairiam em uma espécie do mesmo grupo.
  Na prática {p_intra_obs*100:.1f}% caem ali — confirma que o modelo confunde
  ESPÉCIES PARECIDAS muito mais do que classes não relacionadas.
============================================================
"""
print("\n" + summary)
with open(os.path.join(OUT, "similarity_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write(summary)

print("\nArquivos gerados:")
for f in ["confusion_matrix_by_group.png",
          "confusion_within_big_groups.png",
          "accuracy_vs_group_size.png",
          "top20_confusion_pairs.csv",
          "similarity_summary.txt"]:
    print(f"  eval_test_output/{f}")
