"""
Avalia o modelo GLSim treinado no split de test do CUB-200-2011.
Gera: confusion_matrix.png, classification_report.csv, metrics_summary.txt,
       per_class_bars.png, confusion_matrix.csv.

Uso:
    python eval_test.py [--batch_size 32] [--num_workers 0]
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, accuracy_score,
    top_k_accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "results_train", "cub_vit_b16_16_0", "vit_b16_best.pth")
CLASSNAMES_CSV = os.path.join(BASE_DIR, "data", "cub", "classid_classname.csv")
OUT_DIR = os.path.join(BASE_DIR, "eval_test_output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(ckpt_path, device):
    sys.path.insert(0, BASE_DIR)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        args = SimpleNamespace(**cfg)
    else:
        args = cfg  # already a Namespace / SimpleNamespace
    args.ckpt_path = ckpt_path
    args.device = device
    args.pretrained = False
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0

    from glsim.model_utils.build_model import build_model
    model = build_model(args)
    model.eval()
    return model, args


def build_test_loader(args, batch_size, num_workers):
    from glsim.data_utils.build_transform import build_transform
    from glsim.data_utils.datasets import DatasetImgTarget

    transform = build_transform(args=args, split="test")
    ds = DatasetImgTarget(args, split="test", transform=transform)
    print(f"Test dataset: N={len(ds)}, K={ds.num_classes}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, ds


@torch.no_grad()
def run_eval(model, loader, device):
    all_preds, all_labels, all_topk = [], [], []
    n_total = len(loader.dataset)
    t0 = time.time()
    seen = 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits, dim=-1)
        top5 = probs.topk(5, dim=-1).indices.cpu().numpy()
        preds = top5[:, 0]
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_topk.append(top5)
        seen += imgs.size(0)
        if (batch_idx + 1) % 10 == 0 or seen == n_total:
            elapsed = time.time() - t0
            ips = seen / max(elapsed, 1e-6)
            eta = (n_total - seen) / max(ips, 1e-6)
            print(f"  [{seen:>5d}/{n_total}]  {ips:5.1f} img/s  ETA {eta:5.1f}s")
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_topk, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args_cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    model, args = load_model(CKPT_PATH, device)
    print("Modelo carregado.\n")

    loader, ds = build_test_loader(args, args_cli.batch_size, args_cli.num_workers)
    print(f"Iniciando inferência (batch={args_cli.batch_size})...\n")
    y_pred, y_true, top5 = run_eval(model, loader, device)

    df_cls = pd.read_csv(CLASSNAMES_CSV)
    id2name = dict(zip(df_cls["class_id"], df_cls["class_name"]))
    class_ids = sorted(id2name.keys())
    class_names = [id2name[i] for i in class_ids]
    K = len(class_ids)

    acc1 = accuracy_score(y_true, y_pred)
    acc5 = top_k_accuracy_score(y_true, np.eye(K)[top5].sum(axis=1), k=1, labels=class_ids) if False else \
           float(np.mean([t in topk for t, topk in zip(y_true, top5)]))

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=class_ids)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=class_ids)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=class_ids)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=class_ids)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=class_ids)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=class_ids)

    summary = (
        f"================= Resultados (test split) =================\n"
        f"  N imagens         : {len(y_true)}\n"
        f"  N classes         : {K}\n"
        f"  Top-1 accuracy    : {acc1*100:.2f}%\n"
        f"  Top-5 accuracy    : {acc5*100:.2f}%\n"
        f"  F1  (macro)       : {f1_macro:.4f}\n"
        f"  F1  (weighted)    : {f1_weighted:.4f}\n"
        f"  Precision (macro) : {prec_macro:.4f}\n"
        f"  Precision (weight): {prec_weighted:.4f}\n"
        f"  Recall    (macro) : {rec_macro:.4f}\n"
        f"  Recall    (weight): {rec_weighted:.4f}\n"
        f"==========================================================\n"
    )
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary)

    report = classification_report(y_true, y_pred, labels=class_ids,
                                   target_names=class_names,
                                   zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).T
    df_report.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))
    print(f"Salvo: classification_report.csv ({len(df_report)} linhas)")

    df_pred = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [id2name[i] for i in y_true],
        "pred_name": [id2name[i] for i in y_pred],
        "correct": y_true == y_pred,
    })
    df_pred.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    print(f"Salvo: predictions.csv")

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    np.savetxt(os.path.join(OUT_DIR, "confusion_matrix.csv"), cm.astype(int),
               fmt="%d", delimiter=",")
    print(f"Salvo: confusion_matrix.csv ({cm.shape})")

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, cmap="Blues", square=True, ax=ax,
                cbar_kws={"label": "# imagens", "shrink": 0.6},
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Confusion Matrix — CUB-200-2011 (test split)\n"
                 f"N={len(y_true)} | K={K} | Top-1 acc = {acc1*100:.2f}%",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvo: confusion_matrix.png")

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm_norm, cmap="Blues", square=True, vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "recall por classe", "shrink": 0.6},
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Confusion Matrix Normalizada (recall por classe)\n"
                 f"Top-1 acc = {acc1*100:.2f}% | F1 macro = {f1_macro:.3f}",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_normalized.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvo: confusion_matrix_normalized.png")

    per_class = df_report.loc[class_names, ["precision", "recall", "f1-score", "support"]].copy()
    per_class["class_id"] = class_ids
    sorted_by_f1 = per_class.sort_values("f1-score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 14))
    y = np.arange(20)
    worst = sorted_by_f1.head(20)
    ax.barh(y - 0.25, worst["precision"], 0.25, label="Precision", color="#2196F3")
    ax.barh(y,        worst["recall"],    0.25, label="Recall",    color="#4CAF50")
    ax.barh(y + 0.25, worst["f1-score"],  0.25, label="F1",        color="#FF5722")
    ax.set_yticks(y)
    ax.set_yticklabels([n[:40] for n in worst.index], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_title("Top 20 classes com pior F1-score", fontweight="bold")
    ax.set_xlabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "worst20_classes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvo: worst20_classes.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    f1_sorted = per_class["f1-score"].sort_values(ascending=False).values
    ax.plot(np.arange(K), f1_sorted, color="#FF5722", linewidth=1.2)
    ax.fill_between(np.arange(K), f1_sorted, alpha=0.25, color="#FF5722")
    ax.axhline(f1_macro, color="gray", linestyle="--", linewidth=1, label=f"F1 macro = {f1_macro:.3f}")
    ax.set_title("F1-score por classe (ordenado)", fontweight="bold")
    ax.set_xlabel("Classe (rank por F1)")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvo: f1_distribution.png")

    print(f"\nArquivos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
