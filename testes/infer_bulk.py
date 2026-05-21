"""
Inferência em lote com GLSim (CUB-200-2011) sobre pasta de imagens.
Saídas:
  - resultados.csv  : predição top-1 e top-5 por imagem
  - resumo.png      : distribuição das predições + amostras por classe
"""

import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH      = "../results_train/cub_vit_b16_16_0/vit_b16_best.pth"
CLASSNAMES_CSV = "../data/cub/classid_classname.csv"
IMAGES_PATH    = "todas_imagens"
OUT_CSV        = "resultados.csv"
OUT_PNG        = "resumo.png"
TOP_K          = 5
BATCH_SIZE     = 16
IMAGE_SIZE     = 224
RESIZE_SIZE    = 300

# ── Transform ─────────────────────────────────────────────────────────────────
def build_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(device):
    from types import SimpleNamespace
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    args = cfg if hasattr(cfg, "__dict__") else SimpleNamespace(**cfg)
    args.ckpt_path   = CKPT_PATH
    args.device      = device
    args.pretrained  = False
    args.distributed = False
    args.world_size  = 1
    args.rank = args.local_rank = 0
    from glsim.model_utils.build_model import build_model
    model = build_model(args)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def run_batch(model, batch_tensor, device):
    batch_tensor = batch_tensor.to(device)
    with torch.no_grad():
        out = model(batch_tensor, ret_dist=True)
    logits = out[0] if isinstance(out, tuple) else out
    probs  = torch.softmax(logits, dim=-1)
    topk   = torch.topk(probs, k=TOP_K)
    return topk.indices.cpu().numpy(), topk.values.cpu().numpy()

# ── Plot summary ──────────────────────────────────────────────────────────────
def plot_summary(df, id2name):
    top1_names = df["pred_top1_name"].values
    counter    = Counter(top1_names)
    classes_sorted = sorted(counter, key=counter.get, reverse=True)
    counts = [counter[c] for c in classes_sorted]
    n_cls  = len(classes_sorted)

    fig = plt.figure(figsize=(16, max(8, n_cls * 0.35 + 4)))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.35)

    # Bar chart
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.barh(classes_sorted[::-1], counts[::-1], color="#2196F3", edgecolor="white")
    for bar, val in zip(bars, counts[::-1]):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=7)
    ax1.set_title("Predições GLSim (CUB-200-2011) — top-1 por classe", fontweight="bold")
    ax1.set_xlabel("Nº de imagens classificadas")
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_xlim(0, max(counts) * 1.15)

    # Sample images (up to 12 most predicted classes, 1 image each)
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("Amostras das classes mais preditas", fontweight="bold", fontsize=9)

    sample_cls = classes_sorted[:12]
    n_sample   = len(sample_cls)
    ncols, nrows = 3, (n_sample + 2) // 3
    inner_gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[1],
                                                hspace=0.6, wspace=0.3)
    for i, cls_name in enumerate(sample_cls):
        row_df = df[df["pred_top1_name"] == cls_name].iloc[0]
        img    = Image.open(row_df["filepath"]).convert("RGB").resize((80, 80))
        ax_i   = fig.add_subplot(inner_gs[i // ncols, i % ncols])
        ax_i.imshow(img)
        ax_i.axis("off")
        short = cls_name.replace("_", " ")
        if len(short) > 18:
            short = short[:16] + "…"
        ax_i.set_title(f"{short}\n({counter[cls_name]})", fontsize=6, pad=2)

    plt.suptitle(
        f"Total: {len(df)} imagens  |  {n_cls} classes preditas  |  "
        f"Modelo: GLSim ViT-B/16 CUB (Acc val 90.85%)",
        fontsize=9, y=1.01
    )
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {OUT_PNG}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    id2name = pd.read_csv(CLASSNAMES_CSV, index_col="class_id")["class_name"].to_dict()

    print("Carregando modelo…")
    model = load_model(device)

    transform = build_transform()
    files = sorted(glob.glob(os.path.join(IMAGES_PATH, "*.jpg")))
    print(f"Imagens encontradas: {len(files)}\n")

    records    = []
    batch_imgs = []
    batch_fps  = []

    def flush_batch():
        if not batch_fps:
            return
        tensor = torch.stack(batch_imgs)
        indices, probs = run_batch(model, tensor, device)
        for fp, idx_row, prob_row in zip(batch_fps, indices, probs):
            rec = {"filepath": fp, "filename": os.path.basename(fp)}
            for k in range(TOP_K):
                rec[f"pred_top{k+1}_id"]   = int(idx_row[k])
                rec[f"pred_top{k+1}_name"] = id2name.get(int(idx_row[k]), f"cls_{idx_row[k]}")
                rec[f"pred_top{k+1}_prob"] = round(float(prob_row[k]) * 100, 2)
            records.append(rec)
        batch_imgs.clear()
        batch_fps.clear()

    for fp in tqdm(files, desc="Inferindo"):
        try:
            img = Image.open(fp).convert("RGB")
            batch_imgs.append(transform(img))
            batch_fps.append(fp)
        except Exception as e:
            print(f"  [erro] {fp}: {e}")
            continue
        if len(batch_fps) == BATCH_SIZE:
            flush_batch()

    flush_batch()

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSalvo: {OUT_CSV}  ({len(df)} linhas)")

    print("\n=== Top-10 classes mais preditas ===")
    print(df["pred_top1_name"].value_counts().head(10).to_string())

    plot_summary(df, id2name)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
