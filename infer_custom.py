"""
Inferência com o modelo GLSim treinado em CUB-200-2011.
Uso:
    python infer_custom.py --images_path <pasta_ou_imagem> [--top_k 5]

O script carrega o checkpoint treinado, classifica as imagens e
salva um grid com as predições em results_inference/predictions.png
"""

import argparse
import os
import glob
import sys

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH       = "results_train/cub_vit_b16_16_0/vit_b16_best.pth"
CLASSNAMES_CSV  = "data/cub/classid_classname.csv"
IMAGE_SIZE      = 224
RESIZE_SIZE     = 300          # int(224 * 1.34) rounded


# ── Transform (igual ao split 'test' do projeto) ──────────────────────────────
def build_test_transform():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Load model from checkpoint ─────────────────────────────────────────────────
def load_model(ckpt_path, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]

    # Build args namespace from saved config
    from types import SimpleNamespace
    args = SimpleNamespace(**cfg_dict)
    args.ckpt_path    = ckpt_path
    args.device       = device
    args.pretrained   = False   # weights already in ckpt
    args.distributed  = False
    args.world_size   = 1
    args.rank         = 0
    args.local_rank   = 0

    from glsim.model_utils.build_model import build_model
    model = build_model(args)
    model.eval()
    return model, args


# ── Collect images ─────────────────────────────────────────────────────────────
def collect_images(path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
    return sorted(files)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, img_tensor, device, top_k=5):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor, ret_dist=True)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    logits = logits.squeeze(0)
    probs  = torch.softmax(logits, dim=-1)
    topk   = torch.topk(probs, k=top_k)
    return topk.indices.cpu().tolist(), topk.values.cpu().tolist()


# ── Plot grid ─────────────────────────────────────────────────────────────────
def plot_results(results, out_path):
    n   = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5.2))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (fp, img_pil, indices, probs, classnames) in enumerate(results):
        r, c = divmod(idx, cols)
        ax   = axes[r][c]
        ax.imshow(img_pil)
        ax.axis("off")

        fname = os.path.basename(fp)
        title = f"{fname}\n"
        for rank, (ci, p, cn) in enumerate(zip(indices, probs, classnames)):
            marker = "▶ " if rank == 0 else f"  {rank+1}. "
            title += f"{marker}{cn.split('.')[-1]} ({p*100:.1f}%)\n"

        ax.set_title(title.strip(), fontsize=7, loc="left",
                     fontfamily="monospace", pad=3)

    # hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle("GLSim — CUB-200-2011 Predictions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, required=True,
                        help="Pasta com imagens ou caminho para uma única imagem")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Número de predições top-k (padrão: 5)")
    parser.add_argument("--output", type=str, default="results_inference/predictions.png",
                        help="Caminho de saída do grid de predições")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load class names
    df_cls = pd.read_csv(CLASSNAMES_CSV, index_col="class_id")
    id2name = df_cls["class_name"].to_dict()

    # Load model
    print(f"Loading checkpoint: {CKPT_PATH}")
    model, _ = load_model(CKPT_PATH, device)
    print("Model loaded.\n")

    transform = build_test_transform()
    files = collect_images(args.images_path)
    if not files:
        print(f"Nenhuma imagem encontrada em: {args.images_path}")
        return

    print(f"Imagens encontradas: {len(files)}\n")
    results = []

    for fp in files:
        print(f"  {fp}")
        img_pil = Image.open(fp).convert("RGB")
        img_t   = transform(img_pil)
        indices, probs = run_inference(model, img_t, device, top_k=args.top_k)
        classnames = [id2name.get(i, f"class_{i}") for i in indices]

        for rank, (ci, p, cn) in enumerate(zip(indices, probs, classnames)):
            marker = ">>>" if rank == 0 else f"  {rank+1}."
            print(f"    {marker} [{ci:3d}] {cn:<55} {p*100:5.2f}%")
        print()

        results.append((fp, img_pil, indices, probs, classnames))

    plot_results(results, args.output)


if __name__ == "__main__":
    main()
