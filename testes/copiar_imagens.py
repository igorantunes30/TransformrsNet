import os, shutil
from pathlib import Path

src = Path("D:/mestrado/RNA/SEMINARIO/GLSim-main/testes/especie_passaros")
dst = Path("D:/mestrado/RNA/SEMINARIO/GLSim-main/testes/todas_imagens")
dst.mkdir(exist_ok=True)

count = 0
for img in src.rglob("*.jpg"):
    parts = img.relative_to(src).parts
    if len(parts) == 3:
        split, cls, fname = parts
        new_name = (cls + "__" + split + "__" + fname).replace(" ", "_")
    elif len(parts) == 2:
        new_name = "predict__" + img.name
    else:
        new_name = img.name
    shutil.copy2(img, dst / new_name)
    count += 1

print(f"Copiadas: {count} imagens -> {dst}")
