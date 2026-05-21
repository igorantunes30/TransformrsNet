import os
from pathlib import Path
from collections import Counter

PASTA = Path("D:/mestrado/RNA/SEMINARIO/GLSim-main/testes/todas_imagens")

MANTER = {
    "AMERICAN_GOLDFINCH",       # exata
    "ALBATROSS",                # exata (nome contido)
    "ALTAMIRA_YELLOWTHROAT",    # mesmo gênero (Geothlypis)
    "AFRICAN_EMERALD_CUCKOO",   # mesma família (Cuculidae)
    "AFRICAN_PYGMY_GOOSE",      # mesma família (Anatidae)
    "ALPINE_CHOUGH",            # mesma família (Corvidae)
    "ALBERTS_TOWHEE",           # mesma família (Passerellidae)
}

removidos = Counter()
mantidos  = Counter()

for img in sorted(PASTA.glob("*.jpg")):
    nome = img.name
    # extrai prefixo da espécie (antes do primeiro __)
    prefixo = nome.split("__")[0] if "__" in nome else nome

    if prefixo in MANTER:
        mantidos[prefixo] += 1
    else:
        img.unlink()
        removidos[prefixo] += 1

print(f"\n{'='*55}")
print(f"  Imagens MANTIDAS : {sum(mantidos.values())}")
print(f"  Imagens REMOVIDAS: {sum(removidos.values())}")
print(f"{'='*55}")

print("\nMantidas por espécie:")
for k, v in sorted(mantidos.items()):
    print(f"  {k:<35} {v:>4} imagens")

print("\nRemovidas por espécie:")
for k, v in sorted(removidos.items()):
    print(f"  {k:<35} {v:>4} imagens")
