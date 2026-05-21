# GLSim-CUB-ViT — Classificação Fina de Espécies de Pássaros

> **Seminário de Redes Neurais Artificiais — PPGEE / UFPA**
>
> Treinamento e avaliação do método **GLSim** com backbone **ViT-B/16 (224 px)**
> para classificação fina de **200 espécies de pássaros** (CUB-200-2011).

## Resultados obtidos

Test split do CUB-200-2011 — 5 794 imagens, 200 classes:

| Métrica            | Valor       |
|--------------------|-------------|
| Top-1              | **90,85 %** |
| Top-5              | **98,34 %** |
| F1 macro           | **0,9084**  |
| F1 weighted        | 0,9076      |
| Precision macro    | 0,9111      |
| Recall macro       | 0,9095      |

Resumo bruto em [`eval_test_output/metrics_summary.txt`](./eval_test_output/metrics_summary.txt).

### Análise dos erros (530 / 5 794)

Agrupando as 200 classes por gênero/família a partir do último token do nome
(Albatross, Sparrow, Warbler, Wren, …), obtemos 70 grupos — 37 deles com
duas ou mais espécies. A distribuição dos erros nesses grupos:

| Categoria de erro                        | Quantidade   |
|------------------------------------------|--------------|
| Erros intra-grupo (mesma família)        | 372 (70,2 %) |
| Erros inter-grupo (famílias diferentes)  | 158 (29,8 %) |
| Esperado por chance                      | ≈ 3,5 %      |
| **Lift observado / aleatório**           | **19,8 ×**   |

Ou seja: quando o modelo erra, ele confunde espécies próximas — não classes
não relacionadas. Detalhes em
[`eval_test_output/similarity_summary.txt`](./eval_test_output/similarity_summary.txt).

### Visualizações geradas

Todas em [`eval_test_output/`](./eval_test_output):

- `confusion_matrix.png` / `confusion_matrix_normalized.png` — 200 × 200
- `confusion_matrix_by_group.png` — matriz agregada por gênero/família
- `confusion_within_big_groups.png` — confusões dentro dos maiores grupos
- `top20_confusion_pairs.csv` — pares de espécies mais confundidas
- `worst20_classes.png` — 20 classes com pior F1
- `f1_distribution.png` — histograma de F1 por classe
- `accuracy_vs_group_size.png` — acurácia vs. nº de espécies do grupo
- `metrics_table.png` — sumário visual

Relatório completo: [`RELATORIO.md`](./RELATORIO.md) · [`relatorio.pdf`](./relatorio.pdf).

## Treinamento

- Backbone: ViT-B/16 inicializado com pesos ImageNet-21k
- Imagens: 224 × 224, augmentações `medaugs` (flip + RandAugment leve)
- Otimizador: SGD, lr 0,01, cosine decay com warmup, AMP fp16
- 50 épocas; checkpoint *best* selecionado pelo top-1 de validação

Curva de treino: [`training_curves.png`](./training_curves.png) · log completo em
[`training_log.txt`](./training_log.txt).

Para reproduzir do zero:

```
python tools/train.py \
    --cfg configs/cub_ft_is224_medaugs.yaml \
    --lr 0.01 \
    --model_name vit_b16 \
    --cfg_method configs/methods/glsim.yaml
```

Acompanhamento do treino em tempo real (terminal separado):

```
python dashboard.py
```

> O checkpoint `vit_b16_best.pth` (≈ 710 MB) não está versionado por exceder o
> limite do GitHub. É preciso re-treinar para obter o arquivo.

## Avaliação

Re-gerar todas as métricas e figuras do `eval_test_output/`:

```
python eval_test.py
```

Avaliação rápida via script nativo do GLSim:

```
python tools/train.py --ckpt_path results_train/cub_vit_b16_16_0/vit_b16_best.pth --test_only
```

Estudo da métrica GLS (distribuição de similaridade global-local nas
predições corretas/erradas):

```
python analyze_similarity.py
```

## Inferência em novas fotos

`infer_custom.py` classifica uma imagem ou pasta inteira e salva um grid com
as top-K predições em `results_inference/predictions.png`:

```
python infer_custom.py --images_path minhas_fotos/ --top_k 5
```

Teste com um dataset externo (Bird-525) — para checar generalização fora do
CUB — está em [`testes/`](./testes):

```
python testes/infer_bulk.py
python testes/metricas.py
```

Saídas: [`testes/resultados.csv`](./testes/resultados.csv) e
[`testes/metricas_output/`](./testes/metricas_output) (matriz de confusão,
comparação taxonômica, F1 por classe).

## Estrutura do repositório

```
GLSim-CUB-ViT/
├── glsim/                    # Pacote do modelo
├── tools/                    # train.py, inference.py (originais)
├── configs/                  # YAMLs (dataset, método, augs)
├── data/cub/                 # CSVs dos splits do CUB-200-2011
│
├── infer_custom.py           # Inferência amigável em pasta de imagens
├── eval_test.py              # Avaliação completa + figuras
├── analyze_similarity.py     # Estudo da métrica GLS nas predições
├── dashboard.py              # Monitor de treino no terminal
├── plot_training.py          # Geração das curvas de treino
├── make_metrics_table.py     # Tabela visual de métricas
├── make_group_pairs.py       # Análise de pares confundidos por grupo
├── make_glsim_flow.py        # Diagrama do fluxo do modelo
│
├── eval_test_output/         # Resultados e figuras desta execução
├── testes/                   # Validação out-of-distribution (Bird-525)
├── results_train/            # Checkpoints e crops do treino
├── training_log.txt          # Log textual completo
├── training_curves.png       # Curvas de loss/acc por época
│
├── RELATORIO.md              # Relatório técnico em Markdown
├── relatorio.pdf             # Relatório técnico (PDF)
└── relatorio.tex             # Fonte LaTeX do relatório
```

## Setup

Python 3.10+ com PyTorch + CUDA.

```
pip install -e .
```

## Créditos

Método GLSim (Rios et al., 2024) — [arXiv:2407.12891](https://arxiv.org/abs/2407.12891).
Dataset CUB-200-2011 — Wah et al., Caltech, 2011.
