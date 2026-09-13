# RQ7.1 — Lightweight model compression

**Research question:** How much predictive performance is lost when an image classifier is compressed by post-training quantisation, structured channel pruning, and knowledge distillation to meet a mobile inference budget?

This folder provides a reproducible, Colab-first experiment pipeline.

## Experimental protocol

Use one leakage-audited train/validation/test split for every model variant. For the reportable RQ7.1 experiment, the intended task is the original RGB/color PlantVillage classification problem with all 38 crop-disease classes. The final comparison must report:

- accuracy and macro F1 on the same held-out test set;
- change in accuracy and macro F1 relative to the matching FP32 teacher;
- parameter count and artifact size in MB;
- median and p95 latency at batch size 1 on a CPU-only runtime;
- compression ratio and whether the mobile budget was met.

The required comparisons are:

| ID | Model | Compression |
|---|---|---|
| B1 | EfficientNet-B0 | None, FP32 baseline |
| B2 | MobileNetV3-Small | None, FP32 baseline |
| Q1 | Best baseline | Static INT8 ONNX quantisation |
| P1 | Best baseline | Dependency-aware structured channel pruning + fine-tuning |
| D1 | Best baseline teacher | Knowledge distillation into MobileNetV3-Small 0.5× student |

Each baseline first trains only its linear classification head on frozen ImageNet features, satisfying the course baseline requirement, and then uses a documented fine-tuning stage. The default 25 MB, 100 ms, and 3 percentage-point loss limits in the configuration files are provisional mobile budgets; replace them only after the team records an agreed target device and budget.

## Data layout

Smoke runs use a tiny synthetic three-class dataset only to confirm that the pipeline executes. Final runs require the original, unaugmented, RGB/color PlantVillage data prepared into explicit split directories:

```text
COS801_data/rq71_plantvillage/
├── train/<class-name>/*.jpg
├── val/<class-name>/*.jpg
└── test/<class-name>/*.jpg
```
The `eda` command checks class counts and exact duplicate hashes across splits. A run stops when cross-split duplicates are found. Near-duplicate or source-group checks should also be performed when source metadata is available.

To prepare the recommended Hugging Face PlantVillage RGB/color release into the expected layout:

```bash
pip install datasets scikit-learn
python prepare_plantvillage.py --output /content/drive/MyDrive/COS801_data/rq71_plantvillage
```

The helper expects 38 classes. It uses the dataset's existing train/test split and attempts to create a validation split from the training portion with leaf/source grouping metadata when available. If no grouping column is exposed, it fails safely; rerun with `--allow-image-level-val-split` only if you intentionally accept and document that limitation.

## Colab workflow

Open `notebooks/rq7_1_colab.ipynb` from Google Drive and run the cells in order. `SMOKE_MODE=True` uses a deliberately small budget to verify that the complete pipeline works; it is not a reportable experiment. To run actual PlantVillage data, first prepare `COS801_data/rq71_plantvillage`, then set `SMOKE_MODE=False` in the notebook.

Equivalent command-line examples from this directory:

```bash
python run_experiment.py eda --config configs/smoke.json
python run_experiment.py train --config configs/smoke.json --model mobilenet_v3_small
python run_experiment.py quantize --config configs/smoke.json --checkpoint results/mobilenet_v3_small/baseline.pt
python run_experiment.py prune --config configs/smoke.json --checkpoint results/mobilenet_v3_small/baseline.pt
python run_experiment.py distill --config configs/smoke.json --checkpoint results/mobilenet_v3_small/baseline.pt
```

Replace `configs/smoke.json` with `configs/full.json` for reportable experiments.

## Output policy

- `results/metrics.csv` is the canonical comparison table and should be committed.
- `figures/` contains report-ready plots and should be committed.
- `artifacts/` contains checkpoints and ONNX files and remains ignored because these files are large.
- Every result row records the configuration, seed, model, compression method, software versions, and timestamp.

## Interpretation

RQ7.1 is answered by the Pareto trade-off, not by accuracy alone. The preferred model is the smallest and fastest variant that remains inside the agreed accuracy-loss and latency budgets. Negative results—such as pruning that reduces parameters but not latency—must be retained and explained.
