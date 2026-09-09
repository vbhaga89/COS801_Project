# RQ7.1 — Lightweight model compression

**Research question:** How much predictive performance is lost when an image classifier is compressed by post-training quantisation, structured channel pruning, and knowledge distillation to meet a mobile inference budget?

This folder provides a reproducible, Colab-first experiment pipeline. It deliberately keeps all work local until the team reviews the results and chooses what to commit.

## Experimental protocol

Use one leakage-audited train/validation/test split for every model variant. The final comparison must report:

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

Do not compare variants trained with different splits, augmentations, epoch budgets, or seeds without labelling that comparison as an ablation.

## Data layout

Final runs require explicit split directories:

```text
data/plantvillage/
├── train/<class-name>/*.jpg
├── val/<class-name>/*.jpg
└── test/<class-name>/*.jpg
```

The `eda` command checks class counts and exact duplicate hashes across splits. A run stops when cross-split duplicates are found. Near-duplicate or source-group checks should also be performed when source metadata is available.

## Colab workflow

Open `notebooks/rq7_1_colab.ipynb`, set `PROJECT_ROOT`, `DATA_DIR`, and `OUTPUT_DIR`, then run the cells in order. `SMOKE_MODE=True` uses a deliberately small budget to verify that the complete pipeline works; it is not a reportable experiment.

Equivalent command-line examples from this directory:

```bash
python run_experiment.py eda --config configs/smoke.json
python run_experiment.py train --config configs/smoke.json --model mobilenet_v3_small
python run_experiment.py quantize --config configs/smoke.json --checkpoint results/mobilenet_v3_small/baseline.pt
python run_experiment.py prune --config configs/smoke.json --checkpoint results/mobilenet_v3_small/baseline.pt
python run_experiment.py distill --config configs/smoke.json --checkpoint results/mobilenet_v3_small/baseline.pt
```

Replace `configs/smoke.json` with `configs/full.json` for reportable experiments.

## VS Code CPU smoke workflow

The repository includes `.vscode/tasks.json`. In VS Code, choose **Terminal → Run Task → RQ7.1: Run complete local smoke pipeline**. This creates a deterministic three-class synthetic image dataset, runs the leakage gate, trains both baselines, and exercises all three compression paths. The data and outputs remain under `artifacts/` and are ignored by Git. Synthetic smoke outputs prove only that the software pipeline executes; they are never valid evidence for RQ7.1.

## Output policy

- `results/metrics.csv` is the canonical comparison table and should be committed.
- `figures/` contains report-ready plots and should be committed.
- `artifacts/` contains checkpoints and ONNX files and remains ignored because these files are large.
- Every result row records the configuration, seed, model, compression method, software versions, and timestamp.

## Interpretation

RQ7.1 is answered by the Pareto trade-off, not by accuracy alone. The preferred model is the smallest and fastest variant that remains inside the agreed accuracy-loss and latency budgets. Negative results—such as pruning that reduces parameters but not latency—must be retained and explained.
