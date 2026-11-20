# RQ7.1 — Model Compression

## Overview

RQ7.1 compares compressed crop-disease classifiers with matching FP32
baselines. It evaluates INT8 quantisation, structured pruning, and knowledge
distillation on MobileNetV3-Small and EfficientNet-B0. The reported protocol
uses training seeds 14, 27, and 42, a fixed hybrid-v2 split, accuracy-gated
pruning and quantisation candidates, and CPU inference measurements.

Main questions:

- How much model size can be removed while preserving test accuracy?
- What are the accuracy/size/latency trade-offs for quantisation and pruning?
- Does knowledge distillation improve the compressed student?

## Data

- **Dataset:** [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
  RGB leaf images, retained in the project as 54,299 images across 38 classes.
- **Source licence/terms:** the upstream dataset card declares
  [CC BY-SA 3.0](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/README_HF.md).
  Verify the licence and terms attached to the exact archive before
  redistribution.
- **Fixed manifest:** `splits/train.json`, `splits/val.json`, and
  `splits/test.json` materialise the reviewed train/validation/test assignments.
- **Split protocol:** `splits/split_protocol.json`, `splits/audit.json`,
  `splits/class_coverage.json`, `splits/class_to_idx.json`, and
  `splits/assignments.json` record provenance and validation checks.
- **Staging:** `stage_plantvillage.py` extracts and validates a source archive,
  then creates a local `raw/color/<class>/<image>` layout. It requires no
  source ZIP once staging is complete.

The upstream dataset is downloaded separately. The final experiment excludes
six exact duplicates and separates known author leaf-ID and exact-hash
components. Structural independence is not fully established; complete leaf,
plant, and acquisition-site metadata are unavailable.

Store the archive outside the repository or in a Git-ignored location. Example:

```text
<external-data>/rq71_plantvillage.tar
<external-data>/rq71_plantvillage/
<experiment-output>/
```

## Notebooks

| Notebook | Purpose |
|---|---|
| [01_dataset_setup_eda.ipynb](notebooks/01_dataset_setup_eda.ipynb) | Dataset staging, EDA, and manifest checks |
| [02_train_baselines.ipynb](notebooks/02_train_baselines.ipynb) | Baseline training and checkpoint selection |
| [03_compression.ipynb](notebooks/03_compression.ipynb) | Pruning, distillation, INT8 quantisation, and compression analysis |
| [04_final_evaluation.ipynb](notebooks/04_final_evaluation.ipynb) | Final test, latency, size, calibration, and risk evaluation |
| [Seed 14](notebooks/RQ7_1_Model_Compression_Seed14.ipynb) | Cloud reproduction entry point for seed 14 |
| [Seed 27](notebooks/RQ7_1_Model_Compression_Seed27.ipynb) | Cloud reproduction entry point for seed 27 |
| [Seed 42](notebooks/RQ7_1_Model_Compression_Seed42.ipynb) | Cloud reproduction entry point for seed 42 |

The modular notebooks are stage references. The seed notebooks are the
reported cloud entry points and pin a source commit. Historical saved Kaggle
notebooks must be interpreted as execution evidence for their original source
bundle, not as an unqualified repetition of the final GitHub protocol.

## Directory structure

```text
rq7.1-model-compression/
├── README.md
├── ENVIRONMENT.md
├── requirements.txt
├── configs/                   # Smoke and full experiment configuration
├── notebooks/                 # Stage references and cloud seed entry points
├── splits/                    # Reviewed fixed split and protocol
├── src/rq71/                  # Data, model, engine, and compression modules
├── tests/                     # Protocol and implementation regression tests
├── artifacts/                 # Preserved experiment artefacts
├── figures/                   # Seed figure directories
├── results/                   # JSON/CSV evidence and three-seed summaries
├── stage_plantvillage.py      # Validated local data staging
├── run_experiment.py          # Single-run experiment
├── run_three_seeds.py         # Seeded run launcher
└── analysis/export scripts
```

## Setup and run

From the repository root:

```bash
uv sync
cd rq7.1-model-compression
python check_environment.py
python check_release.py
```

Staging and local preflight:

```bash
python -u stage_plantvillage.py \
  --source /path/to/rq71_plantvillage.tar \
  --output /path/to/local_plantvillage \
  --manifest-dir splits

python -u run_three_seeds.py \
  --config configs/full.json \
  --data-dir /path/to/local_plantvillage \
  --manifest-dir splits \
  --output-root /path/to/results \
  --split-version hybrid-v2 \
  --seeds 14 27 42
```

The launcher performs preflight by default. Add `--execute` to start training.
Run a complete result for each seed rather than using smoke data as final
evidence. For cloud runs, inspect `RUN_FULL`, attach read-only repository access
through a secret if needed, and never write tokens into notebooks or outputs.
Use the compatible torch/torchvision pair documented in `ENVIRONMENT.md`.

## Fixed protocol

| Setting | Value |
|---|---|
| Seeds | 14, 27, 42 |
| Split seed | 16153790 |
| Training batch | 64 |
| Training epochs | 3 head-training + 17 fine-tuning |
| Linear probes | 3 learning rates, 10 epochs each |
| Pruning candidates | 10%, 20%, 30%; 5 recovery epochs each |
| CE/KD students | 20 epochs; KD alpha 0.5, temperature 4 |

Validation selects checkpoints, pruning ratios, and eligible INT8 candidates.
Test data is not used for those selections. A recorded
`ineligible_accuracy_gate` result is a negative scientific outcome, not an
eligible test score.

## Files and generated artifacts

- `results/<seed>/` contains run JSON, configuration, provenance, metrics,
  manifests, and admitted-artifact evaluation records.
- `results/three_seed_summary/` contains consolidated supported tables and a
  standard-library reproducibility script.
- `figures/<seed>/` documents external generated plots.
- `artifacts/` preserves selected run evidence when present.
- Local runs may generate checkpoints, histories, logits, ONNX files,
  quantised models, and generated CSV/PNG files. These are large and typically
  Git-ignored.

Three-seed summary reproduction:

```bash
python results/three_seed_summary/reproduce_summary.py \
  --results-root results --output reproduced_summary
python -m unittest discover -s results/three_seed_summary -p 'test_*.py'
```

## Validation and limitations

```bash
python test_hybrid_integration.py
python test_final_split_defaults.py
python -m unittest discover -s tests -p 'test_*.py'
python check_release.py
```

These checks verify implementation and protocol consistency; they do not certify
a complete cloud run. Report means and sample SD across the three training
seeds on the same fixed split. Smaller models are not necessarily faster, and
hosted CPU latency is a proxy rather than an edge-device benchmark. Selected
pruning ratios are validation-selected and may differ by seed and architecture.

The saved pipeline should disclose data, source, environment, hardware, and
full-vs-smoke status alongside its results. The models are not validated for
field diagnosis or autonomous decisions.
