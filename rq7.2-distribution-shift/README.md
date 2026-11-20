# RQ7.2 — Laboratory-to-Field Generalisation

## Overview

RQ7.2 asks how much crop-disease classifier performance drops when a model
trained on laboratory-style PlantVillage images is evaluated on field-style
PlantDoc images. It compares EfficientNet-B0 and MobileNetV3-Small under:

- no augmentation;
- geometry augmentation;
- colour/lighting augmentation;
- noise/blur augmentation; and
- combined augmentation.

Each architecture is repeated with training seeds 14, 27, and 42.

## Data

### PlantVillage (source/laboratory data)

- Download: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Kaggle mirror: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- The notebooks expect the `raw/color/<class>/<image>` directory layout.
- `color_train.txt` and `color_test.txt` provide fixed laboratory
  train/test membership. They are stored in this folder and must be kept
  unchanged for the reported split.
- Stored copy:
  `rq7.2-distribution-shift/PlantVillage/color_train.txt` and
  `rq7.2-distribution-shift/PlantVillage/color_test.txt` are index copies;
  image storage is local or downloaded separately.

### PlantDoc (independent field data)

- Download: [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- Store images outside source control and set `FIELD_DATASET_ROOT` in the
  notebook. The field labels must be mapped to the PlantVillage taxonomy.
- PlantDoc licence, annotation, and redistribution terms must be verified from
  the source repository before sharing derived images or outputs.

### Optional structural metadata

`plantvillage_groups.csv` may define `index_entry` and `group_id` values for
grouped cross-validation. It is not present in the repository. Without verified
metadata, exact duplicate checks do not establish independence between
different images of the same plant or acquisition group.

## Notebooks

| Notebook | Architecture | Intended training seed |
|---|---|---:|
| [EfficientNet-B0 seed 14](RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed14.ipynb) | EfficientNet-B0 | 14 |
| [EfficientNet-B0 seed 27](RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed27.ipynb) | EfficientNet-B0 | 27 |
| [EfficientNet-B0 seed 42](RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed42.ipynb) | EfficientNet-B0 | 42 |
| [MobileNetV3-Small seed 14](RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed14.ipynb) | MobileNetV3-Small | 14 |
| [MobileNetV3-Small seed 27](RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed27.ipynb) | MobileNetV3-Small | 27 |
| [MobileNetV3-Small seed 42](RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed42.ipynb) | MobileNetV3-Small | 42 |

All notebooks contain the complete pipeline: environment setup, fixed-index
loading, duplicate checks, validation split generation, model training,
checkpointing, augmentation experiments, laboratory/field evaluation,
calibration, and report generation.

## Directory structure

```text
rq7.2-distribution-shift/
├── README.md
├── color_train.txt
├── color_test.txt
├── PlantVillage/
│   ├── color_train.txt
│   └── color_test.txt
├── reports/
│   └── fixed_split_metadata.json
├── *.ipynb                    # Six architecture/seed experiments
├── PlantDoc/                  # Download locally; not guaranteed in Git
└── runs/                      # Generated timestamped notebooks outputs
```

Before running, replace machine-specific `/mnt/d/M IT/` paths with local paths:

```text
DATASET_ROOT=<PlantVillage>/raw/color
FIELD_DATASET_ROOT=<PlantDoc>/<field-root>
TRAIN_INDEX_FILE=<this-folder>/color_train.txt
TEST_INDEX_FILE=<this-folder>/color_test.txt
```

## Setup

TensorFlow/Keras plus the scientific Python stack are required:

```bash
uv sync
uv run python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
PY
```

If the root environment cannot install TensorFlow on a platform, create a
Python environment supported by that TensorFlow release and install:

```bash
python -m pip install tensorflow numpy pandas pillow matplotlib scikit-learn jupyter
```

GPU compatibility depends on the TensorFlow build, OS, driver, CUDA, and cuDNN.
Verify the selected GPU in each notebook and adjust `/GPU:0` or CPU execution
for the host.

## Experiment and output details

The five conditions use independent initialisations/checkpoints. Training uses
an initial stage and fine-tuning stage with early stopping. Default notebook
settings include 224-pixel images, batch size 64, up to 100 epochs, 8 warm-up
epochs, and seed-specific checkpoint/report directories. Read the configuration
cells before execution and do not assume one notebook's settings are identical
to another.

Typical generated output:

```text
runs/<run-name>_<timestamp>/
├── models/        # Saved Keras checkpoints
├── reports/       # CSV/JSON metrics and diagnostics
└── plots/         # Accuracy, calibration, and confusion matrices
```

Metrics include accuracy, macro-F1, laboratory-to-field percentage-point
drop, reliability diagrams, and ECE. Summarise only matched architecture,
augmentation, split, and seed combinations.

## Reproducibility notes

- Set `SEED` to match the notebook filename before running.
- The EfficientNet-B0 seed-27 notebook currently contains `SEED = 14`;
  change it to 27 for a seed-27 run.
- Seed is for model initialisation/training; `SPLIT_SEED` defines the split
  procedure. The seed 27/42 MobileNet notebooks currently set split seed
  values to match their model seed, while the stored metadata records split
  seed 14. For a common split, use and document one fixed `SPLIT_SEED` across
  all runs. Do not silently pool results from different split protocols.
- Keep `color_train.txt` and `color_test.txt` unchanged and verify their
  SHA-256 hashes against `reports/fixed_split_metadata.json`.
- Images, checkpoints, `runs/`, `.keras`, and `.h5` files should remain
  Git-ignored.
- Keep the full package versions, GPU, TensorFlow build, and run timestamp with
  every saved result.

## Files generated

| Artifact | Location | Purpose |
|---|---|---|
| Keras checkpoints | `runs/.../models/` | Reuse or external evaluation |
| Metric reports | `runs/.../reports/` | Laboratory and field accuracy/F1 |
| Diagnostic plots | `runs/.../plots/` | Calibration and confusion analysis |
| Fixed split metadata | `reports/fixed_split_metadata.json` | Index hashes and split status |

## Artifacts and limitations

The field task inherits PlantDoc label mapping, class coverage, background,
illumination, and annotation uncertainty. Exact duplicate checks do not prove
leaf-, plant-, or session-level independence. The assumed 100 W training-power
estimate is illustrative, not a measured training cost. A high laboratory score
does not establish a reliable field classifier.
