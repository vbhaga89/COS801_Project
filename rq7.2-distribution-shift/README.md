# RQ7.2 — Laboratory-to-Field Generalisation of Crop Disease Classifiers

## Overview

This repository contains Jupyter notebooks for **RQ7.2** of the lightweight crop disease diagnosis project:

> How large is the performance drop between laboratory-condition training images and field-condition test images, and which data augmentation strategies reduce it most?

The experiments compare **EfficientNetB0** and **MobileNetV3-Small**, both initialised with ImageNet-pretrained weights. Models are trained on PlantVillage images and evaluated on held-out PlantVillage images (laboratory conditions) and an independent PlantDoc dataset (field conditions). Each architecture is evaluated with a no-augmentation baseline and four augmentation strategies.

## Notebooks

| Architecture | Intended seed | Notebook |
|---|---:|---|
| EfficientNetB0 | 14 | `RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed14(1).ipynb` |
| EfficientNetB0 | 27 | `RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed27(3).ipynb` |
| EfficientNetB0 | 42 | `RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed42(3).ipynb` |
| MobileNetV3-Small | 14 | `RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed14(1).ipynb` |
| MobileNetV3-Small | 27 | `RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed27(3).ipynb` |
| MobileNetV3-Small | 42 | `RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed42(3).ipynb` |

**Important reproducibility check:** The uploaded EfficientNetB0 *Seed27* notebook currently sets `SEED = 14` internally. Change it to `SEED = 27` before executing that notebook. The filename alone does not control the random seed. Keep `SPLIT_SEED = 14` unchanged across runs to preserve the common data split.

## Datasets and split

- **PlantVillage:** Laboratory-condition leaf images organised by disease/class folders. The notebooks use the fixed index files `color_train.txt` and `color_test.txt`; the latter defines the held-out laboratory test set.
- **PlantDoc:** Independent field-condition images used for external evaluation. The field labels must be mapped consistently to the PlantVillage class taxonomy.
- The PlantVillage training index supplies training and validation images. The validation split is generated from this pool without using the fixed test index.
- The notebooks check for exact duplicate image content and can use optional `plantvillage_groups.csv` metadata to check structural groups (leaf, plant, or acquisition session). Without verified group metadata, structural independence is **not established**.

Dataset images and fixed TXT index files are **not included** in these six uploaded notebooks. Obtain and organise them separately, respecting their licences and usage conditions.

### Expected local setup

```text
project/
├── README.md
├── color_train.txt
├── color_test.txt
├── plantvillage_groups.csv          # optional
├── RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed14(1).ipynb
├── RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed27(3).ipynb
├── RQ7_2_EfficientNetB0__Fixed_TXT_Indexes_Seed42(3).ipynb
├── RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed14(1).ipynb
├── RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed27(3).ipynb
├── RQ7_2_MobileNetV3_Fixed_TXT_Indexes_Seed42(3).ipynb
└── runs/                             # generated during execution
```

The dataset directories may live elsewhere. Set `DATASET_ROOT` and `FIELD_DATASET_ROOT` in each notebook to their actual paths. The uploaded notebooks currently use machine-specific paths under `/mnt/d/M IT/`. The fixed index files must be accessible via `TRAIN_INDEX_FILE` and `TEST_INDEX_FILE`. Their entries are expected in the format `raw/color/<class>/<image>`.

## Installation

A Python environment with TensorFlow/Keras, NumPy, pandas, Matplotlib, scikit-learn, Jupyter, and an image-reading backend is required. For example:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install tensorflow numpy pandas matplotlib scikit-learn jupyter pillow
jupyter notebook
```

GPU support depends on the operating system, driver, CUDA/cuDNN compatibility, and TensorFlow build. Verify the installed TensorFlow version and detected GPU in the notebook before running expensive training. The notebook includes an explicit `/GPU:0` test, which may need adjustment on CPU-only systems.

## Experiment design

Each architecture is evaluated under five conditions:

| Condition | Description |
|---|---|
| Baseline | No data augmentation |
| Geometry | Spatial transformations, including flipping, rotation, translation, and zoom |
| Colour/Lighting | Brightness and contrast changes |
| Noise/Blur | Image noise and blurring |
| Combined | Geometry, colour/lighting, and noise/blur together |

The notebooks use separate model initialisations and checkpoints for augmentation conditions, rather than continuing training from a previous condition. Training includes an initial stage and a fine-tuning stage. Consult the notebook configuration cells for the **actual** batch size, learning rates, epoch limits, early-stopping settings, and augmentation parameters; do not assume they match settings from other versions of this project.

### Running the experiments

1. Install dependencies and download the required datasets.
2. Put `color_train.txt` and `color_test.txt` beside the notebooks, or update their paths.
3. Set `DATASET_ROOT` to the PlantVillage `raw/color` class-directory root and `FIELD_DATASET_ROOT` to the PlantDoc class-directory root.
4. Check the training `SEED` in each notebook, especially EfficientNetB0 Seed27. Keep `SPLIT_SEED` consistent.
5. Optionally supply verified `plantvillage_groups.csv` metadata, with columns `index_entry` and `group_id`.
6. Run each notebook from top to bottom in Jupyter, allowing training and evaluation to complete. Check `RUN_AUGMENTATION_EXPERIMENTS` if augmentation experiments are required.
7. Collect outputs from the generated timestamped `runs/` folders and compare results across seeds.

## Evaluation

The notebooks assess performance on laboratory and field data using metrics including:

- **Accuracy:** Percentage of correctly classified images.
- **Macro-F1:** Class-balanced summary of precision and recall.
- **Laboratory-to-field performance drop:** Difference between laboratory and field performance, expressed in percentage points.
- **Calibration:** Reliability diagrams and expected calibration error (ECE), where generated.

Summarise results across the three training seeds using mean and standard deviation. Compare augmentation conditions against the corresponding no-augmentation baseline, and report laboratory and field outcomes separately. Field evaluation is subject to the dataset's label mapping and the class taxonomy retained by the trained model.

## Outputs

Each execution creates a timestamped run directory with subfolders such as:

```text
runs/
└── <run_name_timestamp>/
    ├── models/      # saved model checkpoints
    ├── reports/     # evaluation tables and diagnostic reports
    └── plots/       # figures and reliability visualisations
```

Output names and availability depend on the notebook and which stages complete successfully.

## Reproducibility and limitations

- The fixed TXT files determine laboratory train/test membership; keep them identical for fair comparisons.
- A training seed and the fixed split seed serve different purposes.
- Exact-byte duplicate checks do not by themselves prove independence of different images of the same plant or leaf.
- Field performance can be affected by background complexity, illumination, blur, class imbalance, and differences between PlantVillage and PlantDoc labels.
- A high laboratory score does not guarantee reliable field predictions.
- GPU runtime and training-power estimates should not be treated as measured edge-device inference performance.
- Report actual notebook outputs rather than treating expected performance figures as guaranteed results.

## GitHub recommendations

Avoid committing datasets, trained model checkpoints, generated run outputs, credentials, or local machine paths. A `.gitignore` can include:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
runs/
*.keras
*.h5
PlantVillage/
PlantDoc/
```

Keep the fixed TXT indexes under version control **if redistribution is permitted**, as they are important to reproducibility.

## Project context

This work addresses **RQ7.2: laboratory-to-field generalisation and augmentation** within a broader project on lightweight crop disease diagnosis for low-cost devices. The notebooks and results should be interpreted as research experiments, not as a validated production diagnostic system.
