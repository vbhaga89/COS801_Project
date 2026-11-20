# RQ7.3 — OOD Detection and Abstention

## Overview

RQ7.3 evaluates whether MobileNetV3-Small and EfficientNet-B0 can detect
inputs outside their 30-class PlantVillage training distribution and abstain.
The experiment uses seeds 14, 27, and 42 and compares:

- crop OOD: grape images and corn images excluded from ID training;
- non-plant OOD: selected Caltech-101 categories;
- closed-set ID: the remaining 30 PlantVillage classes;
- post-hoc scores: Maximum Softmax Probability, entropy, prediction margin,
  and energy.

The workflow includes OOD AUROC, FPR@95% TPR, calibration/ECE, and
risk-coverage/AURC.

## Data

### In-distribution PlantVillage data

- Download: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Kaggle mirror: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- The repository copy is under `data/` and is organised in class folders.
- `data/color_train.txt` and `data/color_test.txt` preserve the official
  membership. Grape and corn species are removed from all ID splits.

### Crop OOD data

Grape and corn images remain in the PlantVillage download but are excluded
from ID training, validation, and testing. The experiment writes them to
`ood_grape.csv` and `ood_corn.csv`.

### Non-plant OOD data

- Download/source: [Caltech-101](https://www.kaggle.com/datasets/imbikramsaha/caltech-101)
- The notebook uses 100 images from each of `airplanes`, `Motorbikes`,
  `Faces`, `Leopards`, `watch`, and `chandelier`.
- A prepared copy is stored locally under
  `data/nonplant/caltech101/101_ObjectCategories/`.
- Re-download or verify the source archive if that directory is moved or
  deleted.

Before reuse, verify PlantVillage and Caltech-101 licences and redistribution
terms; do not commit new datasets or archives unless redistribution is allowed.

## Notebooks

Run from this directory in numerical order:

| Notebook | Purpose |
|---|---|
| [01_dataset_setup.ipynb](01_dataset_setup.ipynb) | Scan PlantVillage, build ID/OOD manifests, prepare Caltech-101 non-plant samples, and validate exclusions |
| [02_train_baselines.ipynb](02_train_baselines.ipynb) | Train both 30-class baselines for seeds 14, 27, 42 and save checkpoints/history |
| [03_closed_set_ood.ipynb](03_closed_set_ood.ipynb) | Load models, evaluate ID/crop/non-plant sets, and cache logits for reuse |
| [04_posthoc_methods.ipynb](04_posthoc_methods.ipynb) | Compute MSP, entropy, margin, and energy OOD metrics and distributions |
| [05_final_evaluation.ipynb](05_final_evaluation.ipynb) | Generate calibration and crop/non-plant risk-coverage/AURC results |

Open or execute with `rq7.3-ood-abstention` as the working directory because
all data, model, logits, split, and result paths are relative.

## Directory structure

```text
rq7.3-ood-abstention/
├── README.md
├── 01_dataset_setup.ipynb
├── 02_train_baselines.ipynb
├── 03_closed_set_ood.ipynb
├── 04_posthoc_methods.ipynb
├── 05_final_evaluation.ipynb
├── data/
│   ├── PlantVillage class folders/
│   ├── color_train.txt
│   ├── color_test.txt
│   └── nonplant/caltech101/101_ObjectCategories/
├── splits/
│   ├── train_id.csv
│   ├── val_id.csv
│   ├── test_id.csv
│   ├── ood_grape.csv
│   ├── ood_corn.csv
│   ├── ood_nonplant.csv
│   ├── class_to_idx.json
│   └── split_config.json
├── models/
│   └── <architecture>_seed<14|27|42>.pt
├── logits/
│   └── <architecture>_<dataset>_seed<seed>.pt
└── results/
    ├── *.csv
    ├── *.json
    └── figures/
```

## Setup and run

From the repository root:

```bash
uv sync
cd rq7.3-ood-abstention
uv run jupyter notebook
```

The workflow requires PyTorch/torchvision, NumPy, pandas, Pillow, matplotlib,
and scikit-learn. Verify the PyTorch/torchvision pair and GPU/CUDA support in
`02_train_baselines.ipynb` before full training.

Run all five notebooks in order. Re-running them may overwrite split, model,
logit, CSV, JSON, and PNG files with the same seed and name.

## Files and artifacts generated

### Splits

The setup notebook creates ID train/validation/test manifests and three OOD
manifests plus the class mapping and split configuration.

### Models

`models/` contains one `.pt` checkpoint for each architecture and seed:

```text
efficientnet_b0_seed14.pt
efficientnet_b0_seed27.pt
efficientnet_b0_seed42.pt
mobilenet_v3_small_seed14.pt
mobilenet_v3_small_seed27.pt
mobilenet_v3_small_seed42.pt
```

### Logits

`logits/` contains ID, grape, corn, and non-plant logits/labels for each
architecture and seed. These files decouple post-hoc and final evaluation from
model retraining.

### Results

`results/` contains baseline histories, metadata, closed-set/OOD summaries,
post-hoc metrics, calibration metrics, AURC curves, and seed-combined CSV
files. `results/figures/` contains training curves, OOD AUROC/FPR plots,
energy distributions, reliability diagrams, and risk-coverage curves.

Example filenames:

```text
EfficientNet-B0_reliability_diagram_seed27.png
MobileNetV3-Small_combined_ood_auroc_seed42.png
MobileNetV3-Small_risk_coverage_Nonplant_seed14.png
```

## Reproducibility and interpretation

- Training seeds are 14, 27, and 42.
- Grape and corn are held out by crop species, not randomly sampled from ID.
- Reuse the saved logits for notebook 4 and 5 to avoid different inference
  behaviour from repeated model loading.
- Summaries should report mean and sample SD across matched seeds.
- `AURC` depends on treating OOD inputs as accepted errors for ID-vs-OOD
  risk-coverage curves; state this convention in reports.
- Checkpoint, figure, and metric file names include the architecture and seed
  where appropriate.
- Record PyTorch, torchvision, CUDA, and hardware versions for each run.

Limitations include ID/OOD domain similarity, the small selected non-plant
category set, calibration on the available ID test set, and no deployment
hardware validation. AUROC and abstention metrics demonstrate experimental
behaviour, not suitability for field use.
