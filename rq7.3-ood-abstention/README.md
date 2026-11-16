# RQ 7.3 - Abstention on Out-of-Distribution Inputs

This directory contains the complete workflow for evaluating whether image
classification models can identify inputs that are outside their training
distribution and abstain from making unreliable predictions.

The experiment uses PlantVillage crop images for in-distribution (ID)
classification and evaluates distribution shift using:

- **Grape** images as a held-out crop-species OOD set.
- **Corn** images as a second held-out crop-species OOD set.
- **Non-plant** images sampled from selected Caltech-101 categories.

The ID task contains 30 classes. Grape and corn are excluded from all ID
training, validation, and test splits. The experiments are repeated with seeds
`14`, `27`, and `42`.

## Workflow

Run the notebooks in numerical order:

1. `01_dataset_setup.ipynb`
2. `02_train_baselines.ipynb`
3. `03_closed_set_ood.ipynb`
4. `04_posthoc_methods.ipynb`
5. `05_final_evaluation.ipynb`

The notebooks use paths relative to this directory. Open or run them with
`rq7.3-ood-abstention` as the working directory.

## Notebook descriptions

### 1. Dataset setup

[`01_dataset_setup.ipynb`](./01_dataset_setup.ipynb) prepares and validates the
data used by all later notebooks. It:

- Scans the PlantVillage class folders in `data/`.
- Builds a dataframe containing image paths, class names, and crop species.
- Reads the supplied `color_train.txt` and `color_test.txt` metadata.
- Preserves the official train/test assignment instead of randomly splitting
  individual images.
- Removes Grape and corn from the ID data.
- Creates a stratified 90/10 ID train/validation split.
- Downloads Caltech-101 when necessary and samples 100 images from each of the
  selected non-plant categories:
  `airplanes`, `Motorbikes`, `Faces`, `Leopards`, `watch`, and `chandelier`.
- Runs integrity checks for class coverage, OOD exclusion, and overlap between
  ID splits.

The generated manifests and label mappings are saved in `splits/`.

### 2. Train baseline models

[`02_train_baselines.ipynb`](./02_train_baselines.ipynb) trains the two baseline
classifiers on the ID training set:

- MobileNetV3-Small
- EfficientNet-B0

For each model and seed, it:

- Applies the dataset transforms and loads the ID manifests.
- Trains using the training split and monitors validation performance.
- Selects and saves the best checkpoint.
- Evaluates the model on the ID test set.
- Saves training history, metadata, summary metrics, and training curves.

### 3. Closed-set and OOD confidence evaluation

[`03_closed_set_ood.ipynb`](./03_closed_set_ood.ipynb) loads the trained
checkpoints and evaluates both models on:

- ID test images.
- Grape OOD images.
- Corn OOD images.
- Non-plant OOD images.

It recreates the model architectures, records logits and labels for every
dataset, and calculates the initial closed-set/OOD confidence results. The
logits are saved in `logits/` for reuse by notebooks 4 and 5, avoiding repeated
model inference.

### 4. Post-hoc uncertainty and abstention methods

[`04_posthoc_methods.ipynb`](./04_posthoc_methods.ipynb) evaluates four
post-hoc confidence scores from the saved logits:

- Maximum Softmax Probability (MSP)
- Entropy-based confidence
- Prediction margin
- Energy score

For each model and seed, it computes OOD detection metrics including AUROC,
ID-oriented AUPR, FPR at 95% TPR, and OOD rejection at 95% ID coverage.

The notebook also contains a reusable plotting function followed by separate
cells for seeds `14`, `27`, and `42`. Each seed produces:

- Combined OOD AUROC bar charts.
- Combined OOD FPR@95% TPR bar charts.
- Energy-score distributions for ID, Grape OOD, Corn OOD, and Non-plant OOD.

### 5. Final evaluation

[`05_final_evaluation.ipynb`](./05_final_evaluation.ipynb) performs the final
calibration and abstention analysis using the saved logits. It:

- Calculates expected calibration error and reliability information.
- Builds risk-coverage curves using ID classification errors and OOD inputs
  treated as errors when accepted.
- Calculates the area under each risk-coverage curve (AURC).

As with notebook 4, it uses a reusable plotting function followed by separate
cells for seeds `14`, `27`, and `42`. Each seed produces:

- Reliability diagrams for both models.
- Risk-coverage curves for crop OOD data.
- Risk-coverage curves for non-plant OOD data.
- The actual curves in the risk-coverage curve are MSP, entropy, margin, and energy scoring.

## Directory structure

```text
rq7.3-ood-abstention/
├── 01_dataset_setup.ipynb
├── 02_train_baselines.ipynb
├── 03_closed_set_ood.ipynb
├── 04_posthoc_methods.ipynb
├── 05_final_evaluation.ipynb
├── README.md
├── data/
│   ├── PlantVillage class folders
│   ├── color_train.txt
│   ├── color_test.txt
│   └── nonplant/                 # Downloaded Caltech-101 data
├── splits/                       # CSV manifests and split configuration
├── models/                       # Seed-specific model checkpoints
├── logits/                       # Saved model logits and labels
└── results/
    ├── figures/                  # PNG plots generated by notebooks 2, 4, and 5
    ├── *.csv                     # Metrics, histories, and summaries
    └── *.json                    # Training metadata and configurations
```

## Important generated files

### Split manifests

- `splits/train_id.csv`
- `splits/val_id.csv`
- `splits/test_id.csv`
- `splits/ood_grape.csv`
- `splits/ood_corn.csv`
- `splits/ood_nonplant.csv`
- `splits/class_to_idx.json`
- `splits/split_config.json`

### Model and inference artifacts

Model checkpoints are stored as seed-specific `.pt` files in `models/`.
Inference outputs are stored as seed-specific `.pt` files in `logits/`.

### Results and figures

The `results/` directory contains baseline, closed-set, post-hoc, calibration,
and AURC summaries. All plots are written to `results/figures/` and include the
model name and seed in their filenames, for example:

```text
MobileNetV3-Small_energy_score_distributions_seed42.png
EfficientNet-B0_reliability_diagram_seed27.png
MobileNetV3-Small_risk_coverage_Nonplant_seed14.png
```

## Reproducibility

The notebooks use fixed random seeds where sampling or training is involved.
The primary experiment seeds are `14`, `27`, and `42`. Re-running a notebook
may overwrite files with the same names in `splits/`, `models/`, `logits/`, or
`results/`.
