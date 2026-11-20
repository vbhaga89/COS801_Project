# COS 801 — Lightweight Crop-Disease Diagnosis

This repository contains three independent experiments from a wider study of
crop-disease diagnosis on resource-constrained, low-cost devices. The
experiments share a common PlantVillage-derived starting point, but use
different research questions, datasets, frameworks, and protocols.

## Research questions

### [RQ7.1 — Model compression](rq7.1-model-compression/README.md)

Compares MobileNetV3-Small and EfficientNet-B0 before and after INT8
quantisation, structured pruning, and knowledge distillation. It measures
classification accuracy, model size, and CPU inference latency on a fixed,
audited PlantVillage split. The experiment uses PyTorch and includes three
training seeds and reproducibility checks.

### [RQ7.2 — Laboratory-to-field generalisation](rq7.2-distribution-shift/README.md)

Measures the performance gap between laboratory PlantVillage images and field
images from PlantDoc, and compares five augmentation strategies on
EfficientNet-B0 and MobileNetV3-Small. It uses TensorFlow/Keras and repeats
the comparison across three training seeds.

### [RQ7.3 — OOD detection and abstention](rq7.3-ood-abstention/README.md)

Tests whether MobileNetV3-Small and EfficientNet-B0 can reject grape-crop,
corn-crop, and non-plant inputs rather than making an overconfident closed-set
prediction. It evaluates MSP, entropy, margin, energy, calibration, and
risk-coverage across three seeds using PyTorch.

## Repository layout

```text
COS801_Project/
├── pyproject.toml                 # Shared package configuration
├── uv.lock                        # Resolved dependency lock
├── README.md                      # Project overview
├── rq7.1-model-compression/       # RQ7.1 compression experiment
├── rq7.2-distribution-shift/      # RQ7.2 generalisation experiment
└── rq7.3-ood-abstention/          # RQ7.3 abstention experiment
```

## Environment setup

The project is configured for Python 3.11 through 3.13. Python 3.13 is used
by the recorded RQ7.1 Kaggle smoke environment; RQ7.2 and RQ7.3 should be run
with a compatible local or notebook kernel. PyTorch/torchvision pairs and
TensorFlow GPU builds must remain mutually compatible. Do not mix arbitrary
driver, CUDA, and package versions.

Install `uv`, then create the shared environment from the repository root:

```bash
uv sync
```

Launch Jupyter:

```bash
uv run jupyter notebook
```

Alternatively, create a virtual environment with standard Python tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas pillow scipy matplotlib \
  scikit-learn torch torchvision torch-pruning onnx onnxruntime tensorflow
```

In VS Code, select the repository `.venv` interpreter. For GPU execution,
install the CUDA-compatible TensorFlow or PyTorch stack for the actual
hardware and verify it before starting a full run.

## Dependency policy

The direct dependencies are declared in `pyproject.toml`, and `uv.lock`
captures the resolved environment. The two principal framework stacks are:

| Area | Framework/dependencies |
|---|---|
| RQ7.1 | PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, ONNX, ONNX Runtime, torch-pruning |
| RQ7.2 | TensorFlow/Keras, NumPy, pandas, Pillow, matplotlib, scikit-learn |
| RQ7.3 | PyTorch, torchvision, NumPy, pandas, Pillow, matplotlib, scikit-learn |

Cloud RQ7.1 runs may use the Kaggle/PyTorch environment described in
[rq7.1-model-compression/ENVIRONMENT.md](rq7.1-model-compression/ENVIRONMENT.md)
rather than installing the complete root lock. Those runs should record the
actual kernel, package versions, hardware, source commit, and dataset
fingerprints in their output.

## General reproducibility

- Set the data and output paths before executing a notebook or script.
- For multi-seed experiments, keep the dataset split fixed and change only the
  training seed unless the protocol explicitly calls for a split comparison.
- Record the source commit, package versions, hardware, data archive, fixed
  index or manifest hashes, and random seeds with every run.
- Compare mean and sample standard deviation across seeds only after confirming
  that the seed runs used the same data, configuration, and eligible models.
- Treat smoke, partial, and probe runs as execution checks, not as final
  scientific results.

## Safety and intended use

These experiments study model behaviour; they are not a validated diagnostic,
pesticide, or crop-management system. Results must not be used for autonomous
treatment decisions or interpreted as evidence of field reliability. Verify
dataset licences and redistribution terms before sharing data or trained
weights.
