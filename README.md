# COS 801 — Lightweight Crop Disease Diagnosis

This repository studies crop-disease classification for low-cost offline devices through three linked research questions:

- `rq7.1-model-compression`: accuracy, size, and CPU-latency trade-offs from quantisation, structured pruning, and knowledge distillation;
- `rq7.2-distribution-shift`: generalisation from laboratory-condition images to field images;
- `rq7.3-ood-abstention`: abstention on unfamiliar crops and non-plant inputs.

Each research question has its own reproducible workflow while sharing dependency management through `pyproject.toml` and `uv.lock`.

## Local setup

```bash
uv sync
```

Open the repository root in VS Code and select `.venv` as the Python interpreter. Large datasets, checkpoints, and exported models are not committed.

## Google Colab

RQ7.1 is Colab-first. Open `rq7.1-model-compression/notebooks/rq7_1_colab.ipynb`, mount Google Drive, set the project/data/output paths, and begin with smoke mode. Smoke-mode results validate execution only and must not be reported as final findings.

## Git workflow

Work on one feature branch per research question. Review local changes and experiment outputs before committing; do not commit credentials, datasets, model checkpoints, or private Drive paths.
