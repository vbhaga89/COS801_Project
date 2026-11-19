# Environment specification and evidence

Successful staged synthetic Kaggle smoke used Python 3.13.15, torch 2.11.0+cpu,
torchvision 0.26.0+cpu, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1,
pandas 2.3.3, matplotlib 3.10.0, ONNX 1.22.0, ONNX Runtime 1.31.1,
and torch-pruning 1.6.1. requirements.txt pins these direct analysis/compression
dependencies; it is not a full transitive lock.

The minimal successful T4 diagnostic used torch 2.11.0+cu128. Do not replace
Kaggle's GPU PyTorch build with a CPU wheel. Record the complete GPU environment
with check_environment.py before the official experiment.



