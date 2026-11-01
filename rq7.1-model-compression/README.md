# RQ7.1 — Lightweight model compression

Release candidate on `feature/rq7.1-model-compression`. Compare both architectures
independently; there is no single best-backbone selection step.

## Fixed protocol

- Seeds **14, 27, 42**; one audited 38-class RGB PlantVillage split.
- Training batch size **64**; baseline **20 epochs** (3 head + 17 fine-tuning).
- Pruning recovery: **5 epochs** per ratio 0.1/0.2/0.3.
- CE/KD students: **20 epochs**, fixed alpha 0.5 and temperature 4.
- Separate frozen-feature linear probes: 3 learning-rate trials, 10 epochs each.
- Validation selects checkpoints/hyperparameters; test data do not tune them.

## Structure

`src/rq71/`: reusable models, loading, training, compression, and selection.
`configs/`: budgets. `splits/`: immutable manifests/class mapping/hash protocol.
`tests/`: regression tests. `results/`: curated small evidence and external run
locations. `figures/`: curated report plots.
## Numbered notebooks

1. `notebooks/01_dataset_setup_eda.ipynb`: environment/data preflight and EDA.
2. `notebooks/02_train_baselines.ipynb`: tuned linear probes and both baselines.
3. `notebooks/03_compression.ipynb`: independent compression tracks.
4. `notebooks/04_final_evaluation.ipynb`: selected-model prediction analysis,
   figures, and three-seed aggregation.

Set PROJECT, DATA and RESULTS explicitly. Save notebook 02's exact RUN path for
03/04. Execution is disabled by default. Stages use `run_stage.py`, preserve
config hashes, check prerequisites, and reject silent reruns. 

## Environment and checks

Read [ENVIRONMENT.md](ENVIRONMENT.md). The shared root uv.lock is not asserted to
reproduce the tested Kaggle runtime. Preserve its compatible GPU torch/torchvision
pair; use this folder's scoped requirements. The direct pins are not a complete
transitive lock.

Run from this directory:

```bash
python check_release.py
python -m unittest discover -s tests -p 'test_*.py'
python check_environment.py --require-cuda --output environment.json
python run_three_seeds.py --data-dir DATA --manifest-dir splits --output-root RESULTS
```

The final command only checks data; add `--execute` deliberately to launch.
The default runs all three seeds sequentially. For bounded sessions, use
`--seeds 14 27`, then `--seeds 42`, and aggregate the explicit three seed folders.
No automatic batch-size reduction or official CPU fallback.

## Data and outputs

Train/validation/test counts are **38,146 / 5,449 / 10,704**. Stage images on
runtime-local storage and load only manifest records. Extra physical duplicate
files do not re-enter the split. Never regenerate a random split for each seed.
Exact-hash checks do not prove leaf-group or near-duplicate independence.

Each timestamped run preserves config, logs, models, selection evidence,
predictions and figures. Only admitted final-table artifacts receive prediction
analysis. Temperature is selected on validation NLL.

INT8/pruning accuracy-gate rejection is recorded as **partial**, with no fabricated
eligible test row. Other exceptions are fatal. Summaries check configuration and
split-hash equality and report sample SD and contributing n. Hardware/environment
comparability still needs review. Cloud CPU latency is not target-device proof.

## Verification and release status

The three-seed staged synthetic Kaggle smoke completed, including prediction
analysis and aggregation. Separate real-image hash/decode and GPU startup tests
passed. The bounded real batch64 check logged successful updates/validation
forward passes for both architectures; final run/artifact audit remains required.


Full-data release-candidate run submitted on Kaggle (version `364659822`):
https://www.kaggle.com/code/makungundlovu/rq71-full-data-release-run?scriptVersionId=364659822

