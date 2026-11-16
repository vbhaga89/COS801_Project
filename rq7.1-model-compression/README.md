# RQ7.1 — Model compression

Comparative evaluation of INT8 quantisation, structured pruning and knowledge distillation for **MobileNetV3-Small** and **EfficientNet-B0**. Each compressed model is compared with its matching FP32 baseline;

## Reproduce the reported experiment

Use the sectioned cloud notebooks, one seed per run:

- [Seed 14](notebooks/RQ7_1_Model_Compression_Seed14.ipynb)
- [Seed 27](notebooks/RQ7_1_Model_Compression_Seed27.ipynb)
- [Seed 42](notebooks/RQ7_1_Model_Compression_Seed42.ipynb)

These notebooks support Kaggle and Colab and check out repository commit `c9849c4fed12dd15e05801c860ae37af078e60a4`. They use `stage_plantvillage.py` and the single final `splits/` directory. No source ZIP is required. Attach the audited image archive at the configured path, enable Internet and provide a read-access `GITHUB_TOKEN` secret for this private repository. Kaggle saved background runs require the secret to be attached in advance; Colab can use its secret store or a hidden interactive prompt. Never put tokens in code or outputs.

Review `RUN_FULL` before running all sections; training is disabled by default. Source-commit verification, fixed-manifest checks, GPU/environment checks, image staging and full-data preflight precede training. Every execution creates a timestamped output directory; backups do not implement checkpoint resume. The numbered modular notebooks remain stage references.

Historical saved Kaggle runs used `rq71_hybrid_v2_source.zip`, SHA256 `8bf0c2a502caf94302ff310bf7b25756d5239aff1e95d02924591503550fa3b2`. Their saved versions remain unchanged. The final GitHub notebooks use the cleaned pinned commit; record that provenance separately. Five notebook checks pass, but a complete run of these new checkout entrypoints has not yet been demonstrated.

To run from this directory after installing the scoped requirements:

```bash
python -u stage_plantvillage.py --source /path/to/rq71_plantvillage.tar --output /path/to/NEW_local_images --manifest-dir splits
python -u run_three_seeds.py --config configs/full.json --data-dir /path/to/NEW_local_images --manifest-dir splits --output-root /path/to/results --split-version hybrid-v2 --seeds 42
```

The second command performs preflight only. Add `--execute` to launch training. Run seeds 14, 27 and 42 separately using the same fixed manifests and settings. The launcher defaults to the final protocol and rejects legacy mode; the explicit `--split-version hybrid-v2` above records that choice. The saved Kaggle versions retain their original source bundle; the current cloud reproduction notebooks check out the cleaned pinned GitHub commit.

## Fixed protocol

| Setting | Value |
|---|---|
| Training seeds | 14, 27, 42 |
| Split seed | 16153790 |
| Dataset | 54,299 RGB PlantVillage images, 38 classes |
| Train / validation / test | 38,147 / 5,448 / 10,704 |
| Training batch size | 64 |
| Baseline training | 3 head-training + 17 fine-tuning epochs |
| Frozen-feature linear probes | 3 learning-rate trials, 10 epochs each |
| Pruning candidates | 0.10, 0.20, 0.30; 5 recovery epochs each |
| CE / KD students | 20 epochs each; KD alpha 0.5, temperature 4 |

The fixed hybrid-v2 assignment separates known author leaf-ID and exact-hash components. Six exact duplicate images are excluded from the original collection. Validation selects checkpoints, pruning ratios, and INT8 candidates. Test results are not used by those selection routines.

The repository and final seed notebooks both use `stage_plantvillage.py` and `splits/`. Historical cloud source archives retain their original internal paths as execution evidence. Image assignments, hash validation and training settings are unchanged.

## Results and figures

| Seed | Results | Original report figures | Status |
|---|---|---|---|
| 14 | [Tables and provenance](results/RQ7_1_Model_Compression_Seed14/) | [Figures](figures/RQ7_1_Model_Compression_Seed14/) | Execution completed; MobileNet INT8 rejected |
| 27 | [Tables and manifest](results/RQ7_1_Model_Compression_Seed27/) | [Figures](figures/RQ7_1_Model_Compression_Seed27/) | Execution completed; MobileNet INT8 rejected |
| 42 | [Tables and manifest](results/RQ7_1_Model_Compression_Seed42/) | [Figures](figures/RQ7_1_Model_Compression_Seed42/) | Execution completed; MobileNet INT8 rejected |

An `ineligible_accuracy_gate` outcome is a negative scientific result, not an eligible test score. The manifest can therefore be `partial` even when the notebook execution succeeds. Full models, histories, diagnostics and generated figures remain in the saved Kaggle output. Figure folders document the external outputs; generated image binaries and CSV tables are not tracked.

## Repository layout

- `notebooks/`: cloud reproduction notebooks and modular stage references.
- `src/rq71/` and root Python scripts: reusable training, compression, evaluation and plotting support.
- `configs/`, `splits/`, `tests/`: configuration, final reviewed manifests and regression checks.
- `results/`: JSON evidence, summary documentation and table-reproduction code.
- `figures/`: external-output references; generated plots, datasets and model weights are not tracked.

Read [ENVIRONMENT.md](ENVIRONMENT.md) for dependency guidance. Preserve the compatible runtime PyTorch/torchvision pair and record the actual hardware, package versions and source fingerprints. Useful static and regression checks are:

```bash
python check_release.py
python test_hybrid_integration.py
python test_final_split_defaults.py
python -m unittest discover -s tests -p 'test_*.py'
```

These checks do not certify a complete cloud run.

## Interpretation and limitations

The saved runs pass matched scientific-configuration, source-bundle/source-file, split-hash and recorded-environment checks. Report means and sample SD across the three training seeds on the same fixed split. Smaller artifacts are not necessarily faster; hosted CPU latency is a proxy, not a low-cost-device benchmark. Strict latency comparisons require a common target. Known leaf-ID separation does not resolve missing leaf identities or unscreened near duplicates. Historical test results were inspected before the protocol revision, which must be disclosed; legacy and hybrid-v2 results are not equivalent repetitions.

The three saved runs and 27 admitted-artifact evaluations have been consolidated in [the three-seed summary](results/three_seed_summary/). Report integration and independently stored full-output backups remain separate finalisation tasks. Dataset provenance and the upstream licence declaration are documented below. The models have not been validated for field diagnosis or deployment decisions.


## Reproduce the final tables

The [three-seed summary](results/three_seed_summary/) includes the comparison, calibration, ID selective-risk and supported worst-group tables. No missing MobileNet INT8 score is imputed. Selected MobileNet pruning ratios are 20%, 20%, 10% for seeds 14, 27, 42; EfficientNet selects 20% throughout. These are validation-selected procedures, not identical fixed-ratio test comparisons.

```bash
python results/three_seed_summary/reproduce_summary.py --results-root results --output reproduced_summary
python -m unittest discover -s results/three_seed_summary -p 'test_*.py'
```

These commands use lossless JSON representations of the original comparison tables, saved configurations/provenance and compact evaluation extracts. Generated CSVs are written locally and ignored by Git. They require only Python's standard library, with no images, model weights, cloud credentials or retraining. Five regression tests verify the summary and rejection of mismatched settings, splits, environments and duplicate evaluations.

Reviewed split inputs are stored as JSON tables in `splits/`. Staging and manifest loading reconstruct the runtime CSVs automatically and verify their original SHA256 fingerprints. This changes storage format only: assignments, image counts and the scientific protocol are unchanged. Generated CSVs and images remain available in the saved Kaggle versions and are excluded by the scoped `.gitignore`.

## Data provenance, licensing and intended use

The experiment uses RGB PlantVillage images in controlled acquisition conditions. The project-specific manifest identifies 54,299 retained images and excludes six known exact duplicate copies. This audited count is release-specific and should not be substituted for the counts of other PlantVillage distributions. The fixed manifest hashes and partial leaf-ID grouping limitations are retained in each seed's `split_protocol.json`.

The [authors' dataset card](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/README_HF.md) declares **CC BY-SA 3.0** (checked 4 November 2026). The [authors' loader](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/plant_village.py) also contains a licence-verification caveat. This records the upstream declaration, not independent clearance of every mirror or the exact local image archive. Preserve attribution and verify the archive's accompanying terms before redistributing images or weights. Dataset images and weights are not committed here; do not infer that the code's licence determines their terms.

No human-subject data collection is claimed. Annotation-demographic information and field representativeness are not established. The models are research artifacts, not validated tools for autonomous diagnosis, pesticide recommendations or crop-destruction decisions. Acquisition-site/device subgroup metadata are unavailable; reported groups are crop and disease labels. Energy and monetary cost were not measured. Saved pipeline wall times are approximately 10 h 5 min (seed 14), 8 h 26 min (27) and 8 h 53 min (42), including staging and evaluation, not isolated training times.
