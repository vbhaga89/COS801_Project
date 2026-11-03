# RQ7.1 — Model compression

Comparative evaluation of INT8 quantisation, structured pruning and knowledge distillation for **MobileNetV3-Small** and **EfficientNet-B0**. Each compressed model is compared with its matching FP32 baseline;

## Reproduce the reported experiment

Use the sectioned cloud notebooks, one seed per run:

- [Seed 14](notebooks/RQ7_1_Model_Compression_Seed14.ipynb)
- [Seed 27](notebooks/RQ7_1_Model_Compression_Seed27.ipynb)
- [Seed 42](notebooks/RQ7_1_Model_Compression_Seed42.ipynb)

These notebooks support Kaggle and Colab. Attach the audited image archive and pinned source bundle at the configured paths. Review `RUN_FULL` before running all sections. Source verification, GPU/environment checks, image staging, and full-data preflight precede training. Every execution creates a timestamped output directory; backups do not implement checkpoint resume.

The reported runs execute `rq71_hybrid_v2_source.zip`, SHA256:

```text
8bf0c2a502caf94302ff310bf7b25756d5239aff1e95d02924591503550fa3b2
```

The archive is supplied through the attached Kaggle source dataset or Google Drive. The notebooks verify all 46 source-file fingerprints. Matching execution scripts and fixed manifests are also available directly in this repository; see `hybrid_source_provenance.json` for their fingerprints. Documentation has been updated and the legacy split generator removed without changing training logic. The old `splits/` and numbered modular notebooks remain reference material, not the final hybrid-v2 entrypoint.

To run from this directory after installing the scoped requirements:

```bash
python -u stage_plantvillage.py --source /path/to/rq71_plantvillage.tar --output /path/to/NEW_local_images --manifest-dir splits_hybrid_v2
python -u run_three_seeds.py --config configs/full.json --data-dir /path/to/NEW_local_images --manifest-dir splits_hybrid_v2 --output-root /path/to/results --split-version hybrid-v2 --seeds 42
```

The second command performs preflight only. Add `--execute` to launch training. Run seeds 14, 27 and 42 separately using the same fixed manifests and settings. Always supply `--split-version hybrid-v2`; the retained legacy mode is not the reported protocol. The cloud notebooks continue to use the original pinned bundle so that their execution source remains identical to the saved runs.

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

The repository entrypoint `stage_plantvillage.py` is a filename-only rename of `stage_hybrid_v2.py` in the frozen cloud bundle. Its implementation is unchanged. Cloud notebooks intentionally use the old name inside that verified bundle; direct repository commands use the new name.

## Results and figures

| Seed | Results | Original report figures | Status |
|---|---|---|---|
| 14 | [Tables and provenance](results/RQ7_1_Model_Compression_Seed14/) | [Figures](figures/RQ7_1_Model_Compression_Seed14/) | Execution completed; MobileNet INT8 rejected |
| 27 | [Tables and manifest](results/RQ7_1_Model_Compression_Seed27/) | [Figures](figures/RQ7_1_Model_Compression_Seed27/) | Execution completed; MobileNet INT8 rejected |
| 42 | Pending final audit | Pending final audit | Running at the last recorded status |

An `ineligible_accuracy_gate` outcome is a negative scientific result, not an eligible test score. The manifest can therefore be `partial` even when the notebook execution succeeds. Full models, histories and diagnostics remain in the saved Kaggle output. Curated figure folders include training histories, compression comparisons, reliability diagrams and in-distribution risk–coverage curves. These are not RQ7.3 OOD evaluations.

## Repository layout

- `notebooks/`: cloud reproduction notebooks and modular stage references.
- `src/rq71/` and root Python scripts: reusable training, compression, evaluation and plotting support.
- `configs/`, `splits_hybrid_v2/`, `tests/`: configuration, final manifests and regression checks; `splits/` retains the earlier protocol.
- `results/`, `figures/`: small curated scientific outputs; datasets and model weights are external.

Read [ENVIRONMENT.md](ENVIRONMENT.md) for dependency guidance. Preserve the compatible runtime PyTorch/torchvision pair and record the actual hardware, package versions and source fingerprints. Useful static and regression checks are:

```bash
python check_release.py
python test_hybrid_integration.py
python -m unittest discover -s tests -p 'test_*.py'
```

These checks do not certify a complete cloud run.

## Interpretation and limitations

Report matched-seed metrics and variability only after all three runs are audited. Smaller artifacts are not necessarily faster; hosted CPU latency is a proxy, not a low-cost-device benchmark. Strict latency comparisons require a common target. Known leaf-ID separation does not resolve missing leaf identities or unscreened near duplicates. Historical test results were inspected before the protocol revision, which must be disclosed; legacy and hybrid-v2 results are not equivalent repetitions.

Final aggregation, seed-42 artifact verification and report integration remain pending. Dataset licensing and provenance must be documented in the submission. The models have not been validated for field diagnosis or deployment decisions.
