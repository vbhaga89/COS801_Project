# RQ7_1_Model_Compression_Seed27

RQ7.1 compares quantisation, structured pruning and knowledge distillation independently for MobileNetV3-Small and EfficientNet-B0.

## Execution

- Training seed: **27**; batch size: **64**.
- Source bundle SHA256: `8bf0c2a502caf94302ff310bf7b25756d5239aff1e95d02924591503550fa3b2`.
- Configuration hash: `8a4b68bd5c6b`.
- Split: 38,147 training, 5,448 validation and 10,704 test images; 38 classes.
- Hardware: Tesla T4; inference measured on Intel Xeon CPU at 2.00 GHz.
- Duration: 30,372.9 seconds (8 h 26 min 13 s).
- Python 3.13.15; PyTorch 2.11.0+cu128.
- [Saved Kaggle execution and complete outputs](https://www.kaggle.com/code/makungundlovu/rq7-1-model-compression-seed27?scriptVersionId=365106458).

`final_comparison.csv` and `full_run_manifest.json` are original files downloaded from the saved Kaggle output. `test_results.csv` is a selected-column report table derived from the original CSV without changing numeric values. The saved Kaggle version remains the authoritative record for complete histories, diagnostics, models and the 39 listed figures. This directory is not a complete output archive.

The companion notebook in `../../notebooks/RQ7_1_Model_Compression_Seed27.ipynb` is an output-free reproduction entrypoint. It checks out a pinned GitHub commit, validates the final manifests, stages data locally and checks the complete dataset before training. Supply the image archive at the configured path and provide private-repository read access through the GITHUB_TOKEN secret. Training requires RUN_FULL=True. The saved Kaggle run retains its original bundle provenance; the current notebook is a separate cleaned-source reproduction entrypoint. The curated original report figures are in `../../figures/RQ7_1_Model_Compression_Seed27/`.

## Outcomes

Both baselines, all pruning candidates, both CE/KD comparisons, EfficientNet INT8 and report generation completed. The experiment manifest is `partial` because MobileNet INT8 was rejected by the validation accuracy gate: FP32 ONNX accuracy 0.993759 versus best INT8 accuracy 0.317915. The MobileNet INT8 test split was not evaluated, and that invalid result was excluded from the comparison table.

Twenty-percent pruning was selected independently for both backbones. EfficientNet INT8 reduced artifact size by approximately 3.29 times but increased measured CPU latency. Both pruned models were smaller but slower than their baselines. The MobileNet-teacher KD student passed the baseline-relative accuracy budget; the EfficientNet-teacher KD student did not. Passing a budget does not establish a latency speedup or a multi-seed conclusion.

## Scope and limitations

This directory reports seed 27 only. Multi-seed conclusions require matching seed-14 and seed-42 runs and their variability. Hosted CPU latency is not a low-cost-device benchmark; timings require a common target for strict comparisons. The protocol separates known author leaf-ID and exact-hash components, but unresolved leaf identities and unscreened near duplicates remain limitations. Historical test results were inspected before the validation-protocol revision. Legacy-split results must not be pooled with these results.

Datasets, weights and complete output archives remain outside Git. Scientific gate failures and limitations are retained as part of the result.
