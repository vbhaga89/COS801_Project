# RQ7_1_Model_Compression_Seed42

RQ7.1 compares quantisation, structured pruning and knowledge distillation independently for MobileNetV3-Small and EfficientNet-B0.

## Execution

- Training seed: **42**; batch size: **64**.
- Source bundle SHA256: `8bf0c2a502caf94302ff310bf7b25756d5239aff1e95d02924591503550fa3b2`.
- Configuration hash: `9dbf6b5f31b7`.
- Split: 38,147 training, 5,448 validation and 10,704 test images; 38 classes.
- Hardware: Tesla T4; inference measured on Intel Xeon CPU at 2.00 GHz.
- Duration: 31,992.2 seconds (8 h 53 min 12 s).
- Python 3.13.15; PyTorch 2.11.0+cu128.
- [Saved Kaggle execution and complete outputs](https://www.kaggle.com/code/makungundlovu/rq7-1-model-compression-seed42?scriptVersionId=365291590).

`final_comparison.csv` and `full_run_manifest.json` are original files downloaded from the saved Kaggle output. `test_results.csv` is a selected-column report table derived without changing numeric values. The configuration, split protocol, source provenance and leakage audit are retained unchanged. Historical paths inside those records identify the executed bundle and must not be rewritten to current repository paths. This directory is not a complete model-output archive.

The companion notebook `../../notebooks/RQ7_1_Model_Compression_Seed42.ipynb` is an output-free reproduction entrypoint using a pinned GitHub commit and the final manifests. Supply the image archive and private-repository read access through the GITHUB_TOKEN secret; training requires RUN_FULL=True. The historical execution bundle and the cleaned reproduction entrypoint retain separate provenance. Curated original figures are in `../../figures/RQ7_1_Model_Compression_Seed42/`.

## Outcomes

Both baselines, all pruning candidates, both CE/KD comparisons, EfficientNet INT8 and reporting completed. The scientific manifest is `partial` solely because MobileNet INT8 failed its validation accuracy gate: FP32 ONNX accuracy 0.995228 versus best INT8 accuracy 0.301946. Its test split was not evaluated and no invalid INT8 row is included.

Validation selected 10% nominal pruning for MobileNet and 20% for EfficientNet. EfficientNet INT8 achieved 98.3744% test accuracy with approximately 3.29-fold storage reduction, but higher CPU latency than FP32. Both KD students passed their matching baseline-relative accuracy budgets; both CE controls did not. A valid result or budget pass is not proof of speedup.

## Scope and limitations

These are seed-42 results, not multi-seed averages. Matched three-seed synthesis requires source, split and configuration compatibility checks. Hosted CPU latency is not a low-cost-device benchmark. Known author leaf-ID and exact-hash components are separated, but unresolved identities and unscreened near duplicates remain limitations. Historical test feedback preceded the validation-protocol revision; legacy-split results must not be pooled with these runs. Datasets, weights and full output archives remain outside Git.
