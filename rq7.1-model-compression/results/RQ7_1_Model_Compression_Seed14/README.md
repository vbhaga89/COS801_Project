# RQ7_1_Model_Compression_Seed14

RQ7.1 compares quantisation, structured pruning and knowledge distillation independently for MobileNetV3-Small and EfficientNet-B0.

## Execution

- Training seed: **14**; batch size: **64**.
- Source bundle SHA256: `8bf0c2a502caf94302ff310bf7b25756d5239aff1e95d02924591503550fa3b2`.
- Configuration hash: `f94fcc529664`.
- Split: 38,147 training, 5,448 validation and 10,704 test images; 38 classes.
- Hardware: Tesla T4; inference measured on Intel Xeon CPU at 2.00 GHz.
- Duration: 36,310.4 seconds (10 h 5 min 10 s).
- Python 3.13.15; PyTorch 2.11.0+cu128.
- [Saved Kaggle execution and complete outputs](https://www.kaggle.com/code/makungundlovu/rq7-1-model-compression-seed14?scriptVersionId=364832930).

`test_results.csv` contains selected columns extracted at displayed precision from the saved Kaggle `final_comparison.csv` preview. It is a derived report table, not a byte-identical copy of that original file. The saved Kaggle version is the authoritative record for full manifests, histories, diagnostics, models and the 39 listed figures. The complete output archive has not been verified locally.

The companion notebook is an output-free reproduction entrypoint, not a copy of the executed Kaggle notebook. It checks out a pinned GitHub commit, validates the final manifests, stages data locally and checks the complete dataset before training with an explicit training opt-in. Supply the image archive at the configured path and provide private-repository read access through the GITHUB_TOKEN secret. Training requires RUN_FULL=True. The saved Kaggle run retains its original bundle provenance; the current notebook is a separate cleaned-source reproduction entrypoint.

## Outcomes

Both baselines, all pruning candidates, both CE/KD comparisons, EfficientNet INT8 and report generation completed. The experiment manifest is `partial` because MobileNet INT8 was rejected by the validation accuracy gate: FP32 ONNX accuracy 0.994677 versus best INT8 accuracy 0.254589. No MobileNet INT8 test result was produced.

Twenty-percent pruning was selected independently for both backbones. EfficientNet INT8 reduced artifact size by approximately 3.29 times, but increased measured CPU latency. Both pruned models were smaller but slower than their baselines. KD improved on the corresponding CE controls by 0.44 and 0.95 percentage points with MobileNet and EfficientNet teachers, respectively; both KD students missed the baseline-relative accuracy budget. Budget eligibility does not establish a latency speedup.

## Scope and limitations

Hosted CPU latency is not a low-cost-device benchmark; cross-platform timings require a common target. The protocol separates known author leaf-ID and exact-hash components, but unresolved leaf identities and unscreened near duplicates remain limitations. Historical test results were inspected before the validation-protocol revision. Legacy-split results must not be pooled with these results.

Datasets, weights and output archives are retained outside Git. Scientific gate failures and limitations are part of the result.
