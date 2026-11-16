# RQ7_1_Model_Compression_Seed27

RQ7.1 compares INT8 quantisation, structured pruning and knowledge distillation independently for MobileNetV3-Small and EfficientNet-B0.

## Execution

- Training seed: **27**; batch size: **64**.
- Configuration hash: `8a4b68bd5c6b`.
- Source bundle SHA256: `8bf0c2a502caf94302ff310bf7b25756d5239aff1e95d02924591503550fa3b2`.
- Fixed split: 38,147 training, 5,448 validation and 10,704 test images; 38 classes.
- Hardware: Tesla T4; hosted inference CPU: Intel Xeon at 2.00 GHz.
- Duration: 30,372.9 seconds (8 h 26 min 13 s).
- Python 3.13.15; PyTorch 2.11.0+cu128.
- [Saved execution and full outputs](https://www.kaggle.com/code/makungundlovu/rq7-1-model-compression-seed27?scriptVersionId=365106458).

## Files

`final_comparison.json` is a lossless table representation of the original comparison CSV, with its original SHA256 and serialization metadata. Original manifests, configurations, environments and source-provenance JSONs are unchanged. Generated CSVs remain external; the summary script regenerates report tables locally. `evaluation_metrics.json` is a compact extract of the nine original evaluation files, preserving validation temperature selection, calibration, per-class/per-crop metrics, ID selective risk and source SHA256 fingerprints. Prediction arrays and risk-curve point arrays remain in Kaggle. This is not a complete checkpoint or output backup.

## Outcomes

Both FP32 baselines, all pruning candidates, both CE/KD comparisons, EfficientNet INT8 and reporting completed. The scientific manifest is `partial` solely because MobileNet INT8 failed its validation accuracy gate. No MobileNet INT8 test score was produced or imputed. The selected pruning ratios and all valid metrics remain in the original table. Smaller files and budget eligibility do not establish latency speedup.

The [three-seed summary](../three_seed_summary/) verifies matching scientific settings, split hashes, source fingerprints and recorded environments before combining seeds 14, 27 and 42. It reports means and sample SD, calibration and supported worst-group results. [External figure references](../../figures/RQ7_1_Model_Compression_Seed27/) retain their single-seed meaning.

## Reproduction and limitations

The [cloud notebook](../../notebooks/RQ7_1_Model_Compression_Seed27.ipynb) is an output-free cleaned reproduction entrypoint, not the executed notebook. It checks out pinned source commit `bc89591`, validates the final manifests and stages the configured image archive. Private-repository read access uses the GITHUB_TOKEN secret; training requires RUN_FULL=True. The historical bundle remains unchanged; historical paths in saved records must not be rewritten.

Three seeds share one fixed split. Unknown leaf identities and unscreened near duplicates remain limitations. Historical test feedback preceded protocol revision; legacy-split results are not equivalent repetitions. Hosted CPU latency is not target-device validation. Dataset images, model weights and complete archives remain external.
