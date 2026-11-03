# RQ7.1 three-seed results

Seeds **14, 27, 42**. Each value is mean ± sample standard deviation across three matched training seeds on one fixed split. Accuracy is percent; macro F1 is 0–1; size is MiB; latency is hosted-CPU median in ms.

| Backbone / teacher track | Variant | Accuracy (%) | Macro F1 | Size (MiB) | Latency (ms) |
|---|---|---:|---:|---:|---:|
| efficientnet_b0 | FP32 | 99.418 ± 0.095 | 0.9919 ± 0.0013 | 15.464 ± 0.000 | 14.184 ± 0.715 |
| efficientnet_b0 | Pruning selected | 97.792 ± 0.066 | 0.9703 ± 0.0011 | 10.034 ± 0.000 | 25.642 ± 1.056 |
| efficientnet_b0 | INT8 selected | 98.350 ± 0.505 | 0.9800 ± 0.0054 | 4.707 ± 0.000 | 18.103 ± 0.429 |
| mobilenet_v3_small | FP32 | 99.215 ± 0.037 | 0.9896 ± 0.0009 | 5.948 ± 0.000 | 2.348 ± 0.140 |
| mobilenet_v3_small | Pruning selected | 97.042 ± 1.132 | 0.9605 ± 0.0150 | 4.190 ± 0.563 | 4.511 ± 1.524 |
| efficientnet_b0 | KD student | 96.123 ± 0.472 | 0.9473 ± 0.0047 | 1.637 ± 0.000 | 1.315 ± 0.044 |
| mobilenet_v3_small | KD student | 96.132 ± 0.353 | 0.9491 ± 0.0052 | 1.637 ± 0.000 | 1.285 ± 0.055 |
| efficientnet_b0 | CE student | 95.098 ± 0.240 | 0.9318 ± 0.0064 | 1.637 ± 0.000 | 1.350 ± 0.014 |
| mobilenet_v3_small | CE student | 94.859 ± 0.718 | 0.9327 ± 0.0112 | 1.637 ± 0.000 | 1.281 ± 0.049 |

MobileNet INT8 failed the validation gate for all three seeds. Its test score is unavailable, not zero. MobileNet pruning selected 20%, 20%, 10% for seeds 14, 27, 42; EfficientNet selected 20% in all three. The table compares the validation-selected procedures.

## Reproduction

From the RQ7.1 project directory:

```bash
python results/three_seed_summary/reproduce_summary.py --results-root results --output reproduced_summary
```

The script verifies source-file and bundle fingerprints, scientific configuration, fixed split hashes, recorded environments, row identities and validation-only temperature selection. It reads the original comparison files and compact extracts in the adjacent seed directories. No training, image access or cloud credentials are required.

## Calibration and worst groups

`calibration_summary.csv` reports ECE, NLL and multiclass Brier sum before and after validation-selected temperature scaling. AURC is reported only for the saved uncalibrated ID confidence ranking. `calibration_by_seed.csv` preserves all 54 model/mode records. `worst_groups_by_seed.csv` reports the worst supported class recall and crop accuracy for each of 27 evaluated artifacts, with labels and supports. These are label-derived groups, not sites or devices; the worst-group identity may differ across seeds. Do not treat a mean of per-seed minima as the performance of one fixed group.

Original evaluation files remain in each linked Kaggle output. Compact evaluation extracts preserve scalar metrics, per-class/per-crop records, validation temperature search and original evaluation SHA256; they omit bulky prediction arrays and risk-curve points. No legacy results or exploratory QAT scores are pooled.

## Limitations

Three seeds quantify training randomness on one fixed split, not sampling uncertainty. Historical test feedback preceded protocol revision. Missing leaf identities and unscreened near duplicates remain limitations. Hosted CPU timings share the recorded environment but not a verified identical physical machine or load. Smaller models are not necessarily faster. Energy and monetary cost were not measured; no field or target-device deployment validation is claimed. Full checkpoint archives remain external.

## Fixed-group worst-case summary

`subgroup_summary.csv` first averages each named class recall and crop accuracy across the same three seeds, then selects the lowest mean group for each model/teacher track. The SD belongs to that same selected group. It is not the average of different worst groups. All declared classes/crops have positive support in these records.

| Track | Variant | Worst class | Recall (%) | Worst crop | Accuracy (%) |
|---|---|---|---:|---|---:|
| efficientnet_b0 | FP32 | Corn__maize___Cercospora_leaf_spot_Gray_leaf_spot | 92.52 ± 6.13 | Raspberry | 96.50 ± 1.43 |
| efficientnet_b0 | Pruning selected | Tomato__Early_blight | 81.86 ± 2.46 | Raspberry | 92.59 ± 1.07 |
| efficientnet_b0 | INT8 selected | Tomato__Septoria_leaf_spot | 89.81 ± 10.17 | Potato | 95.92 ± 1.16 |
| mobilenet_v3_small | FP32 | Corn__maize___Cercospora_leaf_spot_Gray_leaf_spot | 95.33 ± 1.62 | Corn | 97.98 ± 0.22 |
| mobilenet_v3_small | Pruning selected | Tomato__Early_blight | 83.57 ± 7.23 | Pepper | 94.95 ± 2.34 |
| efficientnet_b0 | KD student | Corn__maize___Cercospora_leaf_spot_Gray_leaf_spot | 72.90 ± 3.24 | Raspberry | 90.53 ± 3.62 |
| mobilenet_v3_small | KD student | Tomato__Early_blight | 73.80 ± 5.37 | Pepper | 91.46 ± 2.19 |
| efficientnet_b0 | CE student | Strawberry__healthy | 76.25 ± 7.50 | Raspberry | 82.51 ± 10.98 |
| mobilenet_v3_small | CE student | Corn__maize___Cercospora_leaf_spot_Gray_leaf_spot | 75.08 ± 6.89 | Raspberry | 81.07 ± 15.52 |
