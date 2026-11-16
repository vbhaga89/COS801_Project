# RQ7.1 hybrid-v2 manifests

Fixed assignments used for the seed-14, seed-27, and seed-42 experiments.
| Partition | Images | Classes |
|---|---:|---:|
| Train | 38,147 | 38 |
| Validation | 5,448 | 38 |
| Test | 10,704 | 38 |

The split seed is 16153790. Training seeds are 14, 27 and 42. `test.csv` is byte-identical to the earlier test manifest. Known author leaf IDs and exact SHA256-connected components are kept together. Validation includes 4,075 images with resolved author leaf IDs and 1,373 without resolved IDs. We did not establish unknown-leaf independence or perceptual near-duplicate independence.

`assignments.json` stores the reviewed mapping from original `source_path` to destination `path`. The JSON tables preserve column order, string values, line endings and the original CSV SHA256. `stage_plantvillage.py` reconstructs ignored runtime CSVs automatically before validation; directly reusing old split directories does not perform the required reassignment. The staging process verifies image hashes and refuses to overwrite an existing destination. `hybrid_protocol.py` validates the unchanged pinned protocol, assignment, and manifest fingerprints.

`audit.json` and `class_coverage.json` record structural overlap checks and per-class coverage. `split_protocol.json` records counts, provenance, and the original runtime file hashes. JSON tables reproduce the reviewed CSV bytes exactly; no new split is generated.


