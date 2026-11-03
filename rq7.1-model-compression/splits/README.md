# RQ7.1 hybrid-v2 manifests

Fixed assignments used for the seed-14, seed-27, and seed-42 experiments.
| Partition | Images | Classes |
|---|---:|---:|
| Train | 38,147 | 38 |
| Validation | 5,448 | 38 |
| Test | 10,704 | 38 |

The split seed is 16153790. Training seeds are 14, 27 and 42. `test.csv` is byte-identical to the earlier test manifest. Known author leaf IDs and exact SHA256-connected components are kept together. Validation includes 4,075 images with resolved author leaf IDs and 1,373 without resolved IDs. We did not establish unknown-leaf independence or perceptual near-duplicate independence.

`assignments.csv` maps each original `source_path` to its reviewed destination `path`. Use `stage_plantvillage.py`; directly reusing the old split directories does not perform the required reassignment. The staging process verifies image hashes and refuses to overwrite an existing destination. `hybrid_protocol.py` validates the pinned protocol, assignment, and manifest fingerprints.

`audit.json` and `class_coverage.csv` record structural overlap checks and per-class coverage. `split_protocol.json` records counts, provenance, and file hashes. Scientific assignment and manifest files are copied byte-for-byte from the verified source bundle; 


