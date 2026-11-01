# Local release integration review — 1 November 2026

Integrated on feature/rq7.1-model-compression, without commit/push. Overlapping
pre-integration files were preserved outside this repository in the sibling
rq71_branch_before_release_20261101 folder. RQ7.2/RQ7.3 and root environment
specification/lock were not changed.

Verified locally from the integrated project directory:
- Prospective static checks pass, including all four numbered notebooks.
- All 25 tests pass (18.2 seconds), including actual small ONNX inference.
- git diff --check passes; normal LF/CRLF warnings are not test failures.
- Scoped ignore exceptions admit split CSVs and figure PDFs; large model
  artifacts remain ignored.

Five immutable split files are packaged from the byte-verified V18 backup.
Full local image audit covered 54,299 images; bounded cloud GPU checks are
separate, not full-data execution. Synthetic Kaggle staged smoke is not
reportable model performance.

The release commit excludes unrelated edits to legacy notebooks and smoke
configs. Selected source/notebooks/manifests were scanned for credential patterns.
Review still required before final submission: GPU environment capture and
scientific failure/eligibility reporting. Direct requirements are pinned to
observed successful CPU smoke versions but are not a complete transitive lock.
No claim of exact historical Kaggle source equivalence is made. The official
rerun was submitted as Kaggle version 364659822 for seeds 14/27; its completion,
seed 42 and resulting report/artifact audit remain pending.

To run checks: python check_release.py; then python -m unittest discover -s tests
-p 'test_*.py'. The numbered notebooks use run_stage.py; run_three_seeds.py is
the complete explicit launcher and defaults to read-only full-data preflight.
