# Local release integration review — 1 November 2026

Integrated on feature/rq7.1-model-compression. Overlapping
pre-integration files were preserved outside this repository in the sibling
rq71_branch_before_release_20261101 folder. 

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

To run checks: python check_release.py; then python -m unittest discover -s tests
-p 'test_*.py'. The numbered notebooks use run_stage.py; run_three_seeds.py is
the complete explicit launcher and defaults to read-only full-data preflight.
