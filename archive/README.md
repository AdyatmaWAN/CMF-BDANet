# Archive

Pre-consolidation scripts kept for reference, not part of the active pipeline
(`train_fcsnn.py`, `train_mmfemsnet.py`, `train_fujita.py`, `train_svm.py`).

- `run_custom_train_infer.py` — single-config MMF-EMSNet train + MCMAF
  attention-weight visualization (PNG panels per building/class). Not
  reproduced by the grid-search scripts; kept for interpretability work on a
  specific already-chosen config.
- `run_reproduction.py` / `run_reproduction.bat` — wrappers that call
  `run_custom_train_infer.py` with hardcoded "best HPO" parameters per
  scenario from an earlier round of experiments.

`run_all_inference.py` and `run_all_inference_select_best.py` were removed
outright rather than archived: every `train_*.py` script now saves
`predictions.csv` inline during training, and `tools/aggregate_metrics.py`
produces the best-per-config/best-overall rollups those two scripts used to
compute post-hoc. Their logic is fully superseded, not just relocated.
