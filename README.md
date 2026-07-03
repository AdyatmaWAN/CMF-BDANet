# CMF-BDANet

Building damage change detection from pre-/post-event DSM (digital surface
model) + RGB imagery. Five models are trained and compared across three
classification scenarios:

- **FCSNN** — Siamese CNN over pre/post DSM tensors.
- **MMF-EMSNet** — multi-modal fusion network combining DSM-difference
  features with an RGB stream (3-stream or 4-stream variants).
- **FUJITA** — baseline Siamese CNN (from the Fujita et al. method).
- **Moya**, **Hajeb** — hand-crafted-feature SVM baselines (height-difference
  statistics).

**Scenarios** (label remapping applied to the same underlying 5-class data):

| Scenario | Task | Classes |
|---|---|---|
| 1 | binary | class 0 vs class 4 only (other rows dropped) |
| 2 | binary | classes {0,1,2,3} vs class 4 |
| 3 | multiclass | all 5 classes |

## Repository layout

```
configs/            One JSON config per model family (grid definitions)
  fcsnn.json
  mmfemsnet.json
  fujita.json
  svm.json           (covers both Moya and Hajeb)

models/              Model definitions + shared data-loading
  fcsnn.py           FCSNN model + load_dataset() (NPZ loader used by every script)
  mmfemsnet.py       MMF-EMSNet model + DSM/RGB channel extraction + tf.data pipeline
  fujita.py          FUJITA baseline model
  Moya.py            Moya SVM feature extractor (feature_difference)
  Hajeb.py           Hajeb SVM feature extractor (dsm_difference)

utils/               Shared, model-agnostic pipeline code
  experiment.py      Determinism, callbacks, grid-search driver, shared NN train/eval loop
  label_processing.py  Scenario 1/2/3 label remapping
  metrics.py         accuracy/precision/recall/macro-F1/MCC/confusion-matrix

train_fcsnn.py       Grid-search entrypoint for FCSNN
train_mmfemsnet.py   Grid-search entrypoint for MMF-EMSNet
train_fujita.py      Grid-search entrypoint for FUJITA
train_svm.py         Grid-search entrypoint for Moya + Hajeb

tools/
  aggregate_metrics.py   Cross-run aggregation: best-per-config, best-overall

convert_npz_compat.py   One-off tool: make a NumPy-2.x-written NPZ readable by NumPy 1.x

archive/             Pre-consolidation scripts kept for reference (see archive/README.md)
```

## Data format

Every model reads the same NPZ file via `models/fcsnn.py::load_dataset`,
which expects six arrays: `X_train, Y_train, X_val, Y_val, X_test, Y_test`.
`X_*` has shape `(N, H, W, C)` with a fixed channel layout (see
`models/mmfemsnet.py`):

```
0-2   pre-event RGB
3-5   post-event RGB
6     pre-event nDSM        7  pre density     8  pre uncertainty    9  pre dtm-fill
10    post-event nDSM       11 post density    12 post uncertainty  13 post dtm-fill
```

The **dsm_mode** ablation axis (shared by every model) picks which DSM
channels are active: nDSM is always included; density and uncertainty are
each independently toggled, giving 4 modes: `dsm_density_uncertainty`,
`dsm_density`, `dsm_uncertainty`, `dsm_only`.

If you ever hit a numpy 2.x/1.x pickle incompatibility loading an NPZ, run
`python convert_npz_compat.py --input <in.npz> --output <out.npz>` first.

## Running a grid search

Each `train_*.py` does the identical thing: for every combination in its
config's `hpo_grid` × `ablation_grid`, it trains on `X_train`/`Y_train`
(using `X_val`/`Y_val` for early stopping/checkpointing), evaluates once on
`X_test`/`Y_test`, and writes results to a config-derived directory.

```bash
python train_fcsnn.py                                    # full grid, all scenarios
python train_fcsnn.py --scenario 1                        # just scenario 1
python train_fcsnn.py --dsm-mode dsm_only                 # just one DSM mode
python train_mmfemsnet.py --variant 4stream                # just the 4-stream MMF variant
python train_fujita.py --config configs/fujita.json
python train_svm.py --model Moya --scenario 2              # just Moya, scenario 2
```

**Resuming:** every run's result directory already encodes its full set of
parameters, so "already trained" simply means `metrics.csv` exists there
(`utils/experiment.py::is_already_trained`). Re-running any script picks up
where it left off — no separate checkpoint file to manage. A run that
raises an exception logs to `<result_dir>/error.log` and is retried on the
next invocation (it's never marked done).

### Config file shape

Each config in `configs/` has two grids plus shared settings:

```jsonc
{
  "results_root": "results",
  "seed": 1234,
  "epochs": 100,
  "scenarios": [1, 2, 3],
  "datasets": { "dataset_16": "Dataset/NPZ/dataset_16.npz" },
  "callbacks": { "early_stopping_patience": 12, "lr_reduce_patience": 6, ... },

  "ablation_grid": {                 // model/architecture variants
    "dsm_mode": ["dsm_density_uncertainty", "dsm_density", "dsm_uncertainty", "dsm_only"],
    "residual": [true, false],
    "fusion": ["concat", "mcmaf"]
  },
  "hpo_grid": {                      // hyperparameter tuning
    "optimizer": ["Adam", "SGD", "RMSprop", "Nadam", "Adamax"],
    "learning_rate": [0.0001, 0.001],
    "batch_size": [256, 128, 64]
  }
}
```

- `ablation_grid` — architecture/data variants you're comparing (dsm_mode
  everywhere; `residual`/`fusion` for FCSNN and MMF-EMSNet; `variant`
  — `3stream_concat`/`3stream_no_concat`/`4stream` — for MMF-EMSNet only).
- `hpo_grid` — pure hyperparameter tuning: optimizer/lr/batch_size for the
  neural nets, kernel/decision_function/C/gamma for the SVMs.
- Edit these lists to narrow or widen a sweep; `--scenario`/`--dsm-mode`
  CLI flags exist for quick one-off filtering without editing the file.

## What gets saved per run

Every run's result directory (e.g.
`results/scenario_1/FCSNN/dataset_16/dsm_only/residual_True/fusion_concat/Adam_lr0.0001_bs256/`)
contains:

- `metrics.csv` — one row: every grid parameter + accuracy/precision/recall/
  macro-F1/macro-F2/MCC/confusion-matrix cells/train time/etc. Presence of
  this file is what marks the run "done".
- `predictions.csv` — one row per test sample: `sample_index`,
  `source_index` (maps back to the original dataset row, since scenario 1
  drops rows), `y_true`, `y_pred`, `is_correct`, per-class probabilities.
- `classification_report.txt` / `.csv` — sklearn's per-class report.
- Model artifact: `model.keras` + `weights.h5` (+ `best_weights.weights.h5`
  from the checkpoint callback) for the neural nets, `model.pkl` for the SVMs.

Directory shape per model:
```
FCSNN:        scenario_N/FCSNN/<dataset>/<dsm_mode>/residual_<bool>/fusion_<tag>/<opt>_lr<lr>_bs<bs>/
MMF-EMSNet:   scenario_N/MMF/<dataset>/<dsm_mode>/<variant>/residual_<bool>/fusion_<tag>/<opt>_lr<lr>_bs<bs>/
FUJITA:       scenario_N/FUJITA/<dataset>/<dsm_mode>/<opt>_lr<lr>_bs<bs>/
Moya/Hajeb:   scenario_N/<Moya|Hajeb>/<dataset>/<dsm_mode>/dec_<f>_kernel_<k>_C_<c>_gamma_<g>/
```

## Aggregating results

`tools/aggregate_metrics.py` scans every `metrics.csv` under a results root
and produces two tables:

```bash
python tools/aggregate_metrics.py --results results --best-metric f1
```

1. **`aggregated_metrics.csv`** — collapses the hyperparameter grid down to
   the single best HPO combo per `(model, scenario, dataset, ablation-config)`
   group. Answers "what's the best result for this model/config?".
2. **`best_overall.csv`** — collapses stage 1 further, across every model,
   down to the single best row per scenario. Answers "which model/config
   wins outright?".

It groups by exclusion: any column that isn't a known hyperparameter
(`optimizer`, `learning_rate`, `batch_size`, `kernel`, `decision_function`,
`C`, `gamma`) or a per-run metric/bookkeeping column is treated as part of
the ablation identity — so adding a new model or a new ablation column
needs no code changes here.

## Shared pipeline code (`utils/experiment.py`)

This is the module every `train_*.py` script is built on:

- `set_global_determinism()` / `set_tf_determinism()` — seed everything;
  the former must run *before* `import tensorflow`, the latter right after.
- `DSM_MODES` / `dsm_mode_channels()` — the shared dsm_mode ablation axis.
- `make_siamese_dsm_dataset()` — the (pre-DSM, post-DSM) tf.data pipeline
  shared by FCSNN and FUJITA (both take the same two-tensor input shape).
- `make_training_callbacks()` — early stopping / checkpointing / LR
  reduction, all monitoring a custom `val_f1` metric computed each epoch
  (not `val_loss`).
- `train_and_evaluate_nn()` — the full compile → fit → evaluate → save
  loop shared by FCSNN, MMF-EMSNet, and FUJITA. Each script only builds its
  own datasets + model and calls this once; it returns the summary dict
  that becomes the `metrics.csv` row. (`train_svm.py` doesn't use this —
  no epochs/validation/callbacks for an SVM — it has its own equivalent
  logic inline.)
- `grid_search()` — iterates `ablation_grid × hpo_grid`, builds each run's
  result directory, skips already-trained combos, and catches/logs
  per-run exceptions without aborting the whole sweep.
- `build_predictions_frame()`, `compute_f2()`, `save_classification_report()`
  — shared result-saving helpers used identically by every script.

## Archive

`archive/` holds pre-consolidation scripts that aren't part of the grid-search
pipeline but still have unreproduced value (single-config MMF-EMSNet training
with MCMAF attention-weight visualization, and the reproduction wrapper
around it). See `archive/README.md` for why they're kept instead of deleted.
