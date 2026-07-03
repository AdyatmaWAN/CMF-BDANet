# CMF-BDANet

Building damage change detection from pre-/post-event DSM (digital surface
model) + RGB imagery. Five models are trained and compared across five
classification scenarios:

- **FCSNN** — Siamese CNN over pre/post DSM tensors.
- **MMF-EMSNet** — multi-modal fusion network combining DSM-difference
  features with an RGB stream (3-stream or 4-stream variants).
- **FUJITA** — baseline Siamese CNN (from the Fujita et al. method).
- **Moya**, **Hajeb** — hand-crafted-feature SVM baselines (height-difference
  statistics).

**Scenarios** (label remapping applied to the same underlying 5-class data):

| Scenario | Task | Classes | Models |
|---|---|---|---|
| 1 | binary | class 0 vs class 4 only (other rows dropped) | all 5 |
| 2 | binary | classes {0,1,2,3} vs class 4 | all 5 |
| 3 | multiclass | all 5 classes (nominal — classes are unordered categories) | all 5 |
| 4 | binary | classes {0,1} vs {2,3,4} | all 5 |
| 5 | ordinal regression | same 5 classes as scenario 3, but the model is trained to respect their order (0 < 1 < 2 < 3 < 4) instead of treating them as unordered categories | MMF-EMSNet only, for now |

Scenario 5 is explained in full in [Ordinal regression (scenario 5)](#ordinal-regression-scenario-5) below.

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
                     + the CORAL ordinal head (CoralBiases) and its label encode/decode
  fujita.py          FUJITA baseline model
  Moya.py            Moya SVM feature extractor (feature_difference)
  Hajeb.py           Hajeb SVM feature extractor (dsm_difference)

utils/               Shared, model-agnostic pipeline code
  experiment.py      Determinism, callbacks, grid-search driver, shared NN train/eval loop
  label_processing.py  Scenario 1-5 label remapping + is_ordinal_scenario()
  metrics.py         accuracy/precision/recall/macro-F1/MCC/confusion-matrix,
                     plus ordinal MAE/QWK (compute_ordinal_metrics)

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
python train_fcsnn.py --scenario 4                        # the {0,1} vs {2,3,4} binary scenario
python train_fcsnn.py --dsm-mode dsm_only                 # just one DSM mode
python train_mmfemsnet.py --variant 4stream                # just the 4-stream MMF variant
python train_mmfemsnet.py --scenario 5                     # ordinal regression (CORAL) — MMF-EMSNet only
python train_fujita.py --config configs/fujita.json
python train_svm.py --model Moya --scenario 2              # just Moya, scenario 2
```

Scenarios 1-4 work identically across all four scripts. Scenario 5 (ordinal
regression) is currently only wired up for `train_mmfemsnet.py` — its
`--scenario` choices are `[1,2,3,4,5]`, the other three scripts' are
`[1,2,3,4]`, so passing `--scenario 5` to `train_fcsnn.py`/`train_fujita.py`/
`train_svm.py` fails fast with an argparse error instead of silently
training a plain nominal classifier under an ordinal-sounding scenario number.

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
  "scenarios": [1, 2, 3, 4],       // mmfemsnet.json also includes 5 (ordinal)
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
  this file is what marks the run "done". Scenario 5 (ordinal) rows add
  `mae` and `qwk` columns alongside the standard ones — see
  [Ordinal regression](#ordinal-regression-scenario-5).
- `predictions.csv` — one row per test sample: `sample_index`,
  `source_index` (maps back to the original dataset row, since scenario 1
  drops rows), `y_true`, `y_pred`, `is_correct`, per-class probabilities.
  For scenario 5, the per-class probabilities are derived from CORAL's
  threshold probabilities (see below) so the schema is identical to every
  other run — nothing downstream needs to know a run was ordinal.
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

## Ordinal regression (scenario 5)

### The problem with plain multiclass for ordered classes

Scenario 3 treats the 5 damage classes as **nominal** — unordered categories.
A softmax head with categorical cross-entropy loss penalizes *every*
misclassification identically: predicting class 4 when the truth is 0 costs
exactly the same as predicting class 1 when the truth is 0, even though the
first prediction is off by 4 damage levels and the second by only 1. But
these classes are actually **ordinal** — 0 < 1 < 2 < 3 < 4 represents
increasing damage severity — so a model that mixes up adjacent classes
(1 vs 2) is behaving much better than one that mixes up opposite ends (0 vs
4), and neither the loss nor accuracy/macro-F1 can tell the difference.
Scenario 5 reuses scenario 3's exact data but trains and evaluates the model
in a way that's aware of this order.

### CORAL: turning K classes into K-1 ordered yes/no questions

**Source:** Cao, W., Mirjalili, V., & Raschka, S. (2020). *"Rank Consistent
Ordinal Regression for Neural Networks with Application to Age Estimation."*
Pattern Recognition Letters, 140, 325–331
([arXiv:1901.07884](https://arxiv.org/abs/1901.07884)). Builds on the older,
more general idea of decomposing ordinal problems into extended binary
classification (Li & Lin, 2007; Frank & Hall, 2001).

Instead of asking "which of 5 classes is this?" in one shot, CORAL asks
**4 yes/no questions**, one per threshold between adjacent classes:

```
Q0: "Is the damage worse than class 0?"   -> P(y > 0)
Q1: "Is the damage worse than class 1?"   -> P(y > 1)
Q2: "Is the damage worse than class 2?"   -> P(y > 2)
Q3: "Is the damage worse than class 3?"   -> P(y > 3)
```

A true class of 2 means "yes" to Q0 and Q1, "no" to Q2 and Q3 — i.e. the
label `2` gets **encoded** as the binary vector `[1, 1, 0, 0]`
(`encode_coral_labels` in `models/mmfemsnet.py`). Answering all 4 questions
and counting how many are "yes" recovers the class: `sum([1,1,0,0]) = 2`.
This is exactly how prediction works too (`decode_coral_predictions`):
count how many of the 4 sigmoid outputs exceed 0.5.

The reason this needs to be more than "just train 4 independent binary
classifiers" is **rank consistency**: nothing stops 4 independent
classifiers from answering "yes, worse than 2" but "no, not worse than 1" —
a logical contradiction (if it's worse than 2, it must also be worse than
1). CORAL prevents this by construction:

- All 4 thresholds share the exact same learned weight vector `w` over the
  backbone features `g(x)` (a `Dense(1, use_bias=False)` layer named
  `coral_shared_weight` in the model graph) — call this shared value the
  **shared logit**, `z = w·g(x)`.
- Each threshold only adds its own **bias**: `logit_k = z + bias_k`.
- The 4 biases are forced non-increasing (`bias_0 ≥ bias_1 ≥ bias_2 ≥
  bias_3`) via `CoralBiases` in `models/mmfemsnet.py`: `bias_k = bias_0 -
  cumsum(softplus(gap_1..gap_k))`. `softplus` always returns a positive
  number, so each cumulative sum can only grow, so each `bias_k` can only
  shrink as `k` increases. Gradient descent is free to move `bias_0` and the
  gaps however it likes — it can never produce an increasing sequence,
  because the non-increasing shape is baked into the formula, not learned.
  Since a bigger threshold `k` always has a `≤` logit, its sigmoid
  probability `P(y>k)` is always `≤` the previous threshold's — the 4
  answers can never contradict each other.

### The loss function

Each of the 4 thresholds is a genuine binary classification problem — "is
`y` past this threshold, yes or no?" — so each one gets **binary
cross-entropy**:

```
BCE_k = -[ t_k · log(p_k) + (1 - t_k) · log(1 - p_k) ]
```

where `t_k ∈ {0,1}` is the k-th entry of the encoded label and `p_k =
sigmoid(logit_k)` is the model's predicted `P(y > k)`. The total CORAL loss
is just the **sum (equivalently, average) of these 4 binary cross-entropies**:

```
L = BCE_0 + BCE_1 + BCE_2 + BCE_3
```

In code this needs no custom loss function at all: Keras's built-in
`"binary_crossentropy"` string loss, applied to a model output of shape
`(batch, 4)` against an encoded target of shape `(batch, 4)`, already
computes exactly this — elementwise BCE across all 4 thresholds, averaged
over the batch (`train_mmfemsnet.py` passes `loss="binary_crossentropy"`
whenever `is_ordinal_scenario(scenario)` is true; `train_and_evaluate_nn`'s
auto-selected loss would otherwise pick `sparse_categorical_crossentropy`
for a 5-class problem, which is wrong here since the labels aren't
sparse integers anymore, they're the 4-column encoded matrix).

**Why summing/averaging 4 BCE terms is the right objective:** each
threshold's sigmoid output is trained completely independently by its own
BCE term — nothing in the loss itself enforces monotonicity (that's what
the shared-weight + ordered-bias architecture is for). The loss is simply
maximum-likelihood estimation applied 4 times over, once per threshold,
using the shared representation `g(x)`. Because the architecture makes it
*structurally impossible* to represent an inconsistent set of 4 answers,
gradient descent on this simple sum ends up learning a properly ordinal
model, without needing any exotic ranking loss.

### Turning threshold probabilities back into things you can compare

Everything downstream of training (metrics, `predictions.csv`, aggregation)
expects an integer class label and a per-class probability distribution,
not 4 threshold probabilities — so two conversions happen after prediction:

- **`decode_coral_predictions`**: predicted class = count of thresholds
  where `P(y>k) > 0.5`. E.g. `[0.95, 0.90, 0.05, 0.02]` → 2 thresholds
  exceeded → predicted class `2`.
- **`coral_probs_to_class_probs`**: converts the 4 threshold probabilities
  into a proper 5-way distribution via `P(y=k) = P(y>k-1) - P(y>k)`
  (using the conventions `P(y>-1)=1` and `P(y>4)=0`), so `predictions.csv`
  keeps the exact same `prob_class_0..4` columns as every nominal run.

### Evaluation metrics

Standard accuracy/precision/recall/macro-F1/MCC are still computed on the
decoded integer predictions (so ordinal and nominal scenario 3 results stay
comparable), but two ordinal-specific metrics are added
(`compute_ordinal_metrics` in `utils/metrics.py`), because accuracy/F1 still
can't distinguish "off by 1" from "off by 4":

- **MAE** (mean absolute error) — average `|predicted_class - true_class|`.
  Directly answers "on average, how many damage levels off are we?".
- **QWK** (quadratic weighted kappa, `sklearn.metrics.cohen_kappa_score(...,
  weights="quadratic")`) — an agreement score that penalizes a prediction
  error quadratically more the further apart the classes are (a 4-level
  miss counts 16x worse than a 1-level miss), while still discounting
  agreement that could happen by chance the way plain accuracy can't.

## Shared pipeline code (`utils/experiment.py`)

This is the module every `train_*.py` script is built on:

- `set_global_determinism()` / `set_tf_determinism()` — seed everything;
  the former must run *before* `import tensorflow`, the latter right after.
- `DSM_MODES` / `dsm_mode_channels()` — the shared dsm_mode ablation axis.
- `make_siamese_dsm_dataset()` — the (pre-DSM, post-DSM) tf.data pipeline
  shared by FCSNN and FUJITA (both take the same two-tensor input shape).
- `make_training_callbacks()` — early stopping / checkpointing / LR
  reduction, all monitoring a custom `val_f1` metric computed each epoch
  (not `val_loss`). Accepts optional `decode_pred_fn`/`decode_true_fn` hooks
  so `val_f1` can be computed correctly even when the model's raw output
  isn't a plain sigmoid/softmax (e.g. CORAL's threshold probabilities) —
  defaults reproduce the original threshold/argmax behavior.
- `train_and_evaluate_nn()` — the full compile → fit → evaluate → save
  loop shared by FCSNN, MMF-EMSNet, and FUJITA. Each script only builds its
  own datasets + model and calls this once; it returns the summary dict
  that becomes the `metrics.csv` row. (`train_svm.py` doesn't use this —
  no epochs/validation/callbacks for an SVM — it has its own equivalent
  logic inline.) Five optional parameters — `loss`, `decode_pred_fn`,
  `decode_true_fn`, `class_probs_fn`, `extra_metrics_fn` — all default to
  `None` and reproduce today's nominal-classification behavior exactly;
  `train_mmfemsnet.py` supplies CORAL-specific versions of all five only
  when `is_ordinal_scenario(scenario)` is true.
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
