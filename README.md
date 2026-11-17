# CMF-BDANet

Building damage change detection from pre-/post-event DSM (digital surface
model) + RGB imagery. Six models are trained and compared across five
classification scenarios:

- **FCSNN** — Siamese CNN over pre/post DSM tensors. Weight sharing between
  the two towers is switchable (`shared`), and the two embeddings are
  combined by one of three fusion modes (`sub`, `concat`, `mcmaf`).
- **CMF-BDANet** — multi-modal fusion network combining DSM-difference
  features with an RGB stream (3-stream or 4-stream variants).
- **BDDNet** — multi-scale/residual two-stream baseline (optical + lidar).
- **FUJITA** — baseline pseudo-Siamese CNN (Fujita et al.): two unshared
  towers whose features are concatenated.
- **Moya**, **Hajeb** — hand-crafted-feature SVM baselines (height-difference
  statistics).

**The current study** (see [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md))
measures the effect of pseudo-Siamese concatenation in FCSNN. It runs a staged
protocol — hyperparameter search, then ablation, then multi-seed confirmation —
on `dsm_only` data; see [The staged protocol](#the-staged-protocol-stages-0-3).

**Scenarios** (label remapping applied to the same underlying 5-class data):

| Scenario | Task | Classes | Models |
|---|---|---|---|
| 1 | binary | class 0 vs class 4 only (other rows dropped) | all 6 |
| 2 | binary | classes {0,1,2,3} vs class 4 | all 6 |
| 3 | multiclass | all 5 classes (nominal — classes are unordered categories) | all 6 |
| 4 | binary | classes {0,1} vs {2,3,4} | all 6 |
| 5 | ordinal regression | same 5 classes as scenario 3, but the model is trained to respect their order (0 < 1 < 2 < 3 < 4) instead of treating them as unordered categories | FCSNN, CMF-BDANet, BDDNet, FUJITA (not the SVM baselines, no clean CORAL-for-SVM equivalent) |

Scenario 5 is explained in full in [Ordinal regression (scenario 5)](#ordinal-regression-scenario-5) below.

## Repository layout

```
configs/            One JSON config per model family (grid definitions)
  fcsnn.json
  cmf_bdanet.json
  bddnet.json
  fujita.json
  svm.json           (covers both Moya and Hajeb)

models/              Model definitions + shared data-loading
  fcsnn.py           FCSNN model + load_dataset() (NPZ loader used by every script)
  cmf_bdanet.py       CMF-BDANet model + DSM/RGB channel extraction + tf.data pipeline
  bddnet.py          BDD-Net baseline model
  fujita.py          FUJITA baseline model
  ordinal.py         Shared CORAL ordinal head (CoralBiases, add_coral_head) + label
                     encode/decode, used by every neural-net model
  Moya.py            Moya SVM feature extractor (feature_difference)
  Hajeb.py           Hajeb SVM feature extractor (dsm_difference)

utils/               Shared, model-agnostic pipeline code
  experiment.py      Determinism, callbacks, grid-search driver, shared NN train/eval loop
  label_processing.py  Scenario 1-5 label remapping + is_ordinal_scenario()
  metrics.py         accuracy/precision/recall/macro-F1/balanced-accuracy/MCC/
                     confusion-matrix, plus ordinal MAE/QWK (compute_ordinal_metrics)

train_fcsnn.py       Grid-search entrypoint for FCSNN
train_cmf_bdanet.py   Grid-search entrypoint for CMF-BDANet
train_bddnet.py      Grid-search entrypoint for BDDNet
train_fujita.py      Grid-search entrypoint for FUJITA
train_svm.py         Grid-search entrypoint for Moya + Hajeb

tools/
  aggregate_metrics.py   Cross-run aggregation: best-per-config, best-overall
  stage_lib.py           Shared helpers for the staged protocol
  stage0_hpo.py          Stage 0: per-cell hyperparameter search
  stage1_ablation.py     Stage 1: FCSNN ablation + capacity control
  stage2_confirm.py      Stage 2: multi-seed retraining, test set
  stage3_report.py       Stage 3: the paper tables
  stats_compare.py       Bootstrap CIs, paired bootstrap, Holm-Bonferroni
  complexity.py, flops_curve.py   Parameter/FLOP accounting

tests/
  test_fcsnn_shapes.py   Shape/capacity self-check for models/fcsnn.py

docs/
  EXPERIMENT_PLAN.md     The protocol: what runs, in what order, and why

convert_npz_compat.py   One-off tool: make a NumPy-2.x-written NPZ readable by NumPy 1.x

archive/             Pre-consolidation scripts kept for reference (see archive/README.md)
```

## Data format

Every model reads the same NPZ file via `models/fcsnn.py::load_dataset`,
which expects six arrays: `X_train, Y_train, X_val, Y_val, X_test, Y_test`.
`X_*` has shape `(N, H, W, C)` with a fixed channel layout (see
`models/cmf_bdanet.py`):

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

> The shipped configs pin `dsm_mode` to **`dsm_only`**, because that is the
> scope of the current study. Widen the list in `configs/*.json` to bring the
> other three modes back.

If you ever hit a numpy 2.x/1.x pickle incompatibility loading an NPZ, run
`python convert_npz_compat.py --input <in.npz> --output <out.npz>` first.

## Running a grid search

Each `train_*.py` does the identical thing: for every combination in its
config's `hpo_grid` × `ablation_grid`, it trains on `X_train`/`Y_train`
(using `X_val`/`Y_val` for early stopping/checkpointing), evaluates once on
`X_test`/`Y_test`, and writes results to a config-derived directory.

```bash
python train_fcsnn.py                                     # full grid, all scenarios, all seeds
python train_fcsnn.py --scenario 1                        # just scenario 1
python train_fcsnn.py --fusion concat --shared false      # just the pseudo-Siamese concat arm
python train_fcsnn.py --dropout true --dense false        # pin two ablation axes
python train_fcsnn.py --base-filters 22 --shared false    # the capacity-control width
python train_fcsnn.py --seed 12345                        # one seed instead of all five
python train_fcsnn.py --scenario 5                        # ordinal regression (CORAL)
python train_cmf_bdanet.py --variant 4stream              # just the 4-stream CMF-BDANet variant
python train_bddnet.py --base-filters 32                  # just one BDDNet width
python train_fujita.py --scenario 5                       # ordinal regression (CORAL)
python train_svm.py --model Moya --scenario 2             # just Moya, scenario 2
```

**Seeds are a grid axis.** Each config lists `seeds`, and the seed is part of
every result directory, so one invocation trains every architecture/HPO combo
once per seed and each is independently resumable. `--seed N` narrows to one.
This is what makes the multi-seed reporting in Stage 2 possible: `seed`
actually reaches the model's initializers and the shuffle, rather than being a
single module-level constant.

Scenarios 1-4 work identically across all five scripts. Scenario 5 (ordinal
regression via CORAL) is wired up for the four neural nets —
`train_fcsnn.py`, `train_cmf_bdanet.py`, `train_bddnet.py` and
`train_fujita.py` all accept `--scenario 5` and share the same
`models/ordinal.py` head. `train_svm.py`
does not (`--scenario` choices are `[1,2,3,4]`) since there's no clean CORAL
equivalent for the SVM baselines — passing `--scenario 5` to it fails fast
with an argparse error rather than silently training something else under
an ordinal-sounding scenario number.

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
  "seeds": [12345, 23451, 34512, 45123, 51234],
  "epochs": 100,
  "scenarios": [1, 2, 3, 4, 5],
  "datasets": { "dataset_16": "Dataset/NPZ_spatial/dataset_16.npz" },
  "callbacks": { "early_stopping_patience": 12, "lr_reduce_patience": 6, ... },

  "ablation_grid": {                 // model/architecture variants
    "dsm_mode": ["dsm_only"],
    "fusion": ["sub", "concat"],     // "mcmaf" is implemented but not in the grid
    "shared": [true, false],
    "dropout": [true, false],
    "dense": [true, false],
    "base_filters": [32]
  },
  "hpo_grid": {                      // hyperparameter tuning
    "optimizer": ["Adam", "Nadam", "SGD"],
    "learning_rate": [0.0001, 0.001],
    "batch_size": [64, 128, 256]
  },
  "capacity_control": {              // FCSNN only; not a reported arm
    "shared": false,
    "base_filters": 22
  }
}
```

- `ablation_grid` — architecture/data variants you're comparing. For FCSNN
  that is `fusion` / `shared` / `dropout` / `dense` (16 configs); `variant`
  — `3stream_concat`/`3stream_no_concat`/`4stream` — for CMF-BDANet;
  `base_filters` for BDDNet.
- `hpo_grid` — pure hyperparameter tuning: optimizer/lr/batch_size for the
  neural nets, kernel/decision_function/C/gamma for the SVMs. `seeds` is
  merged into this grid at run time.
- `capacity_control` — the parameter-matched width for FCSNN's capacity
  control: two unshared towers at `bf=22` come to 1.038x one shared tower at
  `bf=32` (verified by `tests/test_fcsnn_shapes.py`). See
  [The staged protocol](#the-staged-protocol-stages-0-3).
- Edit these lists to narrow or widen a sweep; the CLI flags above exist for
  quick one-off filtering without editing the file.

**There is no class weighting anywhere**, including in the SVM baselines —
it was removed deliberately so every model is trained under the same
conditions. `train_svm.py` previously passed `class_weight="balanced"` to
`SVC`; it no longer does.

## What gets saved per run

Every run's result directory (e.g.
`results/scenario_1/FCSNN/dataset_16/dsm_only/fusion_concat/shared_False/bf32/do_True/dense_True/Adam_lr0.0001_bs128/seed_12345/`)
contains:

- `metrics.csv` — one row: every grid parameter (including `seed` and
  `params`, the model's parameter count) + accuracy/precision/recall/
  macro-F1/macro-F2/balanced-accuracy/MCC/confusion-matrix cells/train
  time/etc. Presence of
  this file is what marks the run "done". Scenario 5 (ordinal) rows add six
  more columns (`mae`, `rmse`, `off_by_one_accuracy`, `qwk`, `linear_kappa`,
  `spearman`) alongside the standard ones — see
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

Directory shape per model (every one ends in `seed_<n>/`):
```
FCSNN:        scenario_N/FCSNN/<dataset>/<dsm_mode>/fusion_<tag>/shared_<bool>/bf<n>/do_<bool>/dense_<bool>/<opt>_lr<lr>_bs<bs>/seed_<n>/
CMF-BDANet:   scenario_N/CMF/<dataset>/<dsm_mode>/<variant>/residual_<bool>/fusion_<tag>/<opt>_lr<lr>_bs<bs>/seed_<n>/
BDDNet:       scenario_N/BDD/<dataset>/<dsm_mode>/base_filters_<n>/<opt>_lr<lr>_bs<bs>/seed_<n>/
FUJITA:       scenario_N/FUJITA/<dataset>/<dsm_mode>/<opt>_lr<lr>_bs<bs>/seed_<n>/
Moya/Hajeb:   scenario_N/<Moya|Hajeb>/<dataset>/<dsm_mode>/dec_<f>_kernel_<k>_C_<c>_gamma_<g>/seed_<n>/
```

One layout serves all three stages of the protocol below, which is what lets a
later stage reuse an earlier stage's runs instead of retraining them: the path
encodes every parameter, so `metrics.csv` existing is a complete "already
done" check.

## Aggregating results

`tools/aggregate_metrics.py` scans every `metrics.csv` under a results root
and produces two tables:

```bash
python tools/aggregate_metrics.py --results results --best-metric f2
```

`--best-metric` defaults to **f2**, matching the metric training monitors
(see [Selection metric](#selection-metric-macro-f2)). Pass `f1` for the old
behaviour. `seed` is treated as a bookkeeping column, so runs that differ only
by seed collapse into one group rather than fragmenting it.

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

**This also runs automatically, scoped to one scenario at a time.** Every
`train_*.py` script calls `utils/experiment.py::aggregate_scenarios_after_run`
at the end of `main()`, which re-runs both aggregation stages — but pointed
at `results/scenario_N/` instead of all of `results/` — for every scenario
that was part of the run. Since every model's results already live nested
under `results/scenario_N/{MODEL}/...`, scoping the scan that way picks up
*every* model that has ever been trained for that scenario, not just the one
that just finished. So training FCSNN for scenario 4, then later training
CMF-BDANet for scenario 4, leaves
`results/scenario_4/aggregated_metrics.csv` and `best_overall.csv` reflecting
both models — no manual aggregation step needed. This is on by default and
unconditional (it re-scans even if every run in this invocation was skipped
as already-trained, since the point is to keep the aggregate current, not to
gate it on this specific invocation having done new work).

## Selection metric: macro-F2

Every selection in the pipeline — checkpointing, early stopping, LR-plateau
reduction, and which config wins a stage — is made on **validation macro-F2**
(`val_f2`), not macro-F1. F2 weights recall four times precision, which is the
right bias for damage assessment: a missed damaged building costs more than a
false alarm.

`macro_f1` is still computed and saved alongside `macro_f2` in every
`metrics.csv`, so the F1 view of any run remains available — it simply no
longer drives selection. Ordinal runs (scenario 5) override this with QWK
(`val_qwk`), for the reasons in
[Ordinal regression](#ordinal-regression-scenario-5).

One consequence worth stating in a write-up: selection now favours
higher-recall, lower-precision models, and configs chosen under F2 may differ
from those an F1-selected run would have picked.

## The staged protocol (stages 0-3)

The `train_*.py` scripts above run whatever grid their config names. The
`tools/stage*.py` scripts sit on top and run them in the order the study
needs, so that **nothing is ever selected on the test set**:

```bash
python tools/stage0_hpo.py        # per-cell hyperparameter search  (validation only, 1 seed)
```

```bash
python tools/stage1_ablation.py   # FCSNN ablation + capacity control (validation only, 1 seed)
```

```bash
python tools/stage2_confirm.py    # retrain every arm on all 5 seeds, evaluate on test
```

```bash
python tools/stage3_report.py     # the paper tables
```

| Stage | What it does | Output |
|---|---|---|
| 0 | Searches the HPO grid once per *cell* — for FCSNN, each `(sharing x fusion)` combination, plus the capacity-control width. Tuning per cell matters because the study compares those cells against each other, so hyperparameters chosen on one arm and applied to its competitor would bias the comparison. | `results/stage0_hpo.csv` |
| 1 | With hyperparameters frozen, runs the 16-config FCSNN ablation and picks each cell's best. Then the capacity control: the winning fusion, unshared, at the parameter-matched width. | `results/stage1_winners.csv` |
| 2 | Retrains every arm on all five seeds, each stopping independently on its own validation curve, and evaluates the test split once per seed. | per-run `metrics.csv` |
| 3 | Aggregates into the paper tables. | `results/paper/` |

Common flags: `--scenarios 1,2` on every stage; `--select-only` (stages 0-1)
re-runs the selection without retraining; `--dry-run` (stage 2) lists the
invocations it would make.

The stage scripts do not reimplement training. Each narrows a model's config —
pinning the ablation axes and hyperparameters for one cell — writes it to
`results/_stage_configs/`, and invokes the existing `train_*.py` with
`--config`. Selection then reads the resulting `metrics.csv` files back.

### What Stage 3 produces

- **`table_main_scenario_N.csv`** — cross-model comparison, 8 rows: the four
  FCSNN cell winners plus BDDNet, Fujita, Moya and Hajeb.
- **`table_ablation_scenario_N.csv`** — FCSNN only, 17 rows: all 16 ablation
  configs plus the capacity control (`control=True`), with the validation
  score that drove selection and an `is_cell_winner` flag.
- **`table_hpo.csv`** — Stage-0 winning hyperparameters, for a
  reproducibility appendix.
- **`significance.csv`** — paired-bootstrap p-values, Holm-Bonferroni
  corrected.

Both tables are also written combined across scenarios, and carry mean ± std
across seeds plus a bootstrap 95% CI for every metric.

### The capacity control

Unshared (pseudo-Siamese) FCSNN has two independent towers, so ~2x the
parameters of the shared version. A win could therefore be *architecture*
(separate weights letting each tower specialise on pre- vs post-event DSM) or
simply *capacity*. The control holds parameters fixed — two unshared towers at
`base_filters=22` come to 1.038x one shared tower at 32 — so a win that
survives there is attributable to weight sharing rather than budget. It is
tuned at its own width in Stage 0, so "it was tuned for a wider model" is not
available as an explanation for its result.

Note that halving the width would *not* have matched parameters: conv
parameters scale with `C_in x C_out`, so `bf=16` gives only 0.61x. Run
`python tests/test_fcsnn_shapes.py` to print the exact counts.

The `sub` vs `concat` comparison is unaffected by any of this — those arms
have identical tower parameters and differ only in the fusion head.

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
in a way that's aware of this order. It's available for all four neural
nets — FCSNN, CMF-BDANet, BDDNet, FUJITA — all sharing the same CORAL head from
`models/ordinal.py`. It's not available for the SVM baselines (Moya/Hajeb):
`SVC` has no clean CORAL-style ordinal equivalent, so `train_svm.py`
doesn't accept `--scenario 5`.

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
(`encode_coral_labels` in `models/ordinal.py`). Answering all 4 questions
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
  bias_3`) via `CoralBiases` in `models/ordinal.py`: `bias_k = bias_0 -
  cumsum(softplus(gap_1..gap_k))`. `softplus` always returns a positive
  number, so each cumulative sum can only grow, so each `bias_k` can only
  shrink as `k` increases. Gradient descent is free to move `bias_0` and the
  gaps however it likes — it can never produce an increasing sequence,
  because the non-increasing shape is baked into the formula, not learned.
  Since a bigger threshold `k` always has a `≤` logit, its sigmoid
  probability `P(y>k)` is always `≤` the previous threshold's — the 4
  answers can never contradict each other.
- `add_coral_head(x, num_classes)` in `models/ordinal.py` wires the shared
  `Dense(1, use_bias=False)` + `CoralBiases` pair onto a model's penultimate
  feature tensor `x` in one call — `models/fcsnn.py`, `models/cmf_bdanet.py`,
  `models/bddnet.py` and `models/fujita.py` all use this same function for
  their `ordinal=True` head, so there is exactly one CORAL implementation,
  not four.

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
over the batch. Every ordinal-capable script (`train_fcsnn.py`,
`train_cmf_bdanet.py`, `train_fujita.py`) passes `loss="binary_crossentropy"`
whenever `is_ordinal_scenario(scenario)` is true; `train_and_evaluate_nn`'s
auto-selected loss would otherwise pick `sparse_categorical_crossentropy`
for a 5-class problem, which is wrong here since the labels aren't
sparse integers anymore, they're the 4-column encoded matrix.

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
comparable), but `compute_ordinal_metrics` in `utils/metrics.py` adds six
ordinal-specific metrics to every scenario-5 `metrics.csv` row, because
accuracy/F1 alone can't distinguish "off by 1" from "off by 4":

- **`mae`** (mean absolute error) — average `|predicted_class - true_class|`.
  Directly answers "on average, how many damage levels off are we?".
- **`rmse`** (root mean squared error) — like MAE but squares errors before
  averaging, so a few severe misses (off by 4) inflate it far more than
  many mild ones (off by 1). Reported alongside MAE because the two only
  disagree when error *severity* varies — matching MAE with a high RMSE
  means a handful of bad misses are dragging the average up.
- **`off_by_one_accuracy`** — fraction of predictions within ±1 class of the
  truth. The most easily-communicated ordinal metric ("94% of predictions
  are within one damage level").
- **`qwk`** (quadratic weighted kappa) — chance-corrected agreement that
  penalizes an error quadratically more the further apart the classes are
  (a 4-level miss counts 16x worse than a 1-level miss). The standard
  single-number summary for ordinal agreement — this is also what drives
  training/checkpoint monitoring (see below).
- **`linear_kappa`** — the same idea as QWK but penalizes errors linearly
  instead of quadratically. Reported alongside QWK because they diverge
  most when comparing "many mild misses" against "a few severe ones" — if
  QWK is noticeably lower than linear kappa, the model's mistakes skew
  toward large jumps rather than being uniformly off-by-a-little.
- **`spearman`** (Spearman rank correlation) — whether predictions are
  *monotonically* related to the truth, independent of how large the errors
  are. A model can score well here with a mediocre MAE if its mistakes are
  consistently biased in one direction rather than scattered.

### Training-time monitoring and console output

Ordinal scenarios change what gets watched *during* training, not just what
gets reported afterward:

- **Checkpointing / early stopping / LR-plateau reduction.** Every NN
  training run tracks a validation metric each epoch (via
  `make_training_callbacks` in `utils/experiment.py`) that drives
  `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` alike. For
  nominal scenarios this is macro-F2 (`val_f2`, see
  [Selection metric](#selection-metric-macro-f2)). For ordinal scenarios it's
  **QWK** (`val_qwk`, via `models/ordinal.py::ordinal_monitor_metric`)
  instead — an F-measure on decoded ranks would treat every misclassification
  as equally bad, which defeats the purpose of training an ordinal-aware
  model in the first place. Both `train_and_evaluate_nn` and
  `make_training_callbacks` accept `monitor_name`/`monitor_fn` to make this
  swap, and default to `val_f2` when unset.
- **Per-epoch console line.** Follows directly from the above: you'll see
  `- val_f2: 0.8123` per epoch for nominal runs and `- val_qwk: 0.9012` for
  ordinal ones.
- **End-of-run console line.** `train_and_evaluate_nn` accepts a
  `summary_log_fn` hook (default prints `ACC=.. F1=.. F2=.. BAcc=.. MCC=..`); ordinal
  scenarios use `models/ordinal.py::ordinal_summary_log`, which prints
  `ACC=.. MAE=.. QWK=.. OffByOne=..` instead — the full six-metric set
  still lands in `metrics.csv` either way, this only changes what's
  concise enough to scan in the terminal.
- **`metrics.csv` schema note:** the "best value achieved during training"
  column is named `best_val_metric` since it holds either a macro-F2 or a QWK
  value depending on the run; the `monitor_metric` column next to it always
  names which one. This is the column every stage selects on.

## Shared pipeline code (`utils/experiment.py`)

This is the module every `train_*.py` script is built on:

- `set_global_determinism()` / `set_tf_determinism()` — seed everything;
  the former must run *before* `import tensorflow`, the latter right after.
- `DSM_MODES` / `dsm_mode_channels()` — the shared dsm_mode ablation axis.
- `make_siamese_dsm_dataset()` — the (pre-DSM, post-DSM) tf.data pipeline
  shared by FCSNN and FUJITA (both take the same two-tensor input shape).
  Accepts the same `ordinal`/`num_classes` pair as `models/cmf_bdanet.py`'s
  `make_dataset()` to CORAL-encode `Y` before building the dataset.
- `MONITOR_METRIC` — the one place `"val_f2"` is defined; everything that
  monitors or selects reads it.
- `make_training_callbacks()` — early stopping / checkpointing / LR
  reduction, all monitoring a custom validation metric computed each epoch
  (not `val_loss`) — macro-F2 (`val_f2`) by default, QWK (`val_qwk`) for
  ordinal runs. Accepts optional `decode_pred_fn`/`decode_true_fn` hooks so
  the metric can be computed correctly even when the model's raw output
  isn't a plain sigmoid/softmax (e.g. CORAL's threshold probabilities), and
  `monitor_name`/`monitor_fn` to swap which metric is tracked.
- `train_and_evaluate_nn()` — the full compile → fit → evaluate → save
  loop shared by FCSNN, CMF-BDANet, BDDNet, and FUJITA. Each script only builds its
  own datasets + model and calls this once; it returns the summary dict
  that becomes the `metrics.csv` row. (`train_svm.py` doesn't use this —
  no epochs/validation/callbacks for an SVM — it has its own equivalent
  logic inline.) Eight optional parameters — `loss`, `decode_pred_fn`,
  `decode_true_fn`, `class_probs_fn`, `extra_metrics_fn`, `monitor_name`,
  `monitor_fn`, `summary_log_fn` — all default to `None` (or `val_f2`) and
  give plain nominal-classification behaviour; each `train_*.py` supplies
  CORAL-specific versions of all eight (from `models/ordinal.py`) only when
  `is_ordinal_scenario(scenario)` is true.
- `grid_search()` — iterates `ablation_grid × hpo_grid` (with `seeds` merged
  into the latter), builds each run's result directory, skips already-trained
  combos, and catches/logs per-run exceptions without aborting the whole
  sweep.
- `build_predictions_frame()`, `compute_f2()`, `save_classification_report()`
  — shared result-saving helpers used identically by every script.
- `aggregate_scenarios_after_run()` — called once at the end of every
  script's `main()`; re-aggregates each scenario the run touched (see
  [Aggregating results](#aggregating-results)).

## Tests

```bash
python tests/test_fcsnn_shapes.py
```

Plain `assert` + `__main__`, no framework. It checks what the protocol
depends on: all three fusion modes build; unshared has ~2x the parameters of
shared; the capacity-control width lands within 10% of the shared model;
`sub` and `concat` have identical tower parameters (so that comparison is not
confounded by capacity); the `dropout`/`dense` flags change the graph; the
same seed reproduces and different seeds don't; and the two unshared towers
start from *different* weights. It also prints the exact parameter count per
arm — that is what the `params` column reports, and what re-pins the control
width if the arithmetic ever changes.

`python tools/stats_compare.py` runs its own self-check of the bootstrap CI,
the paired bootstrap, and the Holm-Bonferroni correction.

## Archive

`archive/` holds pre-consolidation scripts that aren't part of the grid-search
pipeline but still have unreproduced value (single-config CMF-BDANet training
with MCMAF attention-weight visualization, and the reproduction wrapper
around it). See `archive/README.md` for why they're kept instead of deleted.
