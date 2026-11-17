# Experiment plan — pseudo-siamese concatenation in FCSNN

**Status:** Part 1 (models + infrastructure) and Part 2 (the stage scripts) are
both **implemented and verified**. No training has been run — the dataset is not
in the working tree.

## How to run it

```
python tools/stage0_hpo.py                     # per-cell HPO, 1 seed
python tools/stage1_ablation.py                # 16-config ablation + control
python tools/stage2_confirm.py                 # 5 seeds, test set
python tools/stage3_report.py                  # the paper tables
```

Every stage takes `--scenarios 1,2`, every stage but 3 takes `--select-only` or
`--dry-run`, and all of them resume: a run whose `metrics.csv` exists is
skipped, so a stage can be stopped and restarted freely.

The stage scripts do not reimplement training. Each one narrows a model's
config (pinning the ablation axes and hyperparameters for one cell), writes it
to `results/_stage_configs/`, and invokes the existing `train_*.py` with
`--config`. Selection then reads the resulting `metrics.csv` files back.

Verified by `tests/test_fcsnn_shapes.py` (all checks pass) plus an offline
wiring check: all five configs parse, FCSNN expands to 16 ablation configs x 18
HPO combos x 5 seeds = 1440 distinct result directories, and the capacity
control resolves to `.../bf22/...`. Measured parameter counts:

| Arm | `bf` | Params | vs. shared |
|---|---|---|---|
| shared, sub | 32 | 618,049 | 1.000x |
| unshared, sub | 32 | 1,232,257 | 1.994x |
| **unshared, sub** | **22** | **641,497** | **1.038x** |
| shared, concat | 32 | 621,889 | 1.000x |
| unshared, concat | 32 | 1,236,097 | 1.988x |
| **unshared, concat** | **22** | **644,217** | **1.036x** |

**Paper scope:** the effect of pseudo-siamese concatenation in FCSNN, plus baselines.
CMF-BDANet is made protocol-ready but is **not run here** (separate paper).

## Locked parameters

| Parameter | Value |
|---|---|
| Data | `Dataset/NPZ_spatial/dataset_16.npz`, `dsm_only` |
| Scenarios | 1, 2, 3, 4, 5 |
| Seeds | `12345, 23451, 34512, 45123, 51234` |
| Max epochs / patience | 100 / 12 |
| Optimizers | `Adam, Nadam, SGD` |
| Learning rates | `1e-4, 1e-3` |
| Batch sizes | `64, 128, 256` |
| Class weighting | **none**, anywhere (removed from the SVMs too) |
| `base_filters` | **32** for every reported FCSNN and CMF-BDANet variant; **22** for the capacity-control arm only. BDDNet keeps its own `bf{16,32}` axis |
| HPO scope | per `(sharing x fusion)` cell |
| Selection metric | **`val_f2`** (scenarios 1-4), `val_qwk` (scenario 5) — drives checkpointing, early stopping, LR reduction, and every Stage-0/1 selection |
| Test set | touched only in Stage 2 |
| SVM randomisation | `random_state=seed` only; `std = 0` accepted and reported as-is |

**Ablation grid = 16 configs:** `fusion{sub,concat} x shared{T,F} x dropout{T,F} x dense{T,F}`,
4 per `(sharing x fusion)` cell. Plus 1 capacity-control arm = **17 FCSNN arms**.

Stages 0 and 1 run at seed `12345`, the first of the five, so Stage 2 inherits
those runs instead of repeating them.

---

# Part 1 — Per-file changes

## `models/fcsnn.py` — the real work

1. **Delete `substraction: bool`.** `substraction` + `fusion` collapse into one
   `fusion: str` in `{"sub","concat","mcmaf"}`, resolved by a single
   `if/elif/else` replacing lines 149-175.
   *Why:* today they are two independent params in an `if / if` sequence, so
   `mcmaf` overwrites whatever `substraction` produced; and `train_fcsnn.py:138`
   hardcodes `substraction=True`, making the `"concat"` branch unreachable.
   **Every `fusion=concat` row in the current results is actually a subtraction run.**
2. `fusion="sub"` -> `abs(Subtract([feat_a, feat_b]))`, bit-identical to today.
   Keeps the existing baseline comparable.
3. `fusion="concat"` -> `Concatenate([feat_a, feat_b])`. **The new path the paper is about.**
4. `fusion="mcmaf"` -> existing gated-token block moved into the `elif`.
   Selectable, in no grid.
5. `substraction=` kept as a deprecated kwarg mapping to `fusion="sub"` with a
   warning (`train_fujita.py` and `archive/` still pass it).
6. **Dense block** (lines 118-127): emit **one** `Dense(256)->ReLU`, not
   `num_of_layer` copies; remove the `Dropout(0.5)` that follows the dense stack.
   *Why:* `num_of_layer=3` is currently reused for conv depth *and* dense depth,
   giving `Dense(256)x3`. Diagram 2 shows one "Dense Block(256)", no dropout after.
7. **`base_filters` stays one parameter** — no per-arm `_a`/`_b` split, since the
   capacity control shrinks *both* unshared towers equally. Pinned to **32** for
   every reported variant; the control arm uses **22**.
8. **Add `seed: int` to the constructor.** Every `GlorotUniform(seed=)`, every
   `Dropout(seed=)`, and `add_coral_head(seed=)` read it. Delete module-level
   `SEED = 1234` and `tf.random.set_seed(SEED)` (lines 23-24).
   *Why:* without this all 5 seeds initialise identically and Stage 2's std is
   meaningless. The most important fix in the plan.
9. `residual` stays a parameter, pinned `True` by config — it is the architecture
   in diagram 2, not an ablation axis.
10. **Each unshared tower gets its own seed** (`seed` and `seed + 1`).
    *Why:* `GlorotUniform(seed=s)` is deterministic for a given shape, so the
    old code built both "independent" towers from the same seed and they
    started from **identical weights**. They diverged during training only
    because their inputs differed. Pseudo-Siamese is supposed to mean
    independent parameters from initialisation onward, so this was a real
    (silent) bug in the unshared arm. Same fix applied to `models/fujita.py`,
    whose two towers had the same problem.

**Verified correct, unchanged:** the conv block (lines 88-110,
`Conv->BN->ReLU x3 -> MaxPool(2,2) -> Dropout(0.5)`) matches diagram 1 including
dropout placement; the per-arm `Concat([flat32, flat64, flat128, dense256])`
matches diagram 2 including "concatenation with output of each convolution block".

## `models/cmf_bdanet.py` — made ready, not run

1. `fusion` vocabulary extended to `{"sub","concat","mcmaf"}`, same three-way
   `if/elif/else` as FCSNN; validator at line 309 updated. Existing `mcmaf` /
   `concat` semantics preserved exactly.
2. `seed: int` threaded through `bdd_conv_block` -> `encoder_backbone` ->
   `MCMAFFusion.__init__` -> `build_cmf_bdanet_conv` -> `add_coral_head`.
   Module-level `SEED` / `tf.random.set_seed` (lines 36-37) deleted.
3. `MCMAFFusion.get_config()` added — it is a custom `Layer` with no config, so a
   saved `model.keras` currently fails to reload.
4. `base_filters` pinned to 32 for all variants.
5. `configs/cmf_bdanet.json` reshaped like the others (dataset path, seeds list,
   trimmed optimizers, `dsm_only`), and `train_cmf_bdanet.py` gains the `seed`
   grid key — so one command reruns it under the identical protocol when the
   other paper needs it.

**No architectural change. Not executed in Stages 0-3.**

## `models/bddnet.py` — seed threading only

Delete `SEED = 1234` / `tf.random.set_seed(SEED)` (lines 50-51).
`build_bddnet_model(..., seed=)` feeds `_init(seed)`, `Dropout(seed=)` (line 226),
`add_coral_head(seed=)` (line 229), and `make_dataset`'s `shuffle(seed=)` (line 284).
Architecture untouched.

## `models/fujita.py` — seed threading only

Delete `tensorflow.random.set_seed(1234)` (line 11). Add `seed` to `snn.__init__`.
The four `Conv2D`s and two `Dense`s carry **no `kernel_initializer`** today — they
get explicit `GlorotUniform(seed=...)`, plus `Dropout(seed=)`. Architecture untouched.

> Note for the paper: Fujita's `snn` already builds **two unshared towers and
> concatenates** them (lines 38-43). It is a pseudo-siamese-concat baseline, i.e.
> directly on-topic for the claim.

## `models/Moya.py`, `models/Hajeb.py` — no change

Deterministic feature extractors, no RNG.

## `models/ordinal.py` — no change

`add_coral_head(..., seed=)` already accepts a seed; callers start passing the run
seed instead of their module constant.

## `train_svm.py` — three changes

1. **Remove the balanced class weight.** Lines ~118-121 compute
   `class_weight.compute_class_weight("balanced", ...)` and pass it to `SVC`. Both
   the computation and the `class_weight=` argument are deleted (sklearn default
   `None`). *Expect the SVM baselines to drop on scenarios 1 and 2 relative to
   previously reported numbers.*
2. **Add `random_state=seed` to `SVC`.** The whole randomisation story. With
   `probability=False` and `predict()` running off the decision function, this
   will almost certainly produce five identical runs -> `std = 0`, reported as-is
   and noted in the table caption. No fabricated spread.
3. **Add a validation split.** `train_one` currently fits on train and goes
   straight to `Y_test` (lines 94-135) — there is no `feat_val` anywhere, so the
   SVM baseline is presently selected *on test*. It now takes `feat_val` /
   `Y_val`, scores validation with **macro-F2** (`compute_f2`, the same metric
   the NNs now monitor; `val_qwk` on scenario 5), and writes `best_val_metric` to
   `metrics.csv`. `seed` joins `build_result_dir`. `macro_f2` is already in its
   summary row.

## `utils/experiment.py` — infrastructure

1. **One result-dir layout for all stages**, so `is_already_trained` makes stage
   overlap free:
   `results/scenario_N/FCSNN/dataset_16/dsm_only/fusion_{f}/shared_{T/F}/bf{bf}/do_{T/F}/dense_{T/F}/{opt}_lr{lr}_bs{bs}/seed_{s}/`
   (`bf` is 32 everywhere except the control's 22; the segment stays so BDDNet
   shares the layout.)
2. `seed` becomes a first-class grid key — threaded into
   `set_tf_determinism(seed)`, the model constructor, and
   `make_siamese_dsm_dataset(..., seed)`.
3. `balanced_accuracy_score` added to `utils/metrics.py::compute_metrics`,
   persisted to `metrics.csv`.
4. **No class-weight or sample-weight code at all.**
5. **Validation metric switches from macro-F1 to macro-F2.** In
   `make_training_callbacks`, `score_fn` currently defaults to
   `compute_metrics(...)[3]` (macro-F1, line 276) and `monitor_name` to
   `"val_f1"` (line 248); both become `compute_f2(...)` and `"val_f2"`.
   `train_and_evaluate_nn`'s `monitor_name` default (line 366) follows. This one
   change moves checkpointing, early stopping, LR reduction, the per-epoch console
   line, and the `best_val_metric` that Stages 0-1 select on — they all read the
   same monitor.
6. Each `train_*.py` passes `monitor_name="val_qwk" if ordinal else "val_f2"`
   (currently `"val_f1"` — `train_fcsnn.py:170`, and the same line in
   `train_bddnet.py` / `train_fujita.py` / `train_cmf_bdanet.py`).
   Scenario 5 is unaffected: it keeps `val_qwk`.
7. `macro_f2` is **already** computed and written to `metrics.csv`
   (`compute_f2`, line 160; summary key `macro_f2`, line 458) — no change needed
   for saving it. `macro_f1` continues to be saved alongside, so both are
   reportable.

## `tools/aggregate_metrics.py`

`seed` added to `NON_GROUP_COLUMNS` so it collapses rather than fragmenting
groups; `balanced_accuracy` added. **`best_metric` default flips from `"f1"` to
`"f2"`** (the `--best-metric` CLI default, and
`aggregate_scenarios_after_run(..., best_metric=)` at `utils/experiment.py:558`),
so the automatic post-run aggregation ranks on the same metric the training
monitored. The `f2` branch of `_best_metric_series` already exists and reads
`macro_f2` directly.

## `configs/*.json` (all five)

`datasets.dataset_16 -> "Dataset/NPZ_spatial/dataset_16.npz"` ·
`early_stopping_patience: 12` (unchanged) · `epochs: 100` (unchanged) ·
new `seeds: [12345, 23451, 34512, 45123, 51234]` ·
`hpo_grid.optimizer -> ["Adam","Nadam","SGD"]` · `dsm_mode: ["dsm_only"]` ·
FCSNN `ablation_grid -> {fusion, shared, dropout, dense}`, `residual` pinned true,
`base_filters` pinned 32.

## New files

`tools/stage0_hpo.py` · `tools/stage1_ablation.py` · `tools/stage2_confirm.py` ·
`tools/stage3_report.py` · `tools/stats_compare.py` · `tests/test_fcsnn_shapes.py`

---

# Part 2 — Stages

## Stage 0 — per-cell HPO (validation only, seed 12345)

Reference architecture per cell: `dropout=on, dense=on, residual=on, bf=32`.
HPO grid = `{Adam, Nadam, SGD} x {1e-4, 1e-3} x {64, 128, 256}` = 18.

| Model | Cells | Runs/scenario | Total |
|---|---|---|---|
| FCSNN | 6 — `shared{T,F} x fusion{sub,concat}` at bf=32, plus `unshared x fusion{sub,concat}` at the control width bf=22 | 108 | 540 |
| BDDNet | 2 — `bf{16,32}` | 36 | 180 |
| Fujita | 1 | 18 | 90 |
| Moya (SVM) | its `kernel x dec_func x C x gamma` grid | 16 | 80 |
| Hajeb (SVM) | same grid | 16 | 80 |
| | | | **970** |

**Why the control is tuned here, at both fusions.** Stage 0 runs before Stage 1
has decided which fusion wins, so the control cannot be tuned for only the
fusion it will end up reported at — both are searched and the losing one's 90
runs are not reported. The alternative, letting the control inherit bf=32's
hyperparameters, leaves "the control was tuned for a wider model" as a live
alternative explanation for any gap it shows, which is exactly the objection
the control exists to close. A narrower network can genuinely prefer a
different learning rate or batch size.

> The SVM grid is **16** combos, not 8 as an earlier draft said: 2 kernels x 2
> decision functions x 2 C x 2 gamma. Note `gamma` is ignored by the linear
> kernel, so 4 of the 16 are duplicates of another 4; harmless, and cheaper to
> leave than to special-case.

Output `results/stage0_hpo.csv`: winning `(optimizer, lr, batch_size)` per
(scenario, cell), chosen by **validation macro-F2** (`val_qwk` on scenario 5).
Frozen thereafter. This is also the
baselines' real search — none runs at a single fixed setting.

## Stage 1 — ablation (validation only, seed 12345, HPO frozen)

- FCSNN: 16 configs at `bf=32`; the 4 cell references already exist from
  Stage 0 -> **12 new**.
- Select best per `(sharing x fusion)` cell by **validation macro-F2**
  (`val_qwk` on scenario 5) -> **4 reported arms**.
- BDDNet / Fujita / Moya / Hajeb: with class weighting gone no ablation axis
  remains — their Stage-0 winner *is* their arm. **0 new runs.**

### Capacity control (`bf=22`)

Winning fusion, `shared=False`, `base_filters=22` on both towers, `dropout x dense`
sub-grid. Its reference config (`dropout=on, dense=on`) already exists from
Stage 0, so **3 new runs/scenario**, yielding **1 control arm**. It uses the
hyperparameters Stage 0 tuned **at bf=22**, not bf=32's.

**Why 22 and not 16.** Conv parameters scale with `C_in x C_out`, so halving the
width *quarters* the conv stack rather than halving it. For one tower at 16x16x1
input, `num_of_layer=3`, residual + dense on:

```
P(bf) = 468*bf^2 + 4210*bf + 256
```

| Arm | `bf` | Params | vs. shared |
|---|---|---|---|
| Shared (1 tower) | 32 | ~614 k | 1.00x |
| Unshared (2 towers) | 32 | ~1.23 M | 2.00x |
| Unshared (2 towers) | **22** | **~639 k** | **1.04x** |
| Unshared (2 towers) | 16 | ~372 k | 0.61x |

Exact parity is `bf = 21.5`; `bf=21` gives 0.96x, `bf=22` gives 1.04x. **22** is
chosen so the control is not handicapped — if it still loses to the shared arm,
that cannot be blamed on it having fewer parameters.

(Hand-derived from the layer shapes. `tests/test_fcsnn_shapes.py` prints exact
counts; if they disagree, the test wins and `bf` is re-pinned before Stage 1 runs.)

**15 new/scenario -> 75 total.** Output `results/stage1_winners.csv`. Nothing reported.

## Stage 2 — confirmation (5 seeds, test set)

**Every FCSNN ablation config is multi-seeded, not just the cell winners** — the
ablation table is a reported table, so single-seed rows in it would reproduce the
exact weakness this protocol exists to fix.

Each arm retrained on all five seeds, `max_epochs=100 / patience=12`, each seed
stopping independently (Stage 1's epoch count is never reused). Test evaluated
once per seed.

| Model | Arms | Runs/scenario | New | Total new |
|---|---|---|---|---|
| FCSNN | 17 — 16 ablation + 1 control | 85 | 68 | 340 |
| BDDNet | 1 | 5 | 4 | 20 |
| Fujita | 1 | 5 | 4 | 20 |
| Moya (SVM) | 1 | 5 | 4 | 20 |
| Hajeb (SVM) | 1 | 5 | 4 | 20 |
| | | | | **420** |

Per arm: mean +/- std across seeds for accuracy, precision, recall, macro-F1,
macro-F2, balanced accuracy, MCC, and per-class precision/recall/F1
(+ QWK / MAE / off-by-one / linear kappa / Spearman on scenario 5). Bootstrap 95%
CI from 2000 test-set resamples pooled across seeds, read off `predictions.csv`.
Comparisons by paired bootstrap on seed-matched deltas, Holm-Bonferroni-corrected
across the comparison family. McNemar dropped.

> **Cheaper variant.** `--ablation-seeds 1` multi-seeds only the 4 cell winners +
> control and leaves the other 12 ablation configs at seed 12345. Saves 240 runs
> (total ~970 instead of ~1210), at the cost of an ablation table whose non-winning
> rows are single-seed and must be labelled as such.

## Stage 3 — aggregation (`tools/stage3_report.py`)

Two aggregates, **both produced per scenario**: one file per scenario plus a
combined file carrying a `scenario` column.

### Aggregate A — cross-model, per scenario

`results/paper/table_main_scenario_{1..5}.csv` and combined `table_main.csv`.
**8 rows per scenario:**

| Row | Arm |
|---|---|
| 1 | FCSNN shared + sub |
| 2 | FCSNN shared + concat |
| 3 | FCSNN unshared + sub |
| 4 | FCSNN unshared + concat |
| 5 | BDDNet |
| 6 | Fujita |
| 7 | Moya (SVM) |
| 8 | Hajeb (SVM) |

Each FCSNN row is that cell's Stage-1 validation winner, `bf=32`.

Columns: `scenario, model, arm, shared, fusion, bf, params, optimizer, lr,
batch_size, dropout, dense, n_seeds`, then for every metric a `mean`, `std`, a
paper-ready `mean +/- std` string, and `ci_low` / `ci_high`; then per-class
precision/recall/F1; then the summed confusion matrix. Also written as `.xlsx`.

### Aggregate B — FCSNN ablation, per scenario

`results/paper/table_ablation_scenario_{1..5}.csv` and combined
`table_ablation.csv`. **17 rows per scenario:** all 16 ablation configs plus the
`bf=22` capacity control flagged `control=True`.

Columns: `scenario, fusion, shared, dropout, dense, bf, params, control,
val_f2` (the Stage-1 validation score that drove selection; `val_qwk` on
scenario 5), then
the same mean / std / `mean +/- std` / CI block as Aggregate A, and `is_cell_winner`
marking which row fed Aggregate A.

Sorted `fusion, shared, dropout, dense` so the sub-vs-concat and shared-vs-unshared
blocks read directly off the table.

### Also written

- `results/paper/table_hpo.csv` — Stage-0 winning hyperparameters per
  (scenario, cell), for the reproducibility appendix
- `results/paper/significance.csv` — paired-bootstrap p-values, Holm-corrected,
  per scenario

### Choosing the comparison family (`--family`)

Family size costs power directly: Holm multiplies the smallest p-value by the
family size, and a paired bootstrap with B resamples cannot report a p below
`2/(B+1)` — 0.001 at the default B=2000.

- `--family architecture` (**default**): the four FCSNN arms pairwise (the
  sub-vs-concat and shared-vs-unshared contrasts the paper actually claims)
  plus the best FCSNN arm against each baseline. ~10 comparisons, so the
  smallest reachable Holm-adjusted p is ~0.010.
- `--family all`: every pair of the 8 arms, 28 comparisons, smallest reachable
  Holm-adjusted p ~0.028. Includes comparisons the paper never makes, such as
  Moya vs Hajeb. Use for an appendix, not the headline claim.

`significance.csv` carries `family`, `family_size` and `p_floor` columns so a
reader can tell "not significant" apart from "not enough resamples to show it".

A `--models` flag controls which arms appear in Aggregate A, so adding
CMF-BDANet later is a flag change, not a code change.

### What the capacity control buys

Unshared arms carry ~2x the parameters of shared arms, so an unshared win
otherwise confounds *architecture* (separate weights letting each tower specialise
on pre- vs post-event DSM) with *capacity* (twice the parameters). The `bf=22` row
holds parameters at ~1.04x the shared model, so a win that survives there is
attributable to weight sharing rather than budget.

Read it both ways: if `bf=22` unshared still beats shared, the architectural claim
holds; if it collapses to shared-level performance, the `bf=32` unshared advantage
was capacity — a finding worth reporting, not hiding.

The `sub`-vs-`concat` comparison is unaffected either way: those arms have
identical tower parameters and differ only in the fusion head.

The control is tuned at its own width in Stage 0, so "it was tuned for a wider
model" is not available as an explanation for its result.

Remaining limitation for Methods: parameter count is only one notion of
capacity — FLOPs differ, see `results/flops_curve.csv`.

### Total ~1465 distinct runs (970 + 75 + 420)

~970 with `--ablation-seeds 1`. Resumable throughout — a run is skipped once its
`metrics.csv` exists, and the single shared directory layout means Stages 1-3
inherit everything already trained.

---

# Build order

1. Phase A+B — model and infrastructure changes
2. `tests/test_fcsnn_shapes.py` green: all 3 fusion modes build; unshared `bf=32`
   has ~2x the parameters of shared `bf=32`; **unshared `bf=22` is within 10% of
   shared `bf=32`** (the control is only valid if this holds); `sub` and `concat`
   towers have identical parameter counts; `dropout` / `dense` flags change layer
   count. It **prints the exact parameter count per arm** — that is what the
   `params` column reports, and what re-pins `bf` if 22 turns out wrong.
   Plain `assert` + `__main__`.
3. Stage 0 -> inspect
4. Stage 1 -> inspect
5. Stage 2 -> inspect
6. Stage 3 + `stats_compare.py`

Each stage is a separate script, runnable overnight and inspectable before the next.

# Expect

- **Monitoring F2 instead of F1 shifts what gets selected.** F2 weights recall 4x
  precision, so early stopping, checkpointing and every Stage-0/1 selection will
  favour higher-recall, lower-precision models. That is the right bias for damage
  assessment — a missed damaged building costs more than a false alarm — and
  Methods should say so explicitly rather than leaving it as an unexplained choice.
  Both `macro_f1` and `macro_f2` are still saved and reported, so the F1 view of
  every arm remains available.
- Selected configs may differ from any earlier F1-selected run. Old results are
  not comparable on this axis.
- SVM baselines likely drop on scenarios 1-2 now that `class_weight="balanced"` is gone.
- SVM seeds will almost certainly be identical (`std = 0`).
- Old `results/` runs will not resume under the new layout, and are invalid
  regardless since their `concat` rows are subtractions.
