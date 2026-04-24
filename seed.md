# Seed notes

This tree is the **LLM baseline** for an experiment comparing an LLM-driven
optimizer against a standard metaheuristic optimizer (Optuna with TPE /
CMA-ES / Random samplers) on the same RouteE problem. For fairness, the
LLM should have access to the same *search space* context the optimizer has
access to — i.e. the candidate features, model families, and per-family
hyperparameters, but, as an agent, you should feel free to explore ideas outside of this
space if you think it would be improve the model

## GPU / PyTorch

This environment ships a CUDA build of PyTorch on `linux-64` (via the
`pytorch-gpu` conda-forge metapackage, pinned through `[system-requirements]
cuda = "12"` in `pixi.toml`).

Verified working on this host: Tesla P100-PCIE-12GB, driver 570.133.07,
CUDA runtime 12.9, cuDNN 9.10.2, compute capability 6.0.

## Candidate features (tabular pool)

All features below are derivable from the inference-time inputs described
in `domain.md` (average link speed, average grade, miles, geometry, and
previous-link context). No future-link info is used. Three are always
required (`speed_mph`, `grade_percent`, `miles`); the rest are optional.

| feature                  | definition                                               |
| ------------------------ | -------------------------------------------------------- |
| `speed_mph`              | link average speed (required)                            |
| `grade_percent`          | link average grade (required)                            |
| `miles`                  | link distance (required)                                 |
| `prev_speed_mph`         | speed on previous link within same journey               |
| `speed_delta`            | `speed_mph - prev_speed_mph`                             |
| `prev_miles`             | previous link distance                                   |
| `grade_delta`            | `grade_percent - prev_grade`                             |
| `prev2_speed_mph`        | speed 2 links back                                       |
| `prev3_speed_mph`        | speed 3 links back                                       |
| `prev4_speed_mph`        | speed 4 links back                                       |
| `prev5_speed_mph`        | speed 5 links back                                       |
| `time_seconds`           | link traversal time                                      |
| `prev_time_seconds`      | previous link traversal time                             |
| `abs_bearing_delta`      | `\|heading_delta\|` between previous and current link    |
| `prev_abs_bearing_delta` | previous link's abs bearing delta                        |
| `speed_accel`            | `speed_delta - prev_speed_delta` (approx. jerk)          |

### Sequential feature pool (for CNN / GRU)

Per-timestep features stacked into a look-back window:
`speed_mph, grade_percent, miles, time_seconds, sinuosity, abs_bearing_delta`.
Static (per-link, appended after the recurrent/conv layers):
`link_position`.

Windows are built per-journey; journeys shorter than `seq_len` are dropped.
Sequential models split **by `journey_id`** (not row-level) so sequence
windows never leak across train/test.

## Model families in the optimizer search

Eight families total, split into tabular and sequential:

**Tabular** (row-level 80/20 split via `fixed_utils.train_test_split`):

- `rf` — `RandomForestRegressor`
- `extra_trees` — `ExtraTreesRegressor`
- `hgbr` — `HistGradientBoostingRegressor`
- `xgb` — `XGBRegressor` (`tree_method="hist"`)
- `lgbm` — `LGBMRegressor`
- `mlp` — `MLPRegressor` with `StandardScaler` on inputs

**Sequential** (journey-level 80/20 split; PyTorch; GPU-enabled):

- `cnn` — 1-D Conv stack (`seq_len` in [3,7], 2–4 conv layers,
  channels ∈ {64,128,256}, kernel ∈ {3,5}) + dense head (256→128→1),
  AdamW + CosineAnnealingLR, grad clipping, `GroupNorm(1, C)`.
- `gru` — `nn.GRU` (hidden ∈ {64,128,256}, 1–3 layers) + dense head
  (128→1), AdamW + CosineAnnealingLR, grad clipping.

## Per-family hyperparameter ranges

These are what TPE/CMA-ES/Random sample over:

- **rf / extra_trees**: `n_estimators` 100–2000 (step 100),
  `max_depth ∈ {None,10,20,30,40}`, `min_samples_split` 2–20,
  `max_features` 0.3–1.0, `max_samples` 0.3–0.8.
- **hgbr**: `max_iter` 100–1000 (step 50), `max_depth` 3–15,
  `lr` 0.01–0.3 (log), `l2_regularization` 1e-4–1.0 (log),
  `max_leaf_nodes` 15–255, `min_samples_leaf` 5–50.
- **xgb**: `n_estimators` 100–2000 (step 100), `max_depth` 3–12,
  `lr` 0.01–0.3 (log), `subsample` 0.5–1.0, `colsample_bytree` 0.5–1.0,
  `reg_alpha`, `reg_lambda` 1e-4–10.0 (log).
- **lgbm**: same ranges as xgb.
- **mlp**: `n_layers` 1–4, `layer_size` 64–512 (step 64), `activation ∈ {relu,tanh}`,
  `alpha` 1e-5–0.1 (log), `lr_init` 1e-4–1e-2 (log),
  `batch_size ∈ {256,512,1024,2048}`. `early_stopping=True`, `max_iter=1000`.
- **cnn**: `seq_len` 3–7, `n_layers` 2–4, `channels ∈ {64,128,256}`,
  `kernel_size ∈ {3,5}`, `dropout` 0.0–0.3, `lr` 5e-4–5e-3 (log),
  `batch_size ∈ {1024,2048,4096}`, `weight_decay` 1e-5–1e-2 (log),
  `grad_clip` 0.5–2.0.
- **gru**: `seq_len` 3–7, `hidden_size ∈ {64,128,256}`, `n_layers` 1–3,
  `dropout` 0.0–0.3, `lr`, `batch_size`, `weight_decay`, `grad_clip`
  as per cnn.

The optimizer also runs a **two-phase** variant on demand: phase 1 tunes
family + HPs with all features fixed on; phase 2 fixes the phase-1 winner
and ablates features. The LLM loop does not need to mirror this — it's
free to interleave architectural and feature changes — but it's useful to
know that feature-vs-HP is a deliberate axis the optimizer separates.

