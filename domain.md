# Domain

## Context

Our training data represents simulated vehicle runs over drive cycle traces (typically 1hz).
At each point we simulate the vehicle dynamics and get an energy estimation for that point.
We take these point level results and aggregate them up to the trip/road segment level.
Then, the road segments have attributes like total distance, average speed, average road gradiant, time to traverse, etc.

We model three **powertrain types**, each with its own characteristic energy behavior:

- **BEV (battery electric)** — e.g. 2017 Chevy Bolt. Link energy can be **negative** (regenerative braking). Asymmetric, heavy-tailed target distribution.
- **ICE / Conventional (internal combustion)** — e.g. 2016 Toyota Camry. Link energy is **always non-negative** (fuel consumption only, no regen).
- **PHEV (plug-in hybrid)** — has both a battery and a fuel tank. Link energy is reported in GGE for both fuel sources independently (both converted to gasoline-gallon-equivalent).

Each experiment session targets **exactly one powertrain**. See `## Session Partitioning` below for how branches, results, tags, and `learnings.md` are namespaced, and `program.md` for the generic session protocol that uses that partitioning.

### Model Inference Environment

Note that when we're applying these models for inference, we only have limited data (which is why we're developing these models in the first place).
Our inference environment has the following features:

- Average Speed: The speed is given as an average over the link, either from assuming the speed driven is the posted speed, or, using probe data to gather average speeds.
- Average Road Gradient: The gradient is an average over the link, either from taking the elevation difference of the link endpoints, or, using probe data to sample the gradient at subpoints on a link.
- Link Distance: The distance of the link
  Think about the inference environment as applying these models during a shortest path search in Google Maps where we only have limited information.
  If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.
- Geometry: The link geometry in the well known binary format using the 4326 CRS (latitude and longitude points)

## Constraints

Do not include any features that we do not have in our model inference environment.
For example, we do not have acceleration based data when doing model inference and so we do not want our model trained on acceleration data.
That being said, you could consider novel features like the speed on the previous link or average_speed^2.

If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.

Do not filter or remove data points to reduce error. The model must be able to predict all values in the dataset, including extreme energy rates such as heavy regenerative braking. Filtering outliers artificially lowers RMSE without improving the model's actual predictive capability — we need accurate predictions across the full distribution.

Do not include a feature like link position since at inference time, we will not know the position of a link relative to a whole trajectory.

## Session Partitioning

Research on this domain is partitioned along one axis so that lines of inquiry for each partition do not collide. `program.md`'s generic `<variant>` placeholder resolves to this domain's axis value.

- **Axis name**: `powertrain`
- **Valid values**: `bev` (2017 Chevy Bolt), `ice` (2016 Toyota Camry), `phev` (TBD)
- **Session tag format**: `<powertrain>-<date>` (e.g. `bev-apr17`, `ice-mar5`)
- **Branch name**: `routee-autoresearch/<powertrain>-<date>`
- **Results subdirectory**: `results/<powertrain>/` (so a session's files are `results/<powertrain>/results-<date>.tsv`, `experiments-<date>.jsonl`, `exp-timing-<date>.log`)
- **Persistent cross-session best tag** (Tier 4 in `program.md`): `<powertrain>/best` — e.g. `bev/best`, `ice/best`, `phev/best`
- **`learnings.md` structure**: top-level `Cross-cutting insights` section (pipeline/optimizer/data-representation truths that apply to every powertrain), then one section per powertrain (`BEV (2017 Chevy Bolt)`, `ICE (2016 Toyota Camry)`, `PHEV`), each with What works / What doesn't / Best known config / Open questions.
- **`train.py` selector**: the `POWERTRAIN` constant at the top of `train.py` picks the active partition.
- **Seed ancestor for a new partition's first session**: if `<powertrain>/best` exists, seed `train.py` from that. Otherwise, seed from `bev/best` (inherit the architectural lessons from the earliest-explored powertrain). If neither exists, use `main`.
- **Forks stay within a partition** — do not fork a `bev-*` session from an `ice-*` tag or vice versa; cross-partition lessons flow through `learnings.md → Cross-cutting insights`, not through shared branches.
