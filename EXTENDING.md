# Extending

How to add a new domain or a new optimizer method.

## Add a domain

A domain is a self-contained directory under `domains/<name>/` that
describes one modeling problem. RouteE (`domains/routee/`) is the
reference implementation — copy its structure when adding a new one.

### Minimum for LLM mode

```
domains/<name>/
  __init__.py       empty
  domain.md         human-authored context + constraints
  domain.json       machine-readable partitioning + data config (schema below)
  train.py          scaffold the LLM edits; must declare a partition-selector constant
  seed.md           empty (per-session hints go here at tree creation)
  learnings.md      empty (cross-session knowledge accretes here on main)
  README.md         optional — notes for human maintainers
  data/             parquets or whatever the domain reads
  results/<p>/      one .gitkeep per partition value
```

### `domain.json` schema

```json
{
  "name": "<domain_name>",
  "partition": {
    "axis": "<name>",
    "values": ["<v1>", "<v2>"],
    "train_py_selector": "<NAME>"
  },
  "tag_format": "{partition}-{date}",
  "results_subdir_template": "results/{partition}",
  "data": {
    "path_template": "data/processed/{asset}.parquet",
    "assets_by_partition": {
      "<v1>": "<asset_name>",
      "<v2>": "<asset_name>"
    }
  },
  "seed_ancestor_fallback": "<v1>"
}
```

JSON instead of YAML so the tree-creation script can parse it with the
Python stdlib — no pyyaml dependency on the host running the script.

`tools/new_experiment_tree.sh` reads this file to validate `--partition`
against the declared values and to stamp the right constant in the
tree's `train.py`.

### train.py contract

- Has one constant with the name declared in `domain.json →
  partition.train_py_selector`. The tree creation script rewrites its
  string value to the chosen partition.
- Reads the dataset from `data/processed/<asset>.parquet` (path template
  from `domain.json`). In a tree, `data/` lives at the tree root; in the
  template, under `domains/<name>/data/`.
- Reads shared evaluation helpers from `fixed_utils`:
  `train_test_split` and a metric (default: `evaluate()` returning
  `{"rmse": ...}`). If the domain needs a different metric, replace
  `fixed_utils.py` or have `train.py` call its own.
- Prints its result to stdout as `metric_name: value` (so `grep` + log
  parsing in `program.md` works).

### Minimum for optimizer mode

Add `domains/<name>/search/` with this contract:

```python
# domains/<name>/search/__init__.py
from .data import default_data_path              # optional
from .feature_pipeline import load_and_engineer  # required
from .models import run_trial                    # required
from .search_space import sample_config          # required
from .search_space import sample_config_phase1   # optional (phase 1 search)
from .search_space import sample_config_phase2   # optional (phase 2 ablation)
from .search_space import ALL_FAMILIES           # optional (validates --families)
from .search_space import WARM_START_CONFIGS     # optional (seed trials from priors)
```

Required callables:

- `sample_config(trial: optuna.Trial, families: list[str] | None = None) -> dict`
- `load_and_engineer(data_path: str) -> pd.DataFrame`
- `run_trial(config: dict, df: pd.DataFrame, budget_seconds: float, trial=None) -> float`
  — returns the metric value (lower = better).

Optional but recommended:

- `default_data_path(partition: str) -> pathlib.Path` — lets the driver
  resolve the dataset from `--partition` without a hardcoded `--data-path`.

No ABC, no registry — the driver duck-imports from
`domains.<name>.search`. See `optimizers/common/domain_loader.py`.

### Running trees for a new domain

```bash
tools/new_experiment_tree.sh --name foo-01 --domain foo --mode llm --partition <p1>
tools/new_experiment_tree.sh --name foo-tpe-01 --domain foo --mode optimizer --optimizer tpe --partition <p1>
```

## Add an optimizer method

Each method is a tiny directory under `optimizers/<method>/` with two
files:

```
optimizers/<method>/
  __init__.py     empty
  sampler.py      defines build(seed=42) -> optuna.samplers.BaseSampler
  search.py       CLI entry point — parses args, builds sampler, calls common.driver.run()
```

Template `search.py`:

```python
from optimizers.common.cli import build_parser
from optimizers.common.driver import run
from optimizers.<method>.sampler import build


def main() -> None:
    args = build_parser("<method>").parse_args()
    run("<method>", build(), args)


if __name__ == "__main__":
    main()
```

If the new method is not Optuna-backed (say, Ray Tune or Ax), swap the
`sampler.py` convention for a method-appropriate constructor and extend
`optimizers/common/driver.py` — but keep the TSV/JSONL/timing output
schema so the results are comparable across methods.

Then wire the method into `tools/new_experiment_tree.sh` by adding it to
the `--optimizer` regex validation.
