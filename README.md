# autoresearch

Template for running **autonomous research experiments** that iteratively
improve an ML model for a single optimization objective. Two execution
modes share one harness:

- **LLM mode** — an agent (e.g. Claude Code) edits a scaffold `train.py`
  one change at a time, tagging each experiment, logging reasoning, and
  pushing results. Defined by `program.md`.
- **Optimizer mode** — an Optuna-backed driver (TPE / CMA-ES / Random)
  iterates over a domain-defined search space. Defined by `optimizers/`.

RouteE (vehicle energy prediction) is the reference domain under
`domains/routee/`. Adding a new domain is mechanical — see `EXTENDING.md`.

## Repo layout

```
program.md              LLM experiment protocol (domain-agnostic)
fixed_utils.py          Shared evaluation harness (train/test split + metric)
EXTENDING.md            How to add a domain or an optimizer
tools/                  Tree creation + harness sync
  new_experiment_tree.sh
  sync_harness.sh
  README-trees.md
optimizers/             Pluggable Optuna-backed samplers
  common/               Shared driver, CLI, logging, objective
  tpe/  cmaes/  random/ Per-method entry points
domains/                Domain implementations
  routee/               Reference domain
    domain.md, domain.json, train.py, learnings.md, seed.md
    data/  results/
    search/             Domain hooks for optimizer mode
```

## Quickstart

### Isolated experiment trees

Every run happens in a fresh git repo (a "tree") so the agent or optimizer
cannot see prior sessions via `git log --all`, `git tag -l`, or
accumulated `learnings.md`. Trees are created by `tools/new_experiment_tree.sh`.

Create an LLM tree for BEV:

```bash
tools/new_experiment_tree.sh \
    --name routee-bev-01 \
    --domain routee \
    --mode llm \
    --partition bev
```

Create an optimizer tree for BEV using TPE:

```bash
tools/new_experiment_tree.sh \
    --name routee-bev-tpe-01 \
    --domain routee \
    --mode optimizer \
    --optimizer tpe \
    --partition bev
```

Trees land under `~/repos/routee-autoresearch-trees/<name>/`. The registry
at `~/repos/routee-autoresearch-trees/registry.jsonl` logs provenance for
each tree.

### Running inside a tree

LLM mode — kick off an agent against `program.md`:

```bash
cd ~/repos/routee-autoresearch-trees/routee-bev-01
claude --dangerously-skip-permissions
# then: "Have a look at program.md and let's kick off a new experiment session"
```

Optimizer mode — run the chosen sampler:

```bash
cd ~/repos/routee-autoresearch-trees/routee-bev-tpe-01
pixi run python -m optimizers.tpe.search \
    --tag bev-apr23 \
    --partition bev \
    --n-trials 200 \
    --budget 300
```

Both modes write TSV + JSONL under `results/<partition>/` using the same
schema, so an LLM session and an optimizer session on the same partition
can be compared directly.

### Syncing a harness fix into a live tree

Rare but occasionally needed. Only the framework + domain scaffold files
are synced; tree-owned state (`train.py`, `learnings.md`, `seed.md`,
`results/`, `plans/`) is untouched.

```bash
tools/sync_harness.sh --tree ~/repos/routee-autoresearch-trees/routee-bev-01
```

## Docker

The repo ships a sandboxed Docker image with Claude Code, pixi, and the
full environment pre-installed. See `Dockerfile` and the build/run
commands below.

```bash
docker build --build-arg GIT_TOKEN=your_token_here -t autoresearch .
docker run -it --pids-limit 256 --memory 8g autoresearch
```

Sandbox: 8 GB memory, 256 PIDs max, no host mounts, dropped Linux
capabilities, non-root `researcher` user.

# Acknowledgments
 
This software is built on the "autoresearch" software by github user karpathy available here [link](https://github.com/karpathy/autoresearch) and distributed under the MIT license.

# Metadata

NLR Software Record # SWR 26-090.