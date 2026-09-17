# Audit

Scores both arms' final models the same way, outside either experiment tree.

Each arm reported its own numbers. Those numbers are honest, but the agent
produced them with its own code, on the features it chose. It could not check
whether those features will exist when the model is deployed. This audit checks
that. It answers two questions:

1. **What is the link RMSE when the model only gets inputs RouteE Compass can
   supply?** Compass scores one road link at a time during a route search. It
   knows the current link and the one before it. It does not know which links
   come next, and it has no energy labels.
2. **How long does one prediction take?** A route search calls the model
   millions of times, on one CPU thread.

The numbers are in [`results/report.md`](results/report.md).

## Result in short

The domain-guided model can run in Compass as is, because every input it uses
is available there. The unguided model cannot run in Compass at all: 8 of its
13 features, its sequence model and its journey offset all need things a route
search does not have. Refit on usable inputs only, the unguided recipe is still
worse than the domain-guided model on link RMSE, and about 650 times slower per
link.

## How it works

- `harness.py` is a byte-for-byte copy of the file both arms used. It holds the
  train/test split and the RMSE metric. Check with:

  ```bash
  sha256sum harness.py ../unguided/harness.py ../domain-guided/harness.py
  ```

- `features.py` and `models.py` are copies of each arm's feature code and model
  settings. Nothing is imported from the trees. The audit first reproduces each
  arm's reported number; if that failed, the copies would be wrong.
- `contract.py` lists every feature either arm used and says whether Compass
  can supply it, in one line each.
- `audit_accuracy.py` scores three things: the unguided model as run, the same
  recipe refit on usable inputs, and the domain-guided model.
- `audit_inference.py` times each model per link on one CPU thread.
- `plot.py` draws the two figures in `../img/` from the results.
- `run_audit.py` runs both audits, writes the report, and draws the figures.

The audit reads both arms, so it must not live inside either tree. Do not
symlink it into a tree or add it to a tree's path.

## Running it

```bash
cd audit
pixi install
pixi run python run_audit.py
```

The first run fits the models and takes about half an hour on a GPU host. After
that everything is cached in `cache/`, and re-running is fast.

```bash
pixi run python run_audit.py --accuracy    # link RMSE only
pixi run python run_audit.py --inference   # inference time only (needs cached models)
pixi run python run_audit.py --report      # re-render report.md and figures from cached json
```

## Limits

- **The split is random by row**, so test links share journeys with training
  links. Both arms flagged this. It flatters any feature that averages over a
  journey. A split by journey would be a separate study.
- **The refit is the audit's work, not the arm's.** An agent run under the rules
  might have found a better model. Read the refit as a floor, not a ceiling.
- **The unguided model is not exactly reproducible.** Its sequence model trains
  for 60 seconds of wall clock, so a different host fits a slightly different
  model. The audit allows 5% for that arm and 2% for the deterministic one.
- **One session per arm.** This describes two runs, not a population.
