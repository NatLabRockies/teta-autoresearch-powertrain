# Autoresearch

This is an experiment to research better ML models. One session runs many experiments against a
single scaffold, `train.py`, keeping the changes that help and reverting the ones that don't.

Throughout this document `<tag>` is the session tag agreed at setup (e.g. `bev-may8`).

The `domain.md` defines the problem and the domain context — read it first and treat it as binding.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose something short and dated, e.g. `bev-may8`.
1. **Read the in-scope files**: The repo is small. Read these for full context:
   - `domain.md` — problem definition, constraints, etc. Do not modify.
   - `learnings.md` — accumulated findings from previous sessions. Use it to avoid repeating
     dead ends and to build on what already works.
   - `harness.py` — the fixed point: the time budget, the data split, the evaluation metrics,
     and the result printer. Do not modify.
   - `train.py` — the only file you edit.
1. **Start from `main`'s `train.py` as-is**: it already _is_ the best known configuration. Every
   experiment that did not improve on it was reverted before the next one ran, so `HEAD` always
   carries the best result any session in this tree has produced.
1. **Initialize results files**, with just their headers or empty:
   - `results/results-<tag>.tsv` — header row only
   - `results/experiments-<tag>.jsonl` — empty file
1. **Create a session plan**: Create a plan for the session `plans/plan-<tag>.md` using the template at
   `templates/plan-template.md`. Spend time reviewing `learnings.md` first, then fill in session
   goals, planned experiments, and constraints.
1. **Run the baseline**: Run `train.py` as-is, commit it as `exp0: baseline`, and record the
   result under experiment 0 in both results files.
1. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Metrics: the contract

`harness.evaluate()` returns a dict of named metrics. **That dict is the single source of
truth for what the metrics are**.

### Run metadata

`evaluate()`'s keys define the metrics; `harness.report()` defines everything else worth
recording about a run. It prints a second machine-readable line, `meta: {...}`, carrying the
model family and the feature list. Like the metric names, these are never hardcoded in this
protocol — `train.py` passes them to `report()`:

- `model_family` — a coarse label for the kind of model: `RandomForest`, `MLP`, `CNN`, `GRU`,
  `Linear`. Free text, held in `MODEL_FAMILY`. It exists so results can be grouped afterwards,
  so name the same kind of model the same way every time, and update it whenever an experiment
  changes what kind of model is being fit — a mislabeled row is worse than no label.
- `features` — the columns the model consumes.

Both land in the TSV and the JSONL.

## Experimentation

The training script runs for a **fixed time budget of 10 minutes** (wall clock training time,
excluding startup/compilation), set by `TIME_BUDGET_SECONDS` in `harness.py` and enforced by
killing the run that outlasts it. You launch it as: `pixi run python train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Model
  architecture, optimizer, hyperparameters, features.
  The only fixed obligation is to
  keep scoring with `harness.evaluate()` and reporting with `harness.report()`.

**What you CANNOT do:**

- Modify `harness.py` or `domain.md`. They are read-only. `harness.py` is the fixed point: the
  time budget, the data split, the ground-truth metrics, and the report format.
- Change the time budget. It lives in `harness.py` for that reason, and `train.py` calls
  `run_with_budget(train_model)` with no number of its own.
- Change how the model is scored, or which rows it is scored on. Filtering, reweighting, or
  dropping data before the split changes the exam rather than the model, and any improvement it
  produces is not real.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds
ugly complexity is not worth it. Conversely, removing something and getting equal or better
results is a great outcome — that's a simplification win. When evaluating whether to keep a
change, weigh the complexity cost against the improvement magnitude. A tiny improvement that
adds 20 lines of hacky code? Probably not worth it. The same improvement from deleting code?
Definitely keep. Metrics tied but the code is much simpler? Keep.

**Atomic changes**: Make sure that every experiment only includes one atomic change that can be
pointed to as a cause of the resulting model improvement. For example, if you decide to add a
new feature, only add a single feature and see how the model reacts rather than adding two
features since we won't know which feature resulted in the improvement.

**Stay in budget**: a run that exceeds the time budget is a `crash`, not a slow `keep`. Do not
keep a change that only fits by going over.

## Output Format

Once the script finishes it prints a summary like this:

```
metrics: {"rmse": 0.03959, "trip_rmse": 0.421}
meta: {"model_family": "RandomForest", "features": ["speed_mph", "grade_percent"], "total_seconds": 9.4}
```

Two lines, both parse targets, each fact appearing exactly once. `metrics:` contains exactly what
`evaluate()` returned. `meta:` contains everything about the run that is not a metric.

## Results TSV Format

Tab-separated, NOT comma-separated. Columns are the commit and the model family, then one column
per metric in `evaluate()` order, then status, features, and description:

```
commit	model_family	rmse	trip_rmse	status	features	description
a1b2c3d	RandomForest	0.039590	0.421000	keep	speed_mph,grade_percent	baseline
b2c3d4e	RandomForest	0.035200	0.395000	keep	speed_mph,grade_percent	increase LR to 0.04 (both improved)
c3d4e5f	MLP	0.034800	0.402000	discard	speed_mph,grade_percent	swap to a 2-layer MLP (rmse better, trip_rmse worse)
d4e5f6g	MLP	0.000000	0.000000	crash	speed_mph,grade_percent	double model width (OOM)
```

1. git commit hash (short, 7 chars)
2. `model_family` from the `meta:` line — `-` if the run crashed before printing it
3. one column per metric — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. `features` from the `meta:` line, comma-joined with no spaces — `-` if the run crashed before
   printing it
6. short text description of what this experiment tried

## Experiment Reasoning (JSONL)

For every experiment, append one JSON line to `results/experiments-<tag>.jsonl`. This is the
structured reasoning record that captures _why_ experiments were tried and what was learned.

**Format** (one line per experiment, no pretty-printing):

```json
{
  "exp": 15,
  "commit": "5959293",
  "parent_best": "0271671",
  "status": "keep",
  "started_at": "2026-05-08T19:16:32Z",
  "ended_at": "2026-05-08T19:24:59Z",
  "model_family": "MLP",
  "features": ["speed_mph", "grade_percent", "miles"],
  "metrics": { "rmse": 0.0130, "trip_rmse": 0.395 },
  "best_before": { "rmse": 0.013419, "trip_rmse": 0.421 },
  "delta_pct": { "rmse": -3.1, "trip_rmse": -6.2 },
  "description": "widen the hidden layer 32 -> 64",
  "hypothesis": "the model is under-capacity; the training loss is still falling when the budget ends",
  "observation": "both metrics improved, and trip_rmse moved considerably further than rmse",
  "reasoning": "extra capacity helped, and the lopsided movement suggests the two metrics are limited by different things — worth probing next",
  "tags": ["architecture"]
}
```

**Fields:**

- `exp`: experiment number (int)
- `commit`: short hash of the experiment commit (string)
- `parent_best`: short hash of the current best commit before this experiment (string)
- `status`: `keep`, `discard`, or `crash` (string)
- `started_at` / `ended_at`: UTC timestamps, `date -u +%Y-%m-%dT%H:%M:%SZ`
- `model_family`: from the `meta:` line (string; `null` if the run crashed before printing it)
- `features`: from the `meta:` line, as a JSON array — not comma-joined, this file is already
  JSON (`null` for the same reason)
- `metrics`: every metric `evaluate()` returned (use `0.0` for crashes)
- `best_before`: the same keys, at the current best before this experiment
- `delta_pct`: per-metric percent change from best (`-5.0` means 5% improvement). `null` for
  crashes
- `description`: what was changed (same as TSV, 1 line)
- `hypothesis`: what you expect to happen and why — write BEFORE running (1-2 sentences)
- `observation`: what actually happened, across ALL metrics — write AFTER running (1-2 sentences)
- `reasoning`: why it worked/failed and what it teaches for future experiments (1-2 sentences).
  If metrics disagreed, say which dynamic you think caused the split
- `tags`: category labels like `feature-engineering`, `hypertuning`, `architecture`,
  `data-representation`, `breakthrough`, `dead-end`

Keep hypothesis/observation/reasoning concise — 1-2 sentences each. This is a decision log, not a
paper.

## Git

Everything is commits on `main`. 

- One commit per experiment, message `expN: <short description>`, made _before_ the run.
- Follow-up commits log the outcome: `log expN: <result summary>` or `revert expN: <reason>`.
- **The current best is the `commit` of the most recent `keep` row in the results TSV.** The
  JSONL records it as `parent_best` on every experiment. Because a non-improvement is reverted
  before the next experiment starts, `main`'s `train.py` is always that commit's `train.py`.

## The experiment loop

**Session limits**: A session ends when either **50 experiments** have been run or **8 hours of
wall-clock time** have elapsed since the session started (setup time excluded), whichever comes
first. When the limit is reached, do the final learnings update, record token usage, capture the
transcript, and stop.

Loop until the session limit is reached:

1. **Determine experiment number N** and note the current best commit (the most recent `keep` row
   in the TSV)
2. **Form hypothesis**: Before editing code, decide what you're testing and why. It goes into the
   JSONL, and writing it first is what makes the result interpretable.
3. **Edit `train.py`** with one atomic experimental change
4. **Commit**: `git commit -m "expN: <short description>"`
5. **Record the start time**: `date -u +%Y-%m-%dT%H:%M:%SZ`
6. **Run**: `pixi run python train.py > run.log 2>&1` (redirect everything — do NOT use tee or
   let output flood your context)
7. **Record the end time**. Record it accurately: `ended_at` is also what attributes this
   experiment's share of the session's token cost (see below).
8. **Parse results**: `grep -E "^(metrics|meta): " run.log`. If the `metrics:` line is missing,
   the run crashed — `tail -n 50 run.log` to read the stack trace.
9. **Record results**: Append to both the TSV and the JSONL (with hypothesis, observation,
   reasoning, and the timestamps from steps 5 and 7)
10. **If the result counts as better under `domain.md`:**
    - Commit the updated results files: `git commit -m "log expN: <result summary>"`
    - `main` now carries the new best; nothing else to update
11. **Otherwise, or if the run crashed:**
    - Restore train.py from the best commit: `git checkout <best-commit> -- train.py`, using the
      hash you recorded as `parent_best` for this experiment
    - Commit the revert + updated results files: `git commit -m "revert expN: <short reason>"`
    - Note in the JSONL `observation` which metric regressed and by how much — that's the signal
      for the next hypothesis
12. **Update plan**: Add a progress line to `plans/plan-<tag>.md`
13. **Periodic checkpoint** (~every 20 experiments): update `learnings.md` with new findings, then
    record usage and capture the transcript (see below), and commit

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: if it's something dumb
and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is
fundamentally broken, skip it, log `crash` as the status, and move on.

**Keep going autonomously**: Do not pause to ask the human whether to continue. If you run out of
ideas before the session limit, think harder — re-read `learnings.md` and `domain.md` for new
angles, try combining previous near-misses, or try more radical architectural changes. 
You can also consider reviewing the latest literature to discover ideas about the state of the art.

## Cross-session learnings

`learnings.md` accumulates insights across sessions: what works, what doesn't, the best known
configuration, and open hypotheses.

**At session start:** Read it. Use it to avoid repeating known dead ends and to build on proven
approaches.

**During the session (~every 10 experiments):** Update it with new findings and commit. 

**At session end:** Do a final learnings update before stopping.

## Token cost and transcripts

Run both of these at each periodic checkpoint and again at session end, then commit what they
produce:

```
pixi run python tools/token_usage.py --tag <tag>
pixi run python tools/capture_transcript.py --tag <tag>
git add results/ && git commit -m "capture usage and transcripts through expN"
```

`token_usage.py` writes two files. `results/usage-<tag>.jsonl` is the session view: it appends
a cumulative per-model total, so do not sum its lines — take the latest `snapshot_at` per
model. `results/usage-by-exp-<tag>.jsonl` is the per-experiment view: it is overwritten each
run and holds one line per experiment per model, each covering only that experiment's own window,
so those lines _are_ meant to be summed. Attribution comes from the `ended_at` stamps in
`results/experiments-<tag>.jsonl` — an experiment is charged for everything between the previous
experiment's end and its own, which includes the thinking that produced it.

`capture_transcript.py` copies the raw agent transcripts to `results/transcript-<tag>/` and
writes `results/transcript-audit-<tag>.md`: tool-call counts, the full text of every operator
prompt, and every path referenced outside the tree. 

