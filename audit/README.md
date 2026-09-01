# Audit

Independent scoring of both arms' final models, run **outside** either experiment tree.

Every number in `unguided/` and `domain-guided/` is a *reported* number: produced by the
agent's own harness, on the features it chose, scored by the metric it was given. Those
numbers are honest — neither arm faked anything — but they are self-referential. An agent
optimising a metric on a dataset cannot check whether the inputs it used will exist when
the model is deployed, because that fact is not in the dataset.

This directory supplies the missing check. It answers two questions:

1. **Does the model still work when only Compass-available inputs are supplied?**
2. **Can Compass afford to run it at every link traversal?**

## The finding in one line

The unguided arm's reported lead rests on inputs RouteE Compass cannot supply — and this time
on two whole model components as well as on features: scored under the contract it lands at
0.014050 / 0.004096, worse than the baseline it started from, and even a fair refit inside the
contract loses to the domain-guided model on both metrics while costing 657× more per link. See
[`results/report.md`](results/report.md).

## Why this lives here and not in a tree

A tree must stay an independent sample. This audit reads *both* arms, so it cannot sit
inside either one without giving that arm sight of the other. It lives in the archive,
runs against the same dataset through the same style of symlink, and imports nothing from
either tree — the model definitions in `models.py` and the feature code in `features.py`
are transcriptions, checked by the reproduction test rather than by import.

The one file copied rather than transcribed is `harness.py`, byte-identical to the copy
in each arm (sha256 `feaa278d…`). That is what makes "same held-out rows, same metric"
checkable with one command instead of asserted in prose:

```bash
sha256sum harness.py ../unguided/harness.py ../domain-guided/harness.py
```

Do not symlink this directory into a tree, and do not add it to a tree's path.

## Layout

```
contract.py          domain.md restated as a per-feature ledger — the claim under test
features.py          feature construction: both arms as-run, plus causal substitutes
models.py            both arms' final models, transcribed hyperparameter for hyperparameter
harness.py           verbatim copy of the arms' fixed split and metrics
audit_accuracy.py    reported vs audited scoring
audit_inference.py   per-link inference cost, single-threaded CPU
run_audit.py         runs both, renders results/report.md
results/             report.md, accuracy.json, inference.json (run.log is local)
cache/               trained models (gitignored) — delete to force a refit
```

## Running it

```bash
cd audit
pixi install
pixi run python run_audit.py
```

First run fits four 5-member GBDT ensembles, two CNNs and a 2-member MLP, and takes on
the order of half an hour on a GPU host; everything is cached afterwards, so re-running
or re-rendering the report is fast.

```bash
pixi run python run_audit.py --accuracy    # accuracy only
pixi run python run_audit.py --inference   # inference only (needs cached models)
pixi run python run_audit.py --report      # re-render report.md from cached json
```

## The contract

`contract.py` sorts every feature either arm used into six tiers, each carrying the
sentence of `domain.md` the verdict rests on:

| tier | meaning | in contract |
| --- | --- | --- |
| `LINK` | static link attribute or current-link average | yes |
| `LOOKBACK` | needs the single previously-traversed link | yes |
| `STATE` | causal, but needs more memory than the one permitted link | no, on memory cost |
| `FUTURE` | needs links not yet traversed | no, on causality |
| `TRACE` | exists only in the simulated drive-cycle trace | no, on observability |
| `LABEL` | derived from the target of other links | no — at inference there are no labels |

The tiers are kept apart because they fail for different reasons and the difference is
actionable. A `STATE` feature is something RouteE *could* buy by paying for label memory,
and the audit prices that option. A `FUTURE` feature is unobtainable at any price, because
the route is the thing the search is computing. A `TRACE` feature is unobtainable for a
different reason again — it is an artifact of how the training data was simulated, and no
map has it. A `LABEL` feature is not a feature at all at inference time.

`TRACE` and `LABEL` are new this round, and the reason is worth stating. The previous pair
of trees produced an unguided model whose out-of-contract features were all aggregates of
*observable* quantities — speed, grade, geometry — so every one had a causal prefix
substitute, and the interesting question was what those substitutes cost. This arm's model
reaches further: `gap_seconds` is idle time read off the simulation trace, and
`journey_rate` is a leave-one-out mean of the target itself. Neither has a substitute at
any tier. Filing them as `STATE` would imply a price exists.

### Beyond the feature list

A feature ledger alone would pass two of the unguided model's three parts, so `contract.py`
also carries a `COMPONENTS` table:

- **The sequence member** is a 2-block dilated CNN over each journey's whole link chain.
  Kernel 5 at dilations 1 and 2 gives a receptive field of 13 links — six *ahead* as well
  as six behind. It reads the future structurally, not through any named column.
- **The journey offset** is a per-journey correction built from the training residuals of
  that journey's own links. It needs labels, and it needs the whole journey.

Applying this contract to the unguided arm is not an accusation. That arm was never shown
`domain.md`; it was asked to reduce an error metric and it did, competently and with its
reasoning on the record. The contract measures the distance between what it optimised and
what Compass can execute — which is a property of the brief it was given, not of the work
it did.

## What gets scored

Both arms are scored on the same held-out rows, with the same split seed and the same
`harness.evaluate()` both arms used.

| configuration | what it tells you |
| --- | --- |
| `unguided/reported` | reproduces the archived number — if this fails, nothing else counts |
| `unguided/as-run/gbdt-only` | the tree ensemble alone, still on all 13 features |
| `unguided/as-run/no-offset` | trees + sequence member, without the journey offset |
| `unguided/audited-mean` | ship the trained artifact with the missing inputs simply absent. The floor. |
| `unguided/audited-contract` | missing inputs replaced by the best `LINK`+`LOOKBACK` substitute. **The Compass number.** |
| `unguided/audited-state` | plus `prev2_speed`, the one out-of-contract feature that is genuinely causal |
| `unguided/retrained-contract` | the same GBDT recipe *refit* inside the contract. The fair fight. |
| `unguided/retrained-state` | the same recipe refit with `prev2_speed` allowed — prices the memory option |
| `unguided/retrained-causal-seq` | contract trees blended with a **left-masked** CNN |
| `domain-guided/reported` | reproduces the archived number |
| `domain-guided/audited` | identical to reported; asserted in code, since all 11 features are in contract |

Three of these deserve a note.

The `as-run/*` rows decompose the blend. Because two of the unguided model's three parts
are out of contract for reasons no column name reveals, the only way to show how much of
the reported score rests on each is to take them off one at a time.

The `retrained-*` rows matter because stripping inputs from a model trained to depend on
them is the deployment question, but not a fair comparison of the two research paths — the
unguided arm would have searched differently under the constraint. Refitting its recipe on
the restricted feature set is the honest head-to-head.

`retrained-causal-seq` asks the question the rest of the audit cannot: **was the sequence
model the value, or was its view of the future the value?** Same architecture, same
training budget, same blend weight, but left-masked so it reads only links already
traversed. If it holds most of the gain, then a Compass implementation willing to carry
per-path hidden state has something real to buy, and that is a finding in the unguided
arm's favour. If it does not, the sequence member was mostly reading ahead.

## Reading the inference numbers

Three figures, in increasing order of how far they travel:

- **Measured latency** — microseconds per link in this Python harness, pinned to one
  thread. Compass is Rust, so treat these as pessimistic for every model and read the
  *ratio*.
- **Arithmetic per prediction** — tree-node visits versus multiply-accumulates.
  Language-independent.
- **Search-label state** — how many extra floats each feature set forces every Compass
  search label to carry. This is the figure that decides feasibility rather than speed,
  and it is the one `domain.md` was written about.

The benchmark measures three models, not two. The unguided arm's as-run model is not
implementable at any price, so comparing only against it would price a thing nobody can
buy; the contract refit is the model a Compass team could actually ship from that research
path, and it is the honest cost comparison.

## Limits of this audit

- **The train/test split is random per row**, so test links share journeys with train
  links. Both arms flagged this. It inflates any feature computed over a journey — and
  for this arm's `journey_rate`, which is a leave-one-out mean of the target over the
  journey's *training* links, it is what makes the feature work at all. A journey-level
  or geographic split would be a separate and worthwhile study, and would likely be
  harsher on the unguided arm than this audit is.
- **The causal substitutes are a reconstruction, not the arm's own work.** They are the
  best causal form the audit could write; a session actually run under the constraint
  might find better ones. Read the `retrained-*` rows as a lower bound on what the
  constrained path could reach, not as its ceiling. The left-masked CNN in particular is
  the audit's design, not the arm's.
- **The unguided arm is not exactly reproducible.** Its sequence member trains for a
  wall-clock budget rather than a fixed epoch count, so a different host fits a different
  model. The reproduction check allows ±5% for that arm and ±2% for the deterministic
  domain-guided one.
- **`journey_rate` has a deployment scenario this audit does not price.** A vehicle
  actually driving a route knows its own energy consumed so far, so an on-vehicle
  predictor could supply a causal running form of it. That is a different product from a
  shortest-path search, which is predicting energy rather than observing it, and
  `domain.md` describes the search. The audit scores against the search.
- **One session per arm.** Everything here describes two runs, not a population.
- **The arms differ in more than `domain.md`** — see the root `README.md` on the
  session-start prompt asymmetry. This audit measures deployability, which is a property
  of the finished models; it does not resolve which input caused the difference.
