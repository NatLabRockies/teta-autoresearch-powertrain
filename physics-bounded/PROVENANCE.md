# Provenance — `physics-bounded`

The t=0 record for this tree, plus what the archive copy does and does not guarantee.
Written at import time from the tree's own history.

**Read this first: this tree is not an isolated arm.** Unlike
[`../unguided`](../unguided/PROVENANCE.md) and
[`../domain-guided`](../domain-guided/PROVENANCE.md), it was deliberately given the other two
trees' findings, by operator instruction, and it took them — the transcript audit records reads
of both `../unguided/learnings.md` and `../domain-guided/learnings.md`. It is a follow-on
investigation, not a third sample of the same design. Do not put it in a table beside the other
two as though it were.

## Tree

| | |
| --- | --- |
| harness template | `github.com/NatLabRockies/teta-autoresearch` |
| template commit | `c4dcd724fe88` |
| scaffold (root) commit | `dac3f6c64fd3fdc95bb37cd2fbb2e5a8455df347` |
| scaffolded | 2026-08-31T16:43:53-06:00 |
| tip at import | `60fd8cc649b36b55d9a57c262bb0ff2023d8ebf4` |
| commits | 99 |
| experiments | 46 rows (exp0–exp45), 23 of them `keep` |
| ran in | `~/routee-autoresearch-runs/physics-bounded` |

Generated with `tools/new_tree.sh <dir> --data ~/data/routee-bev`.

## What this tree was given

| | |
| --- | --- |
| `domain.md` | **present**, and rewritten for this tree |
| `physics.py` | **present** — 765 lines, added before the session |
| `seed.md` | **absent** |

The given state is **not** the scaffold commit. The operator committed `a51dbb479257`, *add
physical violation checks*, at 2026-08-31T17:13:02-06:00 — three minutes before the session
opened — adding `physics.py` and rewiring `domain.md`, `harness.py`, `train.py` and
`tools/test_harness.py` around it. Everything from `eeb388b` (exp0) onward is the session's.

Two things separate this tree from the earlier pair, and both change what is worth trying:

1. **Physics is enforced, not advised.** [`physics.py`](physics.py) scores the model over a
   synthetic sweep of every combination of speed, grade and length and checks the predictions
   against physical law — no ground truth involved. Level ground cannot produce net energy; a
   steeper climb cannot cost less than a shallower one; a climb must cost at least the potential
   energy it gains; nothing may exceed what the link could possibly demand. The bounds are built
   from the real FASTSim vehicle constants that generated the dataset (mass 1757.77 kg, Cd 0.29
   over 2.845 m², rolling resistance 0.0073, battery-to-wheel efficiency 0.917), so a violation
   contradicts the model's own training data. `harness.evaluate()` reports `physics_pass`,
   `physics_violation_rate`, `long_link_violation_rate` and `length_invariance` alongside the
   two error metrics, and `domain.md` makes a run with `physics_pass: 0.0` a `discard` whatever
   its RMSE.
2. **The model is not for RouteE Compass.** It may use the whole trip's link chain at inference,
   so the one-link-lookback rule the `domain-guided` arm ran under does not apply here, and
   inference cost is not a competing objective. Forward-reading features are legal in this tree.

The two constraints the earlier `domain.md` carried and this one keeps: do not filter rows to
lower the error, and do not use a link-position feature. `domain.md` also forbids reaching
`physics_pass` by clamping or post-processing the output — the learned function has to be right,
not clipped into the legal range.

## Results

Reported metrics, produced by this tree's own harness. There is **no independent audit of this
tree** — [`../audit/`](../audit/README.md) covers the `unguided` and `domain-guided` arms only.

| | rmse | trip_rmse | physics_pass | length_invariance |
| --- | --- | --- | --- | --- |
| baseline (`eeb388b`, exp0) | 0.013345 | 0.003037 | **0.0** | 8.826788 |
| best (`2b5c37e`, exp45) | 0.005915 | 0.002084 | **1.0** | 0.500424 |

The scaffold RandomForest baseline fails **six of nine** physics checks, with an implied
drivetrain efficiency of 1.29 — it bills a climb for less energy than the height it gains. That
is the finding the tree is built around: an unconstrained regressor has no reason to respect
gravity, and capacity does not give it one.

The final model predicts a link's *total* energy in the shape physics gives it,

```
E(v, g, d, …) = d · [ A + B·g₊ − C·g₋ ] + T
```

with each of the four heads squashed into its own physically legal interval and driven by a
128 → 128 ReLU MLP over 15 network inputs (plus `grade_percent` and `miles`, consumed by the
structure rather than the network). This bounds the total exactly, because the ceiling in
`physics.py` decomposes the same way. All nine checks passed on the first run of that
construction.

Full table in [`results/results-bev-aug31.tsv`](results/results-bev-aug31.tsv), one row per
experiment, with hypothesis and observation in
[`results/experiments-bev-aug31.jsonl`](results/experiments-bev-aug31.jsonl).

## Session record

| | |
| --- | --- |
| session span | 2026-08-31T23:16:37Z → 2026-09-01T05:39:06Z |
| assistant messages | 221 |
| web research | **none** — no WebSearch or WebFetch calls |
| operator prompts | 1, plus one `AskUserQuestion` the session raised itself |
| tool calls | `Bash` 83, `Edit` 10, `Read` 7, `Write` 4, `AskUserQuestion` 1 |

The single operator prompt is the reason this tree is not an isolated arm, and it is reproduced
in full in [`results/transcript-audit-bev-aug31.md`](results/transcript-audit-bev-aug31.md). It
invited the session to read the other trees' learnings and stated the two design changes above.

That audit also lists every out-of-tree path the session touched — 29 distinct references, of
which two are the substantive ones (`../unguided/learnings.md` and
`../domain-guided/learnings.md`); the rest are `/dev/null` and fragments of shell text the
extractor over-reports. The raw transcript sits beside it in `results/transcript-bev-aug31/`
and is committed **verbatim**, via Git LFS (`results/transcript-*/**` in `.gitattributes`).

The tree path had been used by earlier sessions, so the audit windows the transcript to records
at or after the scaffold timestamp and reports what it excluded (170 records here). The window,
not the file, is what makes this a record of *this* run.

Because the transcript is verbatim, it contains whatever the session printed to its own console,
including `.head()` and `describe()` output over the dataset. Published deliberately: the
transcript's value is that it is unedited.

## Dataset

Not in this repository, and not reachable from it.

| | |
| --- | --- |
| path | `~/data/routee-bev/processed/2017_Chevy_Bolt.parquet` |
| sha256 | `420b0d4d5997c2663ef050b326fd535ee9af474970030b814c535120facb738d` |
| size | 212,951,540 bytes |
| rows | 1,638,466 links across 15,247 journeys |

Verified byte-identical at import time to the file the `unguided` and `domain-guided` arms used,
so link and trip errors are comparable across trees even though the design is not.

Simulated 1 Hz drive-cycle traces for a 2017 Chevy Bolt, aggregated to road segments. Schema in
[`../data/README.md`](../data/README.md).

`data` in this directory is a **symlink**, committed as a symlink (git mode `120000`) — 30 bytes
of path text, no data. It points at an absolute path on the machine that ran the study and will
dangle in any clone. That is intentional: the dataset must live outside every git repository, or
`git -C data log` from inside a tree would expose that repository's history to the run.

## What is not guaranteed

Stated plainly, because nothing here should be read as stronger than it is.

- **This tree is not isolated, by design.** It was told to read the other two trees and did. Any
  claim of independence belongs to those trees, not this one.
- **One session.** This is one sample of this condition, not a population.
- The checks are **verified, not enforced**. Nothing prevented the agent from reading an absolute
  path outside its tree. Enforcement would need a container bind-mounting only the tree and the
  dataset.
- `physics.py` bounds are calibrated against the data, not derived from it in closed form. Two
  explicit allowances — `TRANSIENT_KINETIC_MULTIPLE` and `ACCESSORY_WATTS` — cover what link
  averaging hides, and are set a step beyond where violations first reach zero across the 491
  sweep cells holding at least 30 real links. A passing model is passing *against that
  calibration*.
- `physics_pass` is a check on the synthetic sweep, not a proof of correctness on real roads.
- The `analysis/` directory that existed in the working tree at import is **not** included — it
  is local scratch (plots, cached checkpoints), gitignored in the source tree and never
  committed there.
- The model's own pretraining is not controlled and cannot be.

## Archive copy

This directory is a copy, imported with `git read-tree --prefix=physics-bounded/` onto an
`-s ours` merge of the tree's history — so all 99 commits are ancestors of this repository's
HEAD.

Those commits carry their file paths at the tree root, not under `physics-bounded/`, so `git log
physics-bounded/` shows only the import. The history is addressable through its tag:

```bash
git log --oneline import/physics-bounded-bev-aug31
```

The tree itself **stays where it ran** and was not moved, so future sessions on it keep whatever
isolation they are given. Re-sync later with:

```bash
git fetch physics-bounded-src main
git merge -X subtree=physics-bounded physics-bounded-src/main
```

Never add this archive as a remote *inside* the tree — that would give the tree fetchable objects
containing every other tree's entire history.
