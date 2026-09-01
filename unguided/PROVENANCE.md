# Provenance — `unguided`

The t=0 record for this arm, plus what the archive copy does and does not guarantee.
Written at import time from the tree's own history; the shared dataset and isolation sections
are duplicated in [`../domain-guided/PROVENANCE.md`](../domain-guided/PROVENANCE.md) so each arm
stands on its own.

## Tree

| | |
| --- | --- |
| harness template | `github.com/NatLabRockies/teta-autoresearch` |
| template commit | `4cc0f7774b3d` |
| scaffold (root) commit | `60e18ff2bd07eeadee8eea562ef54b23fcbd405b` |
| scaffolded | 2026-08-25T16:57:58-06:00 |
| tip at import | `67355b5b6d3da24982cf869351903763db158ac5` |
| commits | 107 |
| experiments | 50 (exp0–exp49) |
| ran in | `~/routee-autoresearch-runs/unguided` |

Generated with `tools/new_tree.sh <dir> --data ~/data/routee-bev`.

## What this arm was given

| | |
| --- | --- |
| `domain.md` | **absent** — never present at any commit in this tree's history |
| `seed.md` | **absent** |

The two arms' scaffold commits differ by **exactly one file**:

```
$ git diff --stat 60e18ff 409009f
 domain.md | 46 ++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 46 insertions(+)
```

That is the study design working as intended, and it is a change from the previous pair of
trees, where both arms carried differing `seed.md` briefs that confounded the comparison.
There is no `seed.md` in either arm now.

**One difference remains, and it is in the session-start prompt rather than the tree.** This
arm's operator prompt restates the absence of `domain.md` and adds a containment instruction;
the other arm's does not. Verbatim, it was:

> Take a look at program.md and let's kick off a new session. There is no domain.md file in
> this experiment tree. The goal is for you to come up with your own research direction
> without any domain context. Do not reference any other file or folder outside of this repo.

The domain-guided arm received `Take a look at the program.md file and let's kick of a new
experiment session`, then `Go ahead`. So the arms are matched on tree contents but not on
prompt text, and the residual difference is a containment instruction — narrower than the
previous run's confound, but not nothing. Both operator prompts are reproduced in full in each
arm's transcript audit so this can be checked rather than asserted.

## Results

Reported metrics — produced by this arm's own harness, on the features it chose. The
independent audit of these models lives in [`../audit/`](../audit/README.md).

| | rmse | trip_rmse |
| --- | --- | --- |
| baseline (`2dfdc45`) | 0.013345 | 0.003037 |
| best reported (`bc07466`, exp48) | 0.004785 | 0.001231 |

The final model is a blend: `0.70 ×` a 5-member `HistGradientBoostingRegressor` ensemble,
`0.30 ×` a 2-block dilated 1-D CNN over each journey's link chain, plus a distance-weighted
per-journey offset. It uses 13 features, several of which read forward in the journey or
aggregate over the whole of it — which is what the audit exists to price.

Full table in [`results/results-bev-aug25.tsv`](results/results-bev-aug25.tsv), one row per
experiment, with hypothesis and observation in
[`results/experiments-bev-aug25.jsonl`](results/experiments-bev-aug25.jsonl).

## Session record

| | |
| --- | --- |
| session span | 2026-08-25T23:02:48Z → 2026-08-26T04:41:42Z |
| assistant messages | 238 |
| web research | **none** — no WebSearch or WebFetch calls |
| operator prompts | 1 |
| tool calls | `Bash` 102, `Write` 9, `Edit` 9, `Read` 8 |

Neither arm did any web research this time, so the previous run's asymmetry on that axis is
also gone.

The transcript audit at
[`results/transcript-audit-bev-aug25.md`](results/transcript-audit-bev-aug25.md) lists every
out-of-tree path the session touched and the full text of every operator prompt. The raw
transcript sits beside it in `results/transcript-bev-aug25/` and is committed **verbatim**, via
Git LFS (`results/transcript-*/**` in `.gitattributes`).

The tree path had been used by earlier sessions, so the audit windows the transcript to records
at or after the scaffold timestamp and reports what it excluded (1,059 records here). The
window, not the file, is what makes this a record of *this* run.

Because the transcript is verbatim, it contains whatever the session printed to its own
console, including `.head()` and `describe()` output over the dataset. Published deliberately:
the transcript's value is that it is unedited.

## Dataset

Not in this repository, and not reachable from it.

| | |
| --- | --- |
| path | `~/data/routee-bev/processed/2017_Chevy_Bolt.parquet` |
| sha256 | `420b0d4d5997c2663ef050b326fd535ee9af474970030b814c535120facb738d` |
| size | 212,951,540 bytes |
| rows | 1,638,466 links across 15,247 journeys |

Byte-identical to the file both arms used and to the one the previous pair of trees used, so
scores are comparable across runs.

Simulated 1 Hz drive-cycle traces for a 2017 Chevy Bolt, aggregated to road segments. Schema in
[`../data/README.md`](../data/README.md).

`data` in this directory is a **symlink**, committed as a symlink (git mode `120000`) — 40 bytes
of path text, no data. It points at an absolute path on the machine that ran the study and will
dangle in any clone. That is intentional: the dataset must live outside every git repository, or
`git -C data log` from inside a tree would expose that repository's history to an otherwise
isolated run.

## What is not guaranteed

Stated plainly, because the isolation claim should not be read as stronger than it is.

- The checks are **verified, not enforced**. Nothing prevented the agent from reading an absolute
  path outside its tree; the run directory simply gave it little to find. Enforcement would need
  a container bind-mounting only the tree and the dataset.
- The pre-run isolation verifier covers **one** directory level up. Both arms lived under
  `~/routee-autoresearch-runs/`, so `ls ..` revealed that the other arm existed and its tree was
  readable from there. The transcript audit is the evidence on whether either looked; the
  filesystem did not prevent it.
- This arm's session-start prompt *asked* the agent not to look outside the tree. That is an
  instruction, not a control — and it is itself the one remaining asymmetry between the arms.
- The model's own pretraining is not controlled and cannot be.
- Whoever operates a session can leak prior findings by typing them. The operator prompts are in
  the transcript audit so this can be checked rather than asserted.
- **One session per arm.** This is one sample of each condition, not a population.

## Archive copy

This directory is a copy, imported with `git read-tree --prefix=unguided/` onto an `-s ours`
merge of the tree's history — so all 107 commits are ancestors of this repository's HEAD.

Those commits carry their file paths at the tree root, not under `unguided/`, so `git log
unguided/` shows only the import. The history is addressable through its tag:

```bash
git log --oneline import/unguided-bev-aug25
```

The tree itself **stays where it ran** and was not moved, so future sessions on it keep their
isolation. Re-sync later with:

```bash
git fetch unguided-src main
git merge -X subtree=unguided unguided-src/main
```

Never add this archive as a remote *inside* the tree — that would give the tree a remote and
fetchable objects containing the other arm's entire history, which is the exact failure the
separate run location exists to prevent.
