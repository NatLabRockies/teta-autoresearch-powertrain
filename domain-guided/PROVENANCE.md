# Provenance — `domain-guided`

The t=0 record for this arm, plus what the archive copy does and does not guarantee.
Written at import time from the tree's own history; the shared dataset and isolation sections
are duplicated in [`../unguided/PROVENANCE.md`](../unguided/PROVENANCE.md) so each arm stands on
its own.

## Tree

| | |
| --- | --- |
| harness template | `github.com/NatLabRockies/teta-autoresearch` |
| template commit | `4cc0f7774b3d` |
| scaffold (root) commit | `409009f5f6cf6df34eebce9cfac91c8a62127ce7` |
| scaffolded | 2026-08-25T16:57:58-06:00 |
| tip at import | `f6396b5573e078bbdd65209f3fa9bf5a2a400ece` |
| commits | 108 |
| experiments | 51 (exp0–exp50) |
| ran in | `~/routee-autoresearch-runs/domain-guided` |

Generated with `tools/new_tree.sh <dir> --data ~/data/routee-bev`.

## What this arm was given

| | |
| --- | --- |
| `domain.md` | **present** — 46 lines, present from the scaffold commit onward |
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

[`domain.md`](domain.md) describes the RouteE Compass inference environment: link-average
speed, gradient, distance and geometry are available; acceleration and driver behaviour are
not; and if the model uses link sequencing it may look back **one** link and must not read
forward. That last constraint is the one the audit turns into a testable contract.

**One difference remains, and it is in the session-start prompt rather than the tree.** This
arm's operator input was `Take a look at the program.md file and let's kick of a new experiment
session`, then `Go ahead` — two prompts, plus one `AskUserQuestion` the session raised itself.
The unguided arm received a single longer prompt restating the absence of `domain.md` and
adding a containment instruction. So the arms are matched on tree contents but not on prompt
text. Both operator prompts are reproduced in full in each arm's transcript audit so this can
be checked rather than asserted.

## Results

Reported metrics — produced by this arm's own harness, on the features it chose. The
independent audit of these models lives in [`../audit/`](../audit/README.md).

| | rmse | trip_rmse |
| --- | --- | --- |
| baseline (`cc06c6e`) | 0.013345 | 0.003037 |
| best reported (`e315384`, exp50) | 0.006823 | 0.002239 |

The final model is an average of **2 MLPs**, each 11 → 128 → 128 → 1 with ReLU, over 11
features — all of which are static link attributes, current-link averages, or single-link
lookbacks. The session spent its last third buying no accuracy but halving inference cost, to
35,840 multiply-accumulates per link, on the reasoning that `domain.md` makes deployability a
competing objective.

Full table in [`results/results-bev-aug26.tsv`](results/results-bev-aug26.tsv), one row per
experiment, with hypothesis and observation in
[`results/experiments-bev-aug26.jsonl`](results/experiments-bev-aug26.jsonl).

## Session record

| | |
| --- | --- |
| session span | 2026-08-26T17:01:57Z → 2026-08-26T19:46:35Z |
| assistant messages | 301 |
| web research | **none** — no WebSearch or WebFetch calls |
| operator prompts | 2 |
| tool calls | `Bash` 99, `Read` 9, `Edit` 9, `Write` 5, `AskUserQuestion` 1 |

Neither arm did any web research this time, so the previous run's asymmetry on that axis is
also gone.

The transcript audit at
[`results/transcript-audit-bev-aug26.md`](results/transcript-audit-bev-aug26.md) lists every
out-of-tree path the session touched and the full text of every operator prompt. The raw
transcript sits beside it in `results/transcript-bev-aug26/` and is committed **verbatim**, via
Git LFS (`results/transcript-*/**` in `.gitattributes`).

The tree path had been used by earlier sessions, so the audit windows the transcript to records
at or after the scaffold timestamp and reports what it excluded (1,141 records here). The
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
- The arms' session-start prompts were not matched — see above. This arm's was the shorter of
  the two and carried no containment instruction.
- The model's own pretraining is not controlled and cannot be.
- Whoever operates a session can leak prior findings by typing them. The operator prompts are in
  the transcript audit so this can be checked rather than asserted.
- **One session per arm.** This is one sample of each condition, not a population.

## Archive copy

This directory is a copy, imported with `git read-tree --prefix=domain-guided/` onto an
`-s ours` merge of the tree's history — so all 108 commits are ancestors of this repository's
HEAD.

Those commits carry their file paths at the tree root, not under `domain-guided/`, so `git log
domain-guided/` shows only the import. The history is addressable through its tag:

```bash
git log --oneline import/domain-guided-bev-aug26
```

The tree itself **stays where it ran** and was not moved, so future sessions on it keep their
isolation. Re-sync later with:

```bash
git fetch domain-guided-src main
git merge -X subtree=domain-guided domain-guided-src/main
```

Never add this archive as a remote *inside* the tree — that would give the tree a remote and
fetchable objects containing the other arm's entire history, which is the exact failure the
separate run location exists to prevent.
