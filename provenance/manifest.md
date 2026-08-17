# Provenance

Recorded at t=0, before either session started. Everything here is fixed for the duration of
the study; if any of it changes mid-flight, the two arms are no longer comparable.

## Harness

Both trees were generated from the same template commit, by the same command modulo one flag.

| | |
| --- | --- |
| template | `github.com/NatLabRockies/teta-autoresearch` |
| template commit | `e545796abf379630bec1ab73f919f3d0dc658310` |

```bash
tools/new_tree.sh ~/runs/unguarded/tree    --no-domain --data ~/data/routee-bev
tools/new_tree.sh ~/runs/human-guided/tree             --data ~/data/routee-bev
```

## Trees

| arm | scaffold commit | `domain.md` | `seed.md` |
| --- | --- | --- | --- |
| `unguarded` | `c04621d609b1b4214558a359f1db78eb1735dd90` | absent | empty |
| `human-guided` | `4a9b58f996fa7504a6a593702428d1e38150edc1` | present | empty |

Both trees are otherwise byte-identical, including `program.md`, `fixed_utils.py`, and the
starting `train.py`. Both produce the same baseline:

```
metrics: {"rmse": 0.013345, "trip_rmse": 0.003037}
```

`seed.md` is empty in **both** arms deliberately. The single variable under test is the domain
specification. Adding a written brief to one arm would confound guardrails with direction — the
flaw in the earlier pair of runs.

## Dataset

| | |
| --- | --- |
| path | `~/data/routee-bev/processed/2017_Chevy_Bolt.parquet` |
| sha256 | `420b0d4d5997c2663ef050b326fd535ee9af474970030b814c535120facb738d` |
| size | 212,951,540 bytes |

The dataset lives in a plain directory with **no git repository anywhere above it**. This is
load-bearing, not tidiness: when the dataset was previously symlinked into a working repo,
`git -C data log` from inside a tree exposed 307 commits and 40 tags — including
`apr13/milestone-prev-speed` and `apr13/milestone-energy-filter`, which name the exact findings
a fresh run is supposed to reach independently.

## Isolation

`unguarded-isolation.txt` and `human-guided-isolation.txt` are the verifier's output for each
tree, captured before its session began. Both PASS on all ten checks.

Re-run at any time with:

```bash
~/repos/teta-autoresearch/tools/verify_isolation.sh ~/runs/<arm>/tree
```

The checks cover: single-commit history, no tags, no extra branches, no remotes, empty
`learnings.md`, empty `results/` and `plans/`, no `CLAUDE.md` in the tree or any ancestor, no
sibling run one directory up, and `data/` resolving outside every git repo.

## What is not guaranteed

Stated plainly, because the isolation claim should not be read as stronger than it is:

- The checks are **verified, not enforced**. Nothing prevents an agent from reading an absolute
  path outside its tree; the run directories simply give it nothing to find. Enforcement would
  need a container that bind-mounts only the tree and the dataset.
- The sibling check covers **one** directory level. The two arms live at `~/runs/unguarded/tree`
  and `~/runs/human-guided/tree`, so `ls ../..` still reveals that the other arm exists — and its
  tree is readable from there. Neither arm has any reason to look, and the post-hoc transcript
  audit below will show whether either did, but the filesystem does not prevent it. Putting the
  two arms under unrelated roots, or in containers, is what would.
- The model's own pretraining is not controlled and cannot be.
- Whoever operates a session can leak prior findings by typing them. Keep operator input to the
  session-start prompt, and record anything said beyond it.

## Auditing a finished session

`tools/capture_transcript.py` is the counterpart to the checks above: they establish the tree was
clean before a session, it establishes what the session actually did.

```bash
pixi run python tools/capture_transcript.py --tag <tag>
```

It copies the raw Claude Code transcripts to `results/transcript-<tag>/` and writes
`results/transcript-audit-<tag>.md` with three things worth reading before publishing any
comparison:

- **every path referenced outside the tree** — the direct evidence for or against the isolation
  claim. Extraction is best-effort and over-reports (`/dev/null` and `/usr/bin` paths count), so
  the list is a prompt to look, not a verdict.
- **the full text of every operator prompt** — the record of how much direction each arm got.
  If the arms were told different amounts, that is a second variable and the comparison has to
  say so.
- tool-call counts and the session's time span.

Run it at session end, per `program.md`. Snapshots overwrite, so running it mid-session is safe.

**Scope the audit with `--since`.** The `unguarded` tree path already hosted two aborted setup
attempts before its reset, and Claude Code keys transcripts by directory, so those records still
sit alongside whatever the real run writes. Unwindowed, they get folded in — measured at 48
assistant messages across 3 transcript files, of which only one belonged to the current tree.
Pass the scaffold commit's date and the audit reports how many records it excluded:

```bash
pixi run python tools/capture_transcript.py --tag <tag> \
    --since "$(git log -1 --format=%aI $(git rev-list --max-parents=0 HEAD))"
```

The pre-reset transcripts were deliberately left in place rather than deleted — they are the
harness's own records, and a filter is the honest way to exclude them.
