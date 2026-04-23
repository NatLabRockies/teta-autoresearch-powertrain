# Isolated experiment trees

Tools for running autoresearch experiments without cross-tree bias.

## Why

The main repo accumulates knowledge across sessions (`learnings.md`, tags,
committed results, persistent `<partition>/best` pointers). That's great
for incremental progress but fatal for controlled comparisons: any agent
starting on `main` inherits every prior session's conclusions via
`learnings.md` alone.

These tools create per-tree isolated workspaces so that a fresh agent or
optimizer in tree A sees nothing from tree B's history.

## Concepts

- **Template**: this repo. Holds the harness — fixed evaluation
  (`fixed_utils.py`), protocol (`program.md`), optimizer drivers
  (`optimizers/`), tree tooling (`tools/`), and one or more domain
  scaffolds (`domains/<name>/`).
- **Tree**: a separate git repo in `~/repos/routee-autoresearch-trees/<name>/`.
  Contains only the scaffold plus a single initial commit. No prior tags,
  no prior branches, no accumulated `learnings.md`.
- **Registry**: `~/repos/routee-autoresearch-trees/registry.jsonl`. One
  line per tree recording provenance (name, template sha, domain,
  partition, mode, optimizer, seed files).

## Flattened layout

Trees are flattened. The chosen domain's files (`train.py`, `domain.md`,
`seed.md`, `learnings.md`, `data/`) live at the tree root — not under
`domains/<name>/`. This keeps the LLM agent's UX identical across domains
and preserves the existing wording of `program.md`.

Optimizer trees additionally preserve `domains/<name>/search/` under its
subpath so `from domains.<name>.search import run_trial` resolves inside
the tree.

## Create a tree

LLM mode (no intervention):

```bash
tools/new_experiment_tree.sh \
    --name unguided-01 \
    --domain routee \
    --mode llm \
    --partition bev
```

Optimizer mode (TPE over the full RouteE search space):

```bash
tools/new_experiment_tree.sh \
    --name tpe-bev-01 \
    --domain routee \
    --mode optimizer \
    --optimizer tpe \
    --partition bev
```

LLM mode with intervention:

```bash
tools/new_experiment_tree.sh \
    --name guided-01 \
    --domain routee \
    --mode llm \
    --partition bev \
    --seed-md ./hints/guided-01-seed.md
```

With pre-loaded learnings (e.g. to simulate mid-experiment starts):

```bash
tools/new_experiment_tree.sh \
    --name bootstrap-01 \
    --domain routee \
    --mode llm \
    --partition bev \
    --seed-learnings ./priors/bev-minimal.md
```

## Run an agent / optimizer in the tree

```bash
cd ~/repos/routee-autoresearch-trees/<name>
# LLM tree — point an agent at program.md:
claude --dangerously-skip-permissions
# Optimizer tree — invoke the chosen sampler:
pixi run python -m optimizers.tpe.search --tag <tag> --partition bev --n-trials 200
```

## Verify isolation

```bash
cd ~/repos/routee-autoresearch-trees/<name>
git log --all --oneline    # → exactly one commit (initial scaffold)
git tag -l                 # → empty
git branch -a              # → only main
cat learnings.md           # → empty (or the seed file's content)
cat .tree-meta.json        # → provenance
```

## Record LLM token usage for a session

LLM-mode trees emit a per-model token snapshot next to the other session
artifacts:

```bash
cd ~/repos/routee-autoresearch-trees/<name>
pixi run python tools/token_usage.py --tag <variant>-<date>
```

Produces `results/<variant>/usage-<date>.jsonl`, one line per distinct model
observed in Claude Code's session transcripts for the tree. Each line is a
cumulative snapshot with `input_tokens`, `output_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`, plus
`assistant_messages`. Consumers should take the latest `snapshot_at` per
model — lines are not deltas and must not be summed.

The script reads from `~/.claude/projects/<encoded-tree-path>/*.jsonl`
(Claude Code's own transcript store) and recomputes totals from scratch on
every run, so invoking it multiple times in one session cannot double-count.
Only meaningful for LLM-mode trees; optimizer trees have no agent to track.

## Sync a harness change into a live tree

Rare — only when a harness bug fix or protocol change must reach an
in-progress tree. Breaks strict reproducibility from the original
scaffold, so use deliberately.

```bash
tools/sync_harness.sh --tree ~/repos/routee-autoresearch-trees/<name>
```

Copies framework files (`program.md`, `fixed_utils.py`, `pixi.*`,
`pyproject.toml`, etc.) and the tree's domain scaffold (`domain.md`,
`domain.json`) from the template's HEAD into the tree. Optimizer trees
also receive updates to `optimizers/` and `domains/<d>/search/`. Commits
as `harness: sync from template@<sha>`.

Does not touch `train.py`, `learnings.md`, `seed.md`, `results/`,
`plans/`, or `data/`.

## What each tree contains

Copied from template (scaffold, identical across trees of the same mode):

- `program.md`, `fixed_utils.py`, `pyproject.toml`
- `pixi.toml`, `pixi.lock`, `Dockerfile`, `dprint.json`
- `tools/`, `templates/`
- `optimizers/` (optimizer-mode trees only)
- `.gitignore`, `.gitattributes`, `README.md`, `EXTENDING.md`

Flattened from the chosen domain (tree root):

- `domain.md`, `domain.json`, `train.py`

Replaced or reset per tree:

- `seed.md` — empty by default, caller-overridable via `--seed-md`
- `learnings.md` — empty by default, caller-overridable via `--seed-learnings`
- `train.py` — partition selector stamped per `--partition`
- `results/<p>/.gitkeep` — fresh skeleton (one subdir per partition value)
- `.tree-meta.json` — provenance

Symlinked (not copied, read-only):

- `data/` → `domains/<name>/data/` in the template

Preserved at original subpath (optimizer trees only):

- `domains/<name>/search/` — domain hooks imported by the optimizer driver

## Design notes

- **Why separate repos, not orphan branches?** `git log --all`,
  `git tag -l`, `git for-each-ref`, and reflog all happily cross orphan
  branches in a shared repo. Hiding refs is porous. Separate `.git`
  directories are the only airtight boundary git provides for free.

- **Why symlink `data/`?** The parquets are large and read-only per
  protocol. Symlinking saves disk and avoids silent divergence. If you
  need fully archivable trees, replace the symlink with a copy.

- **Why not auto-sync harness updates?** Reproducibility. Each tree
  should be rebuildable from a single scaffold commit. Automatic sync
  would mean trees silently drift based on when you last touched them.
  The explicit `sync_harness.sh` exists for the rare case where a
  harness fix must propagate.

- **Why flatten domain files?** Single-domain-per-tree is a hard
  invariant; no path inside the tree ever references a sibling domain.
  Flattening keeps the LLM agent's mental model identical across
  domains and means `program.md` can reference `train.py` / `domain.md`
  at the root without knowing about the multi-domain template structure.
