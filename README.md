# teta-autoresearch-powertrain

An archive of autonomous-research experiment trees on vehicle powertrain energy prediction.

Each tree is one run of the [`teta-autoresearch`](https://github.com/NatLabRockies/teta-autoresearch)
harness: a session of an agent proposing, running and judging experiments, with every experiment
recorded as a commit. Trees execute **outside this repository**, each in its own directory, and
are imported here with their history intact once a run completes. They stay where they ran; this
is a copy.

## Trees

| tree | session | experiments |
| --- | --- | --- |
| [`unguided`](unguided/) | `bev-aug25` | 50 |
| [`domain-guided`](domain-guided/) | `bev-aug26` | 51 |
| [`physics-bounded`](physics-bounded/) | `bev-aug31` | 46 |

Each carries a `PROVENANCE.md` with what it was given, its session record, its results, the
dataset checksum, and the limits of what it demonstrates. Start there.

[`audit/`](audit/) is an independent re-scoring of the `unguided` and `domain-guided` final
models. It reads both trees, which is why it cannot live inside either one.

## Reading a tree

The current state of a tree is checked out in its directory and read normally — `results/` holds
one row per experiment plus the session's own transcript, and `learnings.md` is what the agent
concluded.

The commit history is imported at the tree root rather than under the tree's directory, so
address it by tag:

```bash
git log --oneline import/physics-bounded-bev-aug31
```

Raw session transcripts are committed verbatim and stored with Git LFS, so cloning needs
`git lfs` installed:

```bash
git lfs install && git clone https://github.com/NatLabRockies/teta-autoresearch-powertrain.git
```

## The dataset

Not in this repository. Each tree's `data` is a committed **symlink** to a path on the machine
that ran the study; it holds no data and will dangle in a clone. See
[`data/README.md`](data/README.md).

## Acknowledgments

This software is built on the "autoresearch" software by github user karpathy available here [link](https://github.com/karpathy/autoresearch) and distributed under the MIT license.

## Metadata

NLR Software Record # SWR 26-090.
