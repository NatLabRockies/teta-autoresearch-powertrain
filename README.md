# teta-autoresearch-powertrain

## What is this?

This repository is an experiment archive for a study investigating the question: "How important is domain specification for an AI-assisted model training task?". It was prepared as part of an invited presentation at [US-RSE 2026](https://us-rse.org/usrse26/) titled _"Steering an LLM AutoResearch Loop with Domain Context: A Case Study with Vehicle Energy Models"_ by Nicholas Reinicke and Robert Fitzgerald ([link to abstract](https://us-rse.org/usrse26/program/talks/#steering-an-llm-autoresearch-loop-with-domain-context-a-case-study-with-vehicle-energy-models)).

Autonomous agents given open-ended modeling objectives on fixed datasets naturally find the most direct path to minimize loss. On tabular and sequential transportation datasets, that path often includes deleting difficult rows, exploiting artifacts of simulation traces, or reading forward into links that a routing engine has not yet traversed. The central question of this study is not simply "which agent reports a lower loss," but rather:

> **Can the agent infer the real-world constraints for a problem or does its reported improvements diverge from real improvements?**

To answer this, each final model is evaluated twice:
- **Reported**: What the agent's internal harness recorded on its selected features and processed dataset.
- **Audited**: The same model re-evaluated outside the experiment tree on a shared held-out set with strict deployment-time feature validity and operational constraints enforced.

The archive contains the complete version-controlled histories of both experimental sessions (see [unguided/PROVENANCE.md](unguided/PROVENANCE.md) and [domain-guided/PROVENANCE.md](domain-guided/PROVENANCE.md)), alongside an independent evaluation harness in [audit/README.md](audit/README.md).

---

## Experimental Setup & Session Descriptions

Both sessions were initialized identically using the [`teta-autoresearch`](https://github.com/NatLabRockies/teta-autoresearch) harness from the same template commit, evaluated against the same powertrain dataset, using the same protocol, metrics, and compute budget:

- **Baseline**: An identical starting baseline using our [existing Chevy Bolt 2017 Random Forest model](https://huggingface.co/NatLabRockies/routee-powertrain-model-library/blob/main/v2/chevrolet/bolt_bev/2017/rf_base_67ae9982/v1/metadata.json) for both sessions (link RMSE: 0.013345, trip RMSE: 0.003037).
- **Diff**: The initial repositories differed by **exactly one file**: [domain-guided/domain.md](domain-guided/domain.md) was present in the guided session and absent in the unguided arm (`git diff --stat 60e18ff 409009f`, 1 file changed, 46 insertions).
- **No web**: Neither session made any web searches or external web fetches during the runs. 
- **Prompts**: Both sessions were run independently with single-prompt initiation. The `domain-guided` session received a standard prompt plus "Go ahead". The `unguided` session received an identical kickoff with an explicit restatement that the domain file was absent and an instruction not to reference external paths. Both prompts are archived verbatim in each session's transcript audit.
- **Domain Specification**: [domain-guided/domain.md](domain-guided/domain.md) specifies the RouteE Compass shortest-path routing environment:
  - Allowed inputs: static link attributes (distance, grade, geometry) and link-average speed.
  - Link sequencing: at most **one** link of lookback; strictly no forward lookahead; no journey-position indexing or metrics.
  - Competing objectives: inference latency and memory overhead are primary constraints because predictions execute millions of times per route query.
- **Execution Isolation**: To prevent cross-session leakage, each session was run in a separate directory outside this repository and was merged into this archive only after completion. The Claude cache in `~/.claude` was wiped between sessions. Sessions were not run concurrently.

---

## Metric Observations

The audit ([audit/README.md](audit/README.md)) rebuilds each final model, checks it reproduces the reported score, and asks whether it could run inside RouteE Compass, which only sees the current link and the one before it.

| session | domain rules | exps | reported link RMSE | usable features | can run in Compass | link RMSE, refit on usable inputs |
| --- | --- | --- | --- | --- | --- | --- |
| unguided | absent | 50 | 0.004795 | 5 of 13 | **no** | 0.007557 |
| domain-guided | present | 51 | 0.006823 | 11 of 11 | **yes** | 0.006823 |

*Baseline: 0.013345 link RMSE. Scored on 327,647 held-out links.*

### Key Metric Takeaways

##### Reported vs. Deployable
| ![link RMSE by model](img/audit-error.png) | ![time to score one million links](img/audit-inference.png) |
| --- | --- |


- **Domain-guided: deployable as is.** Every input it uses is available in Compass, so its reported 0.006823 is the real number.
- **Unguided: not deployable.** Its reported 0.004795 depends on future links, simulation timestamps and energy labels. A route search has none of these, so that number cannot be delivered.
- **Fair comparison.** Refit on usable inputs only, the unguided recipe reaches 0.007557. The domain-guided model is still 10% better. Details are in [audit/results/report.md](audit/results/report.md).

##### RMSE by trial

Unguided:

![unguided RMSE by trial](img/unguided-aug25-rmse.png)

Guided:

![domain-guided RMSE by trial](img/domain-guided-aug26-rmse.png)

*Progression of link RMSE and trip RMSE across experiment iterations (1–51) for unguided vs. domain-guided sessions, annotating key architectural shifts and where out-of-contract features were adopted.*

Improvement categories:

Category | Meaning | Examples
--- | --- | ---
Feature Engineering | Changes to the input features of the ML model | previous speed, edge sinuosity, overall trip energy
Architecture | Change of the ML model type | XGboost, CNN, MLP, GBDT
Hyperparameter Tuning | Parameters of the ML model | learning rate, number of hidden layers
Simplification | reduction of model simplicity without loss of performance | reduce nodes from 256 to 128, remove redundant "speed_delta" feature

---

## Behavioral Observations

The experiment transcripts and commit histories reveal contrasting behavioral patterns between the two agents:

### The Unguided Agent: Exploiting Leakage & Scale
1. **Time Travel**: In experiment 2, the agent added `dke_per_mile` spanning future link speeds, followed by explicit future link lengths (`next_miles`) and exit acceleration terms (`dv_out`). The agent created `journey_rate`, `journey_gge_per_mile`, and post-hoc journey residual offsets, despite having zero inference-time availability. It introduced a 2-block dilated 1-D CNN over the journey link sequence. With a receptive field of $\pm$ 6 links, the model structurally read 6 links into the future.
2. **Data Artifacts**: It discovered `gap_seconds` and `next_gap_seconds` (idle timestamps between link transitions). While correlated with stops, these columns are synthetic artifacts of drive-cycle simulations and do not exist on a static road network.
3. **Compute**: Lacking an inference budget constraint, it stacked an ensemble of 10,000 gradient-boosted trees alongside the CNN.

### The Domain-Guided Agent: Constrained Feature Engineering & Efficiency
1. **Physics**: Guided by [domain.md](domain-guided/domain.md), the agent developed valid single-link lookbacks (`prev_speed_mph`, `ke_delta_per_mile`), static road geometry features (`sinuosity`, `vertices_per_mile`), and turn headings (`junction_turn_degrees`).
2. **Lightweight Model Architecture**: It replaced tree ensembles with a pair of compact MLPs ($11 \to 128 \to 128 \to 1$). Recognizing inference latency as a competing objective, the agent spent its final 15+ experiments preserving its accuracy while halving operations to 35,840 multiply-accumulates (MACs) per link.

---

## Outcomes

### 1. Specification Prevents Phantom Progress
**Without domain rules, the agent optimised the metric it was given using inputs that do not exist at inference time.** 8 of the unguided session's 13 features, its journey offset and its sequence model all need things a route search does not have: future links, simulation-trace timestamps, or energy labels. Its reported 64% error reduction is not a number Compass can deliver.

### 2. Domain Guidance Produced the Better Deployable Model
Refit on usable inputs only, the unguided recipe reaches a link RMSE of 0.007557. The domain-guided model reaches 0.006823, about 10% better.

### 3. Inference Cost Is the Bigger Difference
A RouteE Compass route search scores millions of links, one CPU thread at a time. Measured at batch 512 in the audit's Python harness:
- `domain-guided`: **1.45 microseconds per link**, about 1.4 s for one million links.
- `unguided/retrained`: **945 microseconds per link**, about 945 s for one million links.

The domain-guided model is about **650× faster**. The gap is a model choice: 10,000 boosted trees against two small networks. Only the domain-guided agent was told inference cost mattered.

---

## Archive Layout

directory | description
--- | ---
data/ | dataset notes — the data itself lives outside any git repo
unguided/ | imported tree, full history; see [unguided/PROVENANCE.md](unguided/PROVENANCE.md)
domain-guided/ | imported tree, full history; see [domain-guided/PROVENANCE.md](domain-guided/PROVENANCE.md)
audit/        | independent scoring of both sessions' final models; see [audit/README.md](audit/README.md)


Each session carries its own provenance record ([unguided/PROVENANCE.md](unguided/PROVENANCE.md) and [domain-guided/PROVENANCE.md](domain-guided/PROVENANCE.md)) detailing scaffold commits, prompts, dataset checksums, and isolation boundaries. Full audit definitions and reproduction steps are documented in [audit/README.md](audit/README.md) and [audit/results/report.md](audit/results/report.md).

## Reading the Results

Address an session's commit history through its import tag:

```bash
git log --oneline import/unguided-bev-aug25                    # all 107 commits of that session
git log --grep='^exp' --oneline import/domain-guided-bev-aug26 # just the experiments
git show <commit>                                              # view specific experiment change
```

Experiment logs and metrics inside each session:
- Metric records: [unguided/results/results-bev-aug25.tsv](unguided/results/results-bev-aug25.tsv) and [domain-guided/results/results-bev-aug26.tsv](domain-guided/results/results-bev-aug26.tsv)
- Hypotheses & observations: [unguided/results/experiments-bev-aug25.jsonl](unguided/results/experiments-bev-aug25.jsonl)
- Operator transcripts: [unguided/results/transcript-audit-bev-aug25.md](unguided/results/transcript-audit-bev-aug25.md)
- Agent conclusions: [unguided/learnings.md](unguided/learnings.md) and [domain-guided/learnings.md](domain-guided/learnings.md)

## Running Further Sessions

Sessions must run inside their isolated run directories, never inside this archive:

```bash
~/repos/teta-autoresearch/tools/verify_isolation.sh ~/routee-autoresearch-runs/unguided

cd ~/routee-autoresearch-runs/unguided
claude
# Kickoff prompt: "Have a look at program.md and let's kick off a new experiment session"
```

To merge external session trees into this archive once completed:

```bash
git fetch unguided-src main
git merge -X subtree=unguided unguided-src/main
git tag -a import/unguided-<tag> unguided-src/main -m "..."
```

## Acknowledgments

This software is built on the "autoresearch" software by github user karpathy available [here](https://github.com/karpathy/autoresearch) and distributed under the MIT license.

## Citation

Reinicke, Nicholas and Fitzgerald, Robert. _"Steering an LLM AutoResearch Loop with Domain Context: A Case Study with Vehicle Energy Models"_. US-RSE 2026, San Jose, CA, USA.

## Metadata

NLR Software Record # SWR 26-090.
