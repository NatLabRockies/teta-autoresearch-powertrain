# Audit report

Made by `run_audit.py`. Both models are scored on the same held-out links
(327,647 of 1,638,466), using the same
`harness.py` split and metric the arms used.

## 1. Does the audit reproduce the reported numbers?

| arm | reported link RMSE | reproduced | difference | allowed |
| --- | --- | --- | --- | --- |
| unguided | 0.004785 | 0.004795 | +0.20% | ±5% |
| domain-guided | 0.006823 | 0.006823 | -0.00% | ±2% |

**Yes.** Both arms reproduce within tolerance.

The unguided arm's sequence model trains for 60 seconds of wall clock, not a
fixed number of steps, so a different machine fits a slightly different
model. It is allowed 5%. The domain-guided model is deterministic and is
allowed 2%.

## 2. Which inputs can Compass supply?

Compass scores one link at a time while it searches for a route. It knows
the current link (speed, grade, distance, geometry) and the one link before
it. It does not know which links come next. It has no energy labels.

### unguided: 5 of 13 features usable

| feature | usable | why |
| --- | --- | --- |
| `speed_mph` | yes | average speed of the current link |
| `grade_percent` | yes | average grade of the current link |
| `miles` | yes | length of the current link |
| `dke_per_mile` | **no** | uses the NEXT link's speed; the search has not chosen that link yet |
| `dke_in_link` | yes | kinetic energy change from the previous link's speed to this one |
| `gap_seconds` | **no** | idle time before the link, read from the simulated drive trace; a road map has no such value |
| `prev_miles` | yes | length of the previous link |
| `next_miles` | **no** | length of the NEXT link |
| `next_gap_seconds` | **no** | idle time after the link; from the trace, and about the NEXT link |
| `prev2_speed` | **no** | speed two links back; only one link of lookback is allowed |
| `dv_out` | **no** | speed change into the NEXT link |
| `journey_rate` | **no** | average energy of the journey's other links; there are no energy labels at inference |
| `journey_gge_per_mile` | **no** | same as journey_rate, weighted by distance |

The unguided model also has two parts that are not features. Neither
can run in Compass.

- **sequence model**: a small convolutional network over the whole journey. It reads 6 links ahead as well as 6 links behind.
- **journey offset**: a per-journey correction built from the energy labels of that journey's training links. There are no labels at inference.

### domain-guided: 11 of 11 features usable

| feature | usable | why |
| --- | --- | --- |
| `speed_mph` | yes | average speed of the current link |
| `grade_percent` | yes | average grade of the current link |
| `miles` | yes | length of the current link |
| `prev_speed_mph` | yes | speed of the previous link |
| `ke_delta_per_mile` | yes | kinetic energy change from the previous link's speed to this one |
| `sinuosity` | yes | how curvy the link is; computed from its geometry |
| `junction_turn_degrees` | yes | turn angle from the previous link into this one; from geometry |
| `prev_grade_percent` | yes | grade of the previous link |
| `prev_miles` | yes | length of the previous link |
| `prev_sinuosity` | yes | sinuosity of the previous link |
| `vertices_per_mile` | yes | how many geometry points per mile; a stand-in for road class |

## 3. Link RMSE

Baseline is the random forest both arms started from (0.013345).
Lower is better.

| model | link RMSE | vs baseline | what it is |
| --- | --- | --- | --- |
| `unguided/reported` | 0.004795 | -64.1% | all 13 features, as the arm ran it; cannot run in Compass |
| `unguided/retrained` | 0.007557 | -43.4% | same recipe, refit on the 5 usable features |
| `domain-guided` | 0.006823 | -48.9% | all 11 features usable, as the arm ran it |

What this says:

- The unguided model reported 0.004795, the best number here. But
  that model cannot run in Compass. It needs future links, simulation
  timestamps and energy labels, none of which a route search has.
- Refit on usable inputs only, the unguided recipe gives 0.007557.
  This is the fair comparison.
- The domain-guided model scores 0.006823. Every input it uses is
  available in Compass. The unguided refit is 11% worse.

## 4. Inference time

One CPU thread, like one Compass search thread. Microseconds per link,
median over repeated calls, at several batch sizes.

| batch | unguided as run | unguided retrained | domain-guided |
| --- | --- | --- | --- |
| 1 | 159,875.9 | 160,167.9 | 254.09 |
| 64 | 2,800.3 | 2,931.9 | 5.11 |
| 512 | 925.7 | 944.7 | 1.45 |
| 4,096 | 627.1 | 644.5 | 1.46 |

| model | what it is |
| --- | --- |
| `unguided/as-run` | 10,000 boosted trees (5 seeds x 2,000 trees) |
| `unguided/retrained` | 10,000 boosted trees (5 seeds x 2,000 trees) |
| `domain-guided` | 2 MLPs, 11 -> 128 -> 128 -> 1 |

What this says:

- At batch 512, the domain-guided model is **651x** faster per link
  than the unguided model refit on usable inputs.
- A route search that scores 1,000,000 links would take about
  1.4 s with the domain-guided model and 945 s with the
  unguided refit.
- The gap is a model choice, not a feature choice. The unguided arm built
  10,000 boosted trees. The domain-guided arm built two small networks,
  because `domain.md` told it inference cost mattered.
- These are Python timings. Compass is Rust, so every model would be faster
  there. The ratio is what carries over.

