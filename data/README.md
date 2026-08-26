# Data

The dataset is **not stored in this repository**, and it is not symlinked into it either.

It lives at:

```
~/data/routee-bev/processed/2017_Chevy_Bolt.parquet
sha256 420b0d4d5997c2663ef050b326fd535ee9af474970030b814c535120facb738d
```

## Why it lives outside every git repo

Both trees reach the dataset through a symlink, so wherever that symlink lands is reachable from
inside a run. If it lands inside a git repository, `git -C data log` and `git -C data tag -l`
expose that repository's entire history from within a tree that is otherwise isolated.

`tools/verify_isolation.sh` in the harness checks for this, and `tools/new_tree.sh` warns at
creation time if `--data` resolves inside a repository.

## Schema

One row per road-segment ("link"), from simulated 1 Hz drive-cycle traces for a 2017 Chevy Bolt
aggregated to the segment level.

| column | meaning |
| --- | --- |
| `journey_id` | identifies the trip a link belongs to |
| `link_start_time` | orders links within a journey |
| `speed_mph` | average speed over the link |
| `grade_percent` | average road gradient over the link |
| `miles` | link distance |
| `time_seconds` | link traversal time |
| `geometry` | link geometry, WKB, EPSG:4326 |
| `energy_rate_gge` | **target** — energy per mile in gasoline-gallon-equivalent |

`energy_rate_gge` is negative wherever regenerative braking returns energy to the battery. Those
rows are the heavy tail, they are the hardest to predict, and deleting them is the exploit the
domain-guided arm's `domain.md` forbids.

`journey_id` and `miles` do double duty: both are needed by `harness.evaluate()` to roll link
predictions up into trip totals for `trip_rmse`.

`link_start_time`, `link_end_time`, `time_seconds`, `n_points` and `energy_gge` exist only
because the data was aggregated from simulated 1 Hz traces. None of them has an inference-time
counterpart, which is why `audit/contract.py` gives anything derived from them its own `TRACE`
tier rather than treating it as a memory-cost question.
