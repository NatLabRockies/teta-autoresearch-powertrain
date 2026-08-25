"""Tests for token_usage.py. Run: `pixi run test`."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

import token_usage  # noqa: E402


def _write_transcript(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _assistant(
    model: str, at: str | None = None, sidechain: bool = False, **usage: int
) -> dict:
    record: dict = {
        "message": {"role": "assistant", "model": model, "usage": dict(usage)}
    }
    if at is not None:
        record["timestamp"] = at
    if sidechain:
        record["isSidechain"] = True
    return record


class AggregateByModelTests(unittest.TestCase):
    def test_sums_across_files_grouped_by_model(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "session-a.jsonl",
                [
                    _assistant(
                        "claude-opus-4-7",
                        input_tokens=100,
                        output_tokens=50,
                        cache_creation_input_tokens=200,
                        cache_read_input_tokens=1000,
                    ),
                    _assistant(
                        "claude-sonnet-4-6",
                        input_tokens=10,
                        output_tokens=5,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=20,
                    ),
                    {"message": {"role": "user", "content": "hi"}},  # ignored
                ],
            )
            _write_transcript(
                project / "session-b.jsonl",
                [
                    _assistant(
                        "claude-opus-4-7",
                        input_tokens=25,
                        output_tokens=75,
                        cache_creation_input_tokens=300,
                        cache_read_input_tokens=500,
                    ),
                ],
            )

            totals = token_usage.aggregate_by_model(project)

        self.assertEqual(set(totals), {"claude-opus-4-7", "claude-sonnet-4-6"})
        opus = totals["claude-opus-4-7"]
        self.assertEqual(opus["input_tokens"], 125)
        self.assertEqual(opus["output_tokens"], 125)
        self.assertEqual(opus["cache_creation_input_tokens"], 500)
        self.assertEqual(opus["cache_read_input_tokens"], 1500)
        self.assertEqual(opus["assistant_messages"], 2)
        sonnet = totals["claude-sonnet-4-6"]
        self.assertEqual(sonnet["input_tokens"], 10)
        self.assertEqual(sonnet["assistant_messages"], 1)

    def test_missing_dir_returns_empty(self) -> None:
        with TemporaryDirectory() as td:
            self.assertEqual(
                token_usage.aggregate_by_model(Path(td) / "does-not-exist"), {}
            )

    def test_skips_malformed_lines_and_non_assistant(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            (project / "s.jsonl").write_text(
                "not-json\n"
                + json.dumps({"message": {"role": "user"}})
                + "\n"
                + "\n"
                + json.dumps(_assistant("claude-haiku-4-5", input_tokens=7))
                + "\n"
            )
            totals = token_usage.aggregate_by_model(project)
        self.assertEqual(list(totals), ["claude-haiku-4-5"])
        self.assertEqual(totals["claude-haiku-4-5"]["input_tokens"], 7)


class AppendIdempotencyTests(unittest.TestCase):
    def test_two_runs_yield_equal_cumulative_totals(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td) / ".claude" / "projects" / "proj"
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant("claude-opus-4-7", input_tokens=10, output_tokens=3),
                    _assistant("claude-opus-4-7", input_tokens=20, output_tokens=4),
                ],
            )
            out = Path(td) / "results" / "usage-bev-apr23.jsonl"

            now1 = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
            now2 = datetime(2026, 4, 23, 13, 0, 0, tzinfo=timezone.utc)
            totals = token_usage.aggregate_by_model(project)
            token_usage.append_records(
                out, token_usage.build_records("bev-apr23", totals, now=now1)
            )
            token_usage.append_records(
                out, token_usage.build_records("bev-apr23", totals, now=now2)
            )

            lines = out.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)
        recs = [json.loads(line) for line in lines]
        # Same cumulative totals — second run did NOT double the counts.
        self.assertEqual(recs[0]["input_tokens"], recs[1]["input_tokens"])
        self.assertEqual(recs[0]["output_tokens"], recs[1]["output_tokens"])
        self.assertEqual(recs[0]["input_tokens"], 30)
        self.assertEqual(recs[0]["output_tokens"], 7)
        # Snapshot timestamps differ.
        self.assertNotEqual(recs[0]["snapshot_at"], recs[1]["snapshot_at"])


class ParseStampTests(unittest.TestCase):
    def test_normalizes_to_aware_utc(self) -> None:
        # A naive stamp must not come back naive: attribution compares
        # transcript stamps against experiment stamps, and mixing the two
        # would raise instead of returning a wrong-but-quiet answer.
        naive = token_usage.parse_stamp("2026-05-08T19:00:00")
        aware = token_usage.parse_stamp("2026-05-08T19:00:00Z")
        assert naive is not None and aware is not None
        self.assertEqual(naive, aware)

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(token_usage.parse_stamp("not-a-time"))


class ReadExperimentsTests(unittest.TestCase):
    def _write(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_returns_pairs_sorted_by_end_time(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "experiments-t.jsonl"
            self._write(
                path,
                [
                    {"exp": 2, "ended_at": "2026-05-08T12:00:00Z"},
                    {"exp": 0, "ended_at": "2026-05-08T10:00:00Z"},
                    {"exp": 1, "ended_at": "2026-05-08T11:00:00Z"},
                ],
            )
            found = token_usage.read_experiments(path)
        self.assertEqual([exp for exp, _ in found], [0, 1, 2])

    def test_skips_unusable_lines(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "experiments-t.jsonl"
            path.write_text(
                "not-json\n"
                + json.dumps({"exp": 0})  # no ended_at
                + "\n"
                + json.dumps({"ended_at": "2026-05-08T10:00:00Z"})  # no exp
                + "\n"
                + json.dumps({"exp": 1, "ended_at": "nonsense"})
                + "\n"
                + json.dumps({"exp": 2, "ended_at": "2026-05-08T12:00:00Z"})
                + "\n"
            )
            found = token_usage.read_experiments(path)
        self.assertEqual([exp for exp, _ in found], [2])

    def test_missing_file_returns_empty(self) -> None:
        with TemporaryDirectory() as td:
            self.assertEqual(token_usage.read_experiments(Path(td) / "nope"), [])


class AttributionTests(unittest.TestCase):
    """The per-experiment view: tokens partitioned by `ended_at` boundaries."""

    EXPERIMENTS = [
        (0, token_usage.parse_stamp("2026-05-08T10:00:00Z")),
        (1, token_usage.parse_stamp("2026-05-08T11:00:00Z")),
    ]

    def _experiments(self) -> list:
        return [(exp, when) for exp, when in self.EXPERIMENTS if when is not None]

    def _events(self, project: Path) -> list:
        return token_usage.collect_events(project)

    def test_partitions_by_end_time_including_a_boundary_hit(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    # setup, before exp0 finished -> exp 0
                    _assistant("m", at="2026-05-08T09:30:00Z", input_tokens=1),
                    # exactly on exp0's boundary -> exp 0, not exp 1
                    _assistant("m", at="2026-05-08T10:00:00Z", input_tokens=2),
                    # between the two -> exp 1
                    _assistant("m", at="2026-05-08T10:30:00Z", input_tokens=4),
                    # after the last experiment -> the tail bucket
                    _assistant("m", at="2026-05-08T12:00:00Z", input_tokens=8),
                ],
            )
            buckets = token_usage.attribute_by_experiment(
                self._events(project), self._experiments()
            )

        self.assertEqual(buckets[0]["m"]["input_tokens"], 3)
        self.assertEqual(buckets[1]["m"]["input_tokens"], 4)
        self.assertEqual(buckets[None]["m"]["input_tokens"], 8)
        self.assertEqual(buckets[0]["m"]["assistant_messages"], 2)

    def test_attributed_totals_sum_to_the_session_total(self) -> None:
        # The partition must be exhaustive: nothing between the first record
        # and the last may go uncounted, or per-experiment cost silently
        # understates what the session actually spent.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant("m", at="2026-05-08T09:30:00Z", output_tokens=5),
                    _assistant("m", at="2026-05-08T10:30:00Z", output_tokens=7),
                    _assistant("other", at="2026-05-08T12:00:00Z", output_tokens=9),
                ],
            )
            events = self._events(project)
            session = token_usage.totals_by_model(events)
            buckets = token_usage.attribute_by_experiment(events, self._experiments())

        attributed = sum(
            b["output_tokens"]
            for by_model in buckets.values()
            for b in by_model.values()
        )
        self.assertEqual(attributed, sum(b["output_tokens"] for b in session.values()))

    def test_timestampless_records_count_for_session_but_not_experiments(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [_assistant("m", input_tokens=100)],  # no timestamp at all
            )
            events = self._events(project)
            session = token_usage.totals_by_model(events)
            buckets = token_usage.attribute_by_experiment(events, self._experiments())

        self.assertEqual(session["m"]["input_tokens"], 100)
        self.assertEqual(buckets, {})

    def test_no_experiments_puts_everything_in_the_tail(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [_assistant("m", at="2026-05-08T09:30:00Z", input_tokens=3)],
            )
            buckets = token_usage.attribute_by_experiment(self._events(project), [])
        self.assertEqual(list(buckets), [None])

    def test_records_are_written_in_experiment_order_with_the_tail_last(self) -> None:
        buckets: dict = {
            1: {"m": {"input_tokens": 1}},
            None: {"m": {"input_tokens": 2}},
            0: {"m": {"input_tokens": 3}},
        }
        records = token_usage.build_exp_records("t", buckets)
        self.assertEqual([r["exp"] for r in records], [0, 1, None])
        self.assertTrue(all(r["scope"] == "exp" for r in records))


class WindowTests(unittest.TestCase):
    def test_since_drops_earlier_and_timestampless_records(self) -> None:
        # An aborted session in the same tree path must not be folded in, and
        # a record that cannot prove it is inside the window is not inside it.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant("m", at="2026-05-01T00:00:00Z", input_tokens=1000),
                    _assistant("m", input_tokens=500),
                    _assistant("m", at="2026-05-08T09:00:00Z", input_tokens=7),
                ],
            )
            since = token_usage.parse_stamp("2026-05-08T00:00:00Z")
            totals = token_usage.aggregate_by_model(project, since=since)
        self.assertEqual(totals["m"]["input_tokens"], 7)
        self.assertEqual(totals["m"]["assistant_messages"], 1)


class ContextWindowTests(unittest.TestCase):
    """Occupancy, not cost: a peak and a final reading, never a sum."""

    EXPERIMENTS = [
        (0, token_usage.parse_stamp("2026-05-08T10:00:00Z")),
        (1, token_usage.parse_stamp("2026-05-08T11:00:00Z")),
    ]

    def _experiments(self) -> list:
        return [(exp, when) for exp, when in self.EXPERIMENTS if when is not None]

    def test_occupancy_is_input_plus_both_cache_fields(self) -> None:
        # Output tokens are billed to this message but only enter the window
        # as input on the next one, so they must not be counted here.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant(
                        "m",
                        at="2026-05-08T09:00:00Z",
                        input_tokens=2,
                        cache_creation_input_tokens=6787,
                        cache_read_input_tokens=49397,
                        output_tokens=864,
                    )
                ],
            )
            events = token_usage.collect_events(project)
        self.assertEqual(events[0].context_tokens, 56186)

    def test_last_and_max_over_an_experiment_window(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant("m", at="2026-05-08T09:10:00Z", input_tokens=100),
                    _assistant("m", at="2026-05-08T09:20:00Z", input_tokens=900),
                    _assistant("m", at="2026-05-08T09:30:00Z", input_tokens=400),
                ],
            )
            buckets = token_usage.attribute_by_experiment(
                token_usage.collect_events(project), self._experiments()
            )
        self.assertEqual(buckets[0]["m"]["context_tokens_max"], 900)
        self.assertEqual(buckets[0]["m"]["context_tokens_last"], 400)

    def test_last_follows_the_clock_not_the_order_files_are_read(self) -> None:
        # rglob sorts by path. `session-b` sorts after `session-a` but ran
        # first, and reporting b's reading as "last" would be wrong.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "session-a.jsonl",
                [_assistant("m", at="2026-05-08T09:50:00Z", input_tokens=300)],
            )
            _write_transcript(
                project / "session-b.jsonl",
                [_assistant("m", at="2026-05-08T09:10:00Z", input_tokens=800)],
            )
            buckets = token_usage.attribute_by_experiment(
                token_usage.collect_events(project), self._experiments()
            )
        self.assertEqual(buckets[0]["m"]["context_tokens_last"], 300)
        self.assertEqual(buckets[0]["m"]["context_tokens_max"], 800)

    def test_a_drop_between_experiments_is_preserved(self) -> None:
        # Auto-compaction resets the window. exp1 ending lower than exp0 is
        # the real reading, not something to smooth over.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant("m", at="2026-05-08T09:30:00Z", input_tokens=150_000),
                    _assistant("m", at="2026-05-08T10:30:00Z", input_tokens=20_000),
                ],
            )
            buckets = token_usage.attribute_by_experiment(
                token_usage.collect_events(project), self._experiments()
            )
        self.assertEqual(buckets[0]["m"]["context_tokens_last"], 150_000)
        self.assertEqual(buckets[1]["m"]["context_tokens_last"], 20_000)

    def test_subagents_are_billed_but_excluded_from_occupancy(self) -> None:
        # A subagent runs in its own window, so folding its turns into the
        # main session's peak would report a peak that never happened. Its
        # tokens are still billed to the session.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [_assistant("m", at="2026-05-08T09:10:00Z", input_tokens=500)],
            )
            _write_transcript(
                project / "s" / "subagents" / "sub.jsonl",
                [_assistant("m", at="2026-05-08T09:20:00Z", input_tokens=999_999)],
            )
            events = token_usage.collect_events(project)
            totals = token_usage.totals_by_model(events)
            buckets = token_usage.attribute_by_experiment(events, self._experiments())
        self.assertEqual(totals["m"]["input_tokens"], 1_000_499)
        self.assertEqual(totals["m"]["assistant_messages"], 2)
        self.assertEqual(totals["m"]["context_tokens_max"], 500)
        self.assertEqual(buckets[0]["m"]["context_tokens_max"], 500)
        self.assertEqual(buckets[0]["m"]["context_tokens_last"], 500)

    def test_the_isSidechain_flag_alone_is_enough(self) -> None:
        # Same exclusion when the transcript is not under a subagents/ dir.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [
                    _assistant("m", at="2026-05-08T09:10:00Z", input_tokens=500),
                    _assistant(
                        "m",
                        at="2026-05-08T09:20:00Z",
                        sidechain=True,
                        input_tokens=999_999,
                    ),
                ],
            )
            totals = token_usage.totals_by_model(token_usage.collect_events(project))
        self.assertEqual(totals["m"]["input_tokens"], 1_000_499)
        self.assertEqual(totals["m"]["context_tokens_max"], 500)

    def test_a_window_of_only_subagent_turns_reports_no_occupancy(self) -> None:
        # Not "the context was empty" — not measured. Reporting 0 would be a
        # claim about the main window that these events cannot support.
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s" / "subagents" / "sub.jsonl",
                [_assistant("m", at="2026-05-08T09:10:00Z", input_tokens=42)],
            )
            buckets = token_usage.attribute_by_experiment(
                token_usage.collect_events(project), self._experiments()
            )
        bucket = buckets[0]["m"]
        self.assertEqual(bucket["input_tokens"], 42)
        self.assertNotIn("context_tokens_max", bucket)
        self.assertNotIn("context_tokens_last", bucket)

    def test_occupancy_reaches_the_written_records(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            _write_transcript(
                project / "s.jsonl",
                [_assistant("m", at="2026-05-08T09:10:00Z", input_tokens=1234)],
            )
            events = token_usage.collect_events(project)
            session = token_usage.build_records(
                "t",
                token_usage.totals_by_model(events),
                now=datetime(2026, 5, 8, 13, 0, 0, tzinfo=timezone.utc),
            )
            per_exp = token_usage.build_exp_records(
                "t", token_usage.attribute_by_experiment(events, self._experiments())
            )
        self.assertEqual(session[0]["context_tokens_last"], 1234)
        self.assertEqual(session[0]["context_tokens_max"], 1234)
        self.assertEqual(per_exp[0]["context_tokens_last"], 1234)
        self.assertEqual(per_exp[0]["context_tokens_max"], 1234)


if __name__ == "__main__":
    unittest.main()
