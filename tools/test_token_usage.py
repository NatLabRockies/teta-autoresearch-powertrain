"""Tests for token_usage.py. Run: `pixi run python -m unittest tools.test_token_usage`."""

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


def _assistant(model: str, **usage: int) -> dict:
    return {"message": {"role": "assistant", "model": model, "usage": dict(usage)}}


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


class SplitTagTests(unittest.TestCase):
    def test_simple(self) -> None:
        self.assertEqual(token_usage.split_tag("bev-apr23"), ("bev", "apr23"))

    def test_multi_dash_variant_kept_intact(self) -> None:
        self.assertEqual(
            token_usage.split_tag("bev-tuned-apr23"), ("bev-tuned", "apr23")
        )

    def test_rejects_no_dash(self) -> None:
        with self.assertRaises(ValueError):
            token_usage.split_tag("bev")

    def test_rejects_empty_side(self) -> None:
        with self.assertRaises(ValueError):
            token_usage.split_tag("-apr23")
        with self.assertRaises(ValueError):
            token_usage.split_tag("bev-")


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
            out = Path(td) / "results" / "bev" / "usage-apr23.jsonl"

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


if __name__ == "__main__":
    unittest.main()
