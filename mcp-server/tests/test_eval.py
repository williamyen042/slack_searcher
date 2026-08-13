"""
tests/test_eval.py — Pytest for the eval harness logic.

Tests the metric functions and the end-to-end eval loop against a tiny
hand-crafted fixture corpus (5 messages, 3 queries) with deterministic
stub embeddings.  All assertions are exact — no real network calls.

Run with:
    cd mcp-server
    OPENAI_API_KEY=fake-key .venv/bin/python -m pytest tests/test_eval.py -v

Isolation strategy
------------------
- server.py is imported with the same stubs used by test_search.py.
- _embed is patched with a deterministic stub for the eval loop test.
- _load_corpus is patched to return the fixture messages directly.
- No golden_queries.jsonl file is read; a hand-built list is passed in memory.
"""

import os
import sys
import json
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Env + module stubs (same pattern as test_search.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY",     "fake-key")
os.environ.setdefault("SLACK_BOT_TOKEN",    "xoxb-fake")
os.environ.setdefault("SLACK_CHANNEL_IDS",  "C000001")
os.environ.pop("SLACK_EXPORT_PATH", None)


def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "fastmcp" not in sys.modules:
    _fmcp = _stub_module("fastmcp")

    class _FakeMCP:
        def __init__(self, **kw): pass
        def tool(self):
            def decorator(fn): return fn
            return decorator
        def run(self, **kw): pass

    _fmcp.FastMCP = _FakeMCP

if "slack_sdk" not in sys.modules:
    _sdk = _stub_module("slack_sdk")

    class _FakeWebClient:
        def __init__(self, **kw): pass

    _sdk.WebClient = _FakeWebClient
    _stub_module("slack_sdk.errors", SlackApiError=Exception)

if "dotenv" not in sys.modules:
    _stub_module("dotenv", load_dotenv=lambda *a, **kw: None)  # noqa: E731

# run_eval.py imports dotenv at module level before server; ensure the stub
# absorbs any kwargs (dotenv_path=...) it passes
else:
    # already stubbed by a sibling test file with the same lambda — no-op
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server  # noqa: E402

# ---------------------------------------------------------------------------
# Import eval helpers directly from run_eval.py
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))
import run_eval  # noqa: E402  (eval/run_eval.py)


# ---------------------------------------------------------------------------
# Fixture corpus — 5 messages covering distinct topics
# ---------------------------------------------------------------------------
# message_id = "channel:ts"
_CORPUS = [
    {
        "type": "message",
        "text": "We decided to use JWT for authentication in the new service.",
        "channel": "C0001", "ts": "1000000001.000001",
        "user": "U001",
        "permalink": "https://slack.test/archives/C0001/p1000000001000001",
    },
    {
        "type": "message",
        "text": "Deployment pipeline is blocked due to a Docker image size issue.",
        "channel": "C0001", "ts": "1000000002.000002",
        "user": "U002",
        "permalink": "https://slack.test/archives/C0001/p1000000002000002",
    },
    {
        "type": "message",
        "text": "Standup notes: team velocity is on track for the sprint.",
        "channel": "C0001", "ts": "1000000003.000003",
        "user": "U003",
        "permalink": "https://slack.test/archives/C0001/p1000000003000003",
    },
    {
        "type": "message",
        "text": "ERR_500 in the payment service — pd-1234 opened to track the fix.",
        "channel": "C0002", "ts": "1000000004.000004",
        "user": "U001",
        "permalink": "https://slack.test/archives/C0002/p1000000004000004",
    },
    {
        "type": "message",
        "text": "Agreed: bcrypt with cost factor 12 for password hashing.",
        "channel": "C0002", "ts": "1000000005.000005",
        "user": "U004",
        "permalink": "https://slack.test/archives/C0002/p1000000005000005",
    },
]

# Relevant IDs for each fixture query (channel:ts format)
_GOLDEN = [
    {
        "query":        "query about JWT authentication",
        "relevant_ids": ["C0001:1000000001.000001"],
        "query_type":   "keyword",
    },
    {
        "query":        "why is the deployment pipeline blocked",
        "relevant_ids": ["C0001:1000000002.000002"],
        "query_type":   "semantic",
    },
    {
        "query":        "ERR_500 pd-1234 payment service error",
        "relevant_ids": ["C0002:1000000004.000004"],
        "query_type":   "mixed",
    },
]

# ---------------------------------------------------------------------------
# Deterministic embedding stub
# Maps a keyword in the text → fixed 3-dim unit vector.
# Query is always the first text passed; its vector matches message 0/1/2.
# ---------------------------------------------------------------------------
_VECTOR_MAP = {
    "JWT":        [1.0, 0.0, 0.0],
    "Docker":     [0.0, 1.0, 0.0],
    "Standup":    [0.5, 0.5, 0.0],
    "ERR_500":    [0.0, 0.0, 1.0],
    "bcrypt":     [0.3, 0.3, 0.3],
    # query keywords
    "JWT authentication": [1.0, 0.0, 0.0],
    "deployment pipeline blocked": [0.0, 1.0, 0.0],
    "ERR_500 pd-1234":    [0.0, 0.0, 1.0],
}

def _stub_embed(texts: list[str]):
    import numpy as np
    vecs = []
    for t in texts:
        matched = False
        for kw, vec in _VECTOR_MAP.items():
            if kw in t:
                vecs.append(vec)
                matched = True
                break
        if not matched:
            vecs.append([0.1, 0.1, 0.1])
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Unit tests for metric functions
# ---------------------------------------------------------------------------
class TestRecallAtK(unittest.TestCase):

    def test_perfect_recall_at_5(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant  = ["a", "c"]
        self.assertAlmostEqual(run_eval._recall_at_k(retrieved, relevant, 5), 1.0)

    def test_zero_recall_nothing_found(self):
        retrieved = ["x", "y", "z"]
        relevant  = ["a", "b"]
        self.assertAlmostEqual(run_eval._recall_at_k(retrieved, relevant, 5), 0.0)

    def test_partial_recall(self):
        retrieved = ["a", "x", "x", "x", "x", "b"]
        relevant  = ["a", "b"]
        # only "a" in top-5, "b" is at rank 6
        self.assertAlmostEqual(run_eval._recall_at_k(retrieved, relevant, 5), 0.5)

    def test_empty_relevant_returns_zero(self):
        self.assertAlmostEqual(run_eval._recall_at_k(["a", "b"], [], 5), 0.0)

    def test_k_cutoff_respected(self):
        retrieved = ["x", "x", "x", "a"]   # "a" is at rank 4
        relevant  = ["a"]
        self.assertAlmostEqual(run_eval._recall_at_k(retrieved, relevant, 3), 0.0)
        self.assertAlmostEqual(run_eval._recall_at_k(retrieved, relevant, 4), 1.0)


class TestMRRAtK(unittest.TestCase):

    def test_first_hit_at_rank_1(self):
        self.assertAlmostEqual(run_eval._mrr_at_k(["a", "b", "c"], ["a"], 10), 1.0)

    def test_first_hit_at_rank_2(self):
        self.assertAlmostEqual(run_eval._mrr_at_k(["x", "a", "c"], ["a"], 10), 0.5)

    def test_first_hit_at_rank_3(self):
        self.assertAlmostEqual(run_eval._mrr_at_k(["x", "y", "a"], ["a"], 10), 1/3, places=5)

    def test_no_hit_returns_zero(self):
        self.assertAlmostEqual(run_eval._mrr_at_k(["x", "y", "z"], ["a"], 10), 0.0)

    def test_empty_relevant_returns_zero(self):
        self.assertAlmostEqual(run_eval._mrr_at_k(["a", "b"], [], 10), 0.0)

    def test_k_cutoff_excludes_late_hit(self):
        # hit at rank 5, k=4 → 0
        retrieved = ["x", "x", "x", "x", "a"]
        self.assertAlmostEqual(run_eval._mrr_at_k(retrieved, ["a"], 4), 0.0)
        self.assertAlmostEqual(run_eval._mrr_at_k(retrieved, ["a"], 5), 0.2)


class TestAggregate(unittest.TestCase):

    def test_average(self):
        self.assertAlmostEqual(run_eval._aggregate([0.5, 1.0, 0.0]), 0.5)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(run_eval._aggregate([]), 0.0)


# ---------------------------------------------------------------------------
# Integration test: full eval loop on fixture corpus
# ---------------------------------------------------------------------------
class TestEvalLoop(unittest.TestCase):
    """
    Run run_eval.run_eval() against the fixture corpus and golden set.
    _embed is stubbed deterministically; _load_corpus is patched to return
    _CORPUS.  Asserts that:
      - results file is written with the correct structure
      - overall recall and MRR are > 0 (fixture vectors are designed to hit)
      - per-query_type breakdown keys are present
    """

    def test_eval_loop_produces_results(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"

            # Write a tiny golden JSONL to a temp file
            golden_path = Path(tmp) / "golden.jsonl"
            with open(golden_path, "w") as f:
                for entry in _GOLDEN:
                    f.write(json.dumps(entry) + "\n")

            with patch.object(server, "_embed", side_effect=_stub_embed), \
                 patch.object(run_eval, "_load_corpus", return_value=_CORPUS):
                output = run_eval.run_eval("dense", golden_path, results_dir)

            # Structure checks (inside the with-block so tempdir still exists)
            self.assertEqual(output["mode"], "dense")
            self.assertIn("aggregates", output)
            self.assertIn("per_query",  output)
            self.assertEqual(output["golden_count"], len(_GOLDEN))

            # At least one result file written
            files = list(results_dir.glob("*_dense.json"))
            self.assertEqual(len(files), 1)

        # Aggregates must have overall + each query_type present in golden
        agg = output["aggregates"]
        self.assertIn("overall",  agg)
        self.assertIn("keyword",  agg)
        self.assertIn("semantic", agg)
        self.assertIn("mixed",    agg)

        # Overall metrics should be > 0 because our stub vectors are designed to rank
        # the relevant doc first for each query
        self.assertGreater(agg["overall"]["mrr_at_10"], 0.0,
                           "Expected MRR > 0 with deterministic fixture vectors")
        self.assertGreater(agg["overall"]["recall_at_5"], 0.0,
                           "Expected Recall@5 > 0 with deterministic fixture vectors")

    def test_per_query_result_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden_path = Path(tmp) / "golden.jsonl"
            with open(golden_path, "w") as f:
                f.write(json.dumps(_GOLDEN[0]) + "\n")

            with patch.object(server, "_embed", side_effect=_stub_embed), \
                 patch.object(run_eval, "_load_corpus", return_value=_CORPUS):
                output = run_eval.run_eval("dense", golden_path, Path(tmp) / "r")

        pq = output["per_query"][0]
        for key in ("query", "query_type", "relevant_ids", "retrieved_ids",
                    "recall_at_5", "recall_at_10", "mrr_at_10"):
            self.assertIn(key, pq, f"Missing key '{key}' in per_query result")

    def test_empty_relevant_ids_scores_zero(self):
        """A query with no relevant_ids must score 0 in all metrics."""
        no_judgements = [{"query": "anything", "relevant_ids": [], "query_type": "semantic"}]
        with tempfile.TemporaryDirectory() as tmp:
            golden_path = Path(tmp) / "golden.jsonl"
            with open(golden_path, "w") as f:
                f.write(json.dumps(no_judgements[0]) + "\n")

            with patch.object(server, "_embed", side_effect=_stub_embed), \
                 patch.object(run_eval, "_load_corpus", return_value=_CORPUS):
                output = run_eval.run_eval("dense", golden_path, Path(tmp) / "r")

        pq = output["per_query"][0]
        self.assertEqual(pq["recall_at_5"],  0.0)
        self.assertEqual(pq["recall_at_10"], 0.0)
        self.assertEqual(pq["mrr_at_10"],    0.0)


if __name__ == "__main__":
    unittest.main()
