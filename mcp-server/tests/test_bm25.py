"""
tests/test_bm25.py — Tests for BM25 tokenizer, LexicalIndex, and RRF fusion.

Five groups:
  1. _tokenize_for_bm25 — Slack-specific tokenisation rules
  2. LexicalIndex        — build and search correctness
  3. _rrf_fuse           — math verified against hand-computed values
  4. Mode switching      — RETRIEVAL_MODE routes _rank_chunks() correctly
  5. _rrf_fuse edge cases — k sensitivity, tie-break stability

Run with:
    cd mcp-server
    OPENAI_API_KEY=fake-key .venv/bin/python -m pytest tests/test_bm25.py -v

Isolation strategy
------------------
Same stub pattern as test_search.py. _embed is patched for mode-switching tests.
All BM25/RRF tests are pure — no network calls, no patches needed.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Env + module stubs
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY",    "fake-key")
os.environ.setdefault("SLACK_BOT_TOKEN",   "xoxb-fake")
os.environ.setdefault("SLACK_CHANNEL_IDS", "C000001")
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
    _stub_module("dotenv", load_dotenv=lambda *a, **kw: None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import index   # noqa: E402
import server  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------
def _msg(text: str, channel: str = "C0001", ts: str = "1000000001.000001",
         permalink: str = "https://slack.test/p1") -> dict:
    return {"type": "message", "text": text, "channel": channel,
            "ts": ts, "user": "U001", "permalink": permalink}


# ---------------------------------------------------------------------------
# 1. _tokenize_for_bm25 — Slack-specific tokenisation
# ---------------------------------------------------------------------------
class TestTokenizeForBm25(unittest.TestCase):

    def tok(self, text: str) -> list[str]:
        return server._tokenize_for_bm25(text)

    # ── Case preservation ────────────────────────────────────────────────────
    def test_lowercases_plain_text(self):
        self.assertIn("hello", self.tok("Hello World"))
        self.assertNotIn("Hello", self.tok("Hello World"))

    # ── Preserve special tokens ──────────────────────────────────────────────
    def test_preserves_error_code_with_underscore(self):
        """ERR_500 must survive as a single token."""
        tokens = self.tok("we got ERR_500 in production")
        self.assertIn("err_500", tokens)

    def test_preserves_ticket_id_with_hyphen(self):
        """pd-1234 must survive as a single token."""
        tokens = self.tok("ticket pd-1234 is still open")
        self.assertIn("pd-1234", tokens)

    def test_preserves_aws_region(self):
        """us-east-1 must survive as a single token."""
        tokens = self.tok("deployed to us-east-1 this morning")
        self.assertIn("us-east-1", tokens)

    def test_preserves_username_at_sign(self):
        """@jsmith (plain text, not Slack mention) survives — @ is stripped, 'jsmith' remains."""
        tokens = self.tok("ask @jsmith about this")
        # @ is not a \w char so it's stripped; the word 'jsmith' survives
        self.assertIn("jsmith", tokens)

    # ── Slack markup handling ────────────────────────────────────────────────
    def test_drops_slack_mention(self):
        """<@U123ABC> mention syntax → dropped entirely."""
        tokens = self.tok("ping <@U123ABC> for approval")
        # no token should contain U123ABC
        self.assertFalse(any("u123abc" in t for t in tokens))

    def test_slack_url_with_label_keeps_label(self):
        """<http://example.com|Click here> → 'click' and 'here' kept."""
        tokens = self.tok("see <http://example.com|Click here> for details")
        self.assertIn("click", tokens)
        self.assertIn("here", tokens)

    def test_slack_bare_url_dropped(self):
        """<http://example.com> with no label → dropped."""
        tokens = self.tok("see <http://example.com> for details")
        self.assertFalse(any("example" in t for t in tokens))

    # ── No stemming / no stopword removal ───────────────────────────────────
    def test_stopwords_retained(self):
        """'who', 'owns', 'the' must survive — short queries need them."""
        tokens = self.tok("who owns the finn service")
        for word in ("who", "owns", "the", "finn", "service"):
            self.assertIn(word, tokens, f"Expected '{word}' in tokens: {tokens}")

    def test_no_stemming(self):
        """'running' stays 'running', not 'run'."""
        tokens = self.tok("the service is running fine")
        self.assertIn("running", tokens)
        self.assertNotIn("run", tokens)

    # ── Empty / edge inputs ──────────────────────────────────────────────────
    def test_empty_string(self):
        self.assertEqual(self.tok(""), [])

    def test_only_punctuation(self):
        self.assertEqual(self.tok("!!! ???"), [])

    def test_only_slack_mentions(self):
        """All tokens drop → empty list."""
        self.assertEqual(self.tok("<@U111> <@U222>"), [])

    def test_mixed_content(self):
        """Complex real-world message tokenises without crashing."""
        text = (
            "Hey <@U123ABC>, the ERR_500 in pd-1234 is happening in us-east-1. "
            "See <http://runbook.internal|runbook> for the fix."
        )
        tokens = self.tok(text)
        self.assertIn("err_500", tokens)
        self.assertIn("pd-1234", tokens)
        self.assertIn("us-east-1", tokens)
        self.assertIn("runbook", tokens)
        # Slack mention and URL host should be gone
        self.assertFalse(any("u123abc" in t for t in tokens))
        self.assertFalse(any("runbook.internal" in t for t in tokens))


# ---------------------------------------------------------------------------
# 2. LexicalIndex — build and search
# ---------------------------------------------------------------------------
class TestLexicalIndex(unittest.TestCase):

    def _make_chunks(self, messages: list[dict]) -> list:
        chunks = []
        for msg in messages:
            chunks.extend(server._chunk_message(msg))
        return chunks

    def test_empty_corpus_returns_empty(self):
        idx = server.LexicalIndex([])
        self.assertEqual(idx.search("anything", k=5), [])

    def test_empty_query_returns_empty(self):
        chunks = self._make_chunks([_msg("JWT authentication decision")])
        idx = server.LexicalIndex(chunks)
        self.assertEqual(idx.search("", k=5), [])

    def test_exact_keyword_ranks_first(self):
        """A message containing the exact query token must rank above unrelated ones."""
        msgs = [
            _msg("ERR_500 error in the payment service",
                 ts="1.001", permalink="https://p1"),
            _msg("Standup notes: team velocity is on track",
                 ts="1.002", permalink="https://p2"),
            _msg("Deployment pipeline blocked by Docker image",
                 ts="1.003", permalink="https://p3"),
        ]
        chunks = self._make_chunks(msgs)
        idx = server.LexicalIndex(chunks)
        results = idx.search("ERR_500", k=3)

        self.assertGreater(len(results), 0)
        top_id = results[0][0]
        self.assertEqual(top_id, "C0001:1.001",
                         f"Expected ERR_500 message first, got: {top_id}")

    def test_returns_message_ids_and_scores(self):
        """Each result is a (message_id, float_score) pair."""
        chunks = self._make_chunks([
            _msg("JWT tokens for auth", ts="1.001"),
            _msg("Docker pipeline issue", ts="1.002"),
        ])
        idx = server.LexicalIndex(chunks)
        results = idx.search("JWT", k=5)
        self.assertGreater(len(results), 0)
        for mid, score in results:
            self.assertIsInstance(mid, str)
            self.assertIsInstance(score, float)

    def test_deduplicates_to_one_result_per_message(self):
        """A long message producing multiple chunks should appear at most once."""
        # Create a message long enough to produce >1 chunk
        long_text = ("ERR_500 critical failure in the payment service. " * 60).strip()
        chunks = self._make_chunks([_msg(long_text, ts="1.001")])
        self.assertGreater(len(chunks), 1, "Need >1 chunk to test dedup")

        idx = server.LexicalIndex(chunks)
        results = idx.search("ERR_500", k=10)
        msg_ids = [mid for mid, _ in results]
        self.assertEqual(len(msg_ids), len(set(msg_ids)),
                         "Duplicate message_ids in BM25 results")

    def test_k_limits_results(self):
        msgs = [_msg(f"message number {i}", ts=f"1.{i:03d}") for i in range(10)]
        chunks = self._make_chunks(msgs)
        idx = server.LexicalIndex(chunks)
        results = idx.search("message", k=3)
        self.assertLessEqual(len(results), 3)

    def test_scores_descending(self):
        msgs = [
            _msg("ERR_500 ERR_500 ERR_500 critical error", ts="1.001"),
            _msg("ERR_500 minor issue", ts="1.002"),
            _msg("Standup notes", ts="1.003"),
        ]
        chunks = self._make_chunks(msgs)
        idx = server.LexicalIndex(chunks)
        results = idx.search("ERR_500", k=5)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         f"BM25 scores not descending: {scores}")


# ---------------------------------------------------------------------------
# 3. _rrf_fuse — math verified against hand-computed values
# ---------------------------------------------------------------------------
class TestRrfFuse(unittest.TestCase):

    def test_single_list_score(self):
        """With one list, score(d) = 1/(k+rank). Default k=60."""
        result = dict(server._rrf_fuse([["a", "b", "c"]]))
        # rank-1 → 1/61, rank-2 → 1/62, rank-3 → 1/63
        self.assertAlmostEqual(result["a"], 1/61, places=8)
        self.assertAlmostEqual(result["b"], 1/62, places=8)
        self.assertAlmostEqual(result["c"], 1/63, places=8)

    def test_document_in_both_lists_outranks_single_list(self):
        """A doc appearing in both lists must score higher than one in only one list."""
        # "shared" is rank-1 in both lists; "only_in_A" is rank-1 in list A only
        fused = dict(server._rrf_fuse([
            ["shared", "only_in_a"],
            ["shared", "only_in_b"],
        ]))
        self.assertGreater(fused["shared"], fused["only_in_a"])
        self.assertGreater(fused["shared"], fused["only_in_b"])

    def test_two_lists_additive_scores(self):
        """Score from two lists = sum of individual contributions."""
        # "a" rank-1 in list-0, rank-2 in list-1
        result = dict(server._rrf_fuse([["a", "b"], ["b", "a"]]))
        expected_a = 1/61 + 1/62   # rank-1 list-0 + rank-2 list-1
        expected_b = 1/62 + 1/61   # rank-2 list-0 + rank-1 list-1
        self.assertAlmostEqual(result["a"], expected_a, places=8)
        self.assertAlmostEqual(result["b"], expected_b, places=8)
        # Both equal here so tie-break by message_id (ascending)
        ordered = [mid for mid, _ in server._rrf_fuse([["a", "b"], ["b", "a"]])]
        self.assertEqual(ordered[0], "a")   # "a" < "b" lexicographically

    def test_custom_k_parameter(self):
        """Changing k changes the scores proportionally."""
        result_k60  = dict(server._rrf_fuse([["x"]], k=60))
        result_k10  = dict(server._rrf_fuse([["x"]], k=10))
        self.assertAlmostEqual(result_k60["x"],  1/61, places=8)
        self.assertAlmostEqual(result_k10["x"],  1/11, places=8)
        self.assertGreater(result_k10["x"], result_k60["x"])

    def test_empty_lists_return_empty(self):
        self.assertEqual(server._rrf_fuse([[], []]), [])

    def test_deterministic_tiebreak_by_message_id(self):
        """Equal-score docs must be ordered by message_id ascending (stable)."""
        # Each doc in exactly one list at the same rank → equal scores
        fused = server._rrf_fuse([["beta"], ["alpha"]])
        ids = [mid for mid, _ in fused]
        self.assertEqual(ids, ["alpha", "beta"])

    def test_output_sorted_descending_by_score(self):
        fused = server._rrf_fuse([["a", "b", "c"], ["c", "b", "a"]])
        scores = [s for _, s in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# 4. Mode switching — RETRIEVAL_MODE routes _rank_chunks() correctly
# ---------------------------------------------------------------------------
class TestModeSwitch(unittest.TestCase):
    """
    Use a fixture corpus where dense and BM25 provably disagree:
      - dense  picks the message closest in embedding space to the query
      - bm25   picks the message with the exact query token

    We patch _embed so dense always picks msg-A (semantic match),
    while the exact token "ERR_500" only appears in msg-B (keyword match).
    """

    # Fixture
    _MSG_A = _msg("service authentication failure in production",
                  ts="1.001", permalink="https://p1")
    _MSG_B = _msg("ERR_500 in the payment gateway",
                  ts="1.002", permalink="https://p2")
    _CORPUS = [_MSG_A, _MSG_B]

    # Embedding stub: query + msg-A get aligned vectors, msg-B gets orthogonal
    _VECTOR_MAP = {
        "auth":          [1.0, 0.0, 0.0],
        "authentication":[0.9, 0.1, 0.0],
        "ERR_500":       [0.0, 1.0, 0.0],
        "payment":       [0.0, 0.9, 0.1],
    }

    def _stub_embed(self, texts):
        import numpy as np
        vecs = []
        for t in texts:
            matched = False
            for kw, vec in self._VECTOR_MAP.items():
                if kw in t:
                    vecs.append(vec)
                    matched = True
                    break
            if not matched:
                vecs.append([0.1, 0.1, 0.1])
        return np.array(vecs, dtype=np.float32)

    def _indexed(self):
        """Index the fixture corpus into a throwaway store, as the server reads it."""
        store = index.Store(":memory:")
        index.index_messages(
            store, self._CORPUS, server._OPENAI_EMBED_MODEL,
            self._stub_embed, server._chunk_message,
        )
        chunks, vecs = store.load(server._OPENAI_EMBED_MODEL)
        store.close()
        return chunks, vecs

    def _run(self, mode: str) -> list[dict]:
        original = server.RETRIEVAL_MODE
        try:
            server.RETRIEVAL_MODE = mode
            chunks, vecs = self._indexed()
            with patch.object(server, "_embed", side_effect=self._stub_embed):
                return server._rank_chunks(chunks, vecs, "auth ERR_500 query", k=2)
        finally:
            server.RETRIEVAL_MODE = original

    def test_dense_mode_prefers_semantic_match(self):
        """In dense mode, msg-A (authentication) should rank first."""
        results = self._run("dense")
        self.assertGreater(len(results), 0)
        self.assertIn("authentication", results[0]["text"])

    def test_bm25_mode_prefers_exact_token(self):
        """In bm25 mode, msg-B (ERR_500) should rank first."""
        results = self._run("bm25")
        self.assertGreater(len(results), 0)
        self.assertIn("ERR_500", results[0]["text"])

    def test_hybrid_mode_returns_both_messages(self):
        """In hybrid mode, both messages must appear in results."""
        results = self._run("hybrid")
        texts = " ".join(r["text"] for r in results)
        self.assertIn("authentication", texts)
        self.assertIn("ERR_500", texts)

    def test_all_modes_return_required_keys(self):
        """Every result dict must have the full MCP response schema."""
        for mode in ("dense", "bm25", "hybrid"):
            with self.subTest(mode=mode):
                results = self._run(mode)
                for r in results:
                    for key in ("text", "author", "channel", "ts", "permalink", "score"):
                        self.assertIn(key, r, f"mode={mode}: missing '{key}' in {r}")

    def test_unknown_mode_falls_through_to_dense(self):
        """An unrecognised RETRIEVAL_MODE must not crash — falls through to dense."""
        original = server.RETRIEVAL_MODE
        try:
            server.RETRIEVAL_MODE = "unknown_mode"
            chunks, vecs = self._indexed()
            with patch.object(server, "_embed", side_effect=self._stub_embed):
                results = server._rank_chunks(chunks, vecs, "auth query", k=2)
            # Dense fall-through: should still return results
            self.assertGreater(len(results), 0)
        finally:
            server.RETRIEVAL_MODE = original


if __name__ == "__main__":
    unittest.main()
