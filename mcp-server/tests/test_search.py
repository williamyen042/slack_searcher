"""
Tests for search_slack and get_thread tool behaviour.

Four groups:
  1. Empty-results contract  — tool returns [] → callers get [] (the "not found" path)
  2. Permalink guarantee      — every result dict includes a non-empty permalink
  3. Cosine ranking           — results are ordered by semantic similarity to the query
  4. Routing                  — search_slack() delegates ranking to _rank_chunks()

Run with:
    cd mcp-server
    /path/to/venv/bin/python -m pytest tests/test_search.py -v

Isolation strategy
------------------
Heavy external dependencies (openai, fastmcp, slack_sdk, dotenv) are stubbed
with the same pattern used in test_chunking.py.  The embedding layer (_embed) is
replaced with a deterministic fixture that returns hand-crafted vectors so that
ranking tests are exact, not probabilistic.

Because indexing and querying are now separate, each test builds a real (but
temporary) SQLite index with stubbed vectors and points server.INDEX_PATH at it.
That means these tests exercise the same store the production server reads.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Minimal environment stubs — must be set before server.py is imported.
# Mirror exactly what test_chunking.py does.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY",     "fake-key")
os.environ.setdefault("SLACK_BOT_TOKEN",    "xoxb-fake")
os.environ.setdefault("SLACK_CHANNEL_IDS",  "C000001")
# Ensure live-API mode (not export mode) throughout this file.
os.environ.pop("SLACK_EXPORT_PATH", None)


def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Only stub if the server hasn't been imported yet (test_chunking.py may have
# already installed stubs when the two test files run in the same session).
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

# ---------------------------------------------------------------------------
# Import server (safe now that stubs are in place)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import index   # noqa: E402
import server  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _msg(text: str, channel: str = "C0001", ts: str = "1000000000.000001",
         user: str = "U001", permalink: str = "", thread_ts: str = "") -> dict:
    """Minimal Slack message dict as produced by _fetch_channel_api."""
    m = {
        "type":      "message",
        "text":      text,
        "channel":   channel,
        "ts":        ts,
        "user":      user,
        "permalink": permalink or f"https://slack.test/archives/{channel}/p{ts.replace('.', '')}",
    }
    if thread_ts:
        m["thread_ts"] = thread_ts
    return m


class _IndexFixture(unittest.TestCase):
    """
    Base class that builds a temporary on-disk index and points server at it.

    Subclasses supply `embed()`; `build(messages)` indexes them with that stub
    and forces a reload, so `server.search_slack()` reads a real Store.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="slack-index-test-")
        self._orig_path = server.INDEX_PATH
        server.INDEX_PATH = os.path.join(self._tmpdir, "test_index.db")
        self._patcher = patch.object(server, "_embed", side_effect=self.embed)
        self._patcher.start()
        self.build([])

    def tearDown(self):
        self._patcher.stop()
        server.INDEX_PATH = self._orig_path
        server._load_index(force=True)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def embed(self, texts):
        """Default stub: non-zero vectors so cosine similarity is well-defined."""
        vecs = []
        for t in texts:
            v = np.zeros(16, dtype=np.float32)
            for i, ch in enumerate(t[:16]):
                v[i] = float(ord(ch))
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def build(self, messages: list[dict]) -> None:
        """Index *messages* into the temp store and reload the server's cache."""
        store = index.Store(server.INDEX_PATH)
        index.index_messages(
            store, messages, server._OPENAI_EMBED_MODEL, self.embed, server._chunk_message
        )
        store.close()
        server._load_index(force=True)


# ---------------------------------------------------------------------------
# 1. Empty-results contract
# ---------------------------------------------------------------------------
class TestEmptyResults(_IndexFixture):
    """
    search_slack and get_thread must return [] when there is nothing to return.
    The MCP system prompt then forces the assistant into the "I couldn't find anything" path.
    """

    def test_search_slack_empty_index_returns_empty(self):
        """With nothing indexed, search_slack returns [] rather than guessing."""
        self.assertEqual(server.search_slack(query="auth decision"), [])

    def test_search_slack_channel_filter_matching_nothing_returns_empty(self):
        """A channel filter that matches no indexed chunk produces []."""
        self.build([_msg("Deployment pipeline is blocked.", ts="1.001")])
        self.assertEqual(
            server.search_slack(query="deployment", channels=["C-NOPE"]), []
        )

    def test_search_slack_only_system_subtypes_returns_empty(self):
        """Messages whose subtypes are in _SYSTEM_SUBTYPES never make it into the index."""
        self.build([
            _msg("Alice joined the channel", ts="1.001") | {"subtype": "channel_join"},
            _msg("", ts="1.002") | {"subtype": "bot_add"},
        ])
        self.assertEqual(server.search_slack(query="anything"), [])

    def test_search_slack_only_empty_text_messages_returns_empty(self):
        """Messages with empty text are filtered before chunking."""
        self.build([
            _msg("", ts="1.001"),
            _msg("   ", ts="1.002"),
        ])
        self.assertEqual(server.search_slack(query="hello"), [])

    def test_get_thread_unknown_thread_returns_empty(self):
        """A thread_ts with nothing indexed under it returns []."""
        self.build([_msg("Unrelated message.", ts="9.001")])
        self.assertEqual(
            server.get_thread(channel_id="C0001", thread_ts="1234.001"), []
        )

    def test_get_thread_empty_index_returns_empty(self):
        """get_thread against an empty index returns []."""
        self.assertEqual(server.get_thread(channel_id="C0001", thread_ts="1.001"), [])

    def test_get_reactions_without_token_returns_empty(self):
        """get_reactions is the one tool needing live Slack; no client → []."""
        with patch.object(server, "slack", None):
            self.assertEqual(
                server.get_reactions(channel_id="C0001", message_ts="1.001"), []
            )


# ---------------------------------------------------------------------------
# 2. Permalink guarantee
# ---------------------------------------------------------------------------
class TestPermalinkGuarantee(_IndexFixture):
    """
    Every dict returned by search_slack and get_thread must have a non-empty
    'permalink' key.  This is the structural requirement for citation grounding.
    """

    def test_search_slack_all_results_have_permalink(self):
        """Every result from search_slack must include a non-empty permalink."""
        self.build([
            _msg("Auth decision: JWT tokens", ts="1.001", permalink="https://slack.test/p1"),
            _msg("Deploy blocked by Docker issue", ts="1.002", permalink="https://slack.test/p2"),
        ])
        results = server.search_slack(query="auth", channels=["C0001"])

        self.assertGreater(len(results), 0, "Expected at least one result")
        for r in results:
            self.assertIn("permalink", r, f"Missing 'permalink' key in result: {r}")
            self.assertTrue(r["permalink"], f"Empty permalink in result: {r}")

    def test_permalink_is_built_without_an_api_call(self):
        """
        Permalinks are deterministic, so _permalink must format them locally
        instead of issuing one chat.getPermalink request per message.
        """
        calls = []
        with patch.object(server, "_get_workspace_url", return_value="https://acme.slack.com"), \
             patch.object(server, "_resolve_permalink_api",
                          side_effect=lambda c, t: calls.append((c, t)) or "API"):
            url = server._permalink("C0123", "1699999999.123456")

        self.assertEqual(url, "https://acme.slack.com/archives/C0123/p1699999999123456")
        self.assertEqual(calls, [], "chat.getPermalink must not be called per message")

    def test_permalink_falls_back_to_api_when_workspace_url_unavailable(self):
        """If auth.test fails, fall back to chat.getPermalink rather than returning ''."""
        with patch.object(server, "_get_workspace_url", return_value=""), \
             patch.object(server, "_resolve_permalink_api", return_value="https://fallback/p1"):
            self.assertEqual(server._permalink("C1", "1.1"), "https://fallback/p1")

    def test_get_thread_all_replies_have_permalink(self):
        """Every message returned by get_thread must have a non-empty permalink."""
        self.build([
            _msg("We should use bcrypt for password hashing.", ts="1.001"),
            _msg("Agreed, bcrypt with cost factor 12.", ts="1.002", thread_ts="1.001"),
        ])
        results = server.get_thread(channel_id="C0001", thread_ts="1.001")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("permalink", r, f"Missing 'permalink' in {r}")
            self.assertTrue(r["permalink"], f"Empty permalink in {r}")

    def test_get_thread_result_keys_complete(self):
        """Each get_thread result must expose text, author, ts, and permalink."""
        self.build([
            _msg("LGTM, ship it.", ts="5.001", user="U42", permalink="https://slack.test/p5001"),
        ])
        results = server.get_thread(channel_id="C0001", thread_ts="5.001")

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["text"], "LGTM, ship it.")
        self.assertEqual(r["author"], "U42")
        self.assertEqual(r["ts"], "5.001")
        self.assertEqual(r["permalink"], "https://slack.test/p5001")

    def test_get_thread_is_chronological(self):
        """Replies come back oldest-first so the conversation reads in order."""
        self.build([
            _msg("Parent question?", ts="1.001"),
            _msg("Second reply.", ts="1.003", thread_ts="1.001"),
            _msg("First reply.",  ts="1.002", thread_ts="1.001"),
        ])
        results = server.get_thread(channel_id="C0001", thread_ts="1.001")
        self.assertEqual([r["ts"] for r in results], ["1.001", "1.002", "1.003"])


# ---------------------------------------------------------------------------
# 3. Cosine similarity ranking
# ---------------------------------------------------------------------------
class TestCosineSimilarityRanking(_IndexFixture):
    """
    Verify that search_slack orders results by descending cosine similarity.

    Strategy: replace _embed with a deterministic stub that returns fixed
    vectors keyed on a text fragment.  The same stub is used to build the index
    and to embed the query, so ranking order is an exact assertion rather than a
    probabilistic check that a model update could silently break.

    Vector space (3-dim for clarity):
      query           → [1, 0, 0]
      auth message    → [0.9, 0.1, 0]   (cosine ≈ 0.994)  ← should rank 1st
      deploy message  → [0, 1, 0]       (cosine = 0)       ← should rank last
      standup message → [0.5, 0.5, 0]   (cosine ≈ 0.707)   ← should rank 2nd
    """

    _VECTOR_MAP = {
        "query":   [1.0, 0.0, 0.0],
        "JWT":     [0.9, 0.1, 0.0],   # auth message fragment
        "Docker":  [0.0, 1.0, 0.0],   # deploy message fragment
        "Standup": [0.5, 0.5, 0.0],   # standup message fragment
    }

    def embed(self, texts):
        """Deterministic embedding per text, matched by keyword."""
        vecs = []
        for t in texts:
            for keyword, vec in self._VECTOR_MAP.items():
                if keyword in t:
                    vecs.append(vec)
                    break
            else:
                vecs.append([0.1, 0.1, 0.1])
        return np.array(vecs, dtype=np.float32)

    _FIXTURE = [
        _msg("JWT tokens are used for authentication.", ts="1.001"),
        _msg("Docker image size caused a deployment issue.", ts="1.002"),
        _msg("Standup: velocity looks good.", ts="1.003"),
    ]

    def test_most_similar_message_ranks_first(self):
        """The message whose embedding is closest to the query ranks first."""
        self.build(self._FIXTURE)
        results = server.search_slack(query="query about JWT authentication", limit=3)

        self.assertGreater(len(results), 0)
        self.assertIn("JWT", results[0]["text"],
                      f"Expected JWT message first, got: {results[0]['text']!r}")

    def test_least_similar_message_ranks_last(self):
        """The message furthest from the query direction ranks last."""
        self.build(self._FIXTURE)
        results = server.search_slack(query="query about JWT authentication", limit=3)

        self.assertEqual(len(results), 3)
        self.assertIn("Docker", results[-1]["text"],
                      f"Expected Docker message last, got: {results[-1]['text']!r}")

    def test_scores_are_descending(self):
        """The 'score' field must be monotonically non-increasing across results."""
        self.build(self._FIXTURE)
        results = server.search_slack(query="query about JWT authentication", limit=3)

        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         f"Scores not descending: {scores}")

    def test_limit_is_respected(self):
        """search_slack must return at most `limit` results regardless of corpus size."""
        self.build([
            _msg(f"Message number {i} about various topics.", ts=f"1.{i:03d}")
            for i in range(10)
        ])
        results = server.search_slack(query="anything", limit=4)
        self.assertLessEqual(len(results), 4, f"Expected ≤4 results, got {len(results)}")

    def test_channel_filter_restricts_results(self):
        """Passing `channels` narrows the search to those channels only."""
        self.build([
            _msg("JWT tokens are used for authentication.", channel="C0001", ts="1.001"),
            _msg("JWT rotation policy for the gateway.",     channel="C0002", ts="1.002"),
        ])
        results = server.search_slack(query="query about JWT", channels=["C0002"], limit=5)

        self.assertGreater(len(results), 0)
        self.assertTrue(all(r["channel"] == "C0002" for r in results),
                        f"Channel filter leaked: {[r['channel'] for r in results]}")

    def test_cosine_similarity_pure_math(self):
        """
        Unit test for _cosine_similarity itself — no server logic involved.
        Verifies: sim(v, v) = 1, sim(v, -v) ≈ -1, sim(orthogonal) = 0, sim(53°) = 0.6
        """
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array([
            [1.0, 0.0, 0.0],   # identical  → 1.0
            [-1.0, 0.0, 0.0],  # opposite   → -1.0
            [0.0, 1.0, 0.0],   # orthogonal → 0.0
            [0.6, 0.8, 0.0],   # 53-deg     → 0.6
        ], dtype=np.float32)

        scores = server._cosine_similarity(q, docs)
        self.assertAlmostEqual(float(scores[0]),  1.0,  places=5)
        self.assertAlmostEqual(float(scores[1]), -1.0,  places=5)
        self.assertAlmostEqual(float(scores[2]),  0.0,  places=5)
        self.assertAlmostEqual(float(scores[3]),  0.6,  places=5)


# ---------------------------------------------------------------------------
# 4. Routing — search_slack delegates to _rank_chunks
# ---------------------------------------------------------------------------
class TestRankChunksRouting(_IndexFixture):
    """
    Verify that search_slack() calls _rank_chunks() rather than containing its
    own inlined ranking logic.  This is what keeps the eval harness measuring
    the same code path that serves production queries.
    """

    def test_search_slack_calls_rank_chunks(self):
        """search_slack must delegate ranking, passing the query and limit through."""
        self.build([
            _msg("JWT tokens are used for authentication.", ts="1.001"),
            _msg("Docker image size caused a deployment issue.", ts="1.002"),
        ])

        with patch.object(server, "_rank_chunks", wraps=server._rank_chunks) as spy:
            results = server.search_slack(query="auth", channels=["C0001"], limit=5)

        spy.assert_called_once()
        chunks, vecs, query, k = spy.call_args[0]
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), len(vecs), "one stored vector per chunk")
        self.assertEqual(query, "auth")
        self.assertEqual(k, 5)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("score", r)
            self.assertIn("permalink", r)

    def test_search_slack_embeds_only_the_query(self):
        """
        The whole point of the index: a query costs one embedding call for the
        query text, not one per corpus chunk.
        """
        self.build([
            _msg(f"Message {i} about the deployment pipeline.", ts=f"1.{i:03d}")
            for i in range(20)
        ])

        seen = []

        def _spy_embed(texts):
            seen.append(list(texts))
            return self.embed(texts)

        with patch.object(server, "_embed", side_effect=_spy_embed):
            server.search_slack(query="deployment", limit=3)

        self.assertEqual(len(seen), 1, f"Expected 1 embed call, got {len(seen)}")
        self.assertEqual(seen[0], ["deployment"], "Only the query should be embedded")


if __name__ == "__main__":
    unittest.main()
