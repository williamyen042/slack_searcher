"""
tests/test_index.py — the persistent chunk + embedding store.

Five groups:
  1. Round-trip           — what goes into the store comes back out intact
  2. Incremental indexing — unchanged chunks are never re-embedded (the point)
  3. Edits and deletes    — changed text is re-embedded, deleted messages drop,
                            and a windowed reconcile never touches what it didn't fetch
  4. Model namespacing    — vectors from different models never mix
  5. Cursors              — the incremental fetch watermark survives a reopen

Run with:
    cd mcp-server
    OPENAI_API_KEY=fake-key /path/to/venv/bin/python -m pytest tests/test_index.py -v

No network, no Slack, no OpenAI: embeddings are a deterministic stub, and the
store is a real SQLite database in a temp directory.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

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
    _stub_module("dotenv", load_dotenv=lambda *a, **kw: None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import index   # noqa: E402
import server  # noqa: E402

MODEL = "test-embed-model"


def _msg(text: str, ts: str, channel: str = "C0001", thread_ts: str = "") -> dict:
    m = {
        "type": "message", "text": text, "channel": channel, "ts": ts,
        "user": "U001", "permalink": f"https://slack.test/{channel}/p{ts.replace('.', '')}",
    }
    if thread_ts:
        m["thread_ts"] = thread_ts
    return m


class _StoreCase(unittest.TestCase):
    """Base: a fresh on-disk store per test, plus a counting embed stub."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="slack-index-store-")
        self.path = os.path.join(self._tmpdir, "index.db")
        self.store = index.Store(self.path)
        self.embed_calls: list[list[str]] = []

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def embed(self, texts):
        """Deterministic 8-dim vectors, recording every text it is asked for."""
        self.embed_calls.append(list(texts))
        out = []
        for t in texts:
            v = np.zeros(8, dtype=np.float32)
            for i, ch in enumerate(t[:8]):
                v[i] = float(ord(ch))
            out.append(v)
        return np.array(out, dtype=np.float32)

    @property
    def embedded_texts(self) -> list[str]:
        return [t for batch in self.embed_calls for t in batch]

    def run_index(self, messages) -> tuple[int, int]:
        return index.index_messages(
            self.store, messages, MODEL, self.embed, server._chunk_message
        )


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------
class TestRoundTrip(_StoreCase):

    def test_chunks_and_vectors_survive_a_round_trip(self):
        """Everything written to the store reads back with matching vectors."""
        self.run_index([_msg("We decided to use JWT for auth.", ts="1.001")])
        chunks, vecs = self.store.load(MODEL)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(vecs), 1)
        self.assertEqual(chunks[0].full_text, "We decided to use JWT for auth.")
        self.assertEqual(chunks[0].message_id, "C0001:1.001")
        self.assertEqual(chunks[0].permalink, "https://slack.test/C0001/p1001")
        np.testing.assert_allclose(vecs[0], self.embed(["We decided to use JWT for auth."])[0])

    def test_empty_store_returns_empty(self):
        """An unindexed store yields no chunks and an empty matrix."""
        chunks, vecs = self.store.load(MODEL)
        self.assertEqual(chunks, [])
        self.assertEqual(len(vecs), 0)

    def test_survives_reopen(self):
        """The index is a file, so a new process sees what the last one wrote."""
        self.run_index([_msg("Deployment is blocked.", ts="1.001")])
        self.store.close()

        reopened = index.Store(self.path)
        chunks, vecs = reopened.load(MODEL)
        reopened.close()
        self.store = index.Store(self.path)   # so tearDown has something to close

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(vecs), 1)

    def test_multi_chunk_message_stores_every_chunk(self):
        """A long message stores one row per chunk, all sharing one message_id."""
        long_text = "\n\n".join(
            f"Paragraph {i} about the deployment pipeline in detail. " * 12
            for i in range(4)
        )
        seen, embedded = self.run_index([_msg(long_text, ts="1.001")])
        chunks, vecs = self.store.load(MODEL)

        self.assertGreater(len(chunks), 1, "expected a multi-chunk message")
        self.assertEqual(seen, len(chunks))
        self.assertEqual(embedded, len(chunks))
        self.assertEqual({c.message_id for c in chunks}, {"C0001:1.001"})
        self.assertEqual(len(vecs), len(chunks))
        # full_text is the whole message on every chunk, never a fragment
        # (_chunk_message strips leading/trailing whitespace before storing)
        self.assertTrue(all(c.full_text == long_text.strip() for c in chunks))


# ---------------------------------------------------------------------------
# 2. Incremental indexing — the reason this module exists
# ---------------------------------------------------------------------------
class TestIncremental(_StoreCase):

    def test_second_run_embeds_nothing(self):
        """
        Re-indexing an unchanged corpus must cost zero embedding calls.  This is
        the entire point of the store: without it, every query re-paid for text
        that had not moved.
        """
        corpus = [
            _msg("We decided to use JWT for auth.", ts="1.001"),
            _msg("Deployment blocked by a Docker image size issue.", ts="1.002"),
        ]
        seen1, embedded1 = self.run_index(corpus)
        self.assertEqual(embedded1, seen1, "first run must embed everything")

        self.embed_calls.clear()
        seen2, embedded2 = self.run_index(corpus)

        self.assertEqual(embedded2, 0, "unchanged corpus must not be re-embedded")
        self.assertEqual(self.embed_calls, [], "no embedding call should be made at all")
        self.assertEqual(seen2, seen1)

    def test_only_new_messages_are_embedded(self):
        """Adding one message to a corpus embeds exactly that one message."""
        self.run_index([_msg("First message about auth.", ts="1.001")])
        self.embed_calls.clear()

        self.run_index([
            _msg("First message about auth.", ts="1.001"),
            _msg("Second message about deploys.", ts="1.002"),
        ])

        self.assertEqual(self.embedded_texts, ["Second message about deploys."])
        chunks, _ = self.store.load(MODEL)
        self.assertEqual(len(chunks), 2)

    def test_index_grows_without_duplicating(self):
        """Re-indexing the same message twice replaces it, never duplicates it."""
        msg = [_msg("Same message.", ts="1.001")]
        self.run_index(msg)
        self.run_index(msg)

        chunks, vecs = self.store.load(MODEL)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(vecs), 1)


# ---------------------------------------------------------------------------
# 3. Edits and deletes
# ---------------------------------------------------------------------------
class TestEditsAndDeletes(_StoreCase):

    def test_edited_message_is_re_embedded(self):
        """A message whose text changed must be re-embedded, not served stale."""
        self.run_index([_msg("We decided to use JWT.", ts="1.001")])
        self.embed_calls.clear()

        self.run_index([_msg("We decided to use OAuth instead.", ts="1.001")])

        self.assertEqual(self.embedded_texts, ["We decided to use OAuth instead."])
        chunks, _ = self.store.load(MODEL)
        self.assertEqual(len(chunks), 1, "the edit replaces the row, not appends")
        self.assertEqual(chunks[0].full_text, "We decided to use OAuth instead.")

    def test_delete_removes_every_chunk_of_a_message(self):
        """Deleting by message_id drops all of that message's chunks."""
        self.run_index([
            _msg("Message one.", ts="1.001"),
            _msg("Message two.", ts="1.002"),
        ])
        self.store.delete_messages(["C0001:1.001"], MODEL)

        chunks, vecs = self.store.load(MODEL)
        self.assertEqual([c.message_id for c in chunks], ["C0001:1.002"])
        self.assertEqual(len(vecs), 1)

    def test_message_ids_scoped_by_channel(self):
        """message_ids() filters by channel, which is what --reconcile needs."""
        self.run_index([
            _msg("In channel one.", ts="1.001", channel="C0001"),
            _msg("In channel two.", ts="1.002", channel="C0002"),
        ])
        self.assertEqual(self.store.message_ids(MODEL, "C0001"), {"C0001:1.001"})
        self.assertEqual(self.store.message_ids(MODEL, "C0002"), {"C0002:1.002"})
        self.assertEqual(len(self.store.message_ids(MODEL)), 2)

    def test_message_ids_scoped_by_time(self):
        """
        A windowed --reconcile fetches only part of history, so deletion
        detection must only consider stored messages inside that same window.
        Without this bound, everything older would look deleted.
        """
        self.run_index([
            _msg("Ancient message.", ts="1000.0"),
            _msg("Recent message.",  ts="9000.0"),
        ])
        self.assertEqual(self.store.message_ids(MODEL, since_ts="5000.0"), {"C0001:9000.0"})
        self.assertEqual(len(self.store.message_ids(MODEL, since_ts="0")), 2)
        self.assertEqual(len(self.store.message_ids(MODEL)), 2)

    def test_windowed_reconcile_does_not_delete_older_messages(self):
        """
        The footgun this guards: reconciling a 2-week window must not wipe the
        months of history that window never fetched.
        """
        self.run_index([
            _msg("Ancient, outside the window.", ts="1000.0"),
            _msg("Recent, still in Slack.",      ts="9000.0"),
            _msg("Recent, deleted in Slack.",    ts="9001.0"),
        ])
        window_start = "5000.0"
        # What a windowed fetch would return: only the still-live recent message
        live = {"C0001:9000.0"}

        stale = self.store.message_ids(MODEL, "C0001", since_ts=window_start) - live
        self.store.delete_messages(stale, MODEL)

        remaining = self.store.message_ids(MODEL, "C0001")
        self.assertIn("C0001:1000.0", remaining, "ancient message must survive")
        self.assertIn("C0001:9000.0", remaining, "live message must survive")
        self.assertNotIn("C0001:9001.0", remaining, "deleted message must be dropped")


# ---------------------------------------------------------------------------
# 4. Model namespacing
# ---------------------------------------------------------------------------
class TestModelNamespacing(_StoreCase):

    def test_vectors_from_different_models_do_not_mix(self):
        """
        Vectors from different embedding models are not comparable, so loading
        one model must never return another model's rows.
        """
        msg = [_msg("Shared message text.", ts="1.001")]
        self.run_index(msg)
        index.index_messages(self.store, msg, "other-model", self.embed, server._chunk_message)

        a, va = self.store.load(MODEL)
        b, vb = self.store.load("other-model")
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertEqual(len(va), 1)
        self.assertEqual(len(vb), 1)

    def test_unknown_model_loads_nothing(self):
        """Switching to a model that was never indexed yields an empty result."""
        self.run_index([_msg("Indexed under one model only.", ts="1.001")])
        chunks, vecs = self.store.load("never-indexed-model")
        self.assertEqual(chunks, [])
        self.assertEqual(len(vecs), 0)

    def test_same_text_under_second_model_is_embedded_again(self):
        """The hash check is per-model — a new model re-embeds everything."""
        msg = [_msg("Shared message text.", ts="1.001")]
        self.run_index(msg)
        self.embed_calls.clear()

        index.index_messages(self.store, msg, "other-model", self.embed, server._chunk_message)
        self.assertEqual(self.embedded_texts, ["Shared message text."])


# ---------------------------------------------------------------------------
# 5. Cursors and threads
# ---------------------------------------------------------------------------
class TestCursorsAndThreads(_StoreCase):

    def test_cursor_round_trip(self):
        """The incremental watermark is readable after being written."""
        self.assertIsNone(self.store.get_cursor("C0001"))
        self.store.set_cursor("C0001", "1700000000.000100")
        self.assertEqual(self.store.get_cursor("C0001"), "1700000000.000100")

    def test_cursor_survives_reopen(self):
        """A cron run must resume where the previous process stopped."""
        self.store.set_cursor("C0001", "1700000000.000100")
        self.store.close()

        self.store = index.Store(self.path)
        self.assertEqual(self.store.get_cursor("C0001"), "1700000000.000100")

    def test_cursor_is_per_channel(self):
        """Channels advance independently."""
        self.store.set_cursor("C0001", "1.001")
        self.store.set_cursor("C0002", "2.002")
        self.assertEqual(self.store.get_cursor("C0001"), "1.001")
        self.assertEqual(self.store.get_cursor("C0002"), "2.002")

    def test_thread_returns_parent_and_replies_in_order(self):
        """get_thread reads this: parent plus replies, oldest first."""
        self.run_index([
            _msg("Parent question?", ts="1.001"),
            _msg("Second reply.", ts="1.003", thread_ts="1.001"),
            _msg("First reply.",  ts="1.002", thread_ts="1.001"),
            _msg("Unrelated message.", ts="9.009"),
        ])
        thread = self.store.thread("C0001", "1.001")
        self.assertEqual([c.ts for c in thread], ["1.001", "1.002", "1.003"])

    def test_channels_derived_from_indexed_chunks(self):
        """With no recorded metadata, channels() falls back to what is indexed."""
        self.run_index([
            _msg("In channel one.", ts="1.001", channel="C0001"),
            _msg("In channel two.", ts="1.002", channel="C0002"),
        ])
        self.assertEqual([c["id"] for c in self.store.channels()], ["C0001", "C0002"])

    def test_recorded_channel_metadata_wins(self):
        """When the indexer records channel names, those are returned."""
        self.run_index([_msg("Hello.", ts="1.001", channel="C0001")])
        self.store.upsert_channels([{"id": "C0001", "name": "general", "num_members": 42}])

        chans = self.store.channels()
        self.assertEqual(chans, [{"id": "C0001", "name": "general", "num_members": 42}])


# ---------------------------------------------------------------------------
# 6. Lookback window
# ---------------------------------------------------------------------------
class TestLookbackWindow(unittest.TestCase):
    """
    A cursor only advances past things with a new timestamp.  A reply added to
    an old thread, and an edit to an old message, both leave the parent's ts
    untouched — so fetching strictly from the cursor misses them.  The lookback
    window is what re-walks far enough back to catch them.
    """

    def test_no_cursor_means_full_history(self):
        """The first run has nothing to look back from, so it walks everything."""
        self.assertIsNone(index._lookback_from(None, 48))
        self.assertIsNone(index._lookback_from("", 48))

    def test_window_starts_before_the_cursor(self):
        """oldest must be exactly `hours` earlier than the stored cursor."""
        cursor = "1700000000.000000"
        self.assertAlmostEqual(
            float(index._lookback_from(cursor, 48)), 1700000000.0 - 48 * 3600, places=3
        )
        self.assertAlmostEqual(
            float(index._lookback_from(cursor, 1)), 1700000000.0 - 3600, places=3
        )

    def test_window_never_goes_negative(self):
        """A cursor near epoch must not produce a negative timestamp."""
        self.assertEqual(float(index._lookback_from("100.0", 48)), 0.0)

    def test_zero_lookback_is_the_cursor_itself(self):
        """--lookback-hours 0 reproduces the old cursor-only behaviour."""
        self.assertAlmostEqual(float(index._lookback_from("1700000000.0", 0)), 1700000000.0)

    def test_window_covers_a_reply_to_a_yesterday_thread(self):
        """
        The case that motivated this: a parent from 24h ago receives a reply
        today.  The parent's ts is behind the cursor, so a 48h window must
        reach back far enough to re-fetch it and pick up the reply.
        """
        now = 1700000000.0
        parent_ts = now - 24 * 3600          # posted yesterday
        cursor = f"{now:.6f}"                # cursor advanced past it since

        oldest = float(index._lookback_from(cursor, 48))
        self.assertLess(oldest, parent_ts,
                        "48h window must reach back past a 24h-old thread parent")

        oldest_short = float(index._lookback_from(cursor, 6))
        self.assertGreater(oldest_short, parent_ts,
                           "a 6h window is too short — reconcile is the backstop")


if __name__ == "__main__":
    unittest.main()
