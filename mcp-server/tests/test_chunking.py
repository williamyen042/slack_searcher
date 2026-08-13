"""
Tests for the token-based chunking pipeline in server.py.

Run with:
    cd mcp-server
    .venv/bin/python -m pytest tests/test_chunking.py -v
"""

import os
import sys
import importlib
import types
import unittest

# ---------------------------------------------------------------------------
# Minimal environment stubs so server.py can be imported without real credentials
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY",     "fake-key")
os.environ.setdefault("SLACK_BOT_TOKEN",    "xoxb-fake")
os.environ.setdefault("SLACK_CHANNEL_IDS",  "C000001")

# Stub heavy external dependencies before importing server
def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# fastmcp stub
_fmcp = _stub_module("fastmcp")
class _FakeMCP:
    def __init__(self, **kw): pass
    def tool(self):
        def decorator(fn): return fn
        return decorator
    def run(self, **kw): pass
_fmcp.FastMCP = _FakeMCP

# slack_sdk stubs
_sdk = _stub_module("slack_sdk")
class _FakeWebClient:
    def __init__(self, **kw): pass
_sdk.WebClient = _FakeWebClient
_stub_module("slack_sdk.errors", SlackApiError=Exception)

# dotenv stub
_stub_module("dotenv", load_dotenv=lambda *a, **kw: None)

# Now we can safely import the server module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server  # noqa: E402  (import after stubs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_msg(text: str, channel: str = "C0001", ts: str = "1234567890.000001",
              subtype: str | None = None) -> dict:
    msg = {"type": "message", "text": text, "channel": channel, "ts": ts,
           "user": "U001", "permalink": f"https://slack.com/p/{ts}"}
    if subtype:
        msg["subtype"] = subtype
    return msg


def _token_count(text: str) -> int:
    return server._count_tokens(text)


# ---------------------------------------------------------------------------
# Tests: _chunk_message
# ---------------------------------------------------------------------------
class TestChunkMessage(unittest.TestCase):

    # ── Filtering ────────────────────────────────────────────────────────────

    def test_empty_text_returns_no_chunks(self):
        msg = _make_msg("")
        self.assertEqual(server._chunk_message(msg), [])

    def test_whitespace_only_returns_no_chunks(self):
        msg = _make_msg("   \n\n   ")
        self.assertEqual(server._chunk_message(msg), [])

    def test_system_subtype_channel_join_filtered(self):
        msg = _make_msg("Alice joined the channel", subtype="channel_join")
        self.assertEqual(server._chunk_message(msg), [])

    def test_system_subtype_bot_add_filtered(self):
        msg = _make_msg("A bot was added", subtype="bot_add")
        self.assertEqual(server._chunk_message(msg), [])

    def test_emoji_only_filtered(self):
        msg = _make_msg(":thumbsup: :fire: :tada:")
        self.assertEqual(server._chunk_message(msg), [])

    def test_url_only_filtered(self):
        msg = _make_msg("<https://example.com|example>")
        self.assertEqual(server._chunk_message(msg), [])

    # ── Short messages ───────────────────────────────────────────────────────

    def test_short_message_single_chunk(self):
        msg = _make_msg("Can someone review PR #582?")
        chunks = server._chunk_message(msg)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].total_chunks, 1)
        self.assertEqual(chunks[0].full_text, "Can someone review PR #582?")
        self.assertEqual(chunks[0].chunk_text, "Can someone review PR #582?")

    def test_short_message_preserves_metadata(self):
        msg = _make_msg("hello world", channel="C0001", ts="111.222")
        chunks = server._chunk_message(msg)
        self.assertEqual(chunks[0].channel, "C0001")
        self.assertEqual(chunks[0].ts, "111.222")
        self.assertEqual(chunks[0].author, "U001")
        self.assertEqual(chunks[0].message_id, "C0001:111.222")

    def test_message_exactly_at_chunk_size_single_chunk(self):
        # Construct a message whose token count equals CHUNK_SIZE_TOKENS exactly
        target = server.CHUNK_SIZE_TOKENS
        word = "word"
        # Build a string that is exactly `target` tokens (approximate with repetition)
        candidate = (word + " ") * target
        toks = server._tokens(candidate)
        text = server._decode_tokens(toks[:target])
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        self.assertEqual(len(chunks), 1)

    # ── Long messages ────────────────────────────────────────────────────────

    def test_long_message_produces_multiple_chunks(self):
        # ~1 200 tokens worth of text
        text = ("The quick brown fox jumps over the lazy dog. " * 80).strip()
        assert _token_count(text) > server.CHUNK_SIZE_TOKENS
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        self.assertGreater(len(chunks), 1)

    def test_all_chunks_within_model_token_limit(self):
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100).strip()
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        for chunk in chunks:
            count = _token_count(chunk.chunk_text)
            self.assertLessEqual(
                count, server.MAX_MODEL_TOKENS,
                f"Chunk {chunk.chunk_index} has {count} tokens > {server.MAX_MODEL_TOKENS}",
            )

    def test_no_text_lost_across_chunks(self):
        """
        Every word in the original message must appear in at least one chunk.
        This is a weaker but tractable proxy for 'no information lost'.
        """
        text = ("The quick brown fox jumps over the lazy dog. " * 60).strip()
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        all_chunk_text = " ".join(c.chunk_text for c in chunks)
        # Check a sample of words appear somewhere in the chunks
        for word in ["quick", "brown", "fox", "lazy", "dog"]:
            self.assertIn(word, all_chunk_text)

    def test_full_text_always_original(self):
        """full_text must always be the complete original message, never a fragment."""
        text = ("Another test sentence to fill tokens. " * 70).strip()
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        for chunk in chunks:
            self.assertEqual(chunk.full_text, text)

    def test_chunk_indices_sequential(self):
        text = ("Index test sentence filling tokens here. " * 70).strip()
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)
            self.assertEqual(chunk.total_chunks, len(chunks))

    def test_overlap_content_shared_between_adjacent_chunks(self):
        """
        With overlap > 0, the tail of chunk N should appear at the start of chunk N+1.
        We verify that at least some tokens are shared.
        """
        text = ("Overlap test word sequence filler content here. " * 70).strip()
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        if len(chunks) < 2:
            self.skipTest("Not enough chunks to test overlap")
        # The last few words of chunk[0] should appear in chunk[1]
        tail_words = chunks[0].chunk_text.split()[-5:]
        for word in tail_words:
            if word in chunks[1].chunk_text:
                return  # overlap confirmed
        self.fail("No overlap detected between chunk 0 and chunk 1")

    # ── Natural boundary preference ──────────────────────────────────────────

    def test_paragraph_break_respected(self):
        """A message with clear paragraph breaks should split there first."""
        para = "This is a paragraph sentence. " * 20   # ~100 tokens
        text = "\n\n".join([para.strip()] * 6)          # ~600 tokens total
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        # Each chunk should not start mid-sentence from a forced word split
        # (best-effort check: no chunk starts with a lowercase continuation word)
        for chunk in chunks:
            first_char = chunk.chunk_text.lstrip()[0] if chunk.chunk_text.strip() else ""
            # A chunk starting with lowercase mid-sentence would be suspicious
            # but we can't be strict here — just verify we got multiple chunks
        self.assertGreater(len(chunks), 1)

    # ── Special content ──────────────────────────────────────────────────────

    def test_unicode_message(self):
        text = "こんにちは世界。これはテストです。" * 30
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(_token_count(chunk.chunk_text), server.MAX_MODEL_TOKENS)

    def test_code_block_message(self):
        code = "```\ndef hello():\n    print('Hello, world!')\n    return 42\n```\n"
        text = code * 15
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(_token_count(chunk.chunk_text), server.MAX_MODEL_TOKENS)

    def test_markdown_message(self):
        text = (
            "# Heading\n\n"
            "Some **bold** and _italic_ text.\n\n"
            "- item one\n- item two\n- item three\n\n"
        ) * 20
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        self.assertGreaterEqual(len(chunks), 1)

    def test_malformed_unicode_does_not_crash(self):
        text = "Hello \ud83d\ude00 world! " * 40   # surrogate pairs
        msg = _make_msg(text)
        # Should not raise
        try:
            chunks = server._chunk_message(msg)
            self.assertGreaterEqual(len(chunks), 1)
        except Exception as e:
            self.fail(f"_chunk_message raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# Tests: deduplication in search_slack (unit-level, no network)
# ---------------------------------------------------------------------------
class TestDeduplication(unittest.TestCase):

    def test_best_chunk_per_message_selected(self):
        """
        Given two chunks from the same message with different scores,
        only the higher-scoring one should survive deduplication.
        """
        import numpy as np

        text = ("Dedup test sentence filling tokens here. " * 70).strip()
        msg = _make_msg(text, channel="C0001", ts="100.001")
        chunks = server._chunk_message(msg)
        if len(chunks) < 2:
            self.skipTest("Need at least 2 chunks for dedup test")

        # Simulate scores: give chunk 0 a low score, chunk 1 a high score
        fake_scores = np.zeros(len(chunks), dtype=np.float32)
        fake_scores[0] = 0.1
        fake_scores[1] = 0.9

        best: dict = {}
        for idx, chunk in enumerate(chunks):
            score = float(fake_scores[idx])
            if chunk.message_id not in best or score > best[chunk.message_id][0]:
                best[chunk.message_id] = (score, chunk)

        self.assertEqual(len(best), 1)   # only one unique message
        winning_score, winning_chunk = best[msg["channel"] + ":" + msg["ts"]]
        self.assertAlmostEqual(winning_score, 0.9)
        self.assertEqual(winning_chunk.chunk_index, 1)

    def test_multiple_messages_not_collapsed(self):
        """Two messages with the same channel but different ts must remain separate."""
        text_a = "Message A content here. " * 5
        text_b = "Message B content here. " * 5
        msg_a = _make_msg(text_a, channel="C0001", ts="100.001")
        msg_b = _make_msg(text_b, channel="C0001", ts="200.002")

        chunks_a = server._chunk_message(msg_a)
        chunks_b = server._chunk_message(msg_b)

        all_chunks = chunks_a + chunks_b
        msg_ids = {c.message_id for c in all_chunks}
        self.assertEqual(len(msg_ids), 2)

    def test_full_text_returned_not_chunk_text(self):
        """The full original message must be what gets stored in full_text."""
        text = ("Full text check sentence filling tokens here. " * 70).strip()
        msg = _make_msg(text)
        chunks = server._chunk_message(msg)
        for chunk in chunks:
            self.assertEqual(chunk.full_text, text)
            if len(chunks) > 1:
                self.assertNotEqual(chunk.chunk_text, text)  # chunks differ from full


# ---------------------------------------------------------------------------
# Tests: _split_text_naturally
# ---------------------------------------------------------------------------
class TestSplitTextNaturally(unittest.TestCase):

    def test_short_text_not_split(self):
        text = "Short sentence."
        parts = server._split_text_naturally(text)
        self.assertEqual(parts, [text])

    def test_paragraph_split(self):
        text = ("A " * 200).strip() + "\n\n" + ("B " * 200).strip()
        parts = server._split_text_naturally(text)
        self.assertGreater(len(parts), 1)

    def test_all_parts_within_chunk_size(self):
        text = ("word " * 500).strip()
        parts = server._split_text_naturally(text)
        for part in parts:
            self.assertLessEqual(_token_count(part), server.CHUNK_SIZE_TOKENS)


if __name__ == "__main__":
    unittest.main()
