"""
tests/test_mcp_protocol.py — is this actually an MCP server?

Every other test file stubs `fastmcp.FastMCP` with a no-op class, so they verify
the *functions* (`search_slack`, `get_thread`, …) and never the protocol layer
that turns them into an MCP extension.  A server could pass all 122 of those and
still fail to speak MCP at all.

This file closes that gap.  It launches `server.py` as a real subprocess over
stdio — no stubs, no monkeypatching — and talks JSON-RPC 2.0 to it exactly the
way Claude Code, Claude Desktop, or any other MCP client does:

    initialize  →  notifications/initialized  →  tools/list  →  tools/call

Staying offline
---------------
The server needs an index and an embedding backend.  Both are avoided without
faking the protocol itself:

  * the index is pre-built in-process with arbitrary vectors (embedding quality
    is irrelevant here — this file is about the wire protocol), and
  * the subprocess runs with RETRIEVAL_MODE=bm25, whose query path is pure
    lexical matching and never calls OpenAI.

So this is a real MCP client talking to the real server, with zero network.

Run with:
    cd mcp-server
    /path/to/venv/bin/python -m pytest tests/test_mcp_protocol.py -v
"""

import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import index  # noqa: E402  — stdlib + numpy only, safe regardless of other stubs

_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server.py")
_PY = sys.executable
_MODEL = "text-embedding-3-small"   # server.py's default; the index must match
_TIMEOUT = 20.0

def _fastmcp_available() -> bool:
    """
    Can a *fresh* interpreter import fastmcp?  That is the question that matters,
    because this file talks to a subprocess.  find_spec() alone is unreliable
    here: sibling test modules stub `fastmcp` into sys.modules, and a stub module
    has no __spec__, which makes find_spec raise instead of answer.
    """
    try:
        if importlib.util.find_spec("fastmcp") is not None:
            return True
    except (ImportError, ValueError):
        pass
    return subprocess.run(
        [_PY, "-c", "import fastmcp"], capture_output=True
    ).returncode == 0


_HAS_FASTMCP = _fastmcp_available()


def _fake_embed(texts):
    """Deterministic vectors. Values don't matter — bm25 mode never reads them."""
    out = []
    for t in texts:
        v = np.zeros(8, dtype=np.float32)
        for i, ch in enumerate(t[:8]):
            v[i] = float(ord(ch))
        out.append(v)
    return np.array(out, dtype=np.float32)


def _msg(text, ts, channel="C0001", thread_ts=""):
    m = {
        "type": "message", "text": text, "channel": channel, "ts": ts, "user": "U001",
        "permalink": f"https://acme.slack.com/archives/{channel}/p{ts.replace('.', '')}",
    }
    if thread_ts:
        m["thread_ts"] = thread_ts
    return m


CORPUS = [
    _msg("Seeing ERR_500 in the payment gateway on us-east-1.", "1700000010.000100"),
    _msg("Root cause was the pd-1234 config rollout.", "1700000011.000200",
         thread_ts="1700000010.000100"),
    _msg("We decided to use JWT for authentication.", "1700000020.000100", channel="C0002"),
]


class MCPClient:
    """A minimal MCP client over stdio — enough to exercise the real protocol."""

    def __init__(self, env):
        self.proc = subprocess.Popen(
            [_PY, _SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        self._q: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._next_id = 0

    def _pump(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def _send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline_msgs = []
        while True:
            try:
                msg = self._q.get(timeout=_TIMEOUT)
            except queue.Empty:
                raise AssertionError(
                    f"no response to {method} within {_TIMEOUT}s; saw {deadline_msgs}"
                )
            if msg.get("id") == rid:
                return msg
            deadline_msgs.append(msg)

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def call_tool(self, name, arguments=None):
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def _payload(result: dict):
    """Unwrap a tools/call result into the plain value the tool returned."""
    if "structuredContent" in result:
        sc = result["structuredContent"]
        return sc.get("result", sc)
    content = result.get("content") or []
    if content and content[0].get("type") == "text":
        return json.loads(content[0]["text"])
    return content


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed — run `pip install -e .`")
class TestMCPProtocol(unittest.TestCase):
    """The server must behave like an MCP extension, not just expose Python functions."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="mcp-protocol-test-")
        cls.index_path = os.path.join(cls.tmpdir, "index.db")

        store = index.Store(cls.index_path)
        # Build chunks directly rather than importing server here — sibling test
        # modules may have stubbed server's dependencies in this process.
        chunks = []
        for m in CORPUS:
            c = index.Chunk(
                message_id=f"{m['channel']}:{m['ts']}", chunk_index=0, total_chunks=1,
                channel=m["channel"], ts=m["ts"], author=m["user"],
                full_text=m["text"], permalink=m["permalink"],
                thread_ts=m.get("thread_ts", ""), chunk_text=m["text"],
            )
            chunks.append(c)
        store.upsert(chunks, _fake_embed([c.chunk_text for c in chunks]), _MODEL)
        store.upsert_channels([
            {"id": "C0001", "name": "backend", "num_members": 12},
            {"id": "C0002", "name": "general", "num_members": 40},
        ])
        store.close()

        env = {
            **os.environ,
            "MCP_TRANSPORT":  "stdio",
            "INDEX_PATH":     cls.index_path,
            "OPENAI_API_KEY": "fake-key",
            "RETRIEVAL_MODE": "bm25",      # keeps the query path offline
            "SLACK_EXPORT_PATH": "",
        }
        env.pop("SLACK_BOT_TOKEN", None)   # prove the server runs token-less

        cls.client = MCPClient(env)
        cls.init = cls.client.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "conformance-test", "version": "1"},
        })
        cls.client.notify("notifications/initialized")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ── Handshake ───────────────────────────────────────────────────────────
    def test_initialize_returns_a_valid_result(self):
        """The handshake must succeed and be well-formed JSON-RPC 2.0."""
        self.assertEqual(self.init.get("jsonrpc"), "2.0")
        self.assertNotIn("error", self.init)
        self.assertIn("result", self.init)

    def test_server_identifies_itself(self):
        """serverInfo carries the name a client shows in its MCP panel."""
        self.assertEqual(self.init["result"]["serverInfo"]["name"], "slack-search-mcp")

    def test_protocol_version_negotiated(self):
        self.assertTrue(self.init["result"].get("protocolVersion"))

    def test_declares_tools_capability(self):
        """Without this a client will never ask for the tool list."""
        self.assertIn("tools", self.init["result"].get("capabilities", {}))

    def test_grounding_instructions_reach_the_client(self):
        """
        An IDE host has no custom system prompt, so the grounding contract has to
        travel in the MCP instructions block or it does not apply at all.
        """
        instructions = self.init["result"].get("instructions") or ""
        self.assertTrue(instructions, "no instructions block in initialize result")
        self.assertIn("permalink", instructions.lower())
        self.assertIn("couldn't find anything in slack", instructions.lower())

    # ── Tool discovery ──────────────────────────────────────────────────────
    def test_tools_list_exposes_every_tool(self):
        tools = self.client.request("tools/list")["result"]["tools"]
        self.assertEqual(
            {t["name"] for t in tools},
            {"search_slack", "get_thread", "list_channels", "get_reactions"},
        )

    def test_every_tool_has_a_description_and_schema(self):
        """A client shows the description to the model; no description, no informed call."""
        for t in self.client.request("tools/list")["result"]["tools"]:
            with self.subTest(tool=t["name"]):
                self.assertTrue(t.get("description"), f"{t['name']} has no description")
                self.assertEqual(t["inputSchema"]["type"], "object")

    def test_required_arguments_are_declared(self):
        schemas = {t["name"]: t["inputSchema"]
                   for t in self.client.request("tools/list")["result"]["tools"]}
        self.assertEqual(schemas["search_slack"]["required"], ["query"])
        self.assertEqual(set(schemas["get_thread"]["required"]), {"channel_id", "thread_ts"})
        self.assertEqual(set(schemas["get_reactions"]["required"]), {"channel_id", "message_ts"})
        self.assertFalse(schemas["list_channels"].get("required"))

    # ── Tool invocation ─────────────────────────────────────────────────────
    def test_search_slack_returns_real_indexed_results(self):
        """An end-to-end call: MCP request in, real data from the index out."""
        res = self.client.request("tools/call", {
            "name": "search_slack", "arguments": {"query": "ERR_500", "limit": 3},
        })
        self.assertNotIn("error", res)
        hits = _payload(res["result"])
        self.assertTrue(hits, "search returned nothing over MCP")
        self.assertIn("ERR_500", hits[0]["text"])

    def test_results_carry_the_full_citation_schema(self):
        """Every field the grounding contract depends on must survive the transport."""
        hits = _payload(self.client.call_tool(
            "search_slack", {"query": "ERR_500 payment gateway"})["result"])
        for key in ("text", "author", "channel", "ts", "permalink", "score"):
            self.assertIn(key, hits[0], f"missing '{key}' in MCP tool output")
        self.assertTrue(hits[0]["permalink"].startswith("https://"))

    def test_channel_filter_works_over_the_wire(self):
        hits = _payload(self.client.call_tool(
            "search_slack", {"query": "JWT authentication", "channels": ["C0002"]})["result"])
        self.assertTrue(hits)
        self.assertTrue(all(h["channel"] == "C0002" for h in hits))

    def test_get_thread_returns_parent_and_reply(self):
        msgs = _payload(self.client.call_tool(
            "get_thread", {"channel_id": "C0001", "thread_ts": "1700000010.000100"})["result"])
        self.assertEqual([m["ts"] for m in msgs],
                         ["1700000010.000100", "1700000011.000200"])

    def test_list_channels_returns_indexed_channels(self):
        chans = _payload(self.client.call_tool("list_channels")["result"])
        self.assertEqual({c["name"] for c in chans}, {"backend", "general"})

    def test_get_reactions_degrades_without_a_slack_token(self):
        """The server runs token-less; the one live tool returns [] instead of crashing."""
        res = self.client.call_tool(
            "get_reactions", {"channel_id": "C0001", "message_ts": "1700000010.000100"})
        self.assertNotIn("error", res)
        self.assertEqual(_payload(res["result"]), [])

    def test_no_results_is_an_empty_list_not_an_error(self):
        """The "I couldn't find anything" path depends on [] coming back cleanly."""
        res = self.client.call_tool(
            "search_slack", {"query": "zzzzz nonexistent topic qqqqq"})
        self.assertNotIn("error", res)
        self.assertEqual(_payload(res["result"]), [])

    # ── Error handling ──────────────────────────────────────────────────────
    def test_unknown_tool_is_reported_not_crashed(self):
        """A bad call must not take the server down — later calls still work."""
        res = self.client.call_tool("no_such_tool", {})
        self.assertTrue("error" in res or res.get("result", {}).get("isError"),
                        f"expected an error for an unknown tool, got {res}")

    def test_server_survives_a_bad_call(self):
        self.client.call_tool("no_such_tool", {})
        hits = _payload(self.client.call_tool("search_slack", {"query": "ERR_500"})["result"])
        self.assertTrue(hits, "server stopped answering after a bad tool call")


if __name__ == "__main__":
    unittest.main()
