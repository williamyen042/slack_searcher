"""
index.py  —  persistent chunk + embedding store for slack-search-mcp
====================================================================

Why this exists
---------------
Embedding text is the slow, paid step, and it is a pure function of
(embedding model, chunk text).  The original design re-fetched and re-embedded
the entire corpus on every query, so a message written months ago was paid for
again every time anyone asked anything.

This module splits that in two:

    INDEX (on a schedule)              QUERY (per request, in server.py)
    fetch since stored cursor          embed 1 query vector
    chunk new / changed only     →     matmul against stored vectors
    embed new / changed only           dedup + top-k
    persist to SQLite

Steady-state embedding cost becomes "messages written since the last run"
instead of "the whole workspace, every query".

Usage
-----
Run it with the virtualenv's interpreter — the dependencies are not on the
system Python (`.venv/bin/python`, or activate the venv first):

    .venv/bin/python index.py                       # incremental: configured channels
    .venv/bin/python index.py --channels C01,C02    # incremental: named channels only
    .venv/bin/python index.py --lookback-hours 336  # re-walk the last 2 weeks
    .venv/bin/python index.py --reconcile --lookback-hours 336  # + deletions in window
    .venv/bin/python index.py --reconcile           # full history deep clean
    .venv/bin/python index.py --stats               # what's in the index right now

Design notes
------------
* SQLite, not a vector DB.  Brute-force cosine over a stored matrix is ~100ms
  at a million chunks; approximate nearest neighbour buys nothing at this size
  and costs a service to operate.  This is one file you can delete to rebuild.
* `model` is part of the primary key.  Vectors from different embedding models
  are not comparable, so mixing them is made structurally impossible rather
  than left as a thing you remember not to do.
* `text_hash` detects edits: re-indexing a changed message re-embeds only the
  chunks whose text actually moved.
* A forward-only cursor only advances past things with a NEW timestamp, so it
  cannot see three kinds of change: a reply added to an old thread, an edit to
  an old message, or a deletion.  Two mechanisms cover them — each incremental
  run re-walks LOOKBACK_HOURS before the cursor (cheap: unchanged text costs no
  embeddings), and the slower `--reconcile` pass re-walks full history and drops
  stored messages that no longer come back.

This module imports nothing from server.py at module level (server.py imports
*it*); the CLI does a deferred `import server` inside main() to break the cycle.
"""

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger("slack-search-index")

# Index location — override with INDEX_PATH.  ":memory:" gives a throwaway
# in-process store, which is what the eval harness uses.
DEFAULT_INDEX_PATH: str = os.environ.get(
    "INDEX_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "slack_index.db")
)

# OpenAI's embeddings endpoint caps at 2048 inputs and ~300k tokens per request.
# 256 chunks x <=512 tokens stays comfortably under both.
EMBED_BATCH: int = int(os.environ.get("EMBED_BATCH", "256"))

# How far back before the stored cursor each incremental run re-walks.
#
# A cursor only advances past things with a NEW timestamp, so on its own it
# cannot see a reply added to an old thread or an edit to an old message —
# neither changes the parent's ts, so conversations.history never returns them.
# Re-walking a window catches both.  Re-fetched messages that have not changed
# cost zero embeddings (the text_hash check skips them), so the window is cheap;
# only the fetch is repaid.  Anything older than the window is caught by the
# nightly --reconcile.
LOOKBACK_HOURS: float = float(os.environ.get("INDEX_LOOKBACK_HOURS", "48"))


# ---------------------------------------------------------------------------
# Chunk — one embeddable segment derived from a Slack message
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """One embeddable segment derived from a Slack message."""
    # Identity
    message_id:   str          # unique key: "{channel}:{ts}"
    chunk_index:  int
    total_chunks: int          # filled in after all chunks for a message are known

    # Original message metadata (for returning to caller)
    channel:    str
    ts:         str
    author:     str
    full_text:  str            # the complete original message — never truncated
    permalink:  str = ""
    thread_ts:  str = ""       # parent ts if this message is a thread reply

    # The text this chunk was embedded from
    chunk_text: str = ""

    @property
    def key(self) -> str:
        """Stable identity of this chunk within a model namespace."""
        return f"{self.message_id}#{self.chunk_index}"

    @property
    def text_hash(self) -> str:
        """Changes iff chunk_text changes — this is how edits are detected."""
        return hashlib.sha256(self.chunk_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    message_id   TEXT    NOT NULL,
    chunk_index  INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    channel      TEXT    NOT NULL,
    ts           TEXT    NOT NULL,
    thread_ts    TEXT    NOT NULL DEFAULT '',
    author       TEXT    NOT NULL DEFAULT '',
    permalink    TEXT    NOT NULL DEFAULT '',
    full_text    TEXT    NOT NULL,
    chunk_text   TEXT    NOT NULL,
    text_hash    TEXT    NOT NULL,
    model        TEXT    NOT NULL,
    vec          BLOB    NOT NULL,
    PRIMARY KEY (message_id, chunk_index, model)
);
CREATE INDEX IF NOT EXISTS idx_chunks_channel ON chunks(channel, model);

CREATE TABLE IF NOT EXISTS cursors (
    channel    TEXT PRIMARY KEY,
    last_ts    TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    num_members INTEGER
);
"""


class Store:
    """
    SQLite-backed chunk + vector store.

    Pass ":memory:" for a throwaway in-process index (used by the eval harness
    so evaluation and production run the exact same ranking code).
    """

    def __init__(self, path: str = DEFAULT_INDEX_PATH) -> None:
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # ── Write ───────────────────────────────────────────────────────────────
    def upsert(self, chunks: list[Chunk], vecs: np.ndarray, model: str) -> None:
        """Insert or replace *chunks*, whose vectors are the rows of *vecs*."""
        if not chunks:
            return
        if len(chunks) != len(vecs):
            raise ValueError(f"chunk/vector count mismatch: {len(chunks)} vs {len(vecs)}")

        rows = [
            (
                c.message_id, c.chunk_index, c.total_chunks, c.channel, c.ts,
                c.thread_ts, c.author, c.permalink, c.full_text, c.chunk_text,
                c.text_hash, model,
                np.asarray(vecs[i], dtype=np.float32).tobytes(),
            )
            for i, c in enumerate(chunks)
        ]
        self._db.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(message_id, chunk_index, total_chunks, channel, ts, thread_ts, author, "
            " permalink, full_text, chunk_text, text_hash, model, vec) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._db.commit()

    def delete_messages(self, message_ids, model: str) -> int:
        """Drop every chunk belonging to *message_ids*. Returns rows removed."""
        ids = list(message_ids)
        if not ids:
            return 0
        cur = self._db.executemany(
            "DELETE FROM chunks WHERE message_id = ? AND model = ?",
            [(mid, model) for mid in ids],
        )
        self._db.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(ids)

    def upsert_channels(self, channels: list[dict]) -> None:
        """Record channel metadata so list_channels() needs no Slack call."""
        if not channels:
            return
        self._db.executemany(
            "INSERT OR REPLACE INTO channels (id, name, num_members) VALUES (?,?,?)",
            [(c["id"], c.get("name", c["id"]), c.get("num_members")) for c in channels],
        )
        self._db.commit()

    # ── Read ────────────────────────────────────────────────────────────────
    def hashes(self, model: str) -> dict[str, str]:
        """chunk.key → stored text_hash, for deciding what needs re-embedding."""
        rows = self._db.execute(
            "SELECT message_id, chunk_index, text_hash FROM chunks WHERE model = ?",
            (model,),
        ).fetchall()
        return {f"{r['message_id']}#{r['chunk_index']}": r["text_hash"] for r in rows}

    def message_ids(
        self, model: str, channel: str | None = None, since_ts: str | None = None
    ) -> set[str]:
        """
        Stored message ids, optionally scoped to a channel and to ts >= since_ts.

        The time bound is what makes a windowed --reconcile safe: without it, a
        reconcile that fetched only a window would treat every message older
        than that window as deleted and wipe the index.
        """
        sql = "SELECT DISTINCT message_id FROM chunks WHERE model = ?"
        params: list = [model]
        if channel is not None:
            sql += " AND channel = ?"
            params.append(channel)
        if since_ts is not None:
            sql += " AND CAST(ts AS REAL) >= ?"
            params.append(float(since_ts))
        return {r["message_id"] for r in self._db.execute(sql, params).fetchall()}

    def load(self, model: str) -> tuple[list[Chunk], np.ndarray]:
        """
        Every stored chunk for *model*, plus a parallel (N, D) float32 matrix.
        Returns ([], empty array) when the index is empty.
        """
        rows = self._db.execute(
            "SELECT * FROM chunks WHERE model = ? ORDER BY channel, ts, chunk_index",
            (model,),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)

        chunks = [
            Chunk(
                message_id=r["message_id"], chunk_index=r["chunk_index"],
                total_chunks=r["total_chunks"], channel=r["channel"], ts=r["ts"],
                author=r["author"], full_text=r["full_text"],
                permalink=r["permalink"], thread_ts=r["thread_ts"],
                chunk_text=r["chunk_text"],
            )
            for r in rows
        ]
        vecs = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return chunks, vecs

    def thread(self, channel: str, thread_ts: str) -> list[Chunk]:
        """Every stored message in one thread, chronological, one Chunk each."""
        rows = self._db.execute(
            "SELECT * FROM chunks WHERE channel = ? AND chunk_index = 0 "
            "AND (thread_ts = ? OR ts = ?) ORDER BY CAST(ts AS REAL)",
            (channel, thread_ts, thread_ts),
        ).fetchall()
        return [
            Chunk(
                message_id=r["message_id"], chunk_index=0, total_chunks=r["total_chunks"],
                channel=r["channel"], ts=r["ts"], author=r["author"],
                full_text=r["full_text"], permalink=r["permalink"],
                thread_ts=r["thread_ts"], chunk_text=r["chunk_text"],
            )
            for r in rows
        ]

    def channels(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, name, num_members FROM channels ORDER BY name"
        ).fetchall()
        if rows:
            return [dict(r) for r in rows]
        # No metadata recorded (e.g. export mode) — derive from indexed chunks.
        rows = self._db.execute(
            "SELECT DISTINCT channel FROM chunks ORDER BY channel"
        ).fetchall()
        return [
            {"id": r["channel"], "name": r["channel"].lstrip("#"), "num_members": None}
            for r in rows
        ]

    # ── Cursors ─────────────────────────────────────────────────────────────
    def get_cursor(self, channel: str) -> str | None:
        row = self._db.execute(
            "SELECT last_ts FROM cursors WHERE channel = ?", (channel,)
        ).fetchone()
        return row["last_ts"] if row else None

    def set_cursor(self, channel: str, last_ts: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO cursors (channel, last_ts, indexed_at) VALUES (?,?,?)",
            (channel, last_ts, datetime.now(timezone.utc).isoformat()),
        )
        self._db.commit()

    # ── Introspection ───────────────────────────────────────────────────────
    def stats(self) -> dict:
        total = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        msgs = self._db.execute(
            "SELECT COUNT(DISTINCT message_id) AS n FROM chunks"
        ).fetchone()["n"]
        by_model = self._db.execute(
            "SELECT model, COUNT(*) AS n FROM chunks GROUP BY model"
        ).fetchall()
        cursors = self._db.execute(
            "SELECT channel, last_ts, indexed_at FROM cursors ORDER BY channel"
        ).fetchall()
        return {
            "path":     self.path,
            "chunks":   total,
            "messages": msgs,
            "models":   {r["model"]: r["n"] for r in by_model},
            "cursors":  [dict(r) for r in cursors],
        }


# ---------------------------------------------------------------------------
# Indexing run
# ---------------------------------------------------------------------------
def index_messages(
    store: Store,
    raw_messages: list[dict],
    model: str,
    embed_fn,
    chunk_fn,
) -> tuple[int, int]:
    """
    Chunk *raw_messages*, embed only what is new or edited, and persist.

    Returns (chunks_seen, chunks_embedded).  The gap between the two is the
    whole point of this module — on a steady-state run it should be large.
    """
    all_chunks: list[Chunk] = []
    for msg in raw_messages:
        all_chunks.extend(chunk_fn(msg))

    if not all_chunks:
        return 0, 0

    known = store.hashes(model)
    todo = [c for c in all_chunks if known.get(c.key) != c.text_hash]

    if not todo:
        return len(all_chunks), 0

    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start:start + EMBED_BATCH]
        vecs = embed_fn([c.chunk_text for c in batch])
        store.upsert(batch, vecs, model)
        log.info("  embedded %d/%d chunks", min(start + len(batch), len(todo)), len(todo))

    return len(all_chunks), len(todo)


def _newest_ts(messages: list[dict]) -> str | None:
    tss = [m["ts"] for m in messages if m.get("ts")]
    return max(tss, key=float) if tss else None


def _lookback_from(cursor: str | None, hours: float) -> str | None:
    """
    Where an incremental run should start fetching: *hours* before the stored
    cursor, so edits and thread replies that did not advance the cursor are
    still re-walked.  None (full history) when there is no cursor yet.
    """
    if not cursor:
        return None
    return f"{max(0.0, float(cursor) - hours * 3600):.6f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build/refresh the Slack search index")
    ap.add_argument("--channels", help="Comma-separated channel IDs (default: SLACK_CHANNEL_IDS)")
    ap.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Path to the index file")
    ap.add_argument("--reconcile", action="store_true",
                    help="Also drop messages deleted in Slack. Walks full history unless "
                         "--lookback-hours bounds it to a window.")
    ap.add_argument("--lookback-hours", type=float, default=None,
                    help="Hours before the cursor to re-walk, catching thread replies "
                         f"and edits on older messages (default: {LOOKBACK_HOURS:g}). "
                         "With --reconcile, also bounds which messages deletion "
                         "detection considers.")
    ap.add_argument("--stats", action="store_true", help="Print index stats and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    )

    store = Store(args.index)

    if args.stats:
        st = store.stats()
        print(f"index:    {st['path']}")
        print(f"messages: {st['messages']}")
        print(f"chunks:   {st['chunks']}  {st['models']}")
        for c in st["cursors"]:
            print(f"  {c['channel']:<16} last_ts={c['last_ts']}  indexed_at={c['indexed_at']}")
        return 0

    # Deferred import: server imports this module, so importing it at module
    # level here would be circular.
    import server  # noqa: PLC0415

    model = server._OPENAI_EMBED_MODEL

    if server.USE_EXPORT:
        log.info("export mode: reading %s", server.EXPORT_PATH)
        raw = server._load_export_messages()
        seen, embedded = index_messages(store, raw, model, server._embed, server._chunk_message)
        log.info("export: %d messages → %d chunks, %d embedded", len(raw), seen, embedded)
        store.upsert_channels(server.list_channels())
        print(f"indexed {len(raw)} messages ({embedded} chunks embedded, {seen - embedded} unchanged)")
        return 0

    if args.channels:
        targets = [c.strip() for c in args.channels.split(",") if c.strip()]
    elif server.CHANNEL_IDS:
        targets = server.CHANNEL_IDS
    else:
        targets = [ch["id"] for ch in server.list_channels()]

    if not targets:
        log.error("no channels to index (set SLACK_CHANNEL_IDS or pass --channels)")
        return 1

    store.upsert_channels(server.list_channels())

    total_new = total_embedded = total_deleted = 0
    for channel in targets:
        cursor = store.get_cursor(channel)
        # --reconcile with no explicit window walks all of history; everything
        # else re-walks LOOKBACK_HOURS before the cursor.
        if args.reconcile and args.lookback_hours is None:
            oldest = None
        else:
            hours = LOOKBACK_HOURS if args.lookback_hours is None else args.lookback_hours
            oldest = _lookback_from(cursor, hours)
        raw = server._fetch_channel_api(channel, oldest=oldest)
        for msg in raw:
            if not msg.get("permalink"):
                msg["permalink"] = server._permalink(msg["channel"], msg["ts"])

        seen, embedded = index_messages(store, raw, model, server._embed, server._chunk_message)
        total_new += len(raw)
        total_embedded += embedded

        if args.reconcile:
            # Only messages inside the walked window can be judged deleted —
            # anything older simply wasn't fetched and must not be touched.
            live = {f"{channel}:{m['ts']}" for m in raw}
            stale = store.message_ids(model, channel, since_ts=oldest) - live
            total_deleted += store.delete_messages(stale, model)
            if stale:
                log.info("%s: dropped %d deleted message(s)", channel, len(stale))

        newest = _newest_ts(raw)
        if newest and (not cursor or float(newest) > float(cursor)):
            store.set_cursor(channel, newest)

        log.info("%s: +%d messages → %d chunks (%d embedded)", channel, len(raw), seen, embedded)

    print(
        f"indexed {total_new} messages, embedded {total_embedded} chunks"
        + (f", deleted {total_deleted}" if args.reconcile else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
