"""
slack-search-mcp  —  Slack semantic search over MCP
===================================================

Architecture
------------
Indexing and querying are separate.  `index.py` fetches from Slack, chunks,
embeds, and persists to SQLite on a schedule; this module only reads that
index.  Corpus work happens on a corpus schedule, query work on a query
schedule — see index.py for why.

Chunking pipeline (per message, run by the indexer):
  1. Filter out system sub-type messages (channel_join, etc.)
  2. Split on natural boundaries (paragraphs > lines > sentences)
  3. Pack tokens greedily up to CHUNK_SIZE_TOKENS with CHUNK_OVERLAP_TOKENS overlap
  4. Each chunk carries metadata linking back to the original message
  5. Units are hard-sliced so no chunk can exceed MAX_MODEL_TOKENS

Search pipeline (per query, this module):
  1. Load the stored vectors (cached; reloaded when the index file changes)
  2. Embed the query — one vector, the only paid call on the hot path
  3. Cosine-similarity against all stored chunk vectors
  4. Keep best-scoring chunk per message (dedup)
  5. Return top-K original messages with full text
"""

import logging
import os
import re
import json
import glob as glob_module
import concurrent.futures

import bm25s
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from fastmcp import FastMCP

from index import Chunk, Store, DEFAULT_INDEX_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("slack-search-mcp")

# Resolve .env next to this file, not relative to the process CWD — an MCP
# client launching server.py over stdio sets CWD to the project root, not here.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS: int    = int(os.environ.get("CHUNK_SIZE_TOKENS",    "350"))
CHUNK_OVERLAP_TOKENS: int = int(os.environ.get("CHUNK_OVERLAP_TOKENS",  "60"))
MAX_MODEL_TOKENS: int     = int(os.environ.get("MAX_MODEL_TOKENS",      "512"))

# Retrieval mode: dense (default) | bm25 | hybrid
# "dense"  — OpenAI embeddings + cosine similarity (current behaviour)
# "bm25"   — BM25 lexical index only
# "hybrid" — top-100 from dense + top-100 from BM25, fused with RRF
RETRIEVAL_MODE: str = os.environ.get("RETRIEVAL_MODE", "dense").strip().lower()

# Slack sub-types that carry no user content — skip them entirely
_SYSTEM_SUBTYPES = {
    "channel_join", "channel_leave", "channel_archive", "channel_unarchive",
    "channel_name", "channel_purpose", "channel_topic",
    "group_join", "group_leave", "group_archive", "group_unarchive",
    "group_name", "group_purpose", "group_topic",
    "bot_add", "bot_remove", "bot_enable", "bot_disable",
    "pinned_item", "unpinned_item",
    "ekm_access_denied",
}

# ---------------------------------------------------------------------------
# Mode: export file vs live API
# ---------------------------------------------------------------------------
EXPORT_PATH = os.environ.get("SLACK_EXPORT_PATH", "").strip()
USE_EXPORT  = bool(EXPORT_PATH)

# Slack is only reachable from the indexer now.  The query path reads the
# index, so the serving process runs fine with no token at all — only
# get_reactions still needs live access.
SlackApiError: type[BaseException] = Exception
slack = None

if not USE_EXPORT:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError  # noqa: F811
    _token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if _token:
        slack = WebClient(token=_token)
    else:
        log.info("SLACK_BOT_TOKEN not set — serving from the index only.")

CHANNEL_IDS: list[str] = [
    c.strip() for c in os.environ.get("SLACK_CHANNEL_IDS", "").split(",") if c.strip()
]

# Index location — the query path's only data source.
INDEX_PATH: str = os.environ.get("INDEX_PATH", DEFAULT_INDEX_PATH)

# ---------------------------------------------------------------------------
# OpenAI embeddings
# ---------------------------------------------------------------------------
_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Tokenizer — cl100k_base matches the tokenizer used by OpenAI embedding models.
# Used solely for token-counting / chunking.
_tokenizer = tiktoken.get_encoding("cl100k_base")


# Chunk lives in index.py — it is the shape the store reads and writes, and
# keeping it there lets index.py stay free of any server import.

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------
def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def _tokens(text: str) -> list[int]:
    return _tokenizer.encode(text)


def _decode_tokens(token_ids: list[int]) -> str:
    return _tokenizer.decode(token_ids)


# Re-inserted between packed units — see _chunk_message.
_SEPARATOR_TOKENS: list[int] = _tokenizer.encode("\n")


# ---------------------------------------------------------------------------
# Natural-boundary splitter
# ---------------------------------------------------------------------------
# Priority: paragraph > blank-line > bullet/numbered-list item > sentence > word
_SPLIT_PATTERNS = [
    re.compile(r"\n{2,}"),                          # paragraph break
    re.compile(r"\n"),                              # single newline
    re.compile(r"(?<=\w[.!?])\s+(?=[A-Z\"'])"),    # sentence boundary
    re.compile(r"\s+"),                             # word boundary (fallback)
]


def _split_text_naturally(text: str) -> list[str]:
    """
    Split *text* into the smallest natural units we care about.
    We try paragraph breaks first; if any piece is still over CHUNK_SIZE_TOKENS
    we recurse with the next finer pattern, down to individual words.

    Every returned unit is guaranteed to be <= CHUNK_SIZE_TOKENS: text with no
    natural boundary left (base64 blobs, minified JSON) is hard-sliced on token
    windows rather than returned oversized.
    """
    def _split_with(pattern: re.Pattern, txt: str) -> list[str]:
        parts = [p for p in pattern.split(txt) if p.strip()]
        return parts if parts else [txt]

    def _recursive_split(txt: str, pattern_idx: int) -> list[str]:
        if _count_tokens(txt) <= CHUNK_SIZE_TOKENS:
            return [txt] if txt.strip() else []
        if pattern_idx >= len(_SPLIT_PATTERNS):
            # No boundary left to split on — slice on raw token windows so a
            # single unbroken blob can never produce an oversized chunk.
            toks = _tokens(txt)
            return [_decode_tokens(toks[i:i + CHUNK_SIZE_TOKENS])
                    for i in range(0, len(toks), CHUNK_SIZE_TOKENS)]
        raw_parts = _split_with(_SPLIT_PATTERNS[pattern_idx], txt)
        result = []
        for part in raw_parts:
            if _count_tokens(part) <= CHUNK_SIZE_TOKENS:
                result.append(part)
            else:
                result.extend(_recursive_split(part, pattern_idx + 1))
        return result

    return _recursive_split(text, 0)


# ---------------------------------------------------------------------------
# Core chunking function
# ---------------------------------------------------------------------------
def _chunk_message(msg: dict) -> list[Chunk]:
    """
    Convert one Slack message dict into one or more Chunk objects.

    Returns an empty list for:
    - system sub-type messages (channel_join, bot_add, etc.)
    - messages with no meaningful text
    - messages with only whitespace / URLs / emoji
    """
    # ── Filter system messages ──────────────────────────────────────────────
    subtype = msg.get("subtype", "")
    if subtype in _SYSTEM_SUBTYPES:
        return []

    text = (msg.get("text") or "").strip()
    if not text:
        return []

    # Skip trivially non-semantic content (only emoji / URL / whitespace)
    cleaned = re.sub(r"<[^>]+>", "", text)          # strip Slack mention/URL markup
    cleaned = re.sub(r":[a-z0-9_\-+]+:", "", cleaned)  # strip :emoji:
    cleaned = cleaned.strip()
    if not cleaned:
        return []

    channel   = msg.get("channel", "")
    ts        = msg.get("ts", "")
    author    = msg.get("user") or msg.get("bot_id") or "unknown"
    permalink = msg.get("permalink", "")
    thread_ts = msg.get("thread_ts", "") or ""
    msg_id    = f"{channel}:{ts}"

    token_count = _count_tokens(text)

    # ── Short message → single chunk ────────────────────────────────────────
    if token_count <= CHUNK_SIZE_TOKENS:
        chunk = Chunk(
            message_id=msg_id, chunk_index=0, total_chunks=1,
            channel=channel, ts=ts, author=author,
            full_text=text, permalink=permalink, thread_ts=thread_ts,
            chunk_text=text,
        )
        return [chunk]

    # ── Long message → overlapping chunks ───────────────────────────────────
    # 1. Split into natural units (paragraphs, sentences, words)
    units = _split_text_naturally(text)

    # 2. Greedily pack units into chunks with token-based overlap
    chunks_text: list[str] = []
    current_tokens: list[int] = []

    def _flush(toks: list[int]) -> None:
        # MAX_MODEL_TOKENS is the hard ceiling the embedding model accepts.
        # Units are already <= CHUNK_SIZE_TOKENS, so this only bites on a
        # misconfigured CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS combination.
        chunks_text.append(_decode_tokens(toks[:MAX_MODEL_TOKENS]))

    for unit in units:
        unit_toks = _tokens(unit)

        if len(current_tokens) + len(unit_toks) > CHUNK_SIZE_TOKENS and current_tokens:
            # Flush current window
            _flush(current_tokens)
            # Keep overlap: last CHUNK_OVERLAP_TOKENS tokens
            current_tokens = current_tokens[-CHUNK_OVERLAP_TOKENS:] if CHUNK_OVERLAP_TOKENS else []

        # The splitter consumed the delimiter it split on, so re-insert one.
        # Without this the last token of a unit fuses with the first token of
        # the next ("...in detail.Paragraph 1..."), which corrupts both the
        # embedded text and the BM25 term at every join.
        if current_tokens:
            current_tokens.extend(_SEPARATOR_TOKENS)

        current_tokens.extend(unit_toks)

    if current_tokens:
        _flush(current_tokens)

    # 3. Build Chunk objects
    total = len(chunks_text)
    result: list[Chunk] = []
    for idx, chunk_text in enumerate(chunks_text):
        result.append(Chunk(
            message_id=msg_id, chunk_index=idx, total_chunks=total,
            channel=channel, ts=ts, author=author,
            full_text=text, permalink=permalink, thread_ts=thread_ts,
            chunk_text=chunk_text,
        ))

    return result


# ---------------------------------------------------------------------------
# BM25 tokenizer — Slack-aware, pure function
# ---------------------------------------------------------------------------
# Design requirements (from plan):
#   • lowercase
#   • split on whitespace and most punctuation
#   • PRESERVE tokens with internal underscores, hyphens, or digits intact
#     (ERR_500, pd-1234, us-east-1, @jsmith)
#   • strip Slack markup: <@U123> → drop; <http://…|label> → keep label
#   • do NOT stem, do NOT aggressively remove stopwords

# Compiled once at module load
_RE_SLACK_MENTION  = re.compile(r"<@[A-Z0-9]+>")           # <@U123ABC> → drop
_RE_SLACK_URL      = re.compile(r"<https?://[^|>]+\|([^>]+)>")  # <url|label> → label
_RE_SLACK_URL_BARE = re.compile(r"<https?://[^>]+>")        # <url> → drop
_RE_SPLIT          = re.compile(r"[^\w\-]+")                # split on non-word/non-hyphen


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize *text* for BM25 indexing.

    Transformations applied in order:
      1. Resolve <url|label> → label text
      2. Drop bare <url> and <@mention> Slack markup
      3. Lowercase
      4. Split on whitespace + punctuation, but preserve tokens that contain
         internal underscores, hyphens, or digits (ERR_500, pd-1234, @jsmith)
      5. Drop empty tokens; do not stem or remove stopwords

    Returns a list of string tokens.
    """
    # 1. Resolve labelled Slack URLs → keep the human-readable label
    text = _RE_SLACK_URL.sub(lambda m: m.group(1), text)
    # 2. Drop bare URLs and @mentions
    text = _RE_SLACK_URL_BARE.sub(" ", text)
    text = _RE_SLACK_MENTION.sub(" ", text)
    # 3. Lowercase
    text = text.lower()
    # 4. Split — \w matches [a-z0-9_], so tokens with _ and digits survive;
    #    we also allow hyphens (-) as internal connectors by splitting only on
    #    sequences that contain no word chars and no hyphens.
    tokens = _RE_SPLIT.split(text)
    # 5. Drop empty strings produced by leading/trailing delimiters
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# BM25 lexical index
# ---------------------------------------------------------------------------
class LexicalIndex:
    """
    Thin wrapper around bm25s that operates on the same message records
    as the dense index.  Built from scratch on every call (matching the
    in-memory-only pattern of the dense index — no new persistence infra).

    Usage:
        idx = LexicalIndex(chunks)           # build
        results = idx.search(query, k=100)   # [(message_id, score), ...]
    """

    def __init__(self, chunks: list) -> None:  # chunks: list[Chunk]
        self._chunks = chunks
        self._index  = bm25s.BM25()

        if not chunks:
            return

        # Tokenise every chunk; bm25s expects a list-of-lists-of-strings
        corpus_tokens = [_tokenize_for_bm25(c.chunk_text) for c in chunks]
        self._index.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """
        Return up to *k* (message_id, score) pairs, best first.
        Deduplicates: keeps the highest BM25 score per original message.
        Returns [] if the index is empty.
        """
        if not self._chunks:
            return []

        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []

        # bm25s.retrieve expects a list of token lists (one per query)
        n_results = min(k, len(self._chunks))
        results, scores = self._index.retrieve(
            [query_tokens], corpus=self._chunks, k=n_results, show_progress=False
        )
        # results[0] → list of Chunk objects; scores[0] → parallel float array
        best: dict[str, tuple[float, object]] = {}
        for chunk, score in zip(results[0], scores[0]):
            s = float(score)
            # A BM25 score of 0 means not one query term appeared in this chunk.
            # bm25s pads its top-k with those, and returning them hands the model
            # documents with no lexical relationship to the query at all.  This is
            # not the calibrated relevance threshold (still open — see README);
            # it needs no calibration, because zero overlap is unambiguous.
            if s <= 0.0:
                continue
            if chunk.message_id not in best or s > best[chunk.message_id][0]:
                best[chunk.message_id] = (s, chunk)

        return sorted(
            [(mid, s) for mid, (s, _) in best.items()],
            key=lambda x: x[1],
            reverse=True,
        )


# ---------------------------------------------------------------------------
# RRF fusion (pure function — Phase 3 helper, used by hybrid mode)
# ---------------------------------------------------------------------------
def _rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over *ranked_lists* of message_ids.

    score(d) = Σ  1 / (k + rank_in_list)   for each list containing d
               (ranks are 1-based)

    Tie-break: ascending message_id (deterministic, stable).
    Returns a list of (message_id, rrf_score) sorted descending by score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),   # descending score, then ascending id
    )


# ---------------------------------------------------------------------------
# Core ranking function (chunk → embed → score → dedup → format)
# ---------------------------------------------------------------------------
def _chunk_to_result(score: float, chunk: "Chunk") -> dict:
    """Format a (score, Chunk) pair into the MCP tool response dict."""
    return {
        "text":      chunk.full_text,
        "author":    chunk.author,
        "channel":   chunk.channel,
        "ts":        chunk.ts,
        "permalink": chunk.permalink,
        "score":     round(score, 4),
    }


def _dense_ranked(chunks: list[Chunk], vecs: np.ndarray, query: str) -> dict:
    """
    Score *chunks* against *query* using their stored vectors.

    Only the query is embedded here — one small API call, independent of corpus
    size.  Returns message_id → (score, Chunk) for the best chunk per message.
    """
    query_vec = _embed([query])[0]
    scores    = _cosine_similarity(query_vec, vecs)

    best: dict[str, tuple[float, Chunk]] = {}
    for idx, chunk in enumerate(chunks):
        s = float(scores[idx])
        if chunk.message_id not in best or s > best[chunk.message_id][0]:
            best[chunk.message_id] = (s, chunk)
    return best


def _rank_chunks(
    chunks: list[Chunk],
    vecs: np.ndarray | None,
    query: str,
    k: int,
    lex: "LexicalIndex | None" = None,
) -> list[dict]:
    """
    Rank pre-embedded *chunks* against *query* and return up to *k* results.

    `vecs[i]` must be the stored vector for `chunks[i]`.  Pass vecs=None only in
    bm25 mode, which never reads them.  *lex* is an optional prebuilt BM25 index
    (the server caches one per index load); a fresh one is built if omitted.

    Retrieval mode comes from RETRIEVAL_MODE (default: dense):
      dense  — cosine similarity against stored vectors
      bm25   — BM25 lexical index only
      hybrid — top candidates from both, fused with RRF

    This is the single ranking path used by both search_slack() and the eval
    harness — both read from a Store, so evaluation always measures the code
    that actually serves queries.
    """
    if not chunks:
        log.info("_rank_chunks: index is empty for this query.")
        return []

    log.info("_rank_chunks: mode=%s  %d chunks in scope", RETRIEVAL_MODE, len(chunks))

    # message_id → any Chunk, to recover metadata for BM25-only hits
    chunk_by_msg: dict[str, Chunk] = {}
    for c in chunks:
        chunk_by_msg.setdefault(c.message_id, c)

    if RETRIEVAL_MODE == "bm25":
        lex = lex or LexicalIndex(chunks)
        output = []
        for mid, score in lex.search(query, k=k):
            chunk = chunk_by_msg.get(mid)
            if chunk:
                output.append(_chunk_to_result(score, chunk))
        return output

    if vecs is None or len(vecs) != len(chunks):
        raise ValueError("dense/hybrid ranking needs one stored vector per chunk")

    if RETRIEVAL_MODE == "hybrid":
        # Wider candidate sets from both retrievers before fusing
        CANDIDATE_K = max(k * 10, 100)

        dense_best = _dense_ranked(chunks, vecs, query)
        dense_list = [mid for mid, _ in sorted(
            dense_best.items(), key=lambda x: x[1][0], reverse=True
        )[:CANDIDATE_K]]

        lex = lex or LexicalIndex(chunks)
        bm25_list = [mid for mid, _ in lex.search(query, k=CANDIDATE_K)]

        fused = _rrf_fuse([dense_list, bm25_list])[:k]

        output = []
        for mid, rrf_score in fused:
            # Prefer the dense chunk (it is the best-scoring one for this
            # message); fall back to any chunk for BM25-only hits.
            chunk = dense_best[mid][1] if mid in dense_best else chunk_by_msg.get(mid)
            if chunk:
                output.append(_chunk_to_result(rrf_score, chunk))
        return output

    # ── dense (default) ───────────────────────────────────────────────────────
    best = _dense_ranked(chunks, vecs, query)
    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)[:k]
    return [_chunk_to_result(score, chunk) for score, chunk in ranked]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _embed(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts via OpenAI.
    Returns a 2-D float32 array of shape (len(texts), embedding_dim).
    """
    response = _openai.embeddings.create(input=texts, model=_OPENAI_EMBED_MODEL)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between query_vec and every row of doc_vecs."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
    return (doc_vecs / norms) @ q


# ---------------------------------------------------------------------------
# Export-mode helpers
# ---------------------------------------------------------------------------
def _load_export_messages(channel_names: list[str] | None = None) -> list[dict]:
    """
    Read messages from a Slack export folder.

    Structure:
        <SLACK_EXPORT_PATH>/
            #channel-name/
                2024-01-15.json
            channels.json
            users.json
    """
    workspace_url = os.environ.get("SLACK_WORKSPACE_URL", "").rstrip("/")
    messages: list[dict] = []

    export_root = EXPORT_PATH
    channel_dirs = [
        d for d in os.listdir(export_root)
        if os.path.isdir(os.path.join(export_root, d)) and not d.startswith(".")
    ]
    if channel_names:
        channel_dirs = [
            d for d in channel_dirs
            if d in channel_names or d.lstrip("#") in channel_names
        ]

    for channel_dir in channel_dirs:
        channel_path = os.path.join(export_root, channel_dir)
        day_files = glob_module.glob(os.path.join(channel_path, "*.json"))
        for day_file in sorted(day_files):
            try:
                with open(day_file) as f:
                    day_messages = json.load(f)
                for msg in day_messages:
                    if msg.get("type") != "message":
                        continue
                    if msg.get("subtype") in _SYSTEM_SUBTYPES:
                        continue
                    if not msg.get("text"):
                        continue
                    ts_id     = msg["ts"].replace(".", "")
                    permalink = (
                        f"{workspace_url}/archives/{channel_dir}/p{ts_id}"
                        if workspace_url else ""
                    )
                    messages.append({**msg, "channel": channel_dir, "permalink": permalink})
            except (json.JSONDecodeError, OSError):
                continue

    return messages


# ---------------------------------------------------------------------------
# Live API helpers
# ---------------------------------------------------------------------------
def _fetch_thread_replies_api(channel_id: str, thread_ts: str) -> list[dict]:
    """Fetch all replies in a thread via the Slack API, excluding the parent."""
    replies: list[dict] = []
    try:
        result = slack.conversations_replies(channel=channel_id, ts=thread_ts)
        for msg in result.get("messages", []):
            if msg.get("ts") == thread_ts:
                continue
            if msg.get("type") != "message":
                continue
            if msg.get("subtype") in _SYSTEM_SUBTYPES:
                continue
            if msg.get("text"):
                replies.append({"channel": channel_id, **msg})
    except SlackApiError as e:
        log.warning("conversations.replies failed for %s/%s: %s", channel_id, thread_ts, e)
    return replies


def _fetch_channel_api(channel_id: str, oldest: str | None = None) -> list[dict]:
    """
    Fetch messages and thread replies from a channel via the Slack API.

    *oldest* is the stored cursor: pass it to fetch only what has been written
    since the last index run.  Omit it for a full history walk (--reconcile).
    """
    messages: list[dict] = []
    threaded_parent_ts: list[str] = []
    cursor = None
    try:
        while True:
            kwargs: dict = {"channel": channel_id, "limit": 200}
            if oldest:
                kwargs["oldest"] = oldest
            if cursor:
                kwargs["cursor"] = cursor
            result = slack.conversations_history(**kwargs)
            for msg in result.get("messages", []):
                if msg.get("type") != "message":
                    continue
                if msg.get("subtype") in _SYSTEM_SUBTYPES:
                    continue
                if not msg.get("text"):
                    continue
                messages.append({"channel": channel_id, **msg})
                if msg.get("reply_count", 0) > 0:
                    threaded_parent_ts.append(msg["ts"])
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        log.warning("_fetch_channel_api: failed for %s: %s", channel_id, e.response.get("error", str(e)))

    if threaded_parent_ts:
        with concurrent.futures.ThreadPoolExecutor() as tex:
            thread_results = tex.map(
                lambda ts: _fetch_thread_replies_api(channel_id, ts),
                threaded_parent_ts,
            )
        for replies in thread_results:
            messages.extend(replies)

    return messages


def _resolve_permalink_api(channel: str, ts: str) -> str:
    try:
        resp = slack.chat_getPermalink(channel=channel, message_ts=ts)
        return resp.get("permalink", "")
    except SlackApiError as e:
        log.warning("chat.getPermalink failed for %s/%s: %s", channel, ts, e)
        return ""


_workspace_url: str | None = None   # None = not looked up yet, "" = unavailable


def _get_workspace_url() -> str:
    """
    Workspace base URL, fetched once via auth.test (no extra OAuth scope needed).
    Returns "" if unavailable, which forces the chat.getPermalink fallback.
    """
    global _workspace_url
    if _workspace_url is None:
        try:
            _workspace_url = (slack.auth_test().get("url") or "").rstrip("/")
        except Exception as e:                      # noqa: BLE001 — any failure degrades
            log.warning("auth.test failed, falling back to chat.getPermalink: %s", e)
            _workspace_url = ""
    return _workspace_url


def _permalink(channel: str, ts: str) -> str:
    """
    Build a message permalink.

    Slack permalinks are deterministic — {workspace}/archives/{channel}/p{ts}
    with the dot stripped — so this costs zero API calls after the one-time
    auth.test lookup.  The previous implementation issued one chat.getPermalink
    per message, serially, which dominated query latency and hit rate limits on
    any corpus above a few hundred messages.
    """
    base = _get_workspace_url()
    if not base:
        return _resolve_permalink_api(channel, ts)
    return f"{base}/archives/{channel}/p{ts.replace('.', '')}"


def _discover_channels_api() -> list[dict]:
    """
    Enumerate channels from the Slack API.  Used by the indexer to find what to
    index; the MCP list_channels() tool reads the index instead.
    """
    channels: list[dict] = []
    cursor = None
    if slack is None:
        return channels
    try:
        while True:
            kwargs: dict = {"exclude_archived": True, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            result = slack.conversations_list(**kwargs)
            for ch in result.get("channels", []):
                channels.append({
                    "id":          ch["id"],
                    "name":        ch["name"],
                    "num_members": ch.get("num_members"),
                })
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        log.warning("conversations.list failed: %s", e)
    return channels


# ---------------------------------------------------------------------------
# Index cache
# ---------------------------------------------------------------------------
# Loaded once and reused across queries.  The mtime check means a cron indexing
# run is picked up without restarting the server — otherwise a long-lived
# process would serve a stale snapshot forever.
_index_chunks: list[Chunk] = []
_index_vecs: np.ndarray | None = None
_index_lex: "LexicalIndex | None" = None
_index_mtime: float = -1.0


def _load_index(force: bool = False) -> tuple[list[Chunk], np.ndarray | None]:
    """Return (chunks, vectors) from the index, reloading if the file changed."""
    global _index_chunks, _index_vecs, _index_lex, _index_mtime

    try:
        mtime = os.path.getmtime(INDEX_PATH)
    except OSError:
        mtime = -1.0
        if not _index_chunks:
            log.warning(
                "No index at %s — run `python index.py` to build one.", INDEX_PATH
            )

    if force or mtime != _index_mtime:
        store = Store(INDEX_PATH)
        try:
            _index_chunks, _index_vecs = store.load(_OPENAI_EMBED_MODEL)
        finally:
            store.close()
        # BM25 stats only change when the corpus does, so build once per load
        # rather than once per query.
        _index_lex = LexicalIndex(_index_chunks) if _index_chunks else None
        _index_mtime = mtime
        log.info("Loaded %d chunks from %s", len(_index_chunks), INDEX_PATH)

    return _index_chunks, _index_vecs


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="slack-search-mcp",
    instructions=(
        "You have access to tools that retrieve real messages from a Slack workspace. "
        "Answer ONLY using the content returned by these tools. "
        "Do not use prior knowledge or training data to answer questions about Slack. "
        "Cite every claim by including the permalink of the message it came from. "
        "If the tools return no relevant results, respond with: "
        "'I couldn't find anything in Slack about that.' "
        "Never invent, infer, or paraphrase beyond what the messages say."
    ),
)


@mcp.tool()
def list_channels() -> list[dict]:
    """
    List the Slack channels available to search, with their IDs and names.

    Use this to discover which channels are available before searching.
    Reads the index, so it reflects what is actually searchable — a channel the
    bot can see but has not indexed yet will not appear here.

    Returns:
        List of dicts with keys: id, name, num_members.
    """
    store = Store(INDEX_PATH)
    try:
        return store.channels()
    finally:
        store.close()


@mcp.tool()
def search_slack(query: str, channels: list[str] | None = None, limit: int = 8) -> list[dict]:
    """
    Search indexed Slack messages semantically.

    Reads the prebuilt index — only the query itself is embedded, so cost and
    latency are independent of workspace size.  Results are deduplicated: each
    original Slack message appears at most once, represented by its
    highest-scoring chunk, and 'text' is always the full untruncated message.

    Results are as fresh as the last indexing run (`python index.py`).

    Args:
        query:    Natural-language question or topic to search for.
        channels: Optional list of channel names or IDs to restrict the search to.
                  Defaults to every indexed channel.
        limit:    Maximum number of messages to return (default 8).

    Returns:
        List of dicts with keys: text, author, channel, ts, permalink, score.
    """
    chunks, vecs = _load_index()
    if not chunks:
        log.warning("search_slack: index is empty — run `python index.py` first.")
        return []

    lex = _index_lex

    # ── Optional channel filter ──────────────────────────────────────────────
    if channels:
        wanted = {c.lstrip("#") for c in channels}
        keep = [i for i, c in enumerate(chunks) if c.channel.lstrip("#") in wanted]
        if not keep:
            log.info("search_slack: no indexed chunks in channels %s", sorted(wanted))
            return []
        chunks = [chunks[i] for i in keep]
        vecs = vecs[keep] if vecs is not None and len(vecs) else vecs
        lex = None      # cached index covers the whole corpus, not this subset

    return _rank_chunks(chunks, vecs, query, limit, lex=lex)


@mcp.tool()
def get_thread(channel_id: str, thread_ts: str) -> list[dict]:
    """
    Fetch a full Slack message thread for deeper grounding context.

    Served from the index — the parent and its replies are stored with their
    thread_ts, so this needs no Slack call and returns the same messages that
    search_slack can surface.

    Args:
        channel_id: The Slack channel ID or export folder name containing the thread.
        thread_ts:  The timestamp of the parent message (thread root).

    Returns:
        List of dicts with keys: text, author, ts, permalink — ordered chronologically.
    """
    store = Store(INDEX_PATH)
    try:
        thread = store.thread(channel_id, thread_ts)
    finally:
        store.close()

    return [
        {
            "text":      c.full_text,
            "author":    c.author,
            "ts":        c.ts,
            "permalink": c.permalink,
        }
        for c in thread if c.full_text
    ]


@mcp.tool()
def get_reactions(channel_id: str, message_ts: str) -> list[dict]:
    """
    Fetch emoji reactions for a specific Slack message.

    Use this to understand sentiment or agreement on a message — e.g. how many
    people thumbs-upped a decision, or what reactions a proposal received.

    Args:
        channel_id:  The Slack channel ID containing the message.
        message_ts:  The timestamp (ts) of the message to fetch reactions for.

    Returns:
        List of dicts with keys: emoji (str), count (int), users (list[str]).
        Empty list if the message has no reactions or cannot be found.
    """
    # Reactions change after a message is written, so this is the one tool that
    # cannot be served from the index and still needs a live token.
    if slack is None:
        log.warning("get_reactions needs SLACK_BOT_TOKEN; returning [].")
        return []

    try:
        result = slack.reactions_get(channel=channel_id, timestamp=message_ts)
    except SlackApiError as e:
        log.warning("get_reactions: reactions.get failed for %s/%s: %s", channel_id, message_ts, e)
        return []

    msg = result.get("message", {})
    reactions = msg.get("reactions", [])
    return [
        {
            "emoji": r["name"],
            "count": r["count"],
            "users": r.get("users", []),
        }
        for r in reactions
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "sse")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.environ.get("PORT", 8002))
        mcp.run(transport="sse", host="0.0.0.0", port=port)
