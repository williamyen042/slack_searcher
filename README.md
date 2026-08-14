# Slack Searcher — MCP-powered Slack search, grounded and cited

Ask questions about your Slack workspace. A scheduled indexer keeps a local SQLite index of your messages and their embeddings; at query time the assistant searches that index (dense, BM25, or hybrid) and answers **only** from what it finds — every claim cited with a link back to the original message. If nothing relevant is found, it says so rather than guessing.

> 🏆 **5th place / 93 teams** — IBM's company wide hackathon

---

## What this project is — plain English

> _For anyone who isn't deep into software engineering but wants to understand what was built and why._

**The problem:** Your team's knowledge lives in Slack — decisions, fixes, plans, debates. But finding any of it is painful. You either search manually and give up, or you ask an AI assistant and it confidently makes something up because it has no idea what your team actually said.

**What this does differently:** Instead of guessing, the assistant goes and _reads_ your Slack workspace first. When you ask a question, it searches through your real messages, finds the most relevant ones, and then answers — but only using what it actually found. Every sentence in its answer is backed by a clickable link to the original Slack message. If the answer isn't there, it tells you so.

**Why this matters:** The model can't hallucinate about your Slack because it has no other source of information. It's not drawing on training data or making educated guesses — the only thing it can tell you is what your team actually wrote.

### How it works in plain steps

There are two halves, and they run on completely different schedules.

**Ahead of time — a scheduled job (every ~15 minutes):**

```
1. Fetch whatever has been posted since it last looked
2. Break long messages into smaller readable pieces
3. Convert each piece into a mathematical fingerprint (a vector)
   that captures its meaning
4. Save the pieces and their fingerprints to a local database file
```

It only does this for messages that are new or edited. Everything already saved is
left alone, which is what keeps it cheap.

**When you ask a question:**

```
1. You type a question into the chat interface

2. The assistant decides it needs to search Slack

3. It calls a "search tool" which:
   a. Converts your question into a fingerprint
   b. Compares it against every saved fingerprint in the database
   c. Returns the most similar messages — ranked by relevance

4. The assistant reads those messages and writes an answer
   — citing each Slack message it used with a link

5. You see the answer with clickable source links in the chat UI
```

Nothing in step 3 talks to Slack, so answers come back fast no matter how big the
workspace is. The trade: you only see messages as recent as the last indexing run.

The key insight: **the search tool is the only door into your Slack data**. The model cannot go around it. This is what makes it trustworthy — the constraint is architectural, not just a polite instruction.

### What the three retrieval modes mean (non-technical)

As the project evolved, three different ways of finding messages were built:

| Mode | What it does | Best for |
|---|---|---|
| **Dense** (default) | Finds messages with similar *meaning*, even if they use different words | "What did we decide about security?" |
| **BM25** | Finds messages containing the *exact words* you typed | `ERR_500`, `pd-1234`, `us-east-1` |
| **Hybrid** | Does both, then combines the rankings intelligently | Most real-world queries |

Think of Dense as a librarian who understands context, and BM25 as a word-for-word text search. Hybrid uses both and picks the best results from each.

> Searching for `@someone` does **not** find their Slack mentions. Slack stores a mention
> as an internal user ID (`<@U0123ABC>`), which the indexer discards — so `@jsmith` only
> matches messages where someone typed that name as ordinary text. See
> [Roadmap](#roadmap--future-work).

---

## What this project is — technical deep dive

> _For engineers who want to understand every decision, tradeoff, and design choice._

### The core architectural guarantee

Hallucination is prevented **structurally**, not by asking nicely. The system prompt instructs the model to answer only from tool output — but more importantly, the MCP tool surface is the *only information channel* it has access to at runtime. There is no path through which the model can draw on training data about this specific Slack workspace. If `search_slack()` returns nothing, there is nothing to say.

### Indexing and querying are separate

Embedding text is the slow, paid step, and it is a pure function of (model, chunk text). Doing it on the query path meant a message written three months ago was re-embedded every time anyone asked anything — cost and latency both scaled with corpus size, on every query.

So the pipeline is cut in half:

```
INDEX  (python index.py, on a timer)      QUERY  (per request)
fetch since stored cursor                 embed 1 query vector
chunk new / changed only            →     matmul against stored vectors
embed new / changed only                  dedup + top-k
persist to SQLite
```

Steady-state embedding cost is "messages written since the last run", not "the whole workspace". A query costs one embedding call for the query text and a matrix multiply, whatever the corpus size. The tradeoff: results are as fresh as the last indexing run.

### The RAG pipeline in detail

**Fetch (index time):** Messages are pulled from Slack via `conversations.history` (with thread replies via `conversations.replies`) or from an unzipped export folder. Each run passes `oldest = cursor − INDEX_LOOKBACK_HOURS`, so it fetches new messages plus a rolling re-walk that catches replies and edits on recent threads (see [Why three schedules](#2-mcp-server)). Permalinks are built locally from the workspace URL (one cached `auth.test`) rather than one `chat.getPermalink` per message. Both modes produce the same message dict shape — one env var (`SLACK_EXPORT_PATH`) switches between them.

**Chunk (index time):** Each message is split into overlapping token windows (350 tokens, 60-token overlap) using `tiktoken`'s `cl100k_base` — the identical tokeniser OpenAI's embedding model uses. Character-count or word-count splits are rejected because they don't respect token boundaries (one emoji = up to 4 tokens). Natural boundaries are respected: the splitter tries paragraph breaks first, then single newlines, then sentence boundaries, then words. Every chunk carries the complete original `full_text` so results returned to the model are never truncated.

**Embed (index time):** Batched OpenAI calls of `EMBED_BATCH` chunks (default 256 — the API caps a request at 2048 inputs). Only chunks whose `text_hash` is new or changed are sent, so a re-run over an unchanged corpus makes zero API calls. `text-embedding-3-small` was chosen for setup simplicity — no project IDs, no region endpoints, one API key. Tradeoff: a hard dependency on OpenAI availability, and indexed message content leaves your network. If that isn't acceptable, swap `_embed()` for a locally-hosted model — it is the only function that needs to change.

**Store (`index.py`):** SQLite, not a vector database. Brute-force cosine over a stored matrix is ~100 ms at a million chunks, so approximate nearest neighbour buys nothing at this size and costs a service to operate. `model` is part of the primary key, which makes mixing vectors from different embedding models structurally impossible rather than a thing you remember not to do. A forward-only cursor can't see changes that don't move a timestamp — thread replies, edits, deletions — so the lookback window covers the first two and `--reconcile` covers the third. The index holds message text in plaintext; see [Data at rest](#data-at-rest).

**Rank — `_rank_chunks()`:** Only the query is embedded. Cosine similarity against the stored vectors. `text-embedding-3-small` already returns unit-length vectors, so cosine and dot product are equivalent *for this backend* — the explicit normalisation costs nothing on top of the matmul and means rankings don't silently change if `_embed()` is swapped for a model that doesn't normalise. Best-scoring chunk per `message_id` (dedup). Sort descending. `search_slack()` and the eval harness both read a `Store` and call this same function, so evaluation always measures the code that actually serves queries.

**BM25 — `LexicalIndex`:** A `bm25s` wrapper over the same chunks the dense index loads, built once per index load rather than per query (corpus statistics only change when the corpus does). Addresses the failure mode where dense embeddings generalise exact tokens (`ERR_500` → "server error region") away from the literal string that needs to be found. Uses a hand-written Slack-aware tokeniser instead of NLTK/spaCy because standard NLP tokenisers split on hyphens and underscores — destroying `ERR_500`, `pd-1234`, `us-east-1`. No stemming, no stopword removal: short Slack queries need every word.

**RRF — `_rrf_fuse()`:** Reciprocal Rank Fusion combines dense and BM25 ranked lists. Score normalisation (min-max then sum) is rejected because dense scores (cosine ~0–1) and BM25 scores (term frequency ~0–∞) are incommensurable — one outlier changes the normalisation of every other result. RRF works only on ranks: `score(d) = Σ 1/(k + rank)` over all lists containing `d`, ranks 1-based, k=60 (the empirically validated default from Cormack et al. 2009). Deterministic tie-break by ascending `message_id`.

### Retrieval mode switching

`RETRIEVAL_MODE` is an env var, not a parameter on `search_slack()`. The MCP tool signature is locked — adding a `mode` parameter would change the public API that the system prompt and all external clients depend on. An env var lets you flip modes at server startup with zero interface change.

### The evaluation harness

Built in Phase 1, before any BM25 code, deliberately. The evaluation harness (`eval/run_eval.py`) builds an in-memory index from the corpus and calls `_rank_chunks()` — the same function `search_slack()` delegates to, reading the same `Store` shape. Pass `--index` to evaluate an existing on-disk index at zero embedding cost. It computes Recall@5, Recall@10, and MRR@10 overall and broken down by `query_type` (keyword / semantic / mixed). Every retrieval change produces a committed results JSON file. Changes that don't beat the baseline revert.

**Why Recall+MRR over Precision or NDCG:**
- Precision@k is rejected because false positives (irrelevant results) don't break the LLM — it can ignore them. False negatives (missing the relevant result) do break it. Recall prioritises the failure mode that matters.
- NDCG requires graded relevance labels (0/1/2/3). The golden set has binary labels. NDCG is the wrong metric for binary relevance.

**Why the per-type breakdown matters:** Overall Recall/MRR can mask opposing effects. BM25 might improve keyword queries by +0.3 while degrading semantic queries by −0.3 — aggregate is flat and you'd ship a regression. The breakdown surfaces this.

> **Status:** the committed baseline in `eval/results/` was recorded against an empty corpus (`corpus_size: 0`) and `golden_queries.jsonl` still has empty `relevant_ids`, so every metric is 0.0. The harness runs and `--mode` is honoured; the golden set needs filling in against a real corpus before any mode comparison counts as evidence.

### Test architecture

`server.py` imports `openai`, `slack_sdk`, and `fastmcp` at module load time — before any test setup runs. Module-level stubs install fake versions of those modules into `sys.modules` before `import server`, so the test suite runs with `OPENAI_API_KEY=fake-key` and zero network access. Ranking tests use deterministic hand-crafted 3-dimensional vectors mapped by keyword so ranking order is an exact assertion, not a probabilistic check that model updates could silently break. Tool tests build a real (temporary) SQLite index with those stubbed vectors and point `server.INDEX_PATH` at it, so they exercise the same store production reads rather than a mock of it.

### What's in the roadmap and why it's deferred

| Item | Why deferred |
|---|---|
| Relevance threshold | Needs a different gate per retrieval mode and a filled-in golden set to calibrate against. Shipping an uncalibrated constant is worse than shipping none. |
| Thread-level indexing | Thread is the unit of meaning in Slack; individual messages lose conversational context. Requires index schema changes. |
| Cross-encoder reranking | Only justified if Phase 4 eval shows remaining semantic gap after hybrid. A reranker without a measured gap is wasted latency. |
| Per-user permission filtering | Required before indexing any private channel. Cannot safely filter at query time without resolving caller identity → channel memberships. |
| Per-user erasure (`--forget`) | No targeted delete path yet. See [Data at rest](#data-at-rest) for the manual workaround. |
| `@mention` resolution | Needs a `users.list` ID→name map at index time, plus the `users:read` scope — a scope increase for a feature nobody has asked for yet. |

---

## Architecture

```
                                          index.py  (cron, every ~15 min)
                                            ├─ Slack API (incremental)
Browser (Next.js chat UI)                   └─ OpenAI embeddings
  └─ POST /api/chat (streaming)                    │
       └─ LLM host — system prompt enforces        ▼
            grounding                        ┌───────────────┐
            └─ MCP tools (HTTP/SSE)          │  SQLite index │
                 └─ Python MCP server ───────┤  chunks + vecs│
                      ├─ search_slack        └───────────────┘
                      ├─ get_thread                 ▲
                      ├─ list_channels ─────────────┘
                      └─ get_reactions → Slack API (live; the only one)
```

The serving process reads the index and nothing else, so it runs with no Slack token at all — `get_reactions` is the sole exception, because reaction counts change after a message is written.

## Project structure

```
.
├── mcp-server/
│   ├── server.py              # FastMCP server — MCP tools + ranking (reads the index)
│   ├── index.py               # SQLite chunk/vector store + `python index.py` indexer
│   ├── pyproject.toml         # Python deps (includes bm25s)
│   ├── .env.example           # All env vars documented with defaults
│   ├── eval/
│   │   ├── golden_queries.jsonl   # 20 synthetic eval queries (keyword/semantic/mixed)
│   │   ├── run_eval.py            # Eval runner — Recall@5/10, MRR@10, per query_type
│   │   └── results/               # Committed baseline result files (one per phase gate)
│   └── tests/
│       ├── test_chunking.py   # Chunking pipeline unit tests
│       ├── test_search.py     # search_slack + get_thread tool tests
│       ├── test_eval.py       # Eval metric functions + eval loop on fixture corpus
│       ├── test_bm25.py       # Tokeniser, LexicalIndex, RRF fusion, mode switching
│       └── test_index.py      # Store round-trip, incremental skip, edits, cursors
├── web-client/
│   ├── src/app/
│   │   ├── page.tsx           # Chat UI with streaming + citation rendering
│   │   ├── layout.tsx
│   │   └── api/chat/
│   │       └── route.ts       # Edge function — streaming proxy to the LLM API
│   ├── package.json
│   └── .env.local.example
├── overview/
│   └── index.html             # Static project overview page (open in browser)
└── .mcp.json                  # Registers the MCP server with your MCP client
```

## Setup

### 1. Slack Bot

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → From an app manifest, and paste `slack-manifest.json` (or create from scratch and add the scopes below).
2. Under **OAuth & Permissions**, confirm the Bot Token Scopes:
   - `channels:history`
   - `channels:read`
   - `reactions:read` (only needed for the `get_reactions` tool)
3. **Install App to Workspace** → copy the **Bot User OAuth Token** (`xoxb-…`).
4. Invite the bot to the channels you want it to search: `/invite @your-bot-name`

> **Access model — read this before inviting the bot anywhere.** The bot can only read channels it has been invited to, so channel access is controlled at *ingestion*. For now, only invite it to **public channels where participants have opted in**. This is safe because public-channel content is already visible to anyone in the workspace. Do **not** invite it to private or restricted channels yet — see [Roadmap](#roadmap--future-work) for why that requires query-time permission filtering first.
>
> Note that inviting the bot to a channel means its messages get **copied to disk** in the index, in plaintext. See [Data at rest](#data-at-rest).

### 2. MCP server

```bash
cd mcp-server
cp .env.example .env
# Fill in .env with your tokens and channel IDs
pip install -e .

# 1. Build the index (first run embeds the whole corpus; later runs only the delta)
python index.py
python index.py --stats      # what's in there now

# 2. Serve it
python server.py
# Server runs on http://localhost:8002/sse
```

Keep the index fresh with cron — searches only see what has been indexed:

```cron
# Every 15 min — new messages, plus a 48h re-walk for active threads. Cheap.
*/15 *  * * *  cd /path/to/mcp-server && .venv/bin/python index.py

# Nightly — re-walk 2 weeks and drop anything deleted in that window.
0     3  * * *  cd /path/to/mcp-server && .venv/bin/python index.py --reconcile --lookback-hours 336

# Weekly — full history deep clean, for deletions older than 2 weeks.
0     4  * * 0  cd /path/to/mcp-server && .venv/bin/python index.py --reconcile
```

**Why three schedules.** A cursor only advances past things with a *new* timestamp, and
three kinds of change don't have one: a reply added to an old thread, an edit to an old
message (editing doesn't change `ts`), and a deletion. Re-walking a window catches all
three — and re-walked messages that haven't changed cost **zero embeddings**, because the
`text_hash` check skips them. You only repay the fetch.

That makes it a straight cost/freshness dial, so the three runs pick different points on it:

| Run | Window | Catches | Cost |
|---|---|---|---|
| Every 15 min | 48h | New messages, replies and edits on active threads | A page or two per channel |
| Nightly | 2 weeks | Replies and edits on threads up to 2 weeks old; deletions in that window | Once a day, no embeddings unless something changed |
| Weekly | Full history | Deletions older than 2 weeks | Slowest — run it off-peak |

Tune the frequent run with `--lookback-hours` or `INDEX_LOOKBACK_HOURS`. The deliberate
gap: a reply to a thread older than the nightly window is never indexed. If that matters
for your workspace, widen the nightly window — the cost is fetches, not embeddings.

> `--reconcile` bounds deletion detection to whatever window it walked. A message outside
> that window is untouched, never assumed deleted — otherwise a windowed reconcile would
> wipe every message older than its own window.

The server picks up a new index automatically — it reloads when the file's mtime changes,
so no restart is needed.

`.env` values:

| Variable | Description |
|---|---|
| `SLACK_BOT_TOKEN` | Bot User OAuth Token from step 1 (omit if using export mode) |
| `SLACK_EXPORT_PATH` | Path to an unzipped Slack export — set this *instead of* the bot token to run offline |
| `SLACK_WORKSPACE_URL` | Workspace URL, used to build permalinks in export mode |
| `SLACK_CHANNEL_IDS` | Comma-separated channel IDs to index (e.g. `C0123,C4567`). Blank = every channel the bot can see |
| `OPENAI_API_KEY` | Used for embeddings (`text-embedding-3-small` by default) |
| `INDEX_PATH` | Where the SQLite index lives (default `mcp-server/slack_index.db`) |
| `INDEX_LOOKBACK_HOURS` | How far before the cursor each incremental run re-walks (default 48) |
| `EMBED_BATCH` | Chunks per embedding request while indexing (default 256; API caps at 2048) |
| `OPENAI_EMBED_MODEL` | Embedding model (default `text-embedding-3-small`). Changing it re-embeds everything — vectors are namespaced by model |
| `RETRIEVAL_MODE` | `dense` (default), `bm25`, or `hybrid` |
| `MCP_TRANSPORT` / `PORT` | `sse` (default, port 8002) or `stdio` |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` / `MAX_MODEL_TOKENS` | Chunking knobs (350 / 60 / 512) |

Only `index.py` needs `SLACK_BOT_TOKEN`. The serving process reads the index, so you can
run it without one — `get_reactions` is the only tool that degrades (returns `[]`).

> **Where your data goes:** message text is sent to OpenAI's embeddings API when it is
> *indexed* — once per message, not once per query, and never again unless it's edited.
> `bm25` mode is fully local at query time but still needs the index built. If message
> content must not leave your network, replace `_embed()` with a self-hosted model.
> Message text is also written to the index file in plaintext — see
> [Data at rest](#data-at-rest) before pointing this at a workspace you don't own.

### 3. Web client

```bash
cd web-client
cp .env.local.example .env.local
# Fill in LLM_API_URL and LLM_API_KEY
npm install
npm run dev
# App runs on http://localhost:3000
```

`.env.local` values:

| Variable | Description |
|---|---|
| `LLM_API_URL` | Base URL of an OpenAI-compatible chat completions API |
| `LLM_API_KEY` | API key for that endpoint |

### 4. MCP client connection

`.mcp.json` at the repo root registers the server with any MCP client. It is checked in
and holds **no secrets** — `server.py` loads `mcp-server/.env` from its own directory, so
credentials never need to appear in this file.

**Option A — stdio (recommended)**

The client launches the Python process itself; no separate `python server.py` needed.

```json
{
  "mcpServers": {
    "slack-search": {
      "command": "/absolute/path/to/mcp-server/.venv/bin/python",
      "args": ["/absolute/path/to/mcp-server/server.py"],
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

Use absolute paths — the client's working directory is not guaranteed to be the repo root.

**Option B — HTTP/SSE (if running the server separately)**

Start `python server.py` (port 8002), then:

```json
{
  "mcpServers": {
    "slack-search": { "url": "http://localhost:8002/sse" }
  }
}
```

#### Claude Code

`.mcp.json` at the repo root is exactly Claude Code's project-scope format, so opening
this repo is enough. Project-scoped servers need a one-time approval:

```bash
claude mcp list          # shows: slack-search - ⏸ Pending approval
claude                   # approve when prompted
claude mcp list          # shows: ✔ Connected
```

This works the same in the terminal and in the VS Code / JetBrains extensions — they read
the same config. Claude Desktop is separate: it reads
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) and does not
pick up project `.mcp.json`, so paste the same `mcpServers` block there instead.

**Grounding without a custom system prompt.** The web client injects a grounding system
prompt (`web-client/src/app/api/chat/route.ts`), but an IDE host has no such hook. That is
what `FastMCP(instructions=...)` in `server.py` is for — the MCP client reads it before any
tool call, so the cite-every-claim and say-nothing-if-empty rules travel with the server
rather than with one particular front end. Verified present in the `initialize` response.

With an IDE host wired up, `web-client/` is optional — it is a standalone demo UI, not a
dependency of the MCP server.

## Tests

The MCP server has a pytest suite that runs entirely offline — no Slack token or OpenAI credentials required. All external dependencies are stubbed.

### Running the tests

```bash
cd mcp-server
OPENAI_API_KEY=fake-key .venv/bin/python -m pytest tests/ -v
```

**122 tests, ~0.7s.**

### What is tested

**`tests/test_chunking.py`** — the token-based chunking pipeline

| Class | What it covers |
|---|---|
| `TestChunkMessage` | Filtering (system subtypes, empty text, emoji/URL-only), single-chunk short messages, multi-chunk long messages, token limit enforcement, no text loss across chunks, overlap between adjacent chunks, unicode, code blocks, markdown |
| `TestDeduplication` | Best-scoring chunk per message is kept; two messages with different `ts` are never collapsed; `full_text` is always the original, never a fragment |
| `TestSplitTextNaturally` | Short text is not split; paragraph breaks are respected; all parts stay within `CHUNK_SIZE_TOKENS` |

**`tests/test_search.py`** — `search_slack` and `get_thread` tool behaviour

| Class | What it covers |
|---|---|
| `TestEmptyResults` | Every code path that must return `[]`: empty index, channel filter matching nothing, all messages filtered as system subtypes, all messages empty text, unknown `thread_ts`, `get_thread` on an empty index, `get_reactions` with no Slack client |
| `TestPermalinkGuarantee` | Every `search_slack` result has a non-empty `permalink`; `_permalink()` builds the URL locally with **zero** `chat.getPermalink` calls, and falls back to the API when the workspace URL is unavailable; every `get_thread` reply has a permalink; output schema complete; thread order is chronological |
| `TestCosineSimilarityRanking` | Most-similar message ranks first; least-similar ranks last; `score` is monotonically non-increasing; `limit` respected regardless of corpus size; `channels` filter doesn't leak other channels; `_cosine_similarity` math (identical→1.0, opposite→−1.0, orthogonal→0.0, 53°→0.6) |
| `TestRankChunksRouting` | `search_slack()` delegates to `_rank_chunks()` with one stored vector per chunk — verified via spy; a query makes exactly one embed call, for the query text alone |

**`tests/test_bm25.py`** — BM25 tokeniser, lexical index, RRF fusion, mode switching

| Class | What it covers |
|---|---|
| `TestTokenizeForBm25` | Lowercase, preserves `ERR_500` / `pd-1234` / `us-east-1`, drops `<@mentions>`, keeps `<url\|label>` labels, drops bare URLs, stopwords retained, no stemming, empty/edge inputs |
| `TestLexicalIndex` | Empty corpus, empty query, exact keyword ranks first, returns `(message_id, score)` pairs, deduplicates multi-chunk messages, k-limit respected, scores descending |
| `TestRrfFuse` | Single-list score math (`1/(k+rank)`), double-list beats single, additive scores verified, custom k, empty input, deterministic tie-break by message_id, output sorted descending |
| `TestModeSwitch` | Dense prefers semantic match; BM25 prefers exact token; hybrid returns both; all modes return complete MCP schema keys; unknown mode falls through to dense |

**`tests/test_index.py`** — the persistent chunk + embedding store

| Class | What it covers |
|---|---|
| `TestRoundTrip` | Chunks and vectors survive a write/read cycle; empty store; survives reopen; multi-chunk messages store every chunk with `full_text` intact |
| `TestIncremental` | Re-indexing an unchanged corpus makes **zero** embedding calls; adding one message embeds exactly that message; re-indexing replaces rather than duplicates |
| `TestEditsAndDeletes` | Edited text is re-embedded and replaces the old row; delete removes every chunk of a message; `message_ids()` is scoped by channel *and* by time, so a windowed `--reconcile` never deletes the history it didn't fetch |
| `TestModelNamespacing` | Vectors from different models never mix; unknown model loads nothing; the same text under a second model is embedded again |
| `TestCursorsAndThreads` | Cursor round-trip, survives reopen, per-channel; threads reconstruct parent + replies chronologically; channel metadata |
| `TestLookbackWindow` | No cursor → full history; window starts exactly `hours` before the cursor; never goes negative; `0` reproduces cursor-only behaviour; a 48h window reaches back past a 24h-old thread parent |

**`tests/test_eval.py`** — evaluation harness logic

| Class | What it covers |
|---|---|
| `TestRecallAtK` | Perfect, zero, partial recall; empty relevant; k-cutoff |
| `TestMRRAtK` | Rank 1/2/3 hits; no hit; empty relevant; k-cutoff |
| `TestAggregate` | Average of list; empty list returns 0 |
| `TestEvalLoop` | Full eval loop on 5-message fixture corpus with stubbed embeddings; results file written; per-type breakdown present; empty `relevant_ids` scores zero |

### How the tests are isolated

`server.py` is imported with all heavy dependencies stubbed before import:
- `openai` → real module but pointed at `OPENAI_API_KEY=fake-key`; `_embed()` is patched per-test with deterministic hand-crafted vectors for ranking tests
- `fastmcp` → no-op `FastMCP` class
- `slack_sdk` → bare `WebClient` stub; the query path never calls Slack, so tool tests seed a temp index instead of mocking API responses
- `dotenv` → no-op `load_dotenv` (accepts `**kwargs` so `load_dotenv(dotenv_path=...)` calls don't fail)

No `.env` file, no Slack token, and no index file are needed to run the tests — each test builds its own in a temp directory and deletes it afterwards.

---

## Demo queries to test

- *"What did we decide about the auth approach?"*
- *"Any unresolved action items from #backend this sprint?"*
- *"What's the latest on the deployment pipeline?"*

## How grounding is enforced

1. **Retrieval first:** `search_slack` ranks real indexed Slack messages against the query and returns them with their permalinks.
2. **Grounded system prompt:** the model is explicitly instructed to answer only from the tool's returned context — not from training data.
3. **Citation required:** Every claim must include a `[source](permalink)` link. If no relevant messages are found, the response is: *"I couldn't find anything in Slack about that."*

The constraint is structural — the model has no other information channel to draw from. The residual risk is that it may misattribute or over-generalize *within* the prose that connects the cited sources, which is why every claim ships with a permalink the reader can verify.

---

## Roadmap / future work

These are the things standing between "scoped pilot" and "tool a workspace can safely adopt at scale." All are deliberately deferred, not overlooked.

### 1. Query-time permission filtering (required before indexing any private channel)

**Current behaviour:** access is enforced at *ingestion*. The bot can only read channels it's been invited to, and the pilot is scoped to public channels only. Within that scope this is safe — public content is visible to anyone in the workspace regardless of who's asking.

**The gap:** `search_slack` returns top-K results to whoever queries, without checking whether *that specific user* is a member of the channel a result came from. For public channels this doesn't matter. The moment the index contains a private or restricted channel, a user could receive a message from a channel they were never in — surfaced by the tool, stripped of the context that would have signalled it was sensitive.

**What's required before crossing that line:** filter results by the asking user's Slack membership *at query time* — resolve the caller's user ID → their channel memberships → drop any result from a channel they're not in. This is a distinct engineering phase, not a config change.

> **Bright line:** public-channels-only is not a soft default to relax when someone asks nicely. It is the boundary the current design is safe up to. Indexing anything non-public is blocked on the filtering above.

### 2. ~~Persistent embedding store~~ — done, see `index.py`

Shipped. Model-namespaced vectors, `text_hash` edit detection, per-channel cursors, a
lookback window for thread replies and edits, and `--reconcile` for deletions. What's left
on this thread:

- **Relevance threshold.** `search_slack` still returns its top-K regardless of match
  quality. A single constant across modes is incoherent — cosine has an absolute scale,
  BM25 scores are corpus-relative, RRF scores carry no relevance signal at all — so this
  needs one gate per mode, calibrated against a filled-in golden set.
- **BM25-first hybrid.** Hybrid currently embeds the whole corpus before narrowing to
  candidates, so the candidate limit saves nothing at index time. Running BM25 first
  would make the dense leg O(candidates). Worth measuring before it becomes the default.
- **`float16` vectors.** Halves resident memory (~800 MB → ~400 MB at 100k messages) for
  negligible recall loss, if that ever matters.
- **`index.py --forget`.** A targeted delete path for erasure requests. Today that means
  hand-writing SQL against the index — see [Data at rest](#data-at-rest).

### 3. Consent mechanism for channel inclusion (process, not just code)

"Public channels where people are okay with it" needs an actual signal, not an assumption. Before a channel is indexed, get an explicit yes from its active participants, and provide a clear way to opt a channel back out. Cheap to do at pilot scale; necessary at team or company scale.

### 4. Per-team deployment (scale out to multiple teams)

**Current behaviour:** a single MCP server instance is configured with one set of channel IDs and one bot token, serving whoever opens the client.

**The goal:** each team gets their own scoped Slack bot and their own MCP server instance, so a given team's developers only ever search their own channels — with no cross-team data leakage possible at the API layer.

**How it works:**

| Layer | Per-team |
|---|---|
| Slack App | Each team registers their own bot app on `api.slack.com`, invited only to their channels |
| `SLACK_BOT_TOKEN` | Each bot app has its own token — physically cannot read other teams' channels |
| `SLACK_CHANNEL_IDS` | Scoped to that team's channel list |
| MCP server instance | One deployed process per team (separate container / cloud function, different URL) |
| `.mcp.json` | Each team's client config points to their own server URL |

**Distribution:** once each team's MCP server is deployed as a hosted service, teams receive a one-line `.mcp.json` snippet pointing at their server URL, which can be shipped through whatever config-management path the org already uses — no per-developer setup required.

**Future consolidation:** once query-time permission filtering (item 1 above) is implemented, multiple teams could share a single multi-tenant MCP server that filters results by the requesting user's Slack channel memberships at query time, eliminating the need for separate deployments.

---

## Deploying to a workspace you don't own

Most organisations gate which apps may be installed in their Slack workspace, and this app reads message content — so expect a review. The details differ per org; the substance is the same everywhere.

### OAuth scopes

The bot uses only bot token scopes:

| Scope | Purpose |
|---|---|
| `channels:history` | Read messages from public channels the bot has been invited to |
| `channels:read` | List channels and their metadata |
| `reactions:read` | Read emoji reactions (only needed for `get_reactions`) |

No user tokens, no admin tokens, and no impersonation or write scopes (`chat:write`, `chat:write.customize`) are used anywhere in this project. Drop `reactions:read` and the `get_reactions` tool if you don't need reaction sentiment — a shorter scope list is an easier approval.

### What a reviewer will ask

| Question | Answer for this project |
|---|---|
| What data does it read? | Message text, author IDs, timestamps, thread IDs, and permalinks from public channels the bot is explicitly invited to |
| Where does that data go? | Message text is sent to the embeddings provider **at index time** — once per message, and again only if it is edited. Retrieved messages are sent to the LLM host as tool output on the queries that surface them. |
| **Is it stored?** | **Yes.** A SQLite file (`INDEX_PATH`, default `mcp-server/slack_index.db`) holds the full text of every indexed message, its metadata, and its embedding vector — see [Data at rest](#data-at-rest) below. |
| How long is it kept? | Indefinitely, until the file is deleted or `--reconcile` observes the message gone from Slack. |
| Does it process personal information? | Assume yes, **and it is retained at rest**. Slack messages routinely contain names, email references, and project details. |
| Does it write to Slack? | No. Read-only. |

### Data at rest

Earlier versions rebuilt the index in memory per query and kept nothing. That was a real
security property, and moving to a persistent index gave it up in exchange for cost and
latency. Be straight with your reviewer about the trade:

**What the index file contains.** One row per chunk: `full_text` (the complete original
message), `chunk_text`, `author`, `channel`, `ts`, `thread_ts`, `permalink`, and a
float32 embedding vector. It is not a hash or a digest — the message text is recoverable
in plaintext by anyone who can read the file. Treat it with the same care as a Slack
export.

**Protecting it.** The file has no access control of its own; it inherits the
filesystem's. Put it on an encrypted volume, restrict it to the service account that runs
the indexer and server, and keep it out of backups that travel somewhere less protected.
It is gitignored by default — do not commit it.

**Deletion lag.** A message deleted in Slack stays searchable until a `--reconcile` run
observes it missing. With the schedule above that is up to a day for recent messages and
up to a week for older ones. If someone deletes a message because it should never have
been posted, that lag is your exposure window — run `index.py --reconcile` immediately
rather than waiting for cron, or delete the index file and rebuild.

**Right to erasure.** There is no per-user delete path. Removing one person's messages
means deleting the index and re-indexing without their channels, or writing a targeted
`DELETE FROM chunks WHERE author = ?` against the SQLite file.

Scope the bot to the minimum set of channels necessary and document both that scoping and
the retention answers above wherever your org records data-processing decisions.
