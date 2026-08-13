# Changelog

Notable changes to Slack Searcher. Newest first.

## Unreleased

### Added

- **Persistent index (`mcp-server/index.py`).** Indexing and querying are now separate.
  `python index.py` fetches from Slack, chunks, embeds, and persists to SQLite; the
  server only reads that index. Run it on a timer (`*/15 * * * *`).
  - Incremental by default — a per-channel cursor means each run fetches only what
    was written since the last one.
  - `text_hash` per chunk, so an edited message re-embeds only the chunks whose text
    actually moved. Re-indexing an unchanged corpus costs **zero** embedding calls.
  - `model` is part of the primary key: vectors from different embedding models cannot
    be mixed, and switching models re-embeds rather than silently comparing incomparable
    vectors.
  - Each run re-walks `INDEX_LOOKBACK_HOURS` (default 48) before the cursor. A cursor
    only advances past *new* timestamps, so without this a reply added to an old thread —
    or an edit to an old message, since editing doesn't change `ts` — would never be
    picked up. Re-walked messages that haven't changed cost zero embeddings.
  - `--reconcile` drops messages deleted in Slack. It walks full history by default, or
    just a window when combined with `--lookback-hours` — deletion detection is bounded
    to whatever window was actually walked, so a windowed reconcile never mistakes
    unfetched history for deleted messages. `--stats` prints what is currently indexed.
- `tests/test_index.py` — 20 tests covering round-trip, incremental skip, edits,
  deletes, model namespacing, cursors, and thread reconstruction.
- `search_slack` accepts a `channels` filter that narrows to indexed channels.
- `INDEX_PATH` and `EMBED_BATCH` environment variables.

### Changed

- **`search_slack` no longer fetches from Slack or embeds the corpus.** It embeds the
  query — one call, ~10 tokens — and matmuls against the stored vectors. Query cost and
  latency are now independent of workspace size; previously both scaled linearly with
  the corpus, on every single query.
- `_rank_messages(raw_messages, ...)` → `_rank_chunks(chunks, vecs, ...)`. The eval
  harness builds an in-memory index and calls the same function, so evaluation still
  measures exactly the code that serves production queries.
- `get_thread` and `list_channels` are served from the index — no Slack call. The
  serving process now runs with **no Slack token at all**; only `get_reactions`
  (which reads state that changes after a message is written) still needs one, and
  returns `[]` with a warning when there is no client.
- `eval/run_eval.py` gained `--index` to evaluate an existing on-disk index at zero
  embedding cost.
- The BM25 index is built once per index load rather than once per query.
- `_fetch_channel_api` accepts `oldest` for incremental fetching.

### Fixed

- **Permalinks cost N sequential API calls per query.** `search_slack` issued one
  `chat.getPermalink` per message, serially — seconds to minutes of latency on any
  real corpus, plus rate limiting. Permalinks are deterministic, so they are now built
  locally from the workspace URL after a single cached `auth.test`, with the API call
  kept only as a fallback.
- **Chunk text was corrupted at every unit boundary.** The splitter discarded the
  delimiter it split on and the packer concatenated token IDs, fusing the last token of
  one unit to the first of the next (`"...in detail.Paragraph 1..."` →
  BM25 term `endmarker0paragraph`). A separator token is now re-inserted between packed
  units. Affected the indexed text only; `full_text` was always intact.
- **`MAX_MODEL_TOKENS` was declared but never enforced.** A 12k-character unbroken
  string (base64 blob, minified JSON) produced a single 1500-token chunk against a
  documented 512 cap. Text with no natural boundary left is now hard-sliced on token
  windows, and the emit path clamps at the cap.
- **`run_eval.py --mode` was silently ignored.** It set `RETRIEVAL_MODE` in the
  environment *after* importing `server`, which reads it at import time, so
  `--mode bm25` and `--mode hybrid` evaluated dense and wrote a mislabelled results
  file. It now sets the module attribute directly.
- Six `except SlackApiError: pass` handlers swallowed failures silently, including on
  the permalink path the citation guarantee depends on. All now log a warning.
- `tests/test_search.py` contained two near-identical copies of the same module
  concatenated; Python bound the second copy's classes over the first, so ~16 test
  methods never executed. Removed the stale copy (981 → 493 lines, same tests running).

### Removed

- All WatsonX / Granite code paths, environment variables, and documentation. The
  server has always run OpenAI embeddings; the README described a WatsonX-default
  architecture that did not exist in the code, including a "no real content goes
  through the OpenAI path" rule that `_embed()` violated on every query.
- Dead `ibm_watsonx_ai` module stubs from all four test files, and a
  `server._embedding_model` assignment referring to an attribute that no longer existed.

### Renamed

Project-wide de-branding, no behaviour change:

| Was | Now |
|---|---|
| `Bob × Slack` | `Slack Searcher` |
| `slack-bob-mcp` / `slack-bob-web` | `slack-search-mcp` / `slack-search-web` |
| `BOB_API_URL` / `BOB_API_KEY` | `LLM_API_URL` / `LLM_API_KEY` |
| `.bob/mcp.json` | `.mcp.json` |

The "Slack@IBM compliance" section (APM/IGMS/PIA/ITSS/SSRM) was replaced with a
vendor-neutral "Deploying to a workspace you don't own".

### Known gaps

- **No relevance threshold.** `search_slack` returns its top-K regardless of match
  quality, so a query about something never discussed still returns K weak results.
  The docs described a threshold gate as the structural anti-hallucination layer; it
  was never implemented. It needs one gate per retrieval mode — cosine has an absolute
  scale, BM25 scores are corpus-relative, and RRF scores carry no relevance signal at
  all — and calibration against a filled-in golden set.
- **`golden_queries.jsonl` has empty `relevant_ids`** and the committed baseline was
  recorded against an empty corpus, so every eval metric is currently 0.0.
- **`@mention` search does not work.** Slack stores mentions as `<@U0123ABC>` and the
  tokeniser drops them, so querying `@jsmith` only matches messages that typed the name
  as plain text. Needs a `users.list` ID→name map at ingestion.
- **Hybrid mode still embeds the whole corpus before narrowing candidates.** Running
  BM25 first would make the dense leg O(candidates) instead of O(corpus). Worth
  measuring against the eval harness before making it the default.
