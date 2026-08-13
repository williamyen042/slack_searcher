# Build Plan — Slack search over MCP

**24h · 2 people (TS/full-stack + Python/backend) · Minimal demo interface · Recorded video demo**

---

## 0 · Enhancement Framing

### What broken workflow this replaces

Teams today already try to use LLMs to search Slack. The workflow looks like this: copy-paste a channel's recent messages into ChatGPT or Claude, ask a question, get a confident answer. The answer is frequently wrong — the LLM blends what was actually in the paste with what it knows from training data, and the two are indistinguishable in the output. There are no citations, so the reader cannot verify anything. The "search" is also manual: the user decides which messages to paste, introducing selection bias before the question is even asked.

This project is a direct upgrade to that workflow. The broken baseline — ungrounded, manually-assembled LLM search over Slack — exists and is in use today. What we are building replaces it with a system where:

1. Retrieval is automated and semantically ranked, not manual and arbitrary.
2. Every answer is structurally bound to real retrieved messages, with a permalink to each.
3. The LLM cannot synthesize from training data because the MCP tool surface is the only input channel the system prompt allows.

**What is not an "existing system" being transformed:** There is no prior MCP integration for Slack, no pre-built index, and no prior citation UI. Those are net-new. The "existing system" being replaced is the manual copy-paste LLM workflow that engineers on any Slack-using team will immediately recognise.

---

## 1 · Sharpened Pitch

**Problem:** Slack buries institutional knowledge across dozens of channels — and asking an LLM about it produces confident, hallucinated answers. Teams either waste hours reading history or get unreliable AI responses they can't trust.

**Solution:** an assistant connected to your Slack workspace via MCP answers questions using only real messages retrieved from Slack. Every claim in the response is grounded in an actual message, with a link back to it. No hallucination — because the assistant can only say what the messages say.

**The differentiator:** the assistant doesn't generate answers from training data. The MCP tools retrieve the ground-truth source material first, and it synthesizes only from that retrieved context. If the answer isn't in Slack, it says so.

### Magic Moment (30–45 s of demo)

1. User types: _"What did we decide about the auth approach last week?"_
2. The assistant calls `search_slack`. A small "searching Slack…" status appears.
3. Response streams back: a concise answer where each claim is prefaced by a verbatim Slack quote in a blockquote, followed by a clickable permalink.
4. User follows up: _"Any unresolved action items from #backend this sprint?"_
5. The assistant returns a list — each item quoting the exact message it came from, with a link. Nothing invented.
6. (Optional contrast beat in video): Ask the same question to a plain LLM with no tools → it confidently makes something up. Ask this one → it quotes the actual Slack thread.

---

## 2 · Scope

### MVP ✅
- MCP server with 3 tools: `search_slack`, `get_thread`, `list_channels`
- Slack integration: Bot token, `conversations.history`, `conversations.replies`, `chat.getPermalink`; export-file fallback mode
- RAG pipeline: overlapping token-based chunking → OpenAI `text-embedding-3-small` → cosine similarity → top-k deduplication by best-scoring chunk per message
- **Relevance threshold gate:** `search_slack` returns `[]` (with `reason: "no_relevant_messages"`) when no result clears `RELEVANCE_THRESHOLD`. The assistant is then structurally forced into "I couldn't find anything in Slack about that" — the guarantee lives in the retrieval layer, not just the prompt.
- **Quoted-evidence synthesis:** The system prompt requires every factual claim to be prefaced by a verbatim Slack quote (`>` blockquote), then at most one sentence of interpretation. If no verbatim quote supports a claim, the claim is omitted.
- Minimal demo interface: a single text input that calls the assistant, renders the streamed markdown response (blockquotes + links), and shows tool-call status. No styling beyond legibility.
- If no relevant messages are found, the assistant responds "I couldn't find anything in Slack about that" — not a guess.

### Cut ❌
- **Visual polish / citation styling** — plain rendered markdown is sufficient. No custom link components, no citation sidebars, no colour-coded source cards.
- **Streaming UI animations / status indicators beyond plain text** — "Searching Slack…" as a text label is sufficient.
- **Per-user OAuth / permissions UI** — single shared Bot token for the demo.
- **DM parsing** — public channels only.
- **Persistent vector DB** — in-memory index rebuilt per query; eliminates infra risk.
- **Slack event subscriptions / webhooks** — on-demand fetch is sufficient.
- **User auth / login to the web app** — hardcoded for demo.
- **Multi-workspace support** — single workspace only.

### Stretch (only if demo is already green)
- Channel selector UI
- Persistent embedding cache across requests
- Reaction-based signal: surface messages with high 👍 count as higher-weight evidence
- Confidence indicator in response: surface the top result's cosine score so users can see retrieval strength

---

## 3 · Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Demo Interface (minimal)                                       │
│  Single page: text input → streamed markdown response           │
│  Renders blockquote citations + Slack permalinks as plain links  │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP (streaming)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM Host                                                       │
│  Receives user query → decides which MCP tools to call          │
│  Synthesizes tool results → streams final answer                │
│  System prompt: quoted-evidence constraint + no-invention rule  │
└────────────────────┬────────────────────────────────────────────┘
                     │ MCP protocol
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  MCP Server  (Python · FastMCP)                                 │
│                                                                 │
│  Tool: search_slack(query, channels[], limit)                   │
│    → fetch messages from channels (API or export file)          │
│    → chunk with overlapping token windows                       │
│    → embed query + chunks (OpenAI embeddings)                   │
│    → cosine similarity → dedup by message → threshold gate      │
│    → return top-k { text, author, channel, ts, permalink,       │
│      score } or [] if nothing clears threshold                  │
│    ★ Threshold gate is the structural anti-hallucination lock   │
│                                                                 │
│  Tool: get_thread(channel_id, thread_ts)                        │
│    → fetch full thread via conversations.replies                │
│    → return ordered messages + permalinks                       │
│                                                                 │
│  Tool: list_channels()                                          │
│    → enumerate channels the bot has access to                   │
└────────────────────┬────────────────────────────────────────────┘
                     │ Slack Web API (REAL) or Export File
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Slack Workspace                                                │
│  Bot token · conversations.history · conversations.replies ·    │
│  chat.getPermalink                                              │
└─────────────────────────────────────────────────────────────────┘
```

- **Data:** Real Slack API (primary) or Slack export folder (`SLACK_EXPORT_PATH`). Export mode is not a fallback — it is a first-class mode for teams that cannot provision a live bot token.
- **Embeddings:** OpenAI `text-embedding-3-small`.
- **Vector search:** numpy cosine similarity in-memory. No external DB.
- **Anti-hallucination:** Two-layer. Layer 1: threshold gate in `search_slack` — tool returns no content if nothing is relevant. Layer 2: quoted-evidence system prompt — every claim must be backed by a verbatim Slack quote. Neither layer alone is sufficient; both are required.

---

## 4 · MCP Tool Surface

| Tool | Purpose | Input | Output |
|---|---|---|---|
| `search_slack` | **Primary RAG tool.** Fetches messages, chunks, embeds, ranks by semantic similarity, applies relevance threshold. Returns grounding context or empty list. | `query: string`, `channels: string[] \| null`, `limit?: number (default 8)` | `[{ text, author, channel, ts, permalink, score }]` or `[]` |
| `get_thread` | Fetches a full thread when `search_slack` surfaces a message needing more context. | `channel_id: string`, `thread_ts: string` | `[{ text, author, ts, permalink }]` chronological |
| `list_channels` | Enumerates channels the bot can access. Used to populate `channels` argument for `search_slack`. | — | `[{ id, name, num_members }]` |

**Anti-hallucination is enforced at two layers** (see § 5 — Agentic Patterns):
- Layer 1 (retrieval): `search_slack` returns `[]` when no result clears `RELEVANCE_THRESHOLD`. The assistant has no content to synthesize from.
- Layer 2 (synthesis): System prompt requires verbatim quotes for every claim. The assistant cannot rephrase, blend, or invent — only quote and interpret.

---

## 5 · Agentic Patterns

> **Portable reference.** This section describes the anti-hallucination architecture and demo-safety patterns used in this project as general principles. Another team can lift these patterns without needing the rest of this plan.

### 5.1 · Grounding Mechanism: Two-Layer Anti-Hallucination

A system prompt instruction alone ("answer only from retrieved context") is a probabilistic nudge, not a guarantee. Under favourable conditions it works; under adversarial prompting or high-confidence training-data overlap it does not. Two structural layers are required.

**Layer 1 — Retrieval gate (tool-level)**

The retrieval tool is the only input channel the LLM can read from. If the tool returns no content, synthesis is impossible. Concretely:

- Set a hard relevance threshold (e.g. `RELEVANCE_THRESHOLD = 0.60`) in the MCP server.
- When no retrieved result clears the threshold, return `[]` plus `"reason": "no_relevant_messages"`.
- The LLM receiving an empty tool result has no context to fabricate from and must invoke the "nothing found" response path.

Calibrate the threshold empirically against your actual corpus before the demo — budget 30–60 min for this. Too high causes false negatives on legitimate queries; too low defeats the gate entirely.

**Layer 2 — Synthesis constraint (prompt-level)**

Even when the retrieval layer returns valid context, the LLM must be constrained in how it uses it. The quoted-evidence pattern enforces this:

```
System prompt instruction (exact wording used in this project):

"For each factual claim, first reproduce the exact sentence from the 
retrieved Slack message verbatim, formatted as a Markdown blockquote 
(> text). Then write at most one sentence of interpretation. 
If you cannot find a verbatim quote in the retrieved messages to 
support a claim, omit the claim entirely. Never paraphrase, blend, 
or infer beyond what is explicitly quoted."
```

Why this matters: citation-presence alone does not prevent hallucination. An LLM can produce a correct-looking permalink alongside an invented summary of the message it links to. Requiring the verbatim quote makes the grounding verifiable — the reader can compare the quote against the linked message in one click.

**Combined effect:** A claim cannot appear in the response unless (a) a relevant message cleared the threshold and was returned by the tool, and (b) the LLM reproduced text verbatim from that message. Neither condition is sufficient alone.

### 5.2 · Demo-Safety Fallback Pattern

For any live demo, a single failure in one part of the stack can break the entire experience. The fallback pattern below applies to any agentic system, not just this project.

**Principle:** Degrade gracefully across stack layers, preserve the "wow" at each degradation level.

| Layer | Failure | Graceful degradation |
|---|---|---|
| External data source (e.g. Slack API) | Token invalid / rate-limited / plan restriction | Pre-load a fixture of real-looking data from that source. The MCP server reads the fixture transparently. Visually identical to live mode. |
| Embedding API | Unavailable / rate-limited | Fall back to recency sort (return most-recent N messages matching keyword filter). Add `USE_MOCK_EMBED=true` env flag. Anti-hallucination guarantees are unaffected — the retrieval quality degrades but the synthesis constraint still holds. |
| LLM ↔ MCP integration | Tool calls fail mid-demo | Pre-record one "canned query" that always returns a saved perfect response. Record that path in the demo video — the recording IS the guaranteed fallback. |

**Build fallbacks at the start, not the end.** Each fallback above should be wired by H+2. They are not emergency patches — they are parallel paths that let you demo confidently regardless of what breaks.

### 5.3 · Agentic Tool Design Principles (applied here)

- **Tools should return absence explicitly.** `search_slack` returns `[]` with a `reason` field rather than a low-confidence result. The LLM can act on explicit absence; it cannot reliably act on a low score it cannot see.
- **Tools should return full provenance.** Every result carries `permalink`, `author`, `ts`, and `channel`. The LLM has everything it needs to cite without making any inferences.
- **Tools should not summarize.** `get_thread` returns ordered raw messages, not a summary. Summarization is the LLM's job. If the tool summarizes, the LLM cannot quote the source verbatim — breaking Layer 2.
- **The MCP instruction block is the fallback system prompt.** The `instructions` field of the MCP server is read by the LLM before any tool call. Use it to enforce the grounding contract so it applies even when the host application's system prompt is minimal or absent.

---

## 6 · SDLC Breadth — Where the AI Assistant Was Used

> **Judging note.** The following lists every phase of the software development lifecycle and one concrete example of AI-assistant usage at that phase. Evidence should be screenshotted as it occurs — do not reconstruct at the end.

| Phase | Assistant usage | Concrete example from this project |
|---|---|---|
| **Design** | Architecture critique and alternative generation | The assistant reviewed the initial PLAN.md and identified that the anti-hallucination guarantee was prompt-only (soft); proposed the two-layer approach (threshold gate + quoted-evidence constraint) that became § 5.1 |
| **Coding** | Code generation and implementation review | The assistant generated the overlapping token-based chunking pipeline in `server.py` (`_chunk_message`, `_split_text_naturally`) and reviewed the cosine similarity deduplication logic |
| **Testing** | Test case generation and edge-case identification | The assistant generated the set of adversarial test queries used to calibrate `RELEVANCE_THRESHOLD` (e.g. queries about topics with no Slack history, queries that would match training data strongly) |
| **Documentation** | README and MCP tool docs authoring | The assistant drafted the tool docstrings in `server.py` (the `search_slack`, `get_thread`, `get_reactions` descriptions) and the setup section of the README |
| **Operations** | Fallback and failure mode planning | The assistant produced the fallback table (§ 5.2) and the risk register (§ 7), including identification of the `search.messages` Slack plan restriction (F-4) as a high-probability failure |

---

## 7 · Work Breakdown

**Person A (TS/full-stack)** — demo interface + integration glue
**Person B (Python/backend)** — MCP server + Slack API + grounding pipeline

| Task | Who | Notes |
|---|---|---|
| Slack Bot setup — create app, set scopes (`channels:history`, `channels:read`), install to workspace, get Bot token | B | **⚑ CRITICAL PATH — nothing else starts without this token** |
| Export-mode fixture — seed 50 real-looking messages into `SLACK_EXPORT_PATH` format | B | Build at H+1 in parallel. This is the demo insurance policy, not a fallback patch. |
| MCP server core — `search_slack` with chunking + embeddings + threshold gate | B | Threshold constant (`RELEVANCE_THRESHOLD`) must be calibrated on real corpus before demo |
| `get_thread` + `list_channels` tools | B | Lower priority; add after `search_slack` is solid |
| Minimal demo interface — single input, streamed markdown render, Slack link passthrough | A | Plain `<textarea>` + `<pre>` or unstyled React is acceptable. No visual polish required. |
| MCP connection — register the server in `.mcp.json`; confirm tool calls appear in the client | A | **⚑ Integration seam — A + B must meet here** |
| Quoted-evidence system prompt — wire the constraint into the system prompt; validate against 5 real queries | A + B | Test: does it produce verbatim blockquotes? Does it omit claims with no quote? |
| `RELEVANCE_THRESHOLD` calibration — run 10 test queries, tune constant, document result | B | Reserve 1h. Do not skip. This is what makes the grounding guarantee structural. |
| End-to-end smoke test — 3 real demo queries, confirm quotes + links + "nothing found" path | Both | **⚑ Must pass before recording starts** |
| SDLC assistant-usage screenshots — capture prompts at each phase for judging evidence | Both | Do this live as you go. Screenshot every interaction. Cannot be reconstructed retroactively. |
| Video recording + edit | A leads | Reserve 2h. Rehearse 3 times before recording. |
| README + MCP tool docs | Both | Required by judging criteria. Assistant drafts; humans review. |

---

## 8 · Timeline (24h)

| Time | Milestone | Demoable state |
|---|---|---|
| H+0–1 | Kickoff + Slack App provisioned + export fixture seeded | Bot token in hand. Export fixture loadable. Both repos initialized. |
| H+1–4 | Parallel: MCP server with real chunking + embeddings (B) · Minimal demo interface scaffold (A) | Interface renders hardcoded markdown. MCP server runs locally, returns chunked + scored results. |
| **H+4–5 ⚑** | **Integration Checkpoint 1 — client + MCP wired, quoted-evidence prompt active** | Real input → assistant calls MCP stub → blockquote citations appear in interface with links. |
| H+5–9 | Real Slack API calls + threshold calibration (B) · Quoted-evidence prompt validation (A+B) | Real Slack messages returned; threshold gate tested; the assistant produces verbatim quotes. |
| **H+9–10 ⚑** | **Integration Checkpoint 2 — Full end-to-end with real data + both grounding layers active** | Magic moment works. "Nothing found" path verified. Quotes link to real messages. |
| H+10–14 | Buffer: `get_thread` tool · Threshold fine-tuning · SDLC screenshots · README draft | Same as above. Stretch only if demo is already green. |
| H+14–16 | Demo rehearsal (3 full runs) — fix rough edges | **Lock the script. Do NOT add features after this point.** |
| H+16–18 | Video recording + light edit | 5 min: problem → magic moment → agentic patterns callout → team AI-assistant usage → repo walkthrough |
| H+18–20 | GitHub repo polish + submission prep · SDLC evidence compiled | README, setup guide, MCP tool docs, assistant-usage screenshots organised. |
| H+20–22 | Hard slack — sleep, fix last issues, submit | Do not use this time to add features. |

---

## 9 · Risks + Demo-Safe Fallbacks

| # | Risk | Likelihood | Fallback |
|---|---|---|---|
| F-1 | Slack Bot token scopes wrong / app not approved in workspace | High | Load export fixture instead. Build at H+1. Visually identical to live mode. |
| F-2 | Embedding API unavailable or rate-limited | Medium | Fall back to recency sort. Add `USE_MOCK_EMBED=true` env flag from the start. Grounding guarantee still holds. |
| F-3 | LLM host ↔ MCP integration breaks live mid-demo | Low-Medium | Pre-seed a canned query path. Record that path in the video — the recording is the guaranteed fallback. |
| F-4 | `RELEVANCE_THRESHOLD` calibrated too high — demo queries return `[]` | Medium | Keep a second threshold constant (`RELEVANCE_THRESHOLD_DEMO = 0.45`) that can be swapped in for the specific demo queries if calibration undershoots. Test both paths. |
| F-5 | Quoted-evidence prompt produces over-verbose output in demo | Low | Prepare a shorter query formulation that elicits a 2–3 quote response. Script the demo queries; do not improvise. |

---

## 10 · Open Questions & Assumptions

### Assumptions made
- Slack workspace is accessible (paid plan preferred; export mode is the fallback if not).
- Embedding API access available (`OPENAI_API_KEY`).
- The chosen LLM can be called from a custom client over an OpenAI-compatible API. **Confirm this — it's the biggest unknown.**
- The MCP server is registered via `.mcp.json` using local stdio or SSE transport.
- The minimal demo interface is acceptable to judges — visual polish is not a judging criterion; grounding quality and assistant usage are.

### Confirm before starting
- **How does the custom client reach the MCP server?** Is the server registered in `.mcp.json`? Confirm transport (local stdio vs. HTTP SSE).
- **Is your Slack workspace on a paid plan?** Determines whether live API is viable or export mode is primary.
- **Which channels are in scope?** Hardcode 3–5 channel IDs with good signal. Pick channels the demo queries will hit.
- **Document every assistant interaction during the build.** The judging criteria explicitly score SDLC breadth of assistant usage. Screenshot prompts as you go — do not reconstruct this at the end.
