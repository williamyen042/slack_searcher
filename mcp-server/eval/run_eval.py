#!/usr/bin/env python3
"""
eval/run_eval.py — Retrieval quality evaluation harness for slack-search-mcp.

Usage
-----
    cd mcp-server
    # Baseline dense-only run (requires OPENAI_API_KEY + Slack access):
    OPENAI_API_KEY=sk-... SLACK_EXPORT_PATH=/path/to/export \\
        .venv/bin/python eval/run_eval.py --mode dense

    # Compare all modes:
    .venv/bin/python eval/run_eval.py --mode dense
    .venv/bin/python eval/run_eval.py --mode bm25
    .venv/bin/python eval/run_eval.py --mode hybrid

Output
------
Writes eval/results/<ISO-timestamp>_<mode>.json with:
  - per-query results
  - overall Recall@5, Recall@10, MRR@10
  - per-query_type breakdown of the same metrics

Retrieval
---------
Builds an in-memory index from the corpus, then calls _rank_chunks() — the same
function search_slack() delegates to, reading the same Store shape.  Evaluation
therefore measures the code that actually serves queries, rather than a parallel
path that can drift from it.

By default the corpus is indexed into ":memory:" (embedded once per run, reused
across all golden queries).  Pass --index to evaluate an existing on-disk index
instead, which costs no embedding at all.
Real OpenAI embeddings are used — no stubs.

If relevant_ids in golden_queries.jsonl are empty ([]), the query still runs and
its corpus rank is recorded, but it contributes 0 to recall/MRR scores.  This
allows the harness to be used for corpus exploration before judgements are filled in.
"""

import argparse
import json
import os
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — add mcp-server root to path and set required env vars
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent          # mcp-server/eval/
_SRV_ROOT = _HERE.parent                             # mcp-server/
sys.path.insert(0, str(_SRV_ROOT))

# Ensure env is loaded before importing server (server reads env at module level)
from dotenv import load_dotenv
load_dotenv(dotenv_path=_SRV_ROOT / ".env")

# server.py requires OPENAI_API_KEY at import time
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "ERROR: OPENAI_API_KEY is not set. "
        "Export it or add it to mcp-server/.env before running the eval."
    )

import index   # noqa: E402  (path patched above)
import server  # noqa: E402  (path patched above)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@k: fraction of relevant docs found in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def _mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """MRR@k: reciprocal rank of the first relevant doc in the top-k results."""
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _aggregate(scores: list[float]) -> float:
    """Mean of a list of floats; 0.0 for empty list."""
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Golden query loader (skips comment lines starting with #)
# ---------------------------------------------------------------------------

def _load_golden(path: Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Corpus loader (mirrors what search_slack() does for fetch + permalink)
# ---------------------------------------------------------------------------

def _load_corpus() -> list[dict]:
    """
    Load raw messages from export or live API.  Permalink resolution is included
    so message_id (channel:ts) matches what the golden set records.
    """
    if server.USE_EXPORT:
        # Export mode: permalinks are already embedded by _load_export_messages()
        return server._load_export_messages()

    # Live API mode: fetch all configured channels, resolve permalinks
    if server.CHANNEL_IDS:
        target_channels = server.CHANNEL_IDS
    else:
        target_channels = [ch["id"] for ch in server._discover_channels_api()]

    if not target_channels:
        sys.exit("ERROR: no channels configured (SLACK_CHANNEL_IDS) and channel discovery returned nothing.")

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = ex.map(server._fetch_channel_api, target_channels)
    raw = []
    for msgs in results:
        raw.extend(msgs)

    for msg in raw:
        if not msg.get("permalink"):
            msg["permalink"] = server._permalink(msg["channel"], msg["ts"])

    return raw


def _build_index(index_path: str | None) -> tuple[list, "np.ndarray | None", int]:
    """
    Return (chunks, vecs, corpus_size) for the eval run.

    With --index, read an existing on-disk index (no embedding cost).  Without
    it, fetch the corpus and index it into an in-memory Store: embedded once,
    then reused across every golden query.
    """
    model = server._OPENAI_EMBED_MODEL

    if index_path:
        store = index.Store(index_path)
        chunks, vecs = store.load(model)
        corpus = len({c.message_id for c in chunks})
        store.close()
        print(f"[eval] index: {index_path} — {len(chunks)} chunks / {corpus} messages")
        return chunks, vecs, corpus

    corpus_msgs = _load_corpus()
    print(f"[eval] corpus: {len(corpus_msgs)} raw messages — embedding once into :memory:")
    store = index.Store(":memory:")
    index.index_messages(store, corpus_msgs, model, server._embed, server._chunk_message)
    chunks, vecs = store.load(model)
    store.close()
    print(f"[eval] indexed: {len(chunks)} chunks")
    return chunks, vecs, len(corpus_msgs)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(mode: str, golden_path: Path, results_dir: Path,
             index_path: str | None = None) -> dict:
    print(f"[eval] mode={mode}  loading corpus …")
    chunks, vecs, corpus_size = _build_index(index_path)

    golden = _load_golden(golden_path)
    print(f"[eval] golden set: {len(golden)} queries")

    per_query_results = []

    # Metrics accumulators keyed by query_type + "overall"
    r5_by_type:  dict[str, list[float]] = {"overall": [], "keyword": [], "semantic": [], "mixed": []}
    r10_by_type: dict[str, list[float]] = {"overall": [], "keyword": [], "semantic": [], "mixed": []}
    mrr_by_type: dict[str, list[float]] = {"overall": [], "keyword": [], "semantic": [], "mixed": []}

    for entry in golden:
        query       = entry["query"]
        relevant    = set(entry.get("relevant_ids", []))
        query_type  = entry.get("query_type", "unknown")

        # Call the same function search_slack() uses
        results = server._rank_chunks(chunks, vecs, query, k=10)
        retrieved_ids = [f"{r['channel']}:{r['ts']}" for r in results]

        r5  = _recall_at_k(retrieved_ids, list(relevant), 5)
        r10 = _recall_at_k(retrieved_ids, list(relevant), 10)
        mrr = _mrr_at_k(retrieved_ids, list(relevant), 10)

        per_query_results.append({
            "query":        query,
            "query_type":   query_type,
            "relevant_ids": list(relevant),
            "retrieved_ids": retrieved_ids,
            "recall_at_5":  round(r5,  4),
            "recall_at_10": round(r10, 4),
            "mrr_at_10":    round(mrr, 4),
        })

        for bucket in ("overall", query_type):
            if bucket in r5_by_type:
                r5_by_type[bucket].append(r5)
                r10_by_type[bucket].append(r10)
                mrr_by_type[bucket].append(mrr)

        print(
            f"  {query[:55]:<55}  R@5={r5:.2f}  R@10={r10:.2f}  MRR@10={mrr:.2f}"
        )

    # Build aggregate table
    aggregates: dict[str, dict] = {}
    for bucket in r5_by_type:
        n = len(r5_by_type[bucket])
        if n == 0:
            continue
        aggregates[bucket] = {
            "n":           n,
            "recall_at_5":  round(_aggregate(r5_by_type[bucket]),  4),
            "recall_at_10": round(_aggregate(r10_by_type[bucket]), 4),
            "mrr_at_10":    round(_aggregate(mrr_by_type[bucket]), 4),
        }

    output = {
        "mode":       mode,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "corpus_size": corpus_size,
        "golden_count": len(golden),
        "aggregates": aggregates,
        "per_query":  per_query_results,
    }

    # Write results file
    results_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"{ts_tag}_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[eval] results written → {out_path}")
    print("\n=== Aggregate metrics ===")
    for bucket, agg in aggregates.items():
        print(
            f"  {bucket:<10}  n={agg['n']:>3}  "
            f"R@5={agg['recall_at_5']:.4f}  "
            f"R@10={agg['recall_at_10']:.4f}  "
            f"MRR@10={agg['mrr_at_10']:.4f}"
        )

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval eval harness for slack-search-mcp")
    p.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval mode to evaluate (default: dense)",
    )
    p.add_argument(
        "--golden",
        type=Path,
        default=_HERE / "golden_queries.jsonl",
        help="Path to golden queries JSONL file",
    )
    p.add_argument(
        "--index",
        default=None,
        help="Evaluate an existing on-disk index instead of embedding the corpus",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_HERE / "results",
        help="Directory where result JSON files are written",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # server reads RETRIEVAL_MODE from env at import time, and it is already
    # imported by now — so set the module attribute directly.  Writing the env
    # var here would silently evaluate dense under a bm25/hybrid filename.
    server.RETRIEVAL_MODE = args.mode
    run_eval(args.mode, args.golden, args.results_dir, args.index)
