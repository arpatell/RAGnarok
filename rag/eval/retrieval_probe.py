"""Diagnostic comparison of retrieval channels on the fixed human suite."""
from __future__ import annotations

import concurrent.futures
import json
import sqlite3
from pathlib import Path

from rag.eval.catalog_eval import human_cases
from rag.rag_api import HybridRagService, load_config, _query_tokens


def _sparse_retrieval_text(query):
    filler = {"i", "im", "m", "trying", "remember", "looking", "please", "help", "think", "forgot", "forget", "name", "title", "series", "story", "show", "anime", "manga", "manhwa", "manhua", "sound", "familiar", "something", "somehow", "know"}
    return " ".join(t for t in _query_tokens(query) if t not in filler) or query


def main() -> None:
    with sqlite3.connect("rag/ingestion/catalog.sqlite") as db:
        rows = [json.loads(body) for body, in db.execute("SELECT row_json FROM documents")]
    cases, missing = human_cases(rows, Path("rag/eval/human_search_cases.json"))
    if missing:
        raise RuntimeError("Unresolved test targets")
    service = HybridRagService(load_config())
    output = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for case in cases:
            dense = service._embeddings.embed_query(case["query"])
            sparse = service._bm25.encode_queries(case["query"])
            def retrieve(channel):
                alpha = 0.35 if channel == "cleaned" else channel
                lexical = service._bm25.encode_queries(_sparse_retrieval_text(case["query"])) if channel == "cleaned" else sparse
                d, s = service._scale_hybrid(dense, lexical, alpha)
                found = []
                for namespace in service.config.pinecone_namespaces:
                    found.extend(service._run_namespace_query(namespace=namespace, dense_query=d, sparse_query=s, top_k=40, media_type=case["medium"]))
                return sorted(service._dedupe_candidates(found), key=lambda r: r["score"], reverse=True)
            channels = dict(zip((0.35, "cleaned", 1.0), pool.map(retrieve, (0.35, "cleaned", 1.0))))
            target = case["target_mal_id"]
            ranks = {str(alpha): next((i + 1 for i, r in enumerate(results) if r["mal_id"] == target), None) for alpha, results in channels.items()}
            output.append({**case, "ranks": ranks, "dense_top": [{"title": r["title"], "mal_id": r["mal_id"], "score": r["score"]} for r in channels[1.0][:5]],
                           "channels": {str(a): [{"mal_id": r["mal_id"], "score": r["score"]} for r in rr] for a, rr in channels.items()}})
    path = Path("rag/eval/retrieval_probe_results.json")
    path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    for alpha in ("0.35", "cleaned", "1.0"):
        print(alpha, "hit1", sum(r["ranks"][alpha] == 1 for r in output), "hit10", sum((r["ranks"][alpha] or 1000) <= 10 for r in output), "cases", len(output))


if __name__ == "__main__":
    main()
