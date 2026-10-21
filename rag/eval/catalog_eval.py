"""Catalog-wide coverage and held-out human-query evaluations; no load generation.

Generated source-grounded clues are reported separately from hand-written prompts.
Results include exact medium+MAL identity, per-category hits@1/@10 and title coverage.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import gzip
import json
import os
import random
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rag.catalog_policy import excluded_genres, extract_section

_local = threading.local()


def normalized_title(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), re.UNICODE))


def human_cases(rows: list[dict], path: Path) -> tuple[list[dict], list[dict]]:
    lookup = defaultdict(list)
    identities = {(row["type"], row["mal_id"]): row for row in rows}
    for row in rows:
        for alias in aliases(row):
            if alias:
                lookup[(row["type"], normalized_title(alias))].append(row)
    cases, missing = [], []
    for position, case in enumerate(json.loads(path.read_text(encoding="utf-8"))["cases"]):
        explicit_id = case.get("target_mal_id")
        if explicit_id is not None:
            target = identities.get((case["medium"], explicit_id))
            found = [target] if target else []
        else:
            found = lookup.get((case["medium"], normalized_title(case["target"])), [])
        if not found:
            missing.append(case)
            continue
        # MAL's manga category also contains novels and one-shots. Prefer the
        # serialized comic unless the fixture explicitly identifies an edition.
        target = min(found, key=lambda r: (r.get("display_type") not in {"manga", "manhwa", "manhua", "tv", "movie"}, r["mal_id"]))
        cases.append({**case, "case_id": f"human-{position + 1:04d}", "origin": "human", "target_mal_id": target["mal_id"], "target_title": target["title"]})
    return cases, missing


def aliases(row: dict) -> list[str]:
    value = row.get("alternative_titles", {})
    return list(dict.fromkeys([row["title"], value.get("en", ""), value.get("ja", ""), *value.get("synonyms", []), *row.get("romanized_titles", [])]))


def generated_cases(rows: list[dict], coverage: float, kinds: set[str]) -> list[dict]:
    selected = sorted(rows, key=lambda r: hashlib.sha256(f"{r['type']}:{r['mal_id']}".encode()).hexdigest())[:int(len(rows) * coverage + 0.999)]
    cases = []
    for row in selected:
        base = {"medium": row["type"], "target_mal_id": row["mal_id"], "target_title": row["title"], "origin": "generated"}
        if "title" in kinds:
            cases.append({**base, "kind": "title", "query": row["title"]})
        if "alias" in kinds:
            alt = next((a for a in aliases(row)[1:] if a and a.casefold() != row["title"].casefold()), None)
            if alt:
                cases.append({**base, "kind": "alias", "query": alt})
        if "romanized_alias" in kinds:
            for alt in row.get("romanized_titles", []):
                if alt and normalized_title(alt) != normalized_title(row["title"]):
                    cases.append({**base, "kind": "romanized_alias", "query": alt})
        if "semantic" in kinds:
            synopsis = row.get("synopsis") or extract_section(row["text"], "Synopsis")
            sentences = re.split(r"(?<=[.!?])\s+", synopsis)
            clue = next((s for s in sentences if len(s.split()) >= 8 and row["title"].casefold() not in s.casefold() and not s.startswith("[")), "")
            if clue:
                clue = " ".join(clue.split()[:45])
                lead = ["I'm trying to remember a story. ", "Does this sound familiar? ", "I can't remember the name, but "][row["mal_id"] % 3]
                cases.append({**base, "kind": "semantic", "query": lead + clue})
        if "character" in kinds:
            names = extract_section(row["text"], "Characters").split(";")
            name = re.sub(r"\([^)]*\)", "", names[0]).strip() if names else ""
            if "," in name:
                family, given = name.split(",", 1)
                name = f"{given.strip()} {family.strip()}"
            if row.get("characters_enriched", False) and name and name != "Unknown":
                cases.append({**base, "kind": "character", "query": f"What's the series with a character named {name}?"})
        if "details" in kinds:
            detail = row.get("search_details", {})
            maker = detail.get("Authors") or detail.get("Studios")
            genres = [g for g in row.get("genres", []) if g.lower() not in {"shounen", "seinen", "shoujo", "josei"}]
            year = str(detail.get("Started") or "")[:4]
            if maker and year and genres:
                cases.append({**base, "kind": "details", "query": f"Looking for a {genres[0].lower()} {row['type']} by {maker}, started around {year}."})
    return cases


def evaluate(case: dict, endpoint: str, service=None) -> dict:
    started = time.perf_counter()
    try:
        if service is not None:
            from rag.rag_api import RagSearchRequest
            payload = RagSearchRequest(query=case["query"], media_type=case["medium"])
            body = service.search(payload.query, payload.media_type).model_dump()
            results = body["top_results"]
        else:
            if not hasattr(_local, "client"):
                _local.client = httpx.Client(timeout=30)
            response = _local.client.post(endpoint, json={"query": case["query"], "media_type": case["medium"]})
            response.raise_for_status()
            body = response.json()
            results = body.get("top_results", [])
        target = (case["medium"], case["target_mal_id"])
        matches = [(r.get("media_type") or extract_section(r.get("snippet", ""), "Type").lower(), r.get("mal_id")) for r in results]
        rank = matches.index(target) + 1 if target in matches else None
        leaked = any(excluded_genres(r) or m[0] != case["medium"] for r, m in zip(results, matches))
        return {**case, "rank": rank, "hit1": rank == 1 and not leaked, "hit10": rank is not None and rank <= 10 and not leaked,
                "policy_leak": leaked, "notice": body.get("notice"), "ms": round((time.perf_counter() - started) * 1000),
                "top": [{"title": r["title"], "mal_id": r.get("mal_id"), "media_type": m[0]} for r, m in zip(results[:3], matches[:3])], "error": None}
    except Exception as exc:
        return {**case, "rank": None, "hit1": False, "hit10": False, "error": str(exc), "ms": round((time.perf_counter() - started) * 1000)}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="rag/ingestion/catalog.sqlite")
    parser.add_argument("--corpus", type=Path, default=Path("rag/data/catalog.jsonl.gz"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091/rag/search")
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--cases", type=Path, help="Replay a frozen case set without regenerating prompts after enrichment")
    parser.add_argument("--human-cases", type=Path, default=Path("rag/eval/human_search_cases.json"))
    parser.add_argument("--in-process", action="store_true", help="Use the production search service directly, avoiding HTTP overhead for large correctness runs")
    parser.add_argument("--resume", action="store_true", help="Resume completed identical cases from this output's JSONL")
    parser.add_argument("--kinds", default="title,alias,romanized_alias,semantic,character,details")
    parser.add_argument("--split", choices=["all", "development", "held_out"], default="all")
    parser.add_argument("--generate-only", action="store_true", help="Write deterministic coverage cases without calling retrieval")
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--degraded-retries", type=int, default=0, help="Correctness-only retries after an explicitly reported model-service fallback")
    parser.add_argument("--degraded-retry-seconds", type=float, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-hit1", type=float, default=0.95)
    parser.add_argument("--output", default="rag/eval/catalog_results.json")
    parser.add_argument("--include-pending", action="store_true", help="Expose unindexed catalog entries as coverage failures")
    args = parser.parse_args()
    if args.degraded_retries < 0 or args.degraded_retry_seconds < 0:
        parser.error("Retry bounds must be nonnegative")
    if not 0 < args.coverage <= 1:
        parser.error("coverage must be in (0, 1]")
    if Path(args.database).exists() and (args.include_pending or not args.corpus.exists()):
        with sqlite3.connect(args.database) as db:
            pending = db.execute("SELECT COUNT(*) FROM documents WHERE upserted=0 AND COALESCE(json_extract(row_json, '$.vector_exists'),0)=0").fetchone()[0]
            all_rows = [json.loads(body) | {"characters_enriched": bool(enriched)} for body, enriched in db.execute("SELECT row_json,enriched FROM documents" + ("" if args.include_pending else " WHERE upserted=1 OR json_extract(row_json, '$.vector_exists')=1"))]
    else:
        with gzip.open(args.corpus, "rt", encoding="utf-8") as handle:
            all_rows = [json.loads(line) for line in handle if line.strip()]
        manifest = json.loads(args.corpus.with_name("catalog_manifest.json").read_text())
        pending = max(0, manifest["eligible_titles"] - len(all_rows))
    rows = [row for row in all_rows if not excluded_genres(row)]
    missing_targets = []
    if args.cases:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    elif args.human:
        cases, missing_targets = human_cases(rows, args.human_cases)
    else:
        cases = generated_cases(rows, args.coverage, set(args.kinds.split(",")))
    for case in cases:
        key = f"{case['medium']}:{case['target_mal_id']}"
        case["split"] = "held_out" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0 else "development"
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    if args.limit:
        random.Random(174).shuffle(cases)
        cases = cases[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.generate_only:
        output.write_text(json.dumps({"cases": cases, "eligible_titles": len(rows), "pending_index_titles": pending}, ensure_ascii=True), encoding="utf-8")
        print(json.dumps({"generated_cases": len(cases), "distinct_titles": len({(c['medium'],c['target_mal_id']) for c in cases}), "eligible_titles": len(rows), "missing_human_targets": missing_targets}))
        return
    def identity(case):
        return (case["medium"], case["target_mal_id"], case["kind"], case["query"])
    previous = {}
    fingerprint = hashlib.sha256()
    for path in (Path("rag/rag_api.py"), Path("rag/groq_reranker.py"), Path("rag/evidence_index.py"), Path("rag/catalog_policy.py"), Path("rag/data/evidence.sqlite.gz"), Path(__file__), Path("rag/requirements.txt"), args.corpus, args.human_cases, args.cases or args.human_cases, Path("rag/ingestion/bm25_values.json")):
        if path.exists():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    fingerprint.update(block)
    run_info = {"fingerprint": fingerprint.hexdigest(), "split": args.split, "human": args.human, "kinds": args.kinds, "coverage": args.coverage, "limit": args.limit,
                "degraded_retries": args.degraded_retries, "degraded_retry_seconds": args.degraded_retry_seconds}
    run_info["runtime"] = {name: os.getenv(name, "") for name in ("RAG_LLM_RERANK_ENABLED", "RAG_LLM_PROVIDER", "GROQ_MODEL", "CEREBRAS_MODEL", "PINECONE_INDEX", "PINECONE_NAMESPACE", "HYBRID_DENSE_ALPHA")}
    run_path = output.with_suffix(".run.json")
    if args.resume and output.with_suffix(".jsonl").exists():
        if not run_path.exists() or json.loads(run_path.read_text()) != run_info:
            parser.error("Refusing stale results after code, corpus, fixtures, or selection changed; start a new output run")
        for line in output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines():
            try:
                result = json.loads(line)
                if not result.get("error") and not result.get("notice"):
                    previous[identity(result)] = result
            except json.JSONDecodeError:
                continue
    run_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    results = [previous[identity(c)] for c in cases if identity(c) in previous]
    remaining = [c for c in cases if identity(c) not in previous]
    service = None
    if args.in_process:
        from rag.rag_api import HybridRagService, load_config
        service = HybridRagService(load_config())
    def run_case(case):
        result = evaluate(case, args.endpoint, service)
        degraded = int(bool(result.get("notice")))
        attempts = 1
        while result.get("notice") and attempts <= args.degraded_retries:
            time.sleep(args.degraded_retry_seconds)
            result = evaluate(case, args.endpoint, service)
            degraded += int(bool(result.get("notice")))
            attempts += 1
        return {**result, "evaluation_attempts": attempts, "degraded_attempts": degraded}
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as log, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in results:
            log.write(json.dumps(result, ensure_ascii=True) + "\n")
        for result in pool.map(run_case, remaining):
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=True) + "\n")
            log.flush()
            if len(results) % 1000 == 0:
                print(f"Evaluated {len(results)}/{len(cases)}", flush=True)
    groups = defaultdict(list)
    for result in results:
        groups[result["kind"]].append(result)
    summary = {"cases": len(results), "eligible_titles": len(rows), "pending_index_titles": pending,
               "tested_titles": len({(r['medium'], r['target_mal_id']) for r in results}),
               "hit1": sum(r["hit1"] for r in results) / max(1, len(results)),
               "hit10": sum(r["hit10"] for r in results) / max(1, len(results)),
               "errors": sum(bool(r.get("error")) for r in results),
               "degraded_responses": sum(bool(r.get("notice")) for r in results),
               "degraded_attempts": sum(r.get("degraded_attempts",0) for r in results),
               "policy_leaks": sum(bool(r.get("policy_leak")) for r in results),
               "evaluation_mode": "in_process" if args.in_process else "http",
               "split": args.split,
               "missing_human_targets": missing_targets,
               "groups": {k: {"cases": len(v), "hit1": sum(r["hit1"] for r in v) / len(v), "hit10": sum(r["hit10"] for r in v) / len(v), "tested_titles": len({(r['medium'],r['target_mal_id']) for r in v})} for k, v in groups.items()}}
    summary["title_coverage"] = summary["tested_titles"] / max(1, len(rows) + (0 if args.include_pending else pending))
    summary["macro_category_hit1"] = sum(group["hit1"] for group in summary["groups"].values()) / max(1, len(summary["groups"]))
    output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not results or missing_targets or summary["hit1"] < args.min_hit1 or summary["policy_leaks"] or summary["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
