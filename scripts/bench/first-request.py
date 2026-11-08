"""Cold-process resource check with real local inference and simulated providers.

Not an accuracy or live-provider latency benchmark. Uses the existing Windows
Job budget helper and runs no real Pinecone/Groq requests or ingestion jobs.
"""

import argparse
import gzip
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx


ROOT = Path(__file__).resolve().parents[2]


def worker(port):
    import uvicorn
    from rag import rag_api
    from rag.groq_reranker import GroqReranker

    class FixtureIndex:
        def __init__(self):
            self.calls = 0
            with gzip.open(ROOT / "rag/data/catalog.jsonl.gz", "rt", encoding="utf-8") as handle:
                self.matches = [
                    {"id": str(row["mal_id"]), "score": .5, "metadata": {"title": row["title"], "mal_id": row["mal_id"], "type": row["type"], "chunk_text": row["text"], "genres": row.get("genres", [])}}
                    for row in map(json.loads, itertools.islice(handle, 10))
                ]

        def query(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("Simulated stale first connection")
            return {"matches": self.matches}

    index = FixtureIndex()

    def create_index(*_args, **kwargs):
        assert kwargs["openapi_config"].retries == 0
        return index

    def model_response(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"ordered_candidate_ids": [1], "expanded_title": None, "reasoning": "Fixture response, not a correctness evaluation."})}}], "usage": {"total_tokens": 100}}, headers={"x-ratelimit-limit-tokens": "1000000"})

    def create_model(_key):
        return GroqReranker("fixture", client=httpx.Client(base_url="https://fixture/", transport=httpx.MockTransport(model_response)))

    with patch("rag.rag_api.Pinecone", return_value=SimpleNamespace(Index=create_index)), patch("rag.groq_reranker.GroqReranker", side_effect=create_model):
        uvicorn.run(rag_api.app, host="127.0.0.1", port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--python", default=".tmp/runtime-venv/Scripts/python.exe")
    parser.add_argument("--memory-mb", type=int, default=768)
    parser.add_argument("--cores", type=float, default=.25)
    parser.add_argument("--rag-port", type=int, default=8096)
    parser.add_argument("--backend-port", type=int, default=8796)
    parser.add_argument("--model-cache", type=Path, default=Path(".tmp/first-request-model-cache"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/bench/first-request-budget.json"))
    args = parser.parse_args()
    if args.worker:
        worker(args.rag_port)
        return
    if os.name != "nt":
        parser.error("The shared CPU/RAM cap uses Windows Job objects")
    if args.memory_mb <= 0 or args.cores <= 0:
        parser.error("CPU and memory caps must be positive")

    spec = importlib.util.spec_from_file_location("windows_budget", ROOT / "scripts/bench/windows-budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    for port in (args.rag_port, args.backend_port):
        budget.free_port(port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / ".tmp", prefix="first-request-") as cache:
        env = os.environ | {
            "TMP": cache, "TEMP": cache, "PYTHONPATH": str(ROOT),
            "FASTEMBED_CACHE_PATH": str(args.model_cache.resolve()), "HF_HUB_OFFLINE": "1",
            "RAG_LLM_RERANK_ENABLED": "1", "RAG_LLM_PROVIDER": "groq", "GROQ_API_KEY": "fixture",
            "RAG_TITLE_JIKAN_FALLBACK": "0", "RAG_LOCAL_ONLY": "0", "PORT": str(args.backend_port),
            "RAG_API_URL": f"http://127.0.0.1:{args.rag_port}/rag/search",
        }
        job, processes = budget.Job(args.memory_mb, args.cores), []
        report = {"scope": "Real local model/catalog; simulated providers; distinct uncached requests; not accuracy or production latency", "memory_mb": args.memory_mb, "cpu_cores": args.cores, "requests": []}
        try:
            with args.output.with_suffix(".rag.log").open("w") as rag_log, args.output.with_suffix(".backend.log").open("w") as backend_log:
                processes.append(job.spawn([args.python, "-u", str(Path(__file__).resolve()), "--worker", "--rag-port", str(args.rag_port)], env, rag_log))
                report["rag_startup_ms"] = budget.wait_ready(f"http://127.0.0.1:{args.rag_port}/health", processes, timeout=360)
                processes.append(job.spawn([shutil.which("node"), "--env-file=.env", "apps/backend/dist/server.js"], env, backend_log))
                base = f"http://127.0.0.1:{args.backend_port}"
                report["backend_startup_ms"] = budget.wait_ready(base + "/health", processes)
                for query in ["girl can travel back in time", "a traveler searches for a moving city", "guy finds a notebook that kills people", "a boy sails the seas with a pirate crew"]:
                    start = time.monotonic()
                    response = httpx.get(base + "/api/rag/search", params={"q": query}, timeout=35)
                    response.raise_for_status()
                    body = response.json()
                    record = {"query": query, "ms": round((time.monotonic() - start) * 1000), "status": response.status_code, "results": len(body.get("results", [])), "notice": body.get("notice")}
                    report["requests"].append(record)
                    if not record["results"] or record["notice"]:
                        raise RuntimeError(f"Empty or degraded probe response: {record}")
                report["alive"] = all(process.poll() is None for process in processes)
        except Exception as exc:
            report["failure"] = str(exc)
        finally:
            report["peak_committed_mb"] = job.peak_mb()
            job.close()
            for process in processes:
                process.wait(timeout=10)
            args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if report.get("failure") or not report.get("alive"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
