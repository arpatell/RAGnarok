"""Finite end-to-end smoke benchmark under a shared Windows Job CPU/RAM cap.

Runs runtime-only Python + built Node backend together, leaving room for the OS.
This approximates resource pressure, not GCP CPU speed or upstream network latency.
No deployment changes, Docker, or administrator configuration are required.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
from ctypes import wintypes as w
import json
import gzip
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import httpx


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", w.DWORD),
                ("min_working_set", ctypes.c_size_t), ("max_working_set", ctypes.c_size_t),
                ("active_processes", w.DWORD), ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class CpuLimit(ctypes.Structure):
    _fields_ = [("flags", w.DWORD), ("rate", w.DWORD)]


class Job:
    def __init__(self, memory_mb: int, cores: float):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_mb:
            limits.basic.flags |= 0x200  # JOB_OBJECT_LIMIT_JOB_MEMORY
            limits.job_memory = memory_mb * 1024 * 1024
        self.check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        if cores:
            cpu = CpuLimit(0x1 | 0x4, max(1, round(10000 * cores / (os.cpu_count() or 1))))
            self.check(self.api.SetInformationJobObject(self.handle, 15, ctypes.byref(cpu), ctypes.sizeof(cpu)))

    @staticmethod
    def check(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())

    def spawn(self, command, env, log):
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=0x4 | subprocess.CREATE_NO_WINDOW)
        try:
            self.check(self.api.AssignProcessToJobObject(self.handle, w.HANDLE(process._handle)))
            native = ctypes.WinDLL("ntdll")
            native.NtResumeProcess.argtypes = [w.HANDLE]
            status = native.NtResumeProcess(w.HANDLE(process._handle))
            if status:
                raise RuntimeError(f"NtResumeProcess failed: {status}")
            return process
        except Exception:
            process.kill()
            process.wait()
            raise

    def peak_mb(self):
        limits = ExtendedLimits()
        self.check(self.api.QueryInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None))
        return round(limits.peak_job_memory / 1024 ** 2, 1)

    def close(self):
        self.api.CloseHandle(self.handle)


def free_port(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def wait_ready(url, processes, timeout=300):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("A service exited under the resource budget; inspect benchmark logs")
        try:
            result = httpx.get(url, timeout=3)
            if result.is_success and result.json().get("initialized", "true") == "true":
                return round((time.perf_counter() - started) * 1000)
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(.5)
    raise RuntimeError("Startup deadline exceeded under the resource budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=".tmp/runtime-venv/Scripts/python.exe")
    parser.add_argument("--memory-mb", type=int, default=768)
    parser.add_argument("--cores", type=float, default=.25)
    parser.add_argument("--output", type=Path, default=Path(".tmp/bench/windows-budget.json"))
    parser.add_argument("--rag-port", type=int, default=8095)
    parser.add_argument("--backend-port", type=int, default=8795)
    parser.add_argument("--stress-seconds", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--distinct-queries", type=int, default=256)
    parser.add_argument("--panel-limit", type=int, default=1)
    parser.add_argument("--think-seconds", type=float, default=3.1, help="Per-user pacing below the existing 20 searches/minute/IP limit")
    parser.add_argument("--semantic-every", type=int, default=4, help="One in N stress requests uses a distinct human plot/character clue; 0 disables this mix")
    args = parser.parse_args()
    if args.concurrency < 1 or args.concurrency > 200 or args.stress_seconds < 0 or args.panel_limit < 1 or args.distinct_queries < 1 or args.semantic_every < 0:
        parser.error("Invalid workload bound")
    if os.name != "nt":
        parser.error("Use the existing Docker benchmark on non-Windows systems")
    for port in (args.rag_port, args.backend_port):
        free_port(port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PORT": str(args.backend_port), "RAG_API_URL": f"http://127.0.0.1:{args.rag_port}/rag/search", "RAG_LLM_RERANK_ENABLED": "0", "RAG_TITLE_JIKAN_FALLBACK": "0"}
    env.pop("PYTHONPATH", None)
    job = Job(args.memory_mb, args.cores)
    processes, samples = [], []
    report = {"memory_mb": args.memory_mb, "cpu_cores": args.cores, "logical_host_cpus": os.cpu_count(), "scope": "combined runtime-only Python and built Node backend", "samples": samples}
    try:
        with args.output.with_suffix(".rag.log").open("w") as rag_log, args.output.with_suffix(".backend.log").open("w") as backend_log:
            processes.append(job.spawn([args.python, "-m", "uvicorn", "rag.rag_api:app", "--host", "127.0.0.1", "--port", str(args.rag_port)], env, rag_log))
            report["rag_startup_ms"] = wait_ready(f"http://127.0.0.1:{args.rag_port}/health", processes)
            processes.append(job.spawn([shutil.which("node"), "--env-file=.env", "apps/backend/dist/server.js"], env, backend_log))
            base = f"http://127.0.0.1:{args.backend_port}"
            report["backend_startup_ms"] = wait_ready(base + "/health", processes)
            queries = ["death note manga", "firefly wedding manga", "attack on titan anime", "guy finds notebook that can kill people manga", "Guts Griffith Casca manga", "two brothers try to bring their mother back and one ends up as a suit of armor manga"]
            def run(item):
                phase, query = item
                started = time.perf_counter()
                try:
                    response = httpx.get(base + "/api/rag/search", params={"q": query}, timeout=30)
                    response.raise_for_status()
                    body = response.json()
                    results = body.get("results", [])
                    return {"phase": phase, "query": query, "ms": round((time.perf_counter() - started) * 1000), "status": response.status_code, "top_id": results[0].get("mal_id") if results else None, "empty": not results, "degraded": bool(body.get("notice"))}
                except Exception as exc:
                    return {"phase": phase, "query": query, "ms": round((time.perf_counter() - started) * 1000), "error": str(exc)}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for phase in ("cold", "warm"):
                    samples.extend(pool.map(run, [(phase, query) for query in queries]))
                samples.extend(pool.map(run, [("shared-burst", "one piece manga")] * 3))
                def reader(item):
                    provider, url = item
                    started = time.perf_counter()
                    try:
                        response = httpx.post(base + "/api/ingest", json={"url": url}, timeout=45)
                        response.raise_for_status()
                        chapter = response.json()["chapter"]
                        if chapter["number"] != "1" or chapter["totalPages"] < 2:
                            raise ValueError("Expected readable chapter-one metadata")
                        image_bytes = 0
                        with httpx.Client(timeout=30) as client:
                            for url in chapter["panelUrls"][:args.panel_limit]:
                                image = client.get(base + "/api/image", params={"url": url})
                                image.raise_for_status()
                                if not image.headers.get("content-type", "").startswith("image/"):
                                    raise ValueError("Panel response was not an image")
                                image_bytes += len(image.content)
                        return {"phase": "reader", "provider": provider, "ms": round((time.perf_counter() - started) * 1000), "panels": chapter["totalPages"], "image_bytes": image_bytes}
                    except Exception as exc:
                        return {"phase": "reader", "provider": provider, "ms": round((time.perf_counter() - started) * 1000), "error": str(exc)}
                readers = [("weebcentral", "https://weebcentral.com/chapters/01J76XZ2FVSZYPS9EE49JP5GKH"), ("manhwazone", "https://manhwazone.com/preview/2q32e")]
                for _ in range(2):
                    samples.extend(pool.map(reader, readers))
                if args.stress_seconds:
                    # A deterministic broad title set fills bounded response
                    # caches instead of repeatedly measuring six hot keys.
                    import random
                    rng, varied = random.Random(20261021), []
                    with gzip.open("rag/data/catalog.jsonl.gz", "rt", encoding="utf-8") as handle:
                        for n, line in enumerate(handle):
                            row = json.loads(line)
                            query = row["title"] + " " + row["type"]
                            if len(query) > 190:
                                continue
                            if len(varied) < args.distinct_queries:
                                varied.append(query)
                            else:
                                position = rng.randrange(n + 1)
                                if position < len(varied):
                                    varied[position] = query
                    deadline = time.monotonic() + args.stress_seconds
                    human = json.loads(Path("rag/eval/human_search_cases.json").read_text(encoding="utf-8"))["cases"]
                    memories = [r["query"] + " " + r["medium"] for r in human if r["kind"] in {"recollection", "mixed_clues", "character_trait"}]
                    def stress(worker):
                        records, position = [], worker
                        transport = httpx.HTTPTransport(local_address=f"127.0.0.{worker + 2}")
                        with httpx.Client(timeout=35, transport=transport) as client:
                            while time.monotonic() < deadline:
                                query = varied[position % len(varied)] if position % 5 else queries[position % len(queries)]
                                if args.semantic_every and position % args.semantic_every == 0:
                                    query = memories[(position // args.semantic_every) % len(memories)]
                                started = time.perf_counter()
                                record = {"ms": 0}
                                try:
                                    response = client.get(base + "/api/rag/search", params={"q": query})
                                    response.raise_for_status()
                                    body = response.json()
                                    record["degraded"] = bool(body.get("notice"))
                                    if not body.get("results"):
                                        record["error"] = "Empty search response"
                                except Exception as exc:
                                    record["error"] = str(exc)
                                record["ms"] = round((time.perf_counter() - started) * 1000)
                                records.append(record)
                                position += args.concurrency
                                time.sleep(max(0, args.think_seconds - (time.perf_counter() - started)))
                        return records
                    workload = [r for group in pool.map(stress, range(args.concurrency)) for r in group]
                    ordered = sorted(r["ms"] for r in workload)
                    report["stress"] = {"duration_seconds": args.stress_seconds, "concurrency": args.concurrency,
                        "distinct_title_queries": len(varied), "requests": len(workload),
                        "semantic_every": args.semantic_every,
                        "errors": sum(bool(r.get("error")) for r in workload),
                        "degraded_responses": sum(bool(r.get("degraded")) for r in workload),
                        "p50_ms": ordered[len(ordered)//2] if ordered else None,
                        "p95_ms": ordered[min(len(ordered)-1, int(len(ordered)*.95))] if ordered else None,
                        "max_ms": max(ordered, default=0), "failure_samples": [r for r in workload if r.get("error")][:10]}
            report["errors"] = sum(bool(s.get("error")) or s.get("empty", False) for s in samples) + report.get("stress", {}).get("errors", 0)
            report["degraded_search_responses"] = sum(bool(s.get("degraded")) for s in samples) + report.get("stress", {}).get("degraded_responses", 0)
            report["peak_committed_mb"] = job.peak_mb()
            report["alive_after_workload"] = all(p.poll() is None for p in processes)
    except Exception as exc:
        report["failure"] = str(exc)
        report["peak_committed_mb"] = job.peak_mb()
    finally:
        job.close()
        for process in processes:
            process.wait(timeout=10)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("failure") or report.get("errors") or report.get("degraded_search_responses") or not report.get("alive_after_workload"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
