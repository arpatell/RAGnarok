# Performance Baseline & Benchmarks

This file tracks production performance for the e2-micro deployment.
Record every benchmark run here so before/after comparisons stay honest.

## How to measure

**On the VM (localhost timings + system evidence):**

```bash
RUN_LABEL="baseline-$(date -u +%Y%m%d)" PERF_RECORD=1 \
  bash scripts/gcp/benchmark.sh
```

Captures: endpoint latencies, `free -m`, vmstat CPU-steal sample, service
restart counts, OOM events from the kernel log, recent RAG per-stage timings
and backend API latency lines.

**From your local machine (end-to-end incl. network RTT):**

```bash
BASE_URL=https://YOUR_VM.sslip.io node scripts/bench/run-load.mjs --scenario search-realistic
BASE_URL=https://YOUR_VM.sslip.io RECORD_FILE=PERF.md node scripts/bench/run-load.mjs --scenario all --record
BASE_URL=https://YOUR_VM.sslip.io SOAK_SECONDS=900 node scripts/bench/run-load.mjs --scenario soak
```

Never generate load on the VM itself — the generator would steal the CPU
being measured.

## Pass criteria after optimization

| Metric | Target |
| --- | --- |
| Warmed search p95 (`/api/rag/search`) | ≤ 4s |
| Search errors at 3 concurrent | 0 |
| Reader ingest cache-hit p95 | ≤ 500ms |
| RAG RSS after warm start | < 450 MB |
| OOM kills / restarts during 15-min soak | 0 |
| First search after idle (with warm timer) | ≤ 10s |

## Baseline (pre-optimization)

> Fill in by running `benchmark.sh` on the VM BEFORE deploying the changes.

| Case | HTTP | Total |
| --- | --- | --- |
| health | - | - |
| search: death note | - | - |
| rag/search: death note | - | - |
| rag/search: plot query | - | - |
| meta endpoints | n/a (removed Jikan paths) | - |

Notes
-

## After optimization

> Append runs with `PERF_RECORD=1 benchmark.sh` / `run-load.mjs --record`.

(appended below by tooling)
