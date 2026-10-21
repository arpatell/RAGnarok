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

## Local Resource-Budget Check, 2026-10-20

This is a Windows approximation, **not a production GCP measurement**. The
runtime-only Python service and built Node backend shared a 768 MiB committed
memory limit and a 0.25-core CPU cap. The driver ran outside that budget.
The CPU budget follows [Google's sustained E2 shared-core allocation](https://docs.cloud.google.com/compute/docs/general-purpose-machines#e2_shared-core), not the two vCPUs exposed to the guest.
The serving snapshot contained 83,554 titles; correctness evaluations ran
separately and their failures were not counted as performance successes.

| Observation | Result |
| --- | --- |
| Shared peak committed memory | 722.7 MiB |
| Service exits / HTTP failures in finite workload | 0 / 0 |
| Cached end-to-end search, six-query mix | 23-59 ms |
| Cold title search, three concurrent requests | 461-473 ms |
| Cold WeebCentral chapter open including first image | 3.9 s |
| Cold ManhwaZone open including first image | 3.9 s |
| Cached chapter reopen including first image | 55-78 ms |
| RAG startup under sustained CPU cap | 151.1 s |

The initial startup exceeded the budget before numerical-library thread pools
were limited in code. ManhwaZone's first full-list implementation took 15.7 s
under the same cap; bounded larger Livewire pages reduced repeated cumulative
snapshot parsing while still returning all 201 chapters in the test series.
Pagination falls back to the original page size if that update is rejected.

This was a finite concurrency-three workload, not a 15-minute soak. Real VM
latency, Linux RSS, swap behavior, CPU steal, and external network delays still
need production observation. No VM settings were changed by this benchmark.

The reader timings above begin with a chapter URL, not a search click. Separate
live provider checks resolved seven WeebCentral works in roughly 0.44-0.72 s and
the two supported ManhwaZone fixtures in roughly 1.0-1.2 s; the selected series,
chapter-one title, HTTP response, and series backlink were checked. Cached
resolver calls were below 1 ms in-process. Five other titles were unavailable on
ManhwaZone, not counted as successful resolutions.

Natural-language search can additionally wait on Groq. Its free plan has an
organization-wide token quota; a bounded queue cannot create more capacity.
The app displays a temporary waiting message and returns available retrieval
matches when the provider cannot complete within the request budget. Cached
title/alias lookups do not call Groq. The 8 MiB Python response-cache budget and
4 MiB backend response-cache budget bound growth as more titles are read.

Reproduce after `npm run build`:

```sh
python scripts/bench/windows-budget.py --python .tmp/runtime-venv/Scripts/python.exe --memory-mb 768 --cores .25
```

## Expanded Catalog Pressure Test, 2026-10-20

The final combined Python/Node run used the same 768 MiB / 0.25-core budget,
eight independently addressed simulated users, 384 distinct title keys, and
one human plot/character recollection per two stress requests. Model calls were
disabled, separating retrieval capacity from Groq's organization-wide quota.

| Observation | Result |
| --- | --- |
| Timed mixed workload | 312 requests / 120 seconds |
| HTTP errors, empty responses, service exits | 0 / 0 / 0 |
| Mixed end-to-end p50 / p95 / maximum | 21 / 791 / 1,298 ms |
| Shared peak committed memory | 630.8 MiB |
| RAG / backend startup | 63.6 / 3.6 seconds |

Full response text and character vocabularies now live in the existing disk-backed
index instead of being retained for all 83,554 titles in Python objects. A prior
six-user run peaked at 757.8 MiB; the final workload was more varied, so these
are separate observations, not a controlled identical-workload percentage claim.

Image fetching now has a four-transfer limit, a 32-entry queue with a five-second
wait bound, a shared 20-second transfer/body deadline, an 8 MiB image-size ceiling,
and a 32 MiB byte-bounded cache. Duplicate images still share work. At most two
attempts are made, and Retry-After is not shortened to force more requests.
Seven isolated image tests cover aborts, queue cleanup, byte limits, HTML errors,
cache eviction, coalescing, and upstream throttling.

An earlier 24-panel-per-reader run completed search stress without failures but
had two upstream image timeouts; those were not counted as a successful reader
test. The final run used each provider's first panel, and all 14 app API smoke
checks passed separately. This is not a complete browser-interaction test or a
production-GCP latency guarantee. Cold startup is still substantial, and free
Groq token capacity remains a separate throughput constraint.

```sh
python scripts/bench/windows-budget.py --memory-mb 768 --cores .25 --concurrency 8 --stress-seconds 120 --distinct-queries 384 --semantic-every 2 --panel-limit 1
```

These measurements preceded the subsequent Pinecone monthly-egress rejection.
Remote load testing stopped at that point; the local app now uses a visibly
limited, local-evidence fallback. Its latency must not be presented as equivalent
to the remote hybrid benchmark. Namespace backoff and skipped embedding work
during outages are covered by isolated tests, not additional remote requests.
The benchmark now records degraded search responses and fails its normal run
when any occur, preventing a local-fallback workload from being mistaken for
a successful remote-hybrid benchmark.
