# Search Evaluation

Correctness and resource-pressure tests are intentionally separate. Do not tune
queries to current results or report broad exact-title success as plot-recall
accuracy.

## Fixed Human Queries

`human_search_cases.json` contains hand-written recollections, combined clues,
authors, studios, dates/formats, characters, English aliases, Japanese romaji,
Korean romanizations, and ordinary spelling mistakes. Expected identities are
`(medium, MAL ID)`, not title substrings. Novels, one-shots, adaptations, and
sequels are distinct works. A generic query can remain genuinely ambiguous.

```sh
python -m rag.eval.catalog_eval --human --endpoint http://127.0.0.1:8090/rag/search --output rag/eval/human_results.json
python -m rag.eval.catalog_eval --human --split held_out --endpoint http://127.0.0.1:8090/rag/search --output rag/eval/human_holdout_results.json
```

The default target is 95% hits@1. Failures are retained. The holdout assignment
uses a stable hash of the title identity, so paraphrases of one title cannot
cross the development/holdout boundary.

## Catalog Coverage

```sh
python -m rag.eval.catalog_eval --coverage .9 --in-process --workers 4 --output rag/eval/catalog_results.json
python -m rag.eval.catalog_eval --coverage .9 --kinds title --in-process --workers 4 --output rag/eval/title_results.json
```

The first command checks multiple available fields for a deterministic 90% of
eligible titles. These source-grounded generated clues are **not** independently
written paraphrases. Per-category scores and title coverage are reported
separately. Missing character rosters do not count as character-test coverage.
The in-process mode calls the same production search implementation without
HTTP overhead; it is not a performance benchmark. `--resume` continues an
interrupted identical evaluation, but should not be used after a code, model,
index, or snapshot change.

The evaluator uses the tracked compressed serving snapshot and manifest, matching
what the deployed service sees. `--include-pending` instead reads the local
SQLite checkpoint to expose not-yet-indexed rows. Pending index rows remain in
the coverage denominator. Raw generated cases, reports, and JSONL checkpoints
are ignored by Git.

## Regression and Providers

The existing 280-case suite retains excluded titles as explicit policy-negative
tests. Policy negatives are reported separately from positive hits@1. All
identity comparisons also check the medium, preventing manga/anime ID collisions.
CI preserves its existing regression deployment gate and publishes the expanded
human suite as a visibly failing, nonblocking audit until its 95% hits@1 target
is met. Its failures are not removed or relabeled as successes.

```sh
python -m unittest rag.eval.test_search_contracts -v
node --import tsx --test apps/backend/src/adapters/index.test.ts apps/backend/src/services/webSearch.test.ts apps/backend/src/services/ingest.test.ts apps/backend/src/services/ragProxy.test.ts
node --env-file=.env --import tsx scripts/eval/read-now.mts
node --env-file=.env --import tsx scripts/eval/app-smoke.mts
```

Live provider tests verify series identity, chapter-one labels, successful HTTP
responses, and a backlink to the selected series. Missing provider inventory is
reported separately; known supported WeebCentral titles and the two established
ManhwaZone titles must resolve, rather than being silently marked unavailable.

## Resource Pressure

On Windows, install the existing runtime requirements in an isolated environment,
build the backend, then run:

```sh
python scripts/bench/windows-budget.py --python .tmp/runtime-venv/Scripts/python.exe --memory-mb 768 --cores .25
```

Both servers share the budget; the driver is outside it. This catches startup
allocation failures and workload instability. It does not reproduce GCP CPU
hardware, swap, network, or noisy neighbors. The existing Docker benchmark is
available on systems with Docker. No benchmark requires production SSH changes.

## Current Correctness Audit (2026-10-20)

| Fixed set | Cases | Before hits@1 | After hits@1 |
| --- | ---: | ---: | ---: |
| Hand-written recall and mixed clues | 123 | 58.54% | 82.93% |
| Source-derived plot, character and metadata clues | 2,000 | 74.65% | 88.70% |
| Catalog titles, aliases and romanizations | 153,303 | 96.37% | 96.37% |

The 123-case run used Groq `openai/gpt-oss-20b`; the source-derived and broad
catalog runs disabled model calls, isolating data/deterministic retrieval.
The human run gained 30 first-place matches and lost none. **The 95% recall
target is not met.** Do not substitute macro-category accuracy for overall
hits@1 or describe the source-derived prompts as independent human paraphrases.
Some generic character prompts do not specify which franchise edition, and
creator/year/genre combinations can identify multiple works; strict identity
scoring is retained, not relaxed to improve the headline number.

To replay the same local 2,000 cases after catalog enrichment:

```sh
python -m rag.eval.catalog_eval --cases rag/eval/fixed_clue_cases.json --in-process --output rag/eval/clue_replay.json
```

Groq requests use at most ten candidates with bounded relevant excerpts. The
free-plan quota is shared across users, not multiplied per visitor; the FIFO is
bounded and honors rate-limit headers and Retry-After. Exhausted/failed model
requests return retrieval matches with a temporary UI notice, not a paid-provider
fallback. Live model evaluation consumes the same quota as local app usage.
CI keeps model calls disabled and exercises provider failure/queue contracts
with deterministic mocks.

The existing Deploy VM workflow accepts `GROQ_API_KEY` as a GitHub secret and
updates the existing VM RAG environment and backend timeout before restarting
services. The key is carried over SSH stdin, never embedded in a command line.
Absent that secret, existing VM provider configuration is left unchanged and a
warning is emitted. No new SSH-side setup is required.

## Expanded Audit and Scope

The hand-written suite now has **203 cases across 124 distinct titles**, including
short recollections, less-mainstream works, character traits, explicit editions,
negations, aliases, romanizations, and ordinary spelling errors. The original
123 cases remain in place. No failures were removed to raise the score.

The completed 153,303-case title/alias/romanization replay covered 75,199 of
83,554 titles (90%) and achieved **96.37% hits@1**, without model calls, errors,
or policy/type leaks. The final frozen 280-case gate achieved **272/280 hits@10**;
this is not a 95% plot-recall hits@1 result.

The full 203-case live-provider audit returned 64.04% hits@1, but 163 responses
explicitly fell back because detailed model matching was unavailable. The
evaluator now reports these as `degraded_responses`, not silent model successes.
A separate quota-aware replay was stopped after 28 cases: 20 first-place hits,
three final degraded responses. This partial run is not an aggregate accuracy
estimate. **The 95% human-recall target remains unmet.**

For correctness-only replay, optional `--degraded-retries 2
--degraded-retry-seconds 30` retries explicitly reported provider fallbacks.
Use a single in-process worker with a longer `GROQ_MAX_WAIT_SECONDS` only for
that isolated run; these evaluation waits are not production request settings.
Performance tests disable model calls and report cold, cached, and mixed
workloads separately.

Increasing dense weighting, adding the embedding model's retrieval prefix, and
downweighting biography-only chunks were tested against the fixed human suite.
They reduced overall accuracy and were not adopted. Provider failures preserve
the existing retrieval order rather than publishing experimental fusion order.

Further collection defaults to the top 10,000 titles, balanced between anime
and manga. The current scoped audit found character names for 6,203 titles
(62.03%) and biographies for 4,679 (46.79%); missing data is not described as
complete coverage. Already imported data outside that scope is retained, and
all collected vector updates have been uploaded. Raw character exports remain
local; serving bundles travel through the existing Git deployment.

The final frozen 2,000 source-derived clues achieved 87.05% hits@1 and 98.95%
hits@10 with model calls disabled. This set is distinct from independently
written human recollections and does not establish 95% natural-language recall.

### Provider Allowance Exhaustion

After the completed audits, Pinecone rejected a verification fetch with HTTP 429:
the account's 1,000,000,000-byte monthly egress allowance was exhausted. Live
evaluations contributed traffic; no additional Pinecone data audit or paid upgrade was
performed after this response. The final remote biography-chunk fetch therefore
could not be verified; earlier remote title/chunk checks and all upload
checkpoints had completed.

The service now reports a friendly limited-service notice, uses the existing
local evidence index, and backs off a failing namespace instead of silently
treating missing remote results as a complete search. Embedding work is skipped
while all remote namespaces are cooling down. Exact titles, aliases, and indexed
character-name lookup remain local. This fallback is not equivalent to full
remote semantic retrieval.

When local retrieval has one or no candidates, the optional model may still
recognize a title in its single bounded call. The suggestion must exactly match
an allowed local catalog title/alias in the requested medium and share concrete
query clues with its stored facts before it can be returned. Provider failures
still retain the limited-service notice; this is not an ungrounded substitute
for missing data. Quota/backoff, skipped embedding work, short-clue fallback,
empty-result notices, and grounded recovery have isolated regression tests.

The current local sanity-test server explicitly uses `RAG_LOCAL_ONLY=1` so it
does not make further Pinecone data requests. This is an optional local setting,
not a new required VM setting; production detects provider failures automatically.
No ingestion worker or remote evaluation remains running.

The final local-only sanity check returned Death Note (manga MAL-ID 21) for
"guy finds notebook", with the explicit limited-service notice. The final
contract runs passed 68 Python and 27 backend tests; the production build,
typecheck, and 14 API workflow checks passed. These checks do not replace a
full browser-interaction test or a completed provider-assisted recall audit.
