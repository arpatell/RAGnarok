# RAG Ingestion Artifacts

Generated ingestion outputs are intentionally not committed to GitHub.

Keep these raw files locally:

- `top_4000_documents.jsonl`
- `top_4000_documents_character_enriched.jsonl`
- `top_4000_title_aliases.json`
- checkpoint/log files

`bm25_values.json` is tracked because query and document sparse vectors must share
the same vocabulary. Do not refit it during an incremental import.

The serving snapshot is `rag/data/catalog.jsonl.gz`, with counts and coverage in
`rag/data/catalog_manifest.json`. These are deliberate deployment artifacts, not
raw ingestion dumps. The existing GitHub checkout and service restart deliver
them to the VM automatically; no SSH upload or new environment setting is needed.

From the repository root, using the ingestion Python environment:

```sh
python -m rag.catalog_sync collect
python -m rag.catalog_sync cleanup                 # audit and back up first
python -m rag.catalog_sync cleanup --apply
python -m rag.catalog_sync enrich --limit 200      # resumable, paced MAL roster requests
python -m rag.character_data rosters --limit 500   # IDs and main/supporting roles
python -m rag.character_data profiles --limit 500  # biographies; shared characters fetched once
python -m rag.character_data export               # local compressed character dataset
python -m rag.catalog_sync upsert                  # resumable; deterministic IDs
python -m rag.catalog_sync bundle
python -m rag.catalog_sync status
```

The four excluded genres are Boys Love, Girls Love, Erotica, and Hentai. The
same policy runs before embedding, while building the snapshot, and when serving
results. A title with no available character roster is not advertised as having
character coverage; metadata coverage and character coverage are separate.

`characters.jsonl.gz` and `characters_manifest.json` stay local under this ignored
directory. Full biographies used for retrieval are upserted to Pinecone and kept
in the disk-backed evidence index, not in every in-memory serving title record.
`catalog_sync bundle` also builds `rag/data/evidence.sqlite.gz`:
a read-only field index for creator/year/genre clues. It is unpacked automatically
to the system temporary directory and uses a bounded SQLite page cache, not a new
database service. Commit both serving bundles when intentionally deploying data.

The official MAL API v2 does not provide character rosters. Character ingestion
uses paced public MAL pages, not the retired Jikan public endpoint. Refresh and
profile imports checkpoint every page; running them again resumes pending work.

## Bulk Enrichment

Prefer the validated bulk baseline before per-page gap filling:

```sh
python -m rag.bulk_enrichment download  # four bounded concurrent archive downloads
python -m rag.bulk_enrichment audit     # every row's identity, schema, CRC and source manifest
python -m rag.bulk_enrichment import    # missing fields only; preserves newer MAL values
python -m rag.catalog_sync upsert
python -m rag.catalog_sync bundle
```

Sources and attribution are recorded in `rag/data/SOURCES.md`. No cast is inferred
across an anime/manga adaptation. Historical bulk coverage is not current full-MAL
coverage; missing rosters and recently added works remain gap-fill work.

On an offline workstation with an already installed DirectML or CUDA ONNX runtime,
`upsert --provider DmlExecutionProvider --fp16 --embed-batch-size 32` uses the same
BGE-small weights in GPU-friendly precision. It checks diverse real inputs against
the serving encoder and rejects minimum cosine below 0.999 before writing vectors.
This does not change the serving model or introduce a VM dependency. Sparse
encoding retains the frozen vocabulary, memoizes pure stemming, and overlaps CPU
encoding with GPU inference. Full biographies remain in vectors and the disk
evidence index; compact serving title records do not keep all biographies in RAM.

```sh
python -m rag.full_enrichment run --top-titles 10000 --stages roster profile
python -m rag.full_enrichment status
```

Further collection defaults to the top 10,000 eligible titles, interleaving
5,000 anime and 5,000 manga. Missing ranking numbers fall back to relative MAL
scores rather than putting established titles last. Imported data outside this
scope is retained. Already imported casts and profiles are not downloaded again.
No cast-size, role, or biography-length cutoff is applied inside the scope.
The default stages fill character gaps only; `--stages details roster profile`
also refreshes official API metadata. Account-specific list status is not requested.
Detail responses, compressed source HTML, failures, and per-record checkpoints
remain in the ignored local catalog SQLite database. The official detail API
omits characters, so the worker reads the public character pages separately.

One request at a time, at least one second apart; server Retry-After is honored.
Access denials stop the worker; missing pages and confirmed empty rosters are
recorded separately from successful enrichment. The machine must stay running;
rerun the command after shutdown to resume. Progress is written to
`rag/ingestion/full_enrichment_progress.json`. This is a multi-day offline job,
not something to run on the serving e2-micro. No collector is automatically
installed or started on the VM.

Collection does **not** claim to publish changes to Pinecone or overwrite the
currently tested serving bundles. After collection, audit exclusions, run the
incremental upsert and bundle steps above, and recheck memory and retrieval
quality before deploying. Full biographies materially increase the raw corpus;
never assume the previous resource-budget result applies to the larger corpus.
