# PanelFlow RAG Module

Standalone RAG scripts for bootstrap ingestion, delta sync, and hybrid retrieval API.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r rag/requirements.txt
```

## Required Environment Variables

- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `CEREBRAS_API_KEY`

Optional:

- `PINECONE_NAMESPACE` (default: `baseline`)
- `BM25_VALUES_PATH` (default: `rag/ingestion/bm25_values.json`)
- `CEREBRAS_MODEL` (default: `gpt-oss-120b`)
- `PINECONE_CLOUD` (default: `aws`)
- `PINECONE_REGION` (default: `us-east-1`)

## Layer 1: Baseline Bootstrap

```bash
python rag/bootstrap.py --pinecone-index "$PINECONE_INDEX" --pinecone-namespace baseline --create-index
```

## Layer 2: Delta Sync

```bash
python rag/delta_sync.py --pinecone-index "$PINECONE_INDEX" --pinecone-namespace baseline
```

## Layer 3/4: Hybrid API

```bash
uvicorn rag.rag_api:app --host 0.0.0.0 --port 8090
```
