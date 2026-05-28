# RAG Ingestion Artifacts

Generated ingestion outputs are intentionally not committed to GitHub.

Keep these files locally or upload them directly to the VM/object storage when needed:

- `top_4000_documents.jsonl`
- `top_4000_documents_character_enriched.jsonl`
- `bm25_values.json`
- `top_4000_title_aliases.json`
- checkpoint/log files

The deployed RAG service still needs the relevant runtime artifacts if it is configured to use local BM25/title-alias data.
