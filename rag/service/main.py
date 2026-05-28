from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .models import RagSearchRequest, RagSearchResponse
from .rag_agent import MangaRagAgent

load_dotenv()

app = FastAPI(title="PanelFlow RAG Service", version="1.0.0")


@lru_cache(maxsize=1)
def get_agent() -> MangaRagAgent:
    index_name = os.environ.get("PINECONE_INDEX", "").strip()
    if not index_name:
        raise RuntimeError("PINECONE_INDEX is required.")

    return MangaRagAgent(
        pinecone_index=index_name,
        pinecone_namespace=os.environ.get("PINECONE_NAMESPACE", "manga-rag"),
        bm25_path=os.environ.get("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"),
        top_k=int(os.environ.get("RAG_TOP_K", "10")),
        score_threshold=float(os.environ.get("RAG_SCORE_THRESHOLD", "0.75")),
        recursion_limit=int(os.environ.get("RAG_RECURSION_LIMIT", "6")),
        cerebras_model=os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=RagSearchResponse)
def query_rag(payload: RagSearchRequest) -> RagSearchResponse:
    question = payload.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        agent = get_agent()
        return agent.query(question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc
