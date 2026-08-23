"""Embedding parity check: fastembed ONNX vs sentence-transformers reference.

Run before switching production traffic to a fastembed-backed RAG service:

    .venv/bin/python -m rag.eval.embedding_parity

Verifies that the ONNX runtime path produces vectors equivalent to the
sentence-transformers reference for BAAI/bge-small-en-v1.5 (cosine >= 0.999
on sample queries) so the Pinecone index does not need re-ingestion.
Exits non-zero on mismatch.

Note: requires both fastembed (runtime dep) and sentence-transformers
(ingestion dep) to be installed:
    pip install -r rag/requirements.txt -r rag/requirements-ingestion.txt
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from typing import Any

SAMPLE_QUERIES = [
    "guy finds notebook that can kill people",
    "satoru gojo",
    "blind swordsman wandering feudal japan",
    "maid sama",
    "gods vs humans",
]

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MIN_COSINE = 0.999


def _first_vector(embeddings: Iterable[Any]) -> list[float]:
    iterator = iter(embeddings)
    vector = next(iterator)
    return [safe_float(x) for x in vector]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "sentence-transformers not installed; "
            "pip install -r rag/requirements.txt -r rag/requirements-ingestion.txt",
            file=sys.stderr,
        )
        return 2

    from fastembed import TextEmbedding

    print(f"Loading ONNX model ({MODEL_NAME})...")
    onnx_model = TextEmbedding(model_name=MODEL_NAME)
    print("Loading sentence-transformers reference model...")
    ref_model = SentenceTransformer(MODEL_NAME)

    failures = 0
    for query in SAMPLE_QUERIES:
        onnx_vec = _first_vector(onnx_model.embed([query]))
        ref_vec = [safe_float(x) for x in ref_model.encode(query, normalize_embeddings=True)]

        if len(onnx_vec) != len(ref_vec):
            print(f"FAIL dims onnx={len(onnx_vec)} ref={len(ref_vec)} query={query!r}")
            failures += 1
            continue

        score = cosine(onnx_vec, ref_vec)
        status = "ok" if score >= MIN_COSINE else "FAIL"
        if score < MIN_COSINE:
            failures += 1
        print(f"{status} cosine={score:.6f} dims={len(onnx_vec)} query={query!r}")

    if failures:
        print(f"\n{failures} parity failure(s); DO NOT deploy without re-indexing.")
        return 1

    print("\nParity confirmed; safe to serve fastembed embeddings against the existing index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
