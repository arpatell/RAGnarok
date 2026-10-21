"""Optional FP16 acceleration for offline BGE imports; serving stays unchanged."""
from __future__ import annotations

import numpy as np
from functools import lru_cache


def cache_sparse_stemming(encoder):
    # The frozen BM25 tokenizer is unchanged; only memoize its pure stemmer.
    stemmer = encoder._tokenizer._stemmer
    if not hasattr(stemmer.stem, "cache_info"):
        stemmer.stem = lru_cache(maxsize=131072)(stemmer.stem)


def embed_ordered(model, texts, batch_size):
    # Length grouping avoids padding short metadata records to the longest bio.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    vectors = [None] * len(texts)
    for index, vector in zip(order, model.embed([texts[index] for index in order], batch_size=batch_size), strict=True):
        vectors[index] = vector
    return vectors


def create_embedder(provider, *, fp16=False):
    from fastembed import TextEmbedding
    if not fp16:
        return TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=2, providers=[provider])
    if provider not in {"DmlExecutionProvider", "CUDAExecutionProvider"}:
        raise ValueError("Offline FP16 requires an existing GPU execution provider")
    from fastembed.common.model_description import ModelSource, PoolingType
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
    name = "ragnarok-bge-small-fp16"
    if not any(row["model"] == name for row in TextEmbedding.list_supported_models()):
        TextEmbedding.add_custom_model(model=name, pooling=PoolingType.CLS, normalization=True, dim=384,
            sources=ModelSource(hf="Xenova/bge-small-en-v1.5"), model_file="onnx/model_fp16.onnx")
    return TextEmbedding(model_name=name, threads=2, providers=[provider])


def validate_compatibility(model, texts):
    from fastembed import TextEmbedding
    reference = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=4, providers=["CPUExecutionProvider"])
    a = np.array(list(reference.embed(texts, batch_size=8)), dtype=np.float32)
    b = np.array(list(model.embed(texts, batch_size=8)), dtype=np.float32)
    if a.shape != b.shape or a.shape[1] != 384 or not np.isfinite(b).all():
        raise ValueError("Offline acceleration changed vector dimensions or produced invalid values")
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    minimum = float(cosine.min())
    if minimum < .999:
        raise ValueError(f"Offline acceleration did not preserve BGE vectors: cosine={minimum:.6f}")
    return minimum
