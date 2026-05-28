from __future__ import annotations

import argparse
import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm

LOGGER = logging.getLogger("repair_sparse_and_aliases")
T = TypeVar("T")


@dataclass
class Config:
    pinecone_index: str
    pinecone_namespace: str
    docs_jsonl_path: Path
    bm25_values_path: Path
    text_key: str
    chunk_size: int
    chunk_overlap: int
    fetch_batch_size: int
    upsert_batch_size: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill sparse vectors and character alias metadata into an existing Pinecone index "
            "using the local top_4000 ingestion corpus."
        )
    )
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX", "manga-rag-hybrid-char4000-v1"))
    parser.add_argument("--pinecone-namespace", default=os.getenv("PINECONE_NAMESPACE", "top-4000"))
    parser.add_argument(
        "--docs-jsonl-path",
        default=os.getenv("TOP4000_ENRICHED_JSONL_PATH", "rag/ingestion/top_4000_documents_character_enriched.jsonl"),
    )
    parser.add_argument("--bm25-values-path", default=os.getenv("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"))
    parser.add_argument("--text-key", default=os.getenv("PINECONE_TEXT_KEY", "chunk_text"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")))
    parser.add_argument("--fetch-batch-size", type=int, default=500)
    parser.add_argument("--upsert-batch-size", type=int, default=100)
    args = parser.parse_args()

    return Config(
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        docs_jsonl_path=Path(args.docs_jsonl_path),
        bm25_values_path=Path(args.bm25_values_path),
        text_key=args.text_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fetch_batch_size=max(1, min(1000, args.fetch_batch_size)),
        upsert_batch_size=max(1, min(500, args.upsert_batch_size)),
    )


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\n", " ").split()).strip()


def iter_batches(values: list[T], batch_size: int) -> Iterable[list[T]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def load_source_rows(docs_jsonl_path: Path) -> list[dict[str, Any]]:
    if not docs_jsonl_path.exists():
        raise RuntimeError(f"Missing source docs file: {docs_jsonl_path}")

    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    with docs_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            media_type = row.get("type")
            mal_id = row.get("mal_id")
            title = row.get("title")
            text = row.get("text")
            if media_type not in ("anime", "manga"):
                continue
            if not isinstance(mal_id, int):
                continue
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(text, str) or not text.strip():
                continue

            key = (media_type, mal_id)
            queue_id = row.get("queue_id")
            existing = deduped.get(key)
            if existing is None or len(text) > len(str(existing.get("text", ""))):
                deduped[key] = {
                    "queue_id": queue_id if isinstance(queue_id, int) else 0,
                    "type": media_type,
                    "mal_id": mal_id,
                    "title": clean_text(title),
                    "text": text,
                }

    rows = list(deduped.values())
    rows.sort(key=lambda row: int(row.get("queue_id", 0)))
    return rows


def extract_section(text: str, marker: str) -> str:
    lowered = text.lower()
    marker_idx = lowered.find(marker.lower())
    if marker_idx < 0:
        return ""

    tail = text[marker_idx + len(marker) :]
    stop_markers = ("character search aliases:", "characters:", "synopsis:", "genres:", "type:", "title:")
    lowest_stop = len(tail)
    for stop in stop_markers:
        idx = tail.lower().find(stop)
        if idx > 0:
            lowest_stop = min(lowest_stop, idx)
    return tail[:lowest_stop].strip()


def ascii_fold(value: str) -> str:
    return clean_text(unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii"))


def loose_romanization(value: str) -> str:
    normalized = clean_text(value).lower()
    replacements = (
        ("ou", "o"),
        ("oo", "o"),
        ("uu", "u"),
        ("aa", "a"),
        ("ii", "i"),
        ("oh", "o"),
    )
    for src, dst in replacements:
        normalized = normalized.replace(src, dst)
    return clean_text(normalized)


def build_name_aliases(name: str) -> list[str]:
    clean_name = clean_text(name)
    if not clean_name:
        return []

    variants: list[str] = [clean_name]
    if "," in clean_name:
        last, first = [clean_text(part) for part in clean_name.split(",", 1)]
        if first and last:
            variants.append(f"{first} {last}")
            variants.append(f"{last} {first}")

    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        for candidate in (variant, ascii_fold(variant), loose_romanization(variant), loose_romanization(ascii_fold(variant))):
            normalized = clean_text(candidate)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
    return out


def extract_aliases_for_text(text: str) -> str:
    explicit_aliases = extract_section(text, "Character Search Aliases:")
    aliases: list[str] = []
    seen: set[str] = set()

    if explicit_aliases:
        for entry in explicit_aliases.split(";"):
            alias = clean_text(entry)
            if not alias:
                continue
            key = alias.lower()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
        return "; ".join(aliases)

    characters_block = extract_section(text, "Characters:")
    if not characters_block:
        return ""

    for entry in characters_block.split(";"):
        name = clean_text(entry.split("(", 1)[0])
        if not name or name.lower() == "unknown":
            continue
        for alias in build_name_aliases(name):
            key = alias.lower()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
    return "; ".join(aliases[:180])


def build_chunk_records(config: Config, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[Document] = []
    alias_by_key: dict[str, str] = {}
    for row in rows:
        media_type = str(row["type"])
        mal_id = int(row["mal_id"])
        title = str(row["title"])
        text = str(row["text"])
        item_key = f"{media_type}:{mal_id}"
        alias_by_key[item_key] = extract_aliases_for_text(text)
        docs.append(
            Document(
                page_content=text,
                metadata={"mal_id": mal_id, "title": title, "type": media_type},
            )
        )

    splitter = TokenTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("No chunks produced from source documents.")

    chunk_texts = [chunk.page_content for chunk in chunks]
    bm25 = BM25Encoder().fit(chunk_texts)
    config.bm25_values_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.dump(str(config.bm25_values_path))
    sparse_vectors = bm25.encode_documents(chunk_texts)

    records: list[dict[str, Any]] = []
    per_item_chunk_count: dict[str, int] = {}
    for chunk, sparse in zip(chunks, sparse_vectors, strict=True):
        mal_id = int(chunk.metadata["mal_id"])
        title = str(chunk.metadata["title"])
        media_type = str(chunk.metadata["type"])
        item_key = f"{media_type}:{mal_id}"
        chunk_idx = per_item_chunk_count.get(item_key, 0)
        per_item_chunk_count[item_key] = chunk_idx + 1

        record_id = f"{media_type}-{mal_id}-chunk-{chunk_idx:03d}"
        metadata: dict[str, Any] = {
            "mal_id": mal_id,
            "title": title,
            "type": media_type,
            config.text_key: chunk.page_content,
        }
        aliases = alias_by_key.get(item_key, "")
        if aliases:
            metadata["character_aliases"] = aliases

        records.append(
            {
                "id": record_id,
                "sparse_values": sparse,
                "metadata": metadata,
            }
        )

    return records


def fetch_dense_values(index: Any, namespace: str, ids: list[str], batch_size: int) -> dict[str, list[float]]:
    values_by_id: dict[str, list[float]] = {}
    for batch in tqdm(list(iter_batches(ids, batch_size)), desc="Fetch Dense Values", unit="batch"):
        payload = index.fetch(ids=batch, namespace=namespace)
        vectors = payload.get("vectors") if isinstance(payload, dict) else getattr(payload, "vectors", {})
        if not isinstance(vectors, dict):
            continue
        for vector_id, vector_payload in vectors.items():
            body = vector_payload if isinstance(vector_payload, dict) else vector_payload.to_dict()
            dense_values = body.get("values")
            if isinstance(dense_values, list) and dense_values:
                values_by_id[str(vector_id)] = dense_values
    return values_by_id


def backfill_sparse_and_aliases(config: Config, records: list[dict[str, Any]]) -> dict[str, int]:
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required.")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(config.pinecone_index)

    expected_ids = [str(record["id"]) for record in records]
    dense_values_by_id = fetch_dense_values(index, config.pinecone_namespace, expected_ids, config.fetch_batch_size)

    missing = [vector_id for vector_id in expected_ids if vector_id not in dense_values_by_id]
    if missing:
        LOGGER.warning("Missing %d IDs in index namespace before backfill.", len(missing))

    upserts: list[dict[str, Any]] = []
    for record in records:
        vector_id = str(record["id"])
        dense_values = dense_values_by_id.get(vector_id)
        if dense_values is None:
            continue
        upserts.append(
            {
                "id": vector_id,
                "values": dense_values,
                "sparse_values": record["sparse_values"],
                "metadata": record["metadata"],
            }
        )

    updated = 0
    for batch in tqdm(list(iter_batches(upserts, config.upsert_batch_size)), desc="Pinecone Upsert", unit="batch"):
        index.upsert(vectors=batch, namespace=config.pinecone_namespace)
        updated += len(batch)

    sample_id = upserts[0]["id"] if upserts else ""
    sparse_len = 0
    if sample_id:
        fetched = index.fetch(ids=[sample_id], namespace=config.pinecone_namespace)
        vectors = fetched.get("vectors") if isinstance(fetched, dict) else getattr(fetched, "vectors", {})
        sample = vectors.get(sample_id) if isinstance(vectors, dict) else None
        if sample:
            body = sample if isinstance(sample, dict) else sample.to_dict()
            sparse = body.get("sparse_values") or {}
            sparse_len = len(sparse.get("indices", []))

    return {
        "expected": len(expected_ids),
        "updated": updated,
        "missing": len(missing),
        "sample_sparse_len": sparse_len,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    load_dotenv()
    config = parse_args()

    LOGGER.info("Loading documents from %s", config.docs_jsonl_path)
    rows = load_source_rows(config.docs_jsonl_path)
    LOGGER.info("Loaded deduped rows: %d", len(rows))

    LOGGER.info("Building chunk records and sparse vectors...")
    records = build_chunk_records(config, rows)
    LOGGER.info("Chunk records prepared: %d", len(records))

    stats = backfill_sparse_and_aliases(config, records)
    LOGGER.info(
        "Backfill complete. expected=%d updated=%d missing=%d sample_sparse_len=%d namespace=%s",
        stats["expected"],
        stats["updated"],
        stats["missing"],
        stats["sample_sparse_len"],
        config.pinecone_namespace,
    )


if __name__ == "__main__":
    main()
