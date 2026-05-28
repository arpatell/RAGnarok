from __future__ import annotations

import argparse
import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm

try:
    from rag.top_4000_ingestion import build_dense_embeddings
except ModuleNotFoundError:
    from top_4000_ingestion import build_dense_embeddings

LOGGER = logging.getLogger("top_4000_character_upsert")
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
REQUEST_SLEEP_SECONDS = 1.1

T = TypeVar("T")


@dataclass
class Config:
    pinecone_index: str
    pinecone_namespace: str
    docs_jsonl_path: Path
    enriched_docs_jsonl_path: Path
    bm25_values_path: Path
    checkpoint_path: Path
    text_key: str
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    upsert_batch_size: int
    character_cap: int
    cleanup_stale: bool
    resume: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch character rosters for top_4000 documents from Jikan, add normalized aliases to corpus text, "
            "and upsert refreshed hybrid vectors to Pinecone."
        )
    )
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX", "manga-rag-hybrid"))
    parser.add_argument("--pinecone-namespace", default=os.getenv("PINECONE_NAMESPACE", "top-4000"))
    parser.add_argument("--docs-jsonl-path", default="rag/ingestion/top_4000_documents.jsonl")
    parser.add_argument(
        "--enriched-docs-jsonl-path",
        default="rag/ingestion/top_4000_documents_character_enriched.jsonl",
    )
    parser.add_argument("--bm25-values-path", default="rag/ingestion/bm25_values.json")
    parser.add_argument("--checkpoint-path", default="rag/ingestion/top_4000_character_enrichment_checkpoint.json")
    parser.add_argument("--text-key", default=os.getenv("PINECONE_TEXT_KEY", "chunk_text"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")))
    parser.add_argument("--embed-batch-size", type=int, default=int(os.getenv("EMBED_BATCH_SIZE", "64")))
    parser.add_argument("--upsert-batch-size", type=int, default=int(os.getenv("UPSERT_BATCH_SIZE", "100")))
    parser.add_argument("--character-cap", type=int, default=40)
    parser.add_argument("--cleanup-stale", action="store_true", default=True)
    parser.add_argument("--no-cleanup-stale", dest="cleanup_stale", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    return Config(
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        docs_jsonl_path=Path(args.docs_jsonl_path),
        enriched_docs_jsonl_path=Path(args.enriched_docs_jsonl_path),
        bm25_values_path=Path(args.bm25_values_path),
        checkpoint_path=Path(args.checkpoint_path),
        text_key=args.text_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        character_cap=max(1, args.character_cap),
        cleanup_stale=bool(args.cleanup_stale),
        resume=bool(args.resume),
    )


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\n", " ").split()).strip()


def iter_batches(values: list[T], batch_size: int) -> Iterable[list[T]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"next_index": 0, "failed": []}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"next_index": 0, "failed": []}
    return data


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


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
            if media_type not in ("manga", "anime"):
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
            if existing is None:
                deduped[key] = {
                    "queue_id": queue_id if isinstance(queue_id, int) else 0,
                    "type": media_type,
                    "mal_id": mal_id,
                    "title": clean_text(title),
                    "text": text,
                }
                continue

            existing_text = str(existing.get("text", ""))
            if len(text) > len(existing_text):
                deduped[key] = {
                    "queue_id": queue_id if isinstance(queue_id, int) else int(existing.get("queue_id", 0)),
                    "type": media_type,
                    "mal_id": mal_id,
                    "title": clean_text(title),
                    "text": text,
                }

    rows = list(deduped.values())
    rows.sort(key=lambda row: int(row.get("queue_id", 0)))
    return rows


def parse_text_fields(text: str) -> dict[str, str]:
    lines = text.splitlines()
    data: dict[str, str] = {"type": "", "title": "", "genres": "Unknown", "synopsis": "No synopsis available."}

    for line in lines:
        clean_line = line.strip()
        if clean_line.startswith("Type:"):
            data["type"] = clean_text(clean_line[len("Type:") :])
        elif clean_line.startswith("Title:"):
            data["title"] = clean_text(clean_line[len("Title:") :])
        elif clean_line.startswith("Genres:"):
            data["genres"] = clean_text(clean_line[len("Genres:") :]) or "Unknown"
        elif clean_line.startswith("Synopsis:"):
            data["synopsis"] = clean_text(clean_line[len("Synopsis:") :]) or "No synopsis available."
    return data


def ascii_fold(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return clean_text(folded)


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

    candidates: list[str] = [clean_name]
    if "," in clean_name:
        last, first = [clean_text(part) for part in clean_name.split(",", 1)]
        if first and last:
            candidates.append(f"{first} {last}")
            candidates.append(f"{last} {first}")

    expanded: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for variant in (candidate, ascii_fold(candidate), loose_romanization(candidate), loose_romanization(ascii_fold(candidate))):
            value = clean_text(variant)
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            expanded.append(value)
    return expanded


def extract_character_bundle(character_payload: dict[str, Any], cap: int) -> tuple[list[str], list[str]]:
    rows = character_payload.get("data")
    if not isinstance(rows, list):
        return [], []

    entries: list[str] = []
    aliases: list[str] = []
    alias_seen: set[str] = set()
    for row in rows:
        if len(entries) >= cap:
            break
        if not isinstance(row, dict):
            continue
        character = row.get("character")
        if not isinstance(character, dict):
            continue
        name = clean_text(character.get("name"))
        if not name:
            continue
        role = clean_text(row.get("role")) or "Unknown role"
        entries.append(f"{name} ({role})")

        for alias in build_name_aliases(name):
            key = alias.lower()
            if key in alias_seen:
                continue
            alias_seen.add(key)
            aliases.append(alias)
    return entries, aliases


def compose_enriched_text(
    *,
    media_type: str,
    title: str,
    genres: str,
    synopsis: str,
    character_entries: list[str],
    character_aliases: list[str],
) -> str:
    characters_line = "; ".join(character_entries) if character_entries else "Unknown"
    aliases_line = "; ".join(character_aliases[:140]) if character_aliases else "Unknown"
    return "\n".join(
        [
            f"Type: {media_type}",
            f"Title: {title}",
            f"Genres: {genres or 'Unknown'}",
            f"Synopsis: {synopsis or 'No synopsis available.'}",
            "Characters:",
            characters_line,
            "Character Search Aliases:",
            aliases_line,
        ]
    )


class JikanClient:
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._client = httpx.Client(
            base_url=JIKAN_BASE_URL,
            timeout=timeout_seconds,
            headers={"Accept": "application/json", "User-Agent": "manga-rag-top-4000-character-upsert/1.0"},
        )
        self._next_request_at = 0.0

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        wait_for = self._next_request_at - now
        if wait_for > 0:
            time.sleep(wait_for)
        self._next_request_at = time.monotonic() + REQUEST_SLEEP_SECONDS

    def get_json(self, path: str) -> dict[str, Any]:
        retry_wait = 1.5
        for attempt in range(5):
            self._throttle()
            response = self._client.get(path)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                LOGGER.warning(
                    "Transient Jikan error status=%s path=%s attempt=%d retry_in=%.1fs",
                    response.status_code,
                    path,
                    attempt + 1,
                    retry_wait,
                )
                time.sleep(retry_wait)
                retry_wait *= 1.8
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            return {}
        raise RuntimeError(f"Jikan request failed after retries: {path}")


def build_or_resume_enriched_rows(config: Config, source_rows: list[dict[str, Any]]) -> None:
    config.enriched_docs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(config.checkpoint_path)
    next_index = int(checkpoint.get("next_index", 0))
    failed = checkpoint.get("failed")
    if not isinstance(failed, list):
        failed = []

    if not config.resume:
        next_index = 0
        failed = []
        if config.enriched_docs_jsonl_path.exists():
            config.enriched_docs_jsonl_path.unlink()

    if next_index == 0 and config.enriched_docs_jsonl_path.exists() and config.resume:
        # If we are resuming without checkpoint continuity, infer next index from file line count.
        with config.enriched_docs_jsonl_path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        if 0 < line_count < len(source_rows):
            next_index = line_count
            LOGGER.info("Resume inferred from enriched jsonl line count: %d", line_count)

    mode = "a" if next_index > 0 else "w"
    client = JikanClient(timeout_seconds=20.0)
    progress = tqdm(total=len(source_rows), desc="Character Enrichment", unit="item")
    progress.update(min(next_index, len(source_rows)))
    try:
        with config.enriched_docs_jsonl_path.open(mode, encoding="utf-8") as output:
            for idx in range(next_index, len(source_rows)):
                row = source_rows[idx]
                media_type = str(row["type"])
                mal_id = int(row["mal_id"])
                title = clean_text(row["title"]) or f"{media_type.upper()} MAL {mal_id}"

                fields = parse_text_fields(str(row.get("text", "")))
                genres = fields.get("genres", "Unknown")
                synopsis = fields.get("synopsis", "No synopsis available.")

                character_entries: list[str] = []
                character_aliases: list[str] = []
                try:
                    payload = client.get_json(f"/{media_type}/{mal_id}/characters")
                    character_entries, character_aliases = extract_character_bundle(payload, cap=config.character_cap)
                except Exception as exc:  # noqa: BLE001
                    failed.append(
                        {
                            "index": idx,
                            "type": media_type,
                            "mal_id": mal_id,
                            "title": title,
                            "error": str(exc),
                        }
                    )
                    LOGGER.warning("Character fetch failed for %s/%s: %s", media_type, mal_id, exc)

                enriched_text = compose_enriched_text(
                    media_type=media_type,
                    title=title,
                    genres=genres,
                    synopsis=synopsis,
                    character_entries=character_entries,
                    character_aliases=character_aliases,
                )

                output.write(
                    json.dumps(
                        {
                            "queue_id": int(row.get("queue_id", idx)),
                            "mal_id": mal_id,
                            "title": title,
                            "type": media_type,
                            "text": enriched_text,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

                progress.update(1)
                if (idx + 1) % 25 == 0:
                    save_checkpoint(
                        config.checkpoint_path,
                        {
                            "next_index": idx + 1,
                            "failed": failed[-800:],
                        },
                    )
    finally:
        progress.close()
        client.close()

    save_checkpoint(
        config.checkpoint_path,
        {
            "next_index": len(source_rows),
            "failed": failed[-800:],
        },
    )


def load_enriched_documents(enriched_docs_jsonl_path: Path) -> list[Document]:
    if not enriched_docs_jsonl_path.exists():
        raise RuntimeError(f"Missing enriched docs file: {enriched_docs_jsonl_path}")

    deduped_rows: dict[tuple[str, int], dict[str, Any]] = {}
    with enriched_docs_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mal_id = row.get("mal_id")
            title = row.get("title")
            media_type = row.get("type")
            text = row.get("text")
            if not isinstance(mal_id, int):
                continue
            if media_type not in ("manga", "anime"):
                continue
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            key = (media_type, mal_id)
            existing = deduped_rows.get(key)
            if existing is None or len(text) > len(str(existing.get("text", ""))):
                deduped_rows[key] = {
                    "mal_id": mal_id,
                    "title": clean_text(title),
                    "type": media_type,
                    "text": text,
                }

    documents: list[Document] = []
    for row in deduped_rows.values():
        documents.append(
            Document(
                page_content=str(row["text"]),
                metadata={
                    "mal_id": int(row["mal_id"]),
                    "title": str(row["title"]),
                    "type": str(row["type"]),
                },
            )
        )
    return documents


def extract_ids_from_list_page(page: Any) -> list[str]:
    if page is None:
        return []
    if isinstance(page, list):
        return [str(item) for item in page if isinstance(item, (str, int))]
    if isinstance(page, dict):
        raw_vectors = page.get("vectors")
        if isinstance(raw_vectors, list):
            result: list[str] = []
            for item in raw_vectors:
                if isinstance(item, dict) and "id" in item:
                    result.append(str(item["id"]))
                elif isinstance(item, str):
                    result.append(item)
            return result
        ids = page.get("ids")
        if isinstance(ids, list):
            return [str(item) for item in ids if isinstance(item, (str, int))]
    if hasattr(page, "vectors"):
        vectors = getattr(page, "vectors")
        if isinstance(vectors, list):
            out: list[str] = []
            for item in vectors:
                if hasattr(item, "id"):
                    out.append(str(getattr(item, "id")))
                elif isinstance(item, dict) and "id" in item:
                    out.append(str(item["id"]))
            return out
    return []


def list_namespace_ids(index: Any, namespace: str) -> set[str]:
    ids: set[str] = set()
    for page in index.list(namespace=namespace):
        ids.update(extract_ids_from_list_page(page))
    return ids


def upsert_enriched_documents(config: Config, documents: list[Document]) -> dict[str, int]:
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required.")

    splitter = TokenTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks to upsert.")

    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = build_dense_embeddings()
    dense_vectors: list[list[float]] = []
    for batch in tqdm(list(iter_batches(chunk_texts, config.embed_batch_size)), desc="Dense Embedding", unit="batch"):
        dense_vectors.extend(embeddings.embed_documents(batch))

    bm25 = BM25Encoder().fit(chunk_texts)
    config.bm25_values_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.dump(str(config.bm25_values_path))
    sparse_vectors = bm25.encode_documents(chunk_texts)

    records: list[dict[str, Any]] = []
    per_item_chunk_count: dict[str, int] = {}
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        mal_id = int(chunk.metadata["mal_id"])
        title = str(chunk.metadata["title"])
        media_type = str(chunk.metadata["type"])
        item_key = f"{media_type}:{mal_id}"
        chunk_idx = per_item_chunk_count.get(item_key, 0)
        per_item_chunk_count[item_key] = chunk_idx + 1
        records.append(
            {
                "id": f"{media_type}-{mal_id}-chunk-{chunk_idx:03d}",
                "values": dense,
                "sparse_values": sparse,
                "metadata": {
                    "mal_id": mal_id,
                    "title": title,
                    "type": media_type,
                    config.text_key: chunk.page_content,
                },
            }
        )

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(config.pinecone_index)

    existing_ids = list_namespace_ids(index, config.pinecone_namespace)
    expected_ids = {record["id"] for record in records}

    if config.cleanup_stale:
        stale_ids = sorted(existing_ids - expected_ids)
        for batch in tqdm(list(iter_batches(stale_ids, 1000)), desc="Delete Stale IDs", unit="batch"):
            index.delete(ids=batch, namespace=config.pinecone_namespace)

    upserted = 0
    for batch in tqdm(list(iter_batches(records, config.upsert_batch_size)), desc="Pinecone Upsert", unit="batch"):
        index.upsert(vectors=batch, namespace=config.pinecone_namespace)
        upserted += len(batch)

    existing_after = list_namespace_ids(index, config.pinecone_namespace)
    missing_after = len(expected_ids - existing_after)

    return {
        "documents": len(documents),
        "chunks": len(records),
        "upserted": upserted,
        "existing_before": len(existing_ids),
        "existing_after": len(existing_after),
        "missing_after": missing_after,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    load_dotenv()
    config = parse_args()

    LOGGER.info("Loading source rows from %s", config.docs_jsonl_path)
    source_rows = load_source_rows(config.docs_jsonl_path)
    LOGGER.info("Source rows ready: %d", len(source_rows))

    LOGGER.info("Building character-enriched jsonl at %s", config.enriched_docs_jsonl_path)
    build_or_resume_enriched_rows(config, source_rows)

    documents = load_enriched_documents(config.enriched_docs_jsonl_path)
    LOGGER.info("Enriched documents ready: %d", len(documents))

    stats = upsert_enriched_documents(config, documents)
    LOGGER.info(
        "Character enrichment upsert complete. docs=%d chunks=%d upserted=%d existing_before=%d existing_after=%d missing_after=%d",
        stats["documents"],
        stats["chunks"],
        stats["upserted"],
        stats["existing_before"],
        stats["existing_after"],
        stats["missing_after"],
    )


if __name__ == "__main__":
    main()
