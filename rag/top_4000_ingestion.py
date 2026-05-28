from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

import httpx
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import TokenTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm

LOGGER = logging.getLogger("top_4000_ingestion")
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
TOP_PAGE_LIMIT = 25
TOP_TARGET_COUNT = 2000
TOP_PAGE_MAX = 80
REQUEST_SLEEP_SECONDS = 1.1

HTML_TAG_RE = re.compile(r"<[^>]+>")
BOILERPLATE_RE = re.compile(
    r"\[(?:Written|Adapted)\s+by\s+MAL\s+Rewrite\]|\bSource:\s*[^.\n]+",
    re.IGNORECASE,
)
MULTISPACE_RE = re.compile(r"\s+")

T = TypeVar("T")


@dataclass
class IngestConfig:
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    bm25_values_path: Path
    chunk_size: int
    chunk_overlap: int
    upsert_batch_size: int
    embed_batch_size: int
    text_key: str
    checkpoint_path: Path
    docs_jsonl_path: Path
    create_index: bool
    allow_partial: bool


def parse_args() -> IngestConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch top 2000 anime + top 2000 manga from Jikan, build hybrid embeddings, and upsert to Pinecone."
        )
    )
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX", "manga-rag-hybrid"))
    parser.add_argument("--pinecone-namespace", default=os.getenv("PINECONE_NAMESPACE", "top-4000"))
    parser.add_argument("--pinecone-cloud", default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--pinecone-region", default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--bm25-values-path", default=os.getenv("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")))
    parser.add_argument("--upsert-batch-size", type=int, default=int(os.getenv("UPSERT_BATCH_SIZE", "100")))
    parser.add_argument("--embed-batch-size", type=int, default=int(os.getenv("EMBED_BATCH_SIZE", "64")))
    parser.add_argument("--text-key", default=os.getenv("PINECONE_TEXT_KEY", "chunk_text"))
    parser.add_argument(
        "--checkpoint-path",
        default=os.getenv("TOP4000_CHECKPOINT_PATH", "rag/ingestion/top_4000_checkpoint.json"),
    )
    parser.add_argument(
        "--docs-jsonl-path",
        default=os.getenv("TOP4000_DOCS_JSONL_PATH", "rag/ingestion/top_4000_documents.jsonl"),
    )
    parser.add_argument(
        "--create-index",
        dest="create_index",
        action="store_true",
        default=os.getenv("CREATE_PINECONE_INDEX", "true").strip().lower() == "true",
    )
    parser.add_argument("--no-create-index", dest="create_index", action="store_false")
    parser.add_argument("--allow-partial", action="store_true", help="Proceed with embedding/upsert even if some IDs fail.")
    args = parser.parse_args()

    return IngestConfig(
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        pinecone_cloud=args.pinecone_cloud,
        pinecone_region=args.pinecone_region,
        bm25_values_path=Path(args.bm25_values_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        upsert_batch_size=args.upsert_batch_size,
        embed_batch_size=args.embed_batch_size,
        text_key=args.text_key,
        checkpoint_path=Path(args.checkpoint_path),
        docs_jsonl_path=Path(args.docs_jsonl_path),
        create_index=args.create_index,
        allow_partial=args.allow_partial,
    )


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    text = HTML_TAG_RE.sub(" ", text)
    text = BOILERPLATE_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def iter_batches(values: list[T], batch_size: int) -> Iterable[list[T]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "queue": [],
            "processed_keys": [],
            "failed": [],
            "chunk_upsert_complete": False,
        }
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid checkpoint format at {checkpoint_path}")
    return data


def save_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_processed_queue_ids_from_jsonl(docs_jsonl_path: Path) -> set[int]:
    if not docs_jsonl_path.exists():
        return set()

    queue_ids: set[int] = set()
    with docs_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            queue_id = row.get("queue_id")
            if isinstance(queue_id, int):
                queue_ids.add(queue_id)
    return queue_ids


class JikanRateLimitedClient:
    """Strict global pacing: one request every 1.1 seconds."""

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._client = httpx.AsyncClient(
            base_url=JIKAN_BASE_URL,
            timeout=timeout_seconds,
            headers={"Accept": "application/json", "User-Agent": "manga-rag-top-4000/1.0"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait_for = self._next_request_at - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = loop.time()
            self._next_request_at = now + REQUEST_SLEEP_SECONDS

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        retry_wait = 1.5
        for attempt in range(5):
            await self._throttle()
            response = await self._client.get(path, params=params)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                LOGGER.warning(
                    "Transient Jikan error %s for %s (attempt %d). Retrying in %.1fs",
                    response.status_code,
                    path,
                    attempt + 1,
                    retry_wait,
                )
                await asyncio.sleep(retry_wait)
                retry_wait *= 1.8
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return {}
            return payload
        raise RuntimeError(f"Jikan request failed after retries: {path}")


async def collect_top_queue(client: JikanRateLimitedClient) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []

    for media_type in ("manga", "anime"):
        collected = 0
        for page in range(1, TOP_PAGE_MAX + 1):
            payload = await client.get_json(
                f"/top/{media_type}",
                params={"page": page, "limit": TOP_PAGE_LIMIT},
            )
            rows = payload.get("data")
            if not isinstance(rows, list):
                LOGGER.warning("Unexpected top/%s page %d payload shape.", media_type, page)
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                mal_id = row.get("mal_id")
                if not isinstance(mal_id, int):
                    continue
                title = clean_text(row.get("title") or row.get("title_english") or row.get("title_japanese"))
                if not title:
                    title = f"{media_type.upper()} MAL {mal_id}"
                queue.append(
                    {
                        "queue_id": len(queue),
                        "mal_id": mal_id,
                        "title": title,
                        "type": media_type,
                        "source_page": page,
                    }
                )
                collected += 1

        LOGGER.info("Collected %d top %s entries from pages 1..%d.", collected, media_type, TOP_PAGE_MAX)
        if collected < TOP_TARGET_COUNT:
            LOGGER.warning("Expected ~%d %s items, got %d.", TOP_TARGET_COUNT, media_type, collected)

    return queue


def extract_character_lines(character_payload: dict[str, Any], cap: int = 12) -> list[str]:
    rows = character_payload.get("data")
    if not isinstance(rows, list):
        return []

    lines: list[str] = []
    for row in rows:
        if len(lines) >= cap:
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
        about = clean_text(character.get("about"))
        if about:
            lines.append(f"{name} ({role}): {about}")
        else:
            lines.append(f"{name} ({role})")
    return lines


def synthesize_text(
    *,
    media_type: str,
    title: str,
    full_payload: dict[str, Any],
    character_payload: dict[str, Any],
) -> str:
    data = full_payload.get("data")
    if not isinstance(data, dict):
        data = {}

    synopsis = clean_text(data.get("synopsis")) or "No synopsis available."
    genres_raw = data.get("genres")
    genres: list[str] = []
    if isinstance(genres_raw, list):
        for row in genres_raw:
            if isinstance(row, dict):
                name = clean_text(row.get("name"))
                if name:
                    genres.append(name)
    characters = extract_character_lines(character_payload, cap=12)

    return "\n".join(
        [
            f"Type: {media_type}",
            f"Title: {title}",
            f"Genres: {', '.join(genres) if genres else 'Unknown'}",
            f"Synopsis: {synopsis}",
            "Characters:",
            "; ".join(characters) if characters else "Unknown",
        ]
    )


async def fetch_and_store_documents(
    *,
    client: JikanRateLimitedClient,
    queue: list[dict[str, Any]],
    checkpoint_path: Path,
    docs_jsonl_path: Path,
) -> tuple[int, int]:
    processed_queue_ids = load_processed_queue_ids_from_jsonl(docs_jsonl_path)
    failed: list[dict[str, Any]] = []
    docs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(queue), desc="Deep Fetch (4000 target)", unit="item")
    for item in queue:
        queue_id = int(item.get("queue_id", -1))
        media_type = str(item["type"])
        mal_id = int(item["mal_id"])
        title = str(item["title"])
        if queue_id in processed_queue_ids:
            progress.update(1)
            continue

        try:
            full_payload = await client.get_json(f"/{media_type}/{mal_id}/full")
            character_payload = await client.get_json(f"/{media_type}/{mal_id}/characters")
            merged_text = synthesize_text(
                media_type=media_type,
                title=title,
                full_payload=full_payload,
                character_payload=character_payload,
            )
            row = {
                "queue_id": queue_id,
                "mal_id": mal_id,
                "title": title,
                "type": media_type,
                "text": merged_text,
            }
            with docs_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            processed_queue_ids.add(queue_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed %s/%s: %s", media_type, mal_id, exc)
            failed.append(
                {
                    "queue_id": queue_id,
                    "type": media_type,
                    "mal_id": mal_id,
                    "title": title,
                    "error": str(exc),
                }
            )

        progress.update(1)
        if progress.n % 50 == 0:
            save_checkpoint(
                checkpoint_path,
                {
                    "queue": queue,
                    "processed_keys": sorted(processed_queue_ids),
                    "failed": failed[-500:],
                    "chunk_upsert_complete": False,
                },
            )
    progress.close()

    save_checkpoint(
        checkpoint_path,
        {
            "queue": queue,
            "processed_keys": sorted(processed_queue_ids),
            "failed": failed[-500:],
            "chunk_upsert_complete": False,
        },
    )
    return len(processed_queue_ids), len(failed)


def load_documents_from_jsonl(docs_jsonl_path: Path) -> list[Document]:
    deduped_rows: dict[tuple[str, int], dict[str, str | int]] = {}
    if not docs_jsonl_path.exists():
        return []

    with docs_jsonl_path.open("r", encoding="utf-8") as handle:
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
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(media_type, str) or media_type not in ("anime", "manga"):
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            key = (media_type, mal_id)
            existing = deduped_rows.get(key)
            if existing is None:
                deduped_rows[key] = {
                    "mal_id": mal_id,
                    "title": title.strip(),
                    "type": media_type,
                    "text": text,
                }
                continue

            # Keep the richer synopsis payload when duplicates exist in source rows.
            existing_text = str(existing["text"])
            if len(text) > len(existing_text):
                deduped_rows[key] = {
                    "mal_id": mal_id,
                    "title": title.strip(),
                    "type": media_type,
                    "text": text,
                }

    documents: list[Document] = []
    for row in deduped_rows.values():
        documents.append(
            Document(
                page_content=str(row["text"]),
                metadata={"mal_id": int(row["mal_id"]), "title": str(row["title"]), "type": str(row["type"])},
            )
        )

    return documents


def get_embedding_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def build_dense_embeddings() -> HuggingFaceEmbeddings:
    device = get_embedding_device()
    LOGGER.info("Using HuggingFace embedding device: %s", device)
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_index(pc: Pinecone, config: IngestConfig, dense_dim: int) -> None:
    if pc.has_index(name=config.pinecone_index):
        return

    if not config.create_index:
        raise RuntimeError(f"Index '{config.pinecone_index}' does not exist and --create-index is false.")

    LOGGER.info("Creating Pinecone index %s (dimension=%d)", config.pinecone_index, dense_dim)
    pc.create_index(
        name=config.pinecone_index,
        dimension=dense_dim,
        metric="dotproduct",
        spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region),
    )


def upsert_chunks_hybrid(
    *,
    config: IngestConfig,
    documents: list[Document],
) -> int:
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required.")

    splitter = TokenTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return 0
    LOGGER.info("Chunked %d documents into %d chunks.", len(documents), len(chunks))

    embeddings = build_dense_embeddings()
    chunk_texts = [chunk.page_content for chunk in chunks]
    dense_vectors: list[list[float]] = []
    for batch in tqdm(list(iter_batches(chunk_texts, config.embed_batch_size)), desc="Dense Embedding", unit="batch"):
        dense_vectors.extend(embeddings.embed_documents(batch))

    bm25 = BM25Encoder().fit(chunk_texts)
    config.bm25_values_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.dump(str(config.bm25_values_path))
    sparse_vectors = bm25.encode_documents(chunk_texts)

    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, config, dense_dim=len(dense_vectors[0]))
    index = pc.Index(config.pinecone_index)

    # Required primitive initialization. We use the same index for explicit dense+sparse upsert.
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=config.pinecone_namespace,
        text_key=config.text_key,
    )
    _ = vector_store

    per_item_chunk_count: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        mal_id = int(chunk.metadata["mal_id"])
        title = str(chunk.metadata["title"])
        media_type = str(chunk.metadata["type"])
        item_key = f"{media_type}:{mal_id}"
        chunk_idx = per_item_chunk_count.get(item_key, 0)
        per_item_chunk_count[item_key] = chunk_idx + 1

        metadata = {
            "mal_id": mal_id,
            "title": title,
            "type": media_type,
            config.text_key: chunk.page_content,
        }
        records.append(
            {
                "id": f"{media_type}-{mal_id}-chunk-{chunk_idx:03d}",
                "values": dense,
                "sparse_values": sparse,
                "metadata": metadata,
            }
        )

    for batch in tqdm(list(iter_batches(records, config.upsert_batch_size)), desc="Pinecone Upsert", unit="batch"):
        index.upsert(vectors=batch, namespace=config.pinecone_namespace)

    return len(records)


async def async_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    load_dotenv()
    config = parse_args()

    checkpoint = load_checkpoint(config.checkpoint_path)
    queue = checkpoint.get("queue")
    if not isinstance(queue, list) or not queue:
        client = JikanRateLimitedClient()
        try:
            queue = await collect_top_queue(client)
        finally:
            await client.close()
        save_checkpoint(
            config.checkpoint_path,
            {"queue": queue, "processed_keys": [], "failed": [], "chunk_upsert_complete": False},
        )

    LOGGER.info("Queue size: %d", len(queue))
    if len(queue) < TOP_TARGET_COUNT * 2:
        LOGGER.warning("Queue has fewer than 4000 entries (%d).", len(queue))

    client = JikanRateLimitedClient()
    try:
        processed_count, failed_count = await fetch_and_store_documents(
            client=client,
            queue=queue,
            checkpoint_path=config.checkpoint_path,
            docs_jsonl_path=config.docs_jsonl_path,
        )
    finally:
        await client.close()

    LOGGER.info("Deep fetch complete. processed=%d failed=%d", processed_count, failed_count)
    if failed_count > 0 and not config.allow_partial:
        raise RuntimeError(
            f"Deep fetch had {failed_count} failures. Re-run to retry or use --allow-partial to continue."
        )

    documents = load_documents_from_jsonl(config.docs_jsonl_path)
    if not documents:
        raise RuntimeError("No documents available for chunk/embedding.")

    upserted = upsert_chunks_hybrid(config=config, documents=documents)
    final_checkpoint = load_checkpoint(config.checkpoint_path)
    final_checkpoint["chunk_upsert_complete"] = True
    save_checkpoint(config.checkpoint_path, final_checkpoint)
    LOGGER.info(
        "Ingestion complete. documents=%d upserted_vectors=%d namespace=%s",
        len(documents),
        upserted,
        config.pinecone_namespace,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
