from __future__ import annotations

import argparse
import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

LOGGER = logging.getLogger("delta_sync")
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
MAL_ID_PATTERN = re.compile(r"(?:^|-)mal-(\d+)-chunk-\d+$")

HTML_TAG_RE = re.compile(r"<[^>]+>")
BOILERPLATE_RE = re.compile(
    r"\[(?:Written|Adapted)\s+by\s+MAL\s+Rewrite\]|\bSource:\s*[^.\n]+",
    re.IGNORECASE,
)
MULTISPACE_RE = re.compile(r"\s+")

T = TypeVar("T")


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    text = HTML_TAG_RE.sub(" ", text)
    text = BOILERPLATE_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def iter_batches(values: list[T], batch_size: int) -> Iterable[list[T]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


@dataclass
class DeltaSyncConfig:
    pinecone_index: str
    pinecone_namespace: str
    bm25_values_path: Path
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    upsert_batch_size: int
    jikan_page_limit: int
    jikan_max_pages: int
    jikan_sleep_seconds: float
    character_cap: int
    text_key: str


def parse_args() -> DeltaSyncConfig:
    parser = argparse.ArgumentParser(description="Delta-sync missing anime/manga data from Jikan into Pinecone.")
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX", "manga-rag-hybrid"))
    parser.add_argument("--pinecone-namespace", default=os.getenv("PINECONE_NAMESPACE", "baseline"))
    parser.add_argument("--bm25-values-path", default=os.getenv("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")))
    parser.add_argument("--embed-batch-size", type=int, default=int(os.getenv("EMBED_BATCH_SIZE", "64")))
    parser.add_argument("--upsert-batch-size", type=int, default=int(os.getenv("UPSERT_BATCH_SIZE", "100")))
    parser.add_argument("--jikan-page-limit", type=int, default=int(os.getenv("JIKAN_PAGE_LIMIT", "25")))
    parser.add_argument("--jikan-max-pages", type=int, default=int(os.getenv("JIKAN_MAX_PAGES", "30")))
    parser.add_argument("--jikan-sleep-seconds", type=float, default=float(os.getenv("JIKAN_SLEEP_SECONDS", "1.05")))
    parser.add_argument("--character-cap", type=int, default=int(os.getenv("CHARACTER_CAP", "12")))
    parser.add_argument("--text-key", default=os.getenv("PINECONE_TEXT_KEY", "chunk_text"))
    args = parser.parse_args()

    return DeltaSyncConfig(
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        bm25_values_path=Path(args.bm25_values_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        jikan_page_limit=args.jikan_page_limit,
        jikan_max_pages=args.jikan_max_pages,
        jikan_sleep_seconds=args.jikan_sleep_seconds,
        character_cap=args.character_cap,
        text_key=args.text_key,
    )


class JikanRateLimitedClient:
    """Global request pacing to respect Jikan's 60 req/min limit."""

    def __init__(self, *, sleep_seconds: float, timeout_seconds: float = 20.0) -> None:
        self._sleep_seconds = sleep_seconds
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._client = httpx.AsyncClient(
            base_url=JIKAN_BASE_URL,
            timeout=timeout_seconds,
            headers={"Accept": "application/json", "User-Agent": "manga-rag-delta-sync/1.0"},
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
            self._next_request_at = now + self._sleep_seconds

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        retry_delay = 1.2
        for attempt in range(4):
            await self._throttle()
            response = await self._client.get(path, params=params)
            if response.status_code == 429 and attempt < 3:
                LOGGER.warning("429 from Jikan for %s, retrying in %.2fs", path, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.7
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return {}
            return payload
        raise RuntimeError(f"Jikan request failed after retries for path: {path}")


def _extract_ids_from_list_page(page: Any) -> list[str]:
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


def fetch_known_mal_ids(index: Any, namespace: str) -> set[int]:
    known_ids: set[int] = set()

    try:
        for page in index.list(namespace=namespace):
            ids = _extract_ids_from_list_page(page)
            for vector_id in ids:
                match = MAL_ID_PATTERN.search(vector_id)
                if match:
                    known_ids.add(int(match.group(1)))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to enumerate full index IDs; continuing with partial known_ids: %s", exc)

    LOGGER.info("Known MAL IDs in namespace '%s': %d", namespace, len(known_ids))
    return known_ids


async def list_ids_from_jikan(
    *,
    client: JikanRateLimitedClient,
    media_type: str,
    max_pages: int,
    page_limit: int,
) -> list[int]:
    ids: list[int] = []
    page = 1

    while page <= max_pages:
        payload = await client.get_json(
            f"/{media_type}",
            params={
                "page": page,
                "limit": page_limit,
                "order_by": "members",
                "sort": "desc",
            },
        )

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            break

        for item in data:
            if not isinstance(item, dict):
                continue
            mal_id = item.get("mal_id")
            if isinstance(mal_id, int):
                ids.append(mal_id)

        pagination = payload.get("pagination", {})
        has_next_page = bool(pagination.get("has_next_page"))
        if not has_next_page:
            break
        page += 1

    deduped = list(dict.fromkeys(ids))
    LOGGER.info("Discovered %d candidate %s IDs from Jikan pagination.", len(deduped), media_type)
    return deduped


def _extract_character_lines(character_payload: dict[str, Any], character_cap: int) -> list[str]:
    data = character_payload.get("data")
    if not isinstance(data, list):
        return []

    lines: list[str] = []
    for row in data:
        if len(lines) >= character_cap:
            break
        if not isinstance(row, dict):
            continue
        character = row.get("character")
        if not isinstance(character, dict):
            continue

        name = clean_text(character.get("name"))
        if not name:
            continue

        bio = clean_text(character.get("about")) or clean_text(row.get("role")) or "No bio available."
        lines.append(f"{name}: {bio}")

    return lines


def build_document_from_payloads(
    *,
    media_type: str,
    full_payload: dict[str, Any],
    character_payload: dict[str, Any],
    character_cap: int,
) -> Document | None:
    data = full_payload.get("data")
    if not isinstance(data, dict):
        return None

    mal_id = data.get("mal_id")
    title = clean_text(data.get("title") or data.get("title_english") or data.get("title_japanese"))
    synopsis = clean_text(data.get("synopsis"))

    if not isinstance(mal_id, int) or not title:
        return None

    genres: list[str] = []
    raw_genres = data.get("genres")
    if isinstance(raw_genres, list):
        for genre in raw_genres:
            if isinstance(genre, dict):
                name = clean_text(genre.get("name"))
                if name:
                    genres.append(name)

    character_lines = _extract_character_lines(character_payload, character_cap=character_cap)
    character_block = " | ".join(character_lines) if character_lines else "Unknown"

    page_content = "\n".join(
        [
            f"Type: {media_type}",
            f"Title: {title}",
            f"Genres: {', '.join(genres) if genres else 'Unknown'}",
            f"Synopsis: {synopsis or 'No synopsis available.'}",
            f"Characters (max {character_cap}): {character_block}",
        ]
    )

    return Document(
        page_content=page_content,
        metadata={
            "mal_id": mal_id,
            "title": title,
        },
    )


async def fetch_missing_documents(
    *,
    client: JikanRateLimitedClient,
    media_type: str,
    candidate_ids: list[int],
    known_ids: set[int],
    character_cap: int,
) -> list[Document]:
    documents: list[Document] = []

    for mal_id in candidate_ids:
        if mal_id in known_ids:
            continue

        try:
            full_payload = await client.get_json(f"/{media_type}/{mal_id}/full")
            character_payload = await client.get_json(f"/{media_type}/{mal_id}/characters")
            doc = build_document_from_payloads(
                media_type=media_type,
                full_payload=full_payload,
                character_payload=character_payload,
                character_cap=character_cap,
            )
            if doc is not None:
                documents.append(doc)
        except Exception as exc:  # noqa: BLE001 - continue on bad records
            LOGGER.warning("Skipping %s/%s due to fetch or parse error: %s", media_type, mal_id, exc)

    LOGGER.info("Prepared %d new %s documents after known-id filtering.", len(documents), media_type)
    return documents


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents_in_batches(
    embeddings: HuggingFaceEmbeddings,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    dense_vectors: list[list[float]] = []
    for batch in iter_batches(texts, batch_size):
        dense_vectors.extend(embeddings.embed_documents(batch))
    return dense_vectors


def load_or_fit_bm25(chunks: list[Document], bm25_path: Path) -> BM25Encoder:
    bm25 = BM25Encoder()
    if bm25_path.exists():
        LOGGER.info("Loading BM25 params from %s", bm25_path)
        loaded = bm25.load(str(bm25_path))
        if isinstance(loaded, BM25Encoder):
            return loaded
        return bm25

    LOGGER.warning("BM25 params file not found at %s; fitting from delta chunks.", bm25_path)
    bm25.fit([chunk.page_content for chunk in chunks])
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.dump(str(bm25_path))
    return bm25


def upsert_hybrid_chunks(
    *,
    index: Any,
    namespace: str,
    chunks: list[Document],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[str, list[int] | list[float]]],
    text_key: str,
    batch_size: int,
) -> int:
    records: list[dict[str, Any]] = []
    for idx, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors, strict=True)):
        mal_id = int(chunk.metadata["mal_id"])
        title = str(chunk.metadata["title"])

        metadata = {
            "mal_id": mal_id,
            "title": title,
            text_key: chunk.page_content,
        }

        record = {
            "id": f"mal-{mal_id}-chunk-{idx:06d}",
            "values": dense,
            "sparse_values": sparse,
            "metadata": metadata,
        }
        records.append(record)

    for batch in iter_batches(records, batch_size):
        index.upsert(vectors=batch, namespace=namespace)

    return len(records)


async def async_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    load_dotenv()
    config = parse_args()

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required.")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(config.pinecone_index)

    known_ids = fetch_known_mal_ids(index, config.pinecone_namespace)

    client = JikanRateLimitedClient(sleep_seconds=config.jikan_sleep_seconds)
    try:
        manga_ids, anime_ids = await asyncio.gather(
            list_ids_from_jikan(
                client=client,
                media_type="manga",
                max_pages=config.jikan_max_pages,
                page_limit=config.jikan_page_limit,
            ),
            list_ids_from_jikan(
                client=client,
                media_type="anime",
                max_pages=config.jikan_max_pages,
                page_limit=config.jikan_page_limit,
            ),
        )

        manga_docs, anime_docs = await asyncio.gather(
            fetch_missing_documents(
                client=client,
                media_type="manga",
                candidate_ids=manga_ids,
                known_ids=known_ids,
                character_cap=config.character_cap,
            ),
            fetch_missing_documents(
                client=client,
                media_type="anime",
                candidate_ids=anime_ids,
                known_ids=known_ids,
                character_cap=config.character_cap,
            ),
        )
    finally:
        await client.close()

    documents = manga_docs + anime_docs
    if not documents:
        LOGGER.info("No new manga/anime documents detected; delta sync is up to date.")
        return

    splitter = TokenTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        LOGGER.warning("No chunks produced by splitter; exiting.")
        return

    texts = [chunk.page_content for chunk in chunks]
    embeddings = build_embeddings()
    dense_vectors = embed_documents_in_batches(embeddings, texts, batch_size=config.embed_batch_size)

    bm25 = load_or_fit_bm25(chunks, config.bm25_values_path)
    sparse_vectors = bm25.encode_documents(texts)

    upserted = upsert_hybrid_chunks(
        index=index,
        namespace=config.pinecone_namespace,
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        text_key=config.text_key,
        batch_size=config.upsert_batch_size,
    )
    LOGGER.info(
        "Delta sync complete. New documents=%d, chunks=%d, upserted vectors=%d",
        len(documents),
        len(chunks),
        upserted,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
