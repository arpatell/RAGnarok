"""Resumable offline MAL import, targeted cleanup, and deployable serving snapshot.

Run from the repository root: .venv/Scripts/python.exe -m rag.catalog_sync --help
Uses the existing BM25 vocabulary and deterministic vector IDs. Never refits BM25.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import unicodedata
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from rag.catalog_policy import excluded_genres, extract_section

LOG = logging.getLogger("catalog_sync")
DATA = Path("rag/ingestion")
SOURCE = DATA / "top_4000_documents_character_enriched.jsonl"
FIELDS = "alternative_titles,synopsis,background,genres,media_type,status,mean,num_chapters,num_volumes,num_episodes,authors{first_name,last_name},serialization{name},studios,source,start_date,end_date,start_season"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temp.replace(path)


def row_key(row: dict[str, Any]) -> str:
    return f"{row['type']}:{int(row['mal_id'])}"


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS documents (key TEXT PRIMARY KEY, row_json TEXT NOT NULL, enriched INTEGER NOT NULL DEFAULT 0, upserted INTEGER NOT NULL DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS excluded (key TEXT PRIMARY KEY, row_json TEXT NOT NULL)")
    return db


def store(db: sqlite3.Connection, row: dict[str, Any], *, enriched: bool = False, upserted: bool = False) -> None:
    db.execute("INSERT INTO documents VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET row_json=excluded.row_json,enriched=excluded.enriched,upserted=excluded.upserted",
               (row_key(row), json.dumps(row, ensure_ascii=True), int(enriched), int(upserted)))


class MalClient:
    def __init__(self, interval: float = 0.5) -> None:
        key = os.getenv("MAL_CLIENT_ID") or os.getenv("MYANIMELIST_CLIENT_ID")
        if not key:
            raise RuntimeError("Existing MAL_CLIENT_ID is required")
        self.api = httpx.Client(base_url="https://api.myanimelist.net/v2/", headers={"X-MAL-CLIENT-ID": key}, timeout=25)
        self.web = httpx.Client(headers={"User-Agent": "RAGnarok-catalog/1.0"}, timeout=25, follow_redirects=True)
        self.interval = interval
        self.next_at = 0.0

    def get(self, path: str, *, params: dict | None = None, page: bool = False) -> httpx.Response:
        for attempt in range(4):
            time.sleep(max(0, self.next_at - time.monotonic()))
            self.next_at = time.monotonic() + self.interval
            try:
                response = (self.web if page else self.api).get(path, params=params)
                if response.status_code in (307, 429) or response.status_code >= 500:
                    retry = response.headers.get("Retry-After", "")
                    try:
                        delay = float(retry)
                    except ValueError:
                        try:
                            delay = parsedate_to_datetime(retry).timestamp() - time.time()
                        except (TypeError, ValueError, OverflowError):
                            delay = 2 ** (attempt + 1)
                    time.sleep(max(0, delay))
                    continue
                response.raise_for_status()
                return response
            except httpx.TransportError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"MAL unavailable after retries: {urlsplit(path).path}")


class CharacterParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active = False
        self.is_name = False
        self.parts: list[str] = []
        self.names: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self.active = bool(re.match(r"^https://myanimelist\.net/character/\d+/", href))
            self.is_name = False
            self.parts = []
        elif tag == "h3" and self.active and "h3_character_name" in (dict(attrs).get("class") or ""):
            self.is_name = True

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.active:
            name = " ".join("".join(self.parts).split())
            if self.is_name and name and name not in self.names:
                self.names.append(name)
            self.active = False


def name_aliases(names: list[str], limit: int | None = 140) -> list[str]:
    aliases: dict[str, str] = {}
    for name in names:
        variants = [name]
        if "," in name:
            last, first = (part.strip() for part in name.split(",", 1))
            variants.extend([f"{first} {last}", f"{last} {first}"])
        for variant in variants:
            folded = unicodedata.normalize("NFKD", variant).encode("ascii", "ignore").decode()
            for value in (variant, folded):
                if value.strip():
                    aliases.setdefault(value.casefold(), value)
    return list(aliases.values())[:limit]


def compose(row: dict[str, Any], old_text: str = "", names: list[str] | None = None) -> str:
    genres = ", ".join(row.get("genres", [])) or extract_section(old_text, "Genres") or "Unknown"
    synopsis = row.get("synopsis") or extract_section(old_text, "Synopsis") or "No synopsis available."
    characters = "; ".join(names) if names is not None else extract_section(old_text, "Characters")
    aliases = "; ".join(name_aliases(names)) if names is not None else extract_section(old_text, "Character Search Aliases")
    details = "; ".join(f"{key}: {value}" for key, value in row.get("search_details", {}).items() if value)
    return "\n".join([f"Type: {row['type']}", f"Title: {row['title']}", f"Details: {details or 'Unknown'}", f"Genres: {genres}",
                      f"Synopsis: {synopsis}", "Characters:", characters or "Unknown",
                      "Character Search Aliases:", aliases or "Unknown"])


def seed(db: sqlite3.Connection) -> None:
    if not SOURCE.exists():
        return
    added = 0
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if excluded_genres(row):
                continue
            key = row_key(row)
            if db.execute("SELECT 1 FROM documents WHERE key=?", (key,)).fetchone():
                db.execute("UPDATE documents SET row_json=json_set(row_json, '$.vector_exists', 1) WHERE key=?", (key,))
                continue
            if db.execute("SELECT 1 FROM excluded WHERE key=?", (key,)).fetchone():
                continue
            row["vector_exists"] = True
            row["genres"] = [g.strip() for g in extract_section(row["text"], "Genres").split(",") if g.strip()]
            store(db, row, enriched=True, upserted=True)
            added += 1
    db.commit()
    LOG.info("Seeded %d existing character-enriched titles", added)


def collect(db: sqlite3.Connection, args: argparse.Namespace) -> None:
    seed(db)
    mal = MalClient(args.interval)
    for medium in args.media:
        state_key = f"{medium}:offset"
        state = db.execute("SELECT value FROM state WHERE key=?", (state_key,)).fetchone()
        offset = 0 if args.refresh else int(state[0]) if state else 0
        for _ in range(args.max_pages or 10000):
            try:
                payload = mal.get(f"{medium}/ranking", params={"ranking_type": "all", "offset": offset, "limit": 500, "fields": FIELDS}).json()
            except RuntimeError:
                LOG.warning("Retrying smaller MAL page after upstream failure: %s offset=%d", medium, offset)
                payload = mal.get(f"{medium}/ranking", params={"ranking_type": "all", "offset": offset, "limit": 100, "fields": FIELDS}).json()
            entries = payload.get("data", [])
            excluded = 0
            for item in entries:
                node = item["node"]
                row = {"type": medium, "mal_id": node["id"], "title": node["title"],
                       "catalog_rank": item.get("ranking", {}).get("rank"),
                       "genres": [g["name"] for g in node.get("genres", [])],
                       "alternative_titles": node.get("alternative_titles", {}),
                       "synopsis": html.unescape(node.get("synopsis") or ""),
                       "image_url": node.get("main_picture", {}).get("large") or node.get("main_picture", {}).get("medium"),
                       "display_type": node.get("media_type"), "status": node.get("status"),
                       "chapters": node.get("num_chapters"), "volumes": node.get("num_volumes"),
                       "episodes": node.get("num_episodes"), "score": node.get("mean")}
                authors = [" ".join(filter(None, [a.get("node", {}).get("first_name"), a.get("node", {}).get("last_name")])) for a in node.get("authors", [])]
                row["search_details"] = {"Format": node.get("media_type", "").replace("_", " "),
                                         "Source material": node.get("source", "").replace("_", " "),
                                         "Authors": ", ".join(filter(None, authors)),
                                         "Studios": ", ".join(s.get("name", "") for s in node.get("studios", [])),
                                         "Serialization": ", ".join(s.get("node", {}).get("name", "") for s in node.get("serialization", [])),
                                         "Started": node.get("start_date"), "Ended": node.get("end_date"),
                                         "Status": node.get("status", "").replace("_", " "),
                                         "Season": " ".join(str(v) for v in node.get("start_season", {}).values()),
                                         "Background": html.unescape(node.get("background") or "")}
                key = row_key(row)
                if excluded_genres(row):
                    db.execute("INSERT OR REPLACE INTO excluded VALUES (?, ?)", (key, json.dumps(row)))
                    db.execute("DELETE FROM documents WHERE key=?", (key,))
                    excluded += 1
                    continue
                db.execute("DELETE FROM excluded WHERE key=?", (key,))
                existing = db.execute("SELECT row_json, enriched, upserted FROM documents WHERE key=?", (key,)).fetchone()
                old = json.loads(existing[0]) if existing else {}
                row["vector_exists"] = bool(old.get("vector_exists") or (existing and existing[2]))
                if old.get("character_source"):
                    row["character_source"] = old["character_source"]
                for field in ("characters", "character_schema_version", "character_page_missing"):
                    if field in old:
                        row[field] = old[field]
                row["text"] = compose(row, old.get("text", ""))
                biographies = extract_section(old.get("text", ""), "Character Details")
                if biographies:
                    row["text"] += "\nCharacter Details:\n" + biographies
                store(db, row, enriched=bool(existing and existing[1]), upserted=bool(existing and existing[2] and old.get("text") == row["text"]))
            offset += len(entries)
            db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (state_key, str(offset)))
            db.commit()
            LOG.info("MAL %s offset=%d received=%d excluded=%d stored=%d", medium, offset, len(entries), excluded, db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            if not entries or not payload.get("paging", {}).get("next"):
                break


def cleanup(args: argparse.Namespace) -> None:
    from pinecone import Pinecone

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = DATA / "cleanup" / stamp
    backup.mkdir(parents=True, exist_ok=True)
    blocked: dict[str, dict] = {}
    sources = [SOURCE, DATA / "top_4000_documents.jsonl"]
    for source in sources:
        if source.exists():
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if excluded_genres(row):
                        blocked[row_key(row)] = {"type": row["type"], "mal_id": row["mal_id"], "title": row["title"], "excluded_genres": excluded_genres(row)}
    if args.database.exists():
        with open_db(args.database) as db:
            for key, body in db.execute("SELECT key,row_json FROM excluded"):
                row = json.loads(body)
                blocked[key] = {"type": row["type"], "mal_id": row["mal_id"], "title": row["title"], "excluded_genres": excluded_genres(row)}
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX"])
    namespace = os.getenv("PINECONE_NAMESPACE", "top-4000")
    all_ids = [identifier for page in index.list(namespace=namespace) for identifier in page]
    delete_ids: set[str] = set()
    backup_file = backup / "vectors.jsonl"
    with backup_file.open("w", encoding="utf-8") as handle:
        for start in range(0, len(all_ids), 250):
            payload = index.fetch(ids=all_ids[start:start + 250], namespace=namespace)
            for identifier, vector in payload.vectors.items():
                meta = vector.metadata or {}
                key = f"{meta.get('type')}:{int(meta.get('mal_id', 0))}"
                if key in blocked or excluded_genres(meta):
                    delete_ids.add(identifier)
                    handle.write(json.dumps(vector.to_dict(), ensure_ascii=True) + "\n")
    manifest = {"index": os.environ["PINECONE_INDEX"], "namespace": namespace,
                "apply": args.apply, "excluded_titles": list(blocked.values()),
                "delete_ids": sorted(delete_ids), "vector_backup": str(backup_file)}
    write_json(backup / "manifest.json", manifest)
    LOG.info("Cleanup audit: %d excluded titles, %d/%d vectors; backup=%s", len(blocked), len(delete_ids), len(all_ids), backup)
    if not args.apply:
        return
    for start in range(0, len(delete_ids), 500):
        index.delete(ids=sorted(delete_ids)[start:start + 500], namespace=namespace)
    for source in sources:
        if not source.exists():
            continue
        shutil.copy2(source, backup / source.name)
        temp = source.with_suffix(".clean.tmp")
        with source.open(encoding="utf-8") as incoming, temp.open("w", encoding="utf-8") as outgoing:
            for line in incoming:
                if not excluded_genres(json.loads(line)):
                    outgoing.write(line)
        temp.replace(source)
    for start in range(0, len(delete_ids), 500):
        ids = sorted(delete_ids)[start:start + 500]
        remaining = index.fetch(ids=ids, namespace=namespace).vectors
        if remaining:
            time.sleep(3)
            remaining = index.fetch(ids=ids, namespace=namespace).vectors
        if remaining:
            raise RuntimeError(f"Cleanup verification still finds {len(remaining)} deleted vectors")
    LOG.info("Verified targeted deletion and cleaned local source files")


def enrich(db: sqlite3.Connection, args: argparse.Namespace) -> None:
    mal = MalClient(max(1.0, args.interval))
    done, attempted = 0, 0
    for key, body in db.execute("SELECT key, row_json FROM documents WHERE enriched=0 AND COALESCE(json_extract(row_json, '$.character_page_missing'),0)=0 ORDER BY COALESCE(json_extract(row_json, '$.catalog_rank'), 999999), rowid").fetchall():
        if args.limit and attempted >= args.limit:
            break
        attempted += 1
        row = json.loads(body)
        try:
            response = mal.get(f"https://myanimelist.net/{row['type']}/{row['mal_id']}/_/characters", page=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            db.execute("UPDATE documents SET row_json=json_set(row_json, '$.character_page_missing', 1) WHERE key=?", (key,))
            db.commit()
            LOG.warning("MAL roster unavailable for %s; kept metadata, not marked enriched", key)
            continue
        parser = CharacterParser()
        parser.feed(response.text)
        if "h3_character_name" not in response.text and "No characters" not in response.text:
            raise RuntimeError(f"Unexpected MAL character page for {key}; checkpoint preserved")
        row["text"] = compose(row, row["text"], parser.names[:40])
        row["character_source"] = "mal_character_page"
        store(db, row, enriched=True)
        db.commit()
        done += 1
        LOG.info("Enriched %s characters=%d completed=%d", key, len(parser.names), done)


def upsert(db: sqlite3.Connection, args: argparse.Namespace) -> None:
    from rag.offline_embedding import cache_sparse_stemming, create_embedder, embed_ordered, validate_compatibility
    from concurrent.futures import ThreadPoolExecutor
    from pinecone import Pinecone
    from pinecone_text.sparse import BM25Encoder
    from langchain_text_splitters import TokenTextSplitter

    LOG.setLevel(logging.INFO)
    if not LOG.handlers:
        LOG.addHandler(logging.StreamHandler())
        LOG.propagate = False

    bm25 = BM25Encoder().load("rag/ingestion/bm25_values.json")
    cache_sparse_stemming(bm25)
    model = create_embedder(args.provider, fp16=args.fp16)
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX"])
    namespace = os.getenv("PINECONE_NAMESPACE", "top-4000")
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    rows = db.execute("SELECT key,row_json,enriched FROM documents WHERE upserted=0 ORDER BY enriched DESC, COALESCE(json_extract(row_json, '$.catalog_rank'),999999),rowid").fetchall()
    if args.limit:
        rows = rows[:args.limit]
    if args.fp16 and rows:
        step = max(1, len(rows) // 12)
        sample_texts = [json.loads(body)["text"][:2400] for _, body, _ in rows[::step][:12]]
        minimum = validate_compatibility(model, sample_texts)
        LOG.info("Validated offline FP16 BGE compatibility minimum_cosine=%.6f samples=%d", minimum, len(sample_texts))
    for start in range(0, len(rows), args.document_batch_size):
        batch = rows[start:start + args.document_batch_size]
        records: list[dict] = []
        texts: list[str] = []
        sanitized: list[tuple[str, str]] = []
        for key, body, enriched in batch:
            row = json.loads(body)
            if excluded_genres(row):
                raise RuntimeError(f"Blocked genre reached embedding stage: {key}")
            if not enriched and not row.get("characters"):
                row["text"] = compose(row, row["text"], [])
            row["vector_exists"] = True
            sanitized.append((json.dumps(row), key))
            aliases = [alias.strip() for alias in extract_section(row["text"], "Character Search Aliases").split(";") if alias.strip()]
            for n, chunk in enumerate(splitter.split_text(row["text"])):
                # Full casts belong in searchable chunks, not duplicated in
                # every vector's metadata (large casts exceed Pinecone's cap).
                chunk_aliases = []
                alias_bytes = 0
                chunk_folded = chunk.casefold()
                for alias in aliases:
                    size = len(alias.encode("utf-8")) + 2
                    if alias and alias.casefold() in chunk_folded and alias_bytes + size <= 4096:
                        chunk_aliases.append(alias)
                        alias_bytes += size
                metadata = {"mal_id": row["mal_id"], "title": row["title"], "type": row["type"],
                            "genres": row["genres"], "chunk_text": chunk,
                            "character_aliases": "; ".join(chunk_aliases)}
                records.append({"id": f"{row['type']}-{row['mal_id']}-chunk-{n:03d}", "metadata": metadata})
                texts.append(chunk)
        LOG.info("Embedding titles=%d vectors=%d batch_size=%d", len(batch), len(texts), args.embed_batch_size)
        with ThreadPoolExecutor(max_workers=1) as pool:
            sparse_pending = pool.submit(bm25.encode_documents, texts)
            dense_vectors = embed_ordered(model, texts, args.embed_batch_size)
            sparse_vectors = sparse_pending.result()
        for record, dense, sparse in zip(records, dense_vectors, sparse_vectors, strict=True):
            record.update(values=dense.tolist(), sparse_values=sparse)
        for offset in range(0, len(records), 100):
            index.upsert(vectors=records[offset:offset + 100], namespace=namespace)
        db.executemany("UPDATE documents SET row_json=? WHERE key=?", sanitized)
        db.executemany("UPDATE documents SET upserted=1 WHERE key=?", [(key,) for key, _, _ in batch])
        db.commit()
        LOG.info("Upserted titles=%d/%d vectors_in_batch=%d", start + len(batch), len(rows), len(records))


def bundle(db: sqlite3.Connection) -> None:
    # Transliteration happens offline; the serving VM only reads stored aliases.
    from korean_romanizer.romanizer import Romanizer
    path = Path("rag/data/catalog.jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    count = 0
    with_characters = 0
    with_biographies = 0
    with temp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        for body, enriched in db.execute("SELECT row_json,enriched FROM documents WHERE upserted=1 OR json_extract(row_json, '$.vector_exists')=1 ORDER BY key"):
            row = json.loads(body)
            if excluded_genres(row):
                continue
            names = extract_section(row.get("text", ""), "Characters")
            row["character_roster_checked"] = bool(enriched)
            row["characters_enriched"] = bool((enriched or row.get("characters")) and names and names != "Unknown")
            with_characters += int(row["characters_enriched"])
            with_biographies += int(bool(extract_section(row.get("text", ""), "Character Details")))
            # Keep the full biographies in Pinecone and the disk index, not in
            # every in-memory title record on the 1 GB serving VM.
            roster_names = [n.strip() for n in names.split(";") if n.strip() and n.strip() != "Unknown"]
            aliases = [a.strip() for a in extract_section(row["text"], "Character Search Aliases").split(";") if a.strip() and a.strip() != "Unknown"]
            row["character_count"] = len(row.get("characters") or roster_names)
            row["text"] = compose(row, row["text"], roster_names[:24])
            row["text"] = row["text"].rsplit("Character Search Aliases:\n", 1)[0] + "Character Search Aliases:\n" + ("; ".join(aliases[:100]) or "Unknown")
            row.pop("characters", None)
            native = row.get("alternative_titles", {}).get("ja", "")
            if re.search(r"[\uac00-\ud7a3]", native):
                row["romanized_titles"] = [Romanizer(native).romanize()]
            zipped.write((json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode())
            count += 1
    temp.replace(path)
    write_json(path.with_name("catalog_manifest.json"), {
        "schema_version": 1,
        "source": "MAL API v2 ranking catalog",
        "eligible_titles": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "indexed_titles": count,
        "character_pages_checked": db.execute("SELECT COUNT(*) FROM documents WHERE enriched=1").fetchone()[0],
        "titles_with_character_names": with_characters,
        "titles_with_character_biographies": with_biographies,
        "structured_character_rosters": db.execute("SELECT COUNT(DISTINCT title_key) FROM title_characters").fetchone()[0] if db.execute("SELECT 1 FROM sqlite_master WHERE name='title_characters'").fetchone() else 0,
        "titles_with_full_mal_details": db.execute("SELECT COUNT(*) FROM documents WHERE json_extract(row_json,'$.mal_details_schema_version')>=1").fetchone()[0],
        "unique_character_profiles": db.execute("SELECT COUNT(*) FROM characters").fetchone()[0] if db.execute("SELECT 1 FROM sqlite_master WHERE name='characters'").fetchone() else 0,
        "excluded_genres": ["Boys Love", "Girls Love", "Erotica", "Hentai"],
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "bm25_vocabulary": "rag/ingestion/bm25_values.json",
    })
    LOG.info("Deployable catalog: titles=%d bytes=%d path=%s", count, path.stat().st_size, path)
    from rag.evidence_index import build
    evidence = DATA / "evidence.sqlite"
    build(path, evidence, source_db=db)
    packed = path.with_name("evidence.sqlite.gz")
    staging = packed.with_suffix(".tmp")
    with evidence.open("rb") as source, staging.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        shutil.copyfileobj(source, compressed)
    staging.replace(packed)
    LOG.info("Bundled disk-backed evidence index")


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["cleanup", "collect", "enrich", "upsert", "bundle", "status"])
    parser.add_argument("--database", type=Path, default=DATA / "catalog.sqlite")
    parser.add_argument("--apply", action="store_true", help="Apply targeted remote deletions; otherwise audit only")
    parser.add_argument("--media", nargs="+", default=["manga", "anime"], choices=["manga", "anime"])
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="Refresh existing MAL fields from the first ranking page")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=128)
    parser.add_argument("--provider", default="CPUExecutionProvider", help="Offline embedding provider; never affects the serving runtime")
    parser.add_argument("--fp16", action="store_true", help="Offline GPU acceleration of the same BGE model; checks CPU vector compatibility before writing")
    args = parser.parse_args()
    if args.document_batch_size < 1 or args.embed_batch_size < 1 or args.limit < 0:
        parser.error("Batch sizes must be positive and limits must be nonnegative")
    if args.command == "cleanup":
        cleanup(args)
        return
    with open_db(args.database) as db:
        if args.command == "collect":
            collect(db, args)
        elif args.command == "enrich":
            enrich(db, args)
        elif args.command == "upsert":
            upsert(db, args)
        elif args.command == "bundle":
            seed(db)
            bundle(db)
        else:
            print(json.dumps({"total": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                              "by_stage": db.execute("SELECT enriched,upserted,COUNT(*) FROM documents GROUP BY enriched,upserted").fetchall(),
                              "pagination": db.execute("SELECT * FROM state").fetchall()}))


if __name__ == "__main__":
    main()
