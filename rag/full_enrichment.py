"""Checkpointed, complete MAL catalog details and cast/profile collection, offline only.

No per-title cast limit, supporting-character exclusion, or biography truncation.
Raw responses stay local. Publishing is deliberately separate from collection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import html
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from dotenv import load_dotenv

from rag.catalog_policy import excluded_genres
from rag.catalog_sync import MalClient, open_db, store, write_json
from rag.character_data import ProfileParser, RosterParser, schema, update_title

LOG = logging.getLogger(__name__)
VERSION = 1
COMMON_FIELDS = "id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,genres,created_at,updated_at,media_type,status,pictures,background,related_anime,related_manga,recommendations"
DETAIL_FIELDS = {
    "anime": COMMON_FIELDS + ",num_episodes,start_season,broadcast,source,average_episode_duration,rating,studios,statistics",
    "manga": COMMON_FIELDS + ",num_volumes,num_chapters,authors{first_name,last_name},serialization{name}",
}
USER_AGENT = "RAGnarok-catalog/1.0"


def setup(db, top_titles=0):
    schema(db)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS mal_details (title_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS mal_pages (kind TEXT NOT NULL, key TEXT NOT NULL, url TEXT NOT NULL, html_gzip BLOB NOT NULL, PRIMARY KEY(kind,key));
        CREATE TABLE IF NOT EXISTS enrichment_tasks (
            kind TEXT NOT NULL, key TEXT NOT NULL, priority INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            checked_at REAL, retry_at REAL NOT NULL DEFAULT 0, error TEXT,
            PRIMARY KEY(kind,key));
        CREATE INDEX IF NOT EXISTS enrichment_pending ON enrichment_tasks(kind,status,priority,key);
        CREATE TEMP TABLE IF NOT EXISTS full_priorities (key TEXT PRIMARY KEY, priority INTEGER NOT NULL);
        DELETE FROM full_priorities;
    """)
    # Rank each medium separately so a manga with no ranking-list rank does not
    # wait behind the entire anime catalog. Within a medium prefer MAL rank/score.
    db.execute("""INSERT INTO full_priorities
        WITH ranked AS (SELECT key,json_extract(row_json,'$.type') AS medium,
            json_extract(row_json,'$.catalog_rank') AS mal_rank,
            COALESCE(json_extract(row_json,'$.score'),0) AS score,
            ROW_NUMBER() OVER (PARTITION BY json_extract(row_json,'$.type')
                ORDER BY COALESCE(json_extract(row_json,'$.score'),0) DESC,key) AS score_rank
            FROM documents)
        SELECT key,ROW_NUMBER() OVER (PARTITION BY medium
            ORDER BY COALESCE(mal_rank,score_rank),score DESC,key) FROM ranked""")
    if top_titles:
        # Interleave medium-specific ranks, rather than filling the budget with
        # anime before reaching manga that is missing a ranking-list rank.
        db.execute("""DELETE FROM full_priorities WHERE key NOT IN (
            SELECT key FROM full_priorities ORDER BY priority,key LIMIT ?)
        """, (top_titles,))
    for kind in ("details", "roster"):
        db.execute("""INSERT OR IGNORE INTO enrichment_tasks(kind,key,priority)
            SELECT ?,key,priority FROM full_priorities""", (kind,))
        db.execute("""UPDATE enrichment_tasks SET priority=(SELECT priority FROM full_priorities p WHERE p.key=enrichment_tasks.key)
            WHERE kind=? AND key IN (SELECT key FROM full_priorities)""", (kind,))
    seed_profiles(db)
    db.commit()


def seed_profiles(db):
    # Deduplicate shared characters across all titles, including supporting roles.
    db.execute("""INSERT OR IGNORE INTO enrichment_tasks(kind,key,priority)
        SELECT 'profile',CAST(t.character_id AS TEXT),MIN(r.priority)
        FROM title_characters t JOIN documents d ON d.key=t.title_key
        JOIN full_priorities scope ON scope.key=t.title_key
        JOIN enrichment_tasks r ON r.kind='roster' AND r.key=t.title_key
        WHERE COALESCE(json_extract(d.row_json,'$.excluded_by_policy'),0)=0
            AND NOT EXISTS (SELECT 1 FROM characters c WHERE c.mal_id=t.character_id)
        GROUP BY t.character_id""")


def verify_robots(client):
    response = client.get("https://myanimelist.net/robots.txt", page=True)
    rules = RobotFileParser()
    rules.parse(response.text.splitlines())
    for path in ("/anime/1/_/characters", "/manga/1/_/characters", "/character/1/_"):
        if not rules.can_fetch(USER_AGENT, "https://myanimelist.net" + path):
            raise RuntimeError("MAL robots.txt disallows catalog collection; stopped without bypassing it")


def save_page(db, kind, key, response):
    if urlsplit(str(response.url)).hostname not in {"myanimelist.net", "www.myanimelist.net"}:
        raise ValueError("MAL redirected to an unexpected host")
    db.execute("INSERT OR REPLACE INTO mal_pages VALUES(?,?,?,?)",
               (kind, key, str(response.url), gzip.compress(response.content, mtime=0)))


def verify_page_identity(response, medium, identifier):
    match = re.match(r"^/(anime|manga|character)/(\d+)(?:/|$)", urlsplit(str(response.url)).path)
    if not match or match[1] != medium or int(match[2]) != int(identifier):
        raise ValueError("MAL redirected to a different title or character; response not imported")


def apply_details(db, key, node):
    body, enriched, indexed = db.execute("SELECT row_json,enriched,upserted FROM documents WHERE key=?", (key,)).fetchone()
    row = json.loads(body)
    if node.get("id") != row["mal_id"] or not node.get("title"):
        raise ValueError(f"MAL detail identity mismatch for {key}")
    db.execute("INSERT OR REPLACE INTO mal_details VALUES(?,?)", (key, json.dumps(node, ensure_ascii=True)))
    before = row.get("text", "")
    row.update(title=node["title"], alternative_titles=node.get("alternative_titles") or {},
               synopsis=html.unescape(node.get("synopsis") or ""),
               genres=[g["name"] for g in node.get("genres", [])],
               display_type=node.get("media_type"), status=node.get("status"),
               score=node.get("mean"), chapters=node.get("num_chapters"),
               volumes=node.get("num_volumes"), episodes=node.get("num_episodes"))
    if node.get("rank"):
        row["catalog_rank"] = node["rank"]
    picture = node.get("main_picture") or {}
    row["image_url"] = picture.get("large") or picture.get("medium") or row.get("image_url")
    authors = [" ".join(filter(None, (a.get("node", {}).get("first_name"), a.get("node", {}).get("last_name")))) for a in node.get("authors", [])]
    details = {
        "Format": (node.get("media_type") or "").replace("_", " "),
        "Source material": (node.get("source") or "").replace("_", " "),
        "Authors": ", ".join(filter(None, authors)),
        "Studios": ", ".join(s["name"] for s in node.get("studios", [])),
        "Serialization": ", ".join(s.get("node", {}).get("name", "") for s in node.get("serialization", [])),
        "Started": node.get("start_date"), "Ended": node.get("end_date"),
        "Status": (node.get("status") or "").replace("_", " "),
        "Season": " ".join(str(v) for v in (node.get("start_season") or {}).values()),
        "Background": html.unescape(node.get("background") or ""),
    }
    for field, label in (("num_chapters", "Chapters"), ("num_volumes", "Volumes"), ("num_episodes", "Episodes")):
        if node.get(field):
            details[label] = str(node[field])
    if node.get("average_episode_duration"):
        details["Episode duration"] = f"{node['average_episode_duration']} seconds"
    if node.get("rating"):
        details["Age rating"] = node["rating"]
    # Keep sequel/adaptation identity structured; don't turn related works into aliases.
    row.update(search_details=details, mal_details_schema_version=VERSION,
               mal_fields_returned=sorted(node), metadata_source="mal_api_v2_details")
    if excluded_genres(row):
        db.execute("INSERT OR REPLACE INTO excluded VALUES(?,?)", (key, json.dumps(row)))
        row["excluded_by_policy"] = True
        for kind in ("roster",):
            db.execute("UPDATE enrichment_tasks SET status='excluded' WHERE kind=? AND key=?", (kind, key))
    store(db, row, enriched=bool(enriched), upserted=bool(indexed))
    update_title(db, key, complete=True)
    # Metadata-only changes still require refreshed vectors when text changes.
    after = json.loads(db.execute("SELECT row_json FROM documents WHERE key=?", (key,)).fetchone()[0])["text"]
    if before != after:
        db.execute("UPDATE documents SET upserted=0 WHERE key=?", (key,))


def collect_details(db, client, key):
    medium, identifier = key.split(":")
    response = client.get(f"{medium}/{identifier}", params={"fields": DETAIL_FIELDS[medium]})
    apply_details(db, key, response.json())
    return "done"


def collect_roster(db, client, key):
    medium, identifier = key.split(":")
    response = client.get(f"https://myanimelist.net/{medium}/{identifier}/_/characters", page=True)
    verify_page_identity(response, medium, identifier)
    parser = RosterParser()
    parser.feed(response.text)
    # A challenge page or parser break must never be recorded as an empty roster.
    if not parser.records and "No characters" not in response.text:
        raise ValueError(f"Unrecognized MAL roster for {key}")
    save_page(db, "roster", key, response)
    db.execute("DELETE FROM title_characters WHERE title_key=?", (key,))
    db.executemany("INSERT INTO title_characters VALUES(?,?,?,?)",
                   [(key, r["mal_id"], r["name"], r["role"]) for r in parser.records.values()])
    db.execute("UPDATE documents SET enriched=1 WHERE key=?", (key,))
    update_title(db, key, complete=True)
    return "done" if parser.records else "empty"


def collect_profile(db, client, key):
    response = client.get(f"https://myanimelist.net/character/{int(key)}/_", page=True)
    verify_page_identity(response, "character", key)
    parser = ProfileParser()
    parser.feed(response.text)
    profile = parser.result()
    if not profile["heading"]:
        raise ValueError(f"Unrecognized MAL character profile {key}")
    save_page(db, "profile", key, response)
    profile.update(mal_id=int(key), source_url=str(response.url), profile_schema_version=2)
    db.execute("INSERT OR REPLACE INTO characters VALUES(?,?)", (int(key), json.dumps(profile, ensure_ascii=True)))
    for title_key, in db.execute("SELECT title_key FROM title_characters WHERE character_id=?", (int(key),)).fetchall():
        update_title(db, title_key, complete=True)
    return "done" if profile["about"] else "empty"


HANDLERS = {"details": collect_details, "roster": collect_roster, "profile": collect_profile}


def run_batch(db, client, kind, limit):
    tasks = db.execute("""SELECT key FROM enrichment_tasks WHERE kind=?
        AND (status='pending' OR (status='failed' AND retry_at<=?))
        AND ((kind='profile' AND NOT EXISTS (
            SELECT 1 FROM characters c WHERE c.mal_id=CAST(enrichment_tasks.key AS INTEGER))
            AND EXISTS (SELECT 1 FROM title_characters t JOIN full_priorities p
                ON p.key=t.title_key WHERE t.character_id=CAST(enrichment_tasks.key AS INTEGER)))
          OR (kind!='profile' AND key IN (SELECT key FROM full_priorities)))
        AND (kind!='roster' OR NOT EXISTS (
            SELECT 1 FROM title_characters t WHERE t.title_key=enrichment_tasks.key))
        ORDER BY priority,key LIMIT ?""", (kind, time.time(), limit)).fetchall()
    for key, in tasks:
        status, error = "done", None
        try:
            with db:
                allowed = True
                if kind == "profile":
                    allowed = bool(db.execute("""SELECT 1 FROM title_characters t JOIN documents d ON d.key=t.title_key
                        WHERE t.character_id=? AND COALESCE(json_extract(d.row_json,'$.excluded_by_policy'),0)=0 LIMIT 1""", (int(key),)).fetchone())
                status = HANDLERS[kind](db, client, key) if allowed else "excluded"
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            status = "missing" if code in {404, 410} else "failed"
            error = f"HTTP {code}"
            if code in {401, 403}:
                # Do not evade access restrictions or keep hammering rejected credentials.
                raise RuntimeError(f"MAL rejected {kind} access (HTTP {code}); collection paused") from None
        except (httpx.TransportError, ValueError, RuntimeError) as exc:
            status, error = "failed", f"{type(exc).__name__}: {str(exc)[:200]}"
        with db:
            db.execute("""UPDATE enrichment_tasks SET status=?,attempts=attempts+1,
                checked_at=?,retry_at=?,error=? WHERE kind=? AND key=?""",
                (status, time.time(), time.time() + 3600 if error else 0, error, kind, key))
        LOG.info("%s %s %s%s", kind, key, status, f" ({error})" if error else "")
        if status == "failed":
            # A broken page is isolated, but repeated upstream trouble must slow the job.
            time.sleep(10)
    if kind == "roster":
        with db:
            seed_profiles(db)
    return len(tasks)


def coverage(db):
    stages = {kind: {status: n for status, n in db.execute("SELECT status,COUNT(*) FROM enrichment_tasks WHERE kind=? GROUP BY status", (kind,))} for kind in HANDLERS}
    total = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    complete = db.execute("""SELECT COUNT(*) FROM enrichment_tasks d
        JOIN enrichment_tasks r ON r.key=d.key AND r.kind='roster'
        WHERE d.kind='details' AND d.status='done' AND r.status IN ('done','empty')
        AND NOT EXISTS (SELECT 1 FROM title_characters t
            LEFT JOIN enrichment_tasks p ON p.kind='profile' AND p.key=CAST(t.character_id AS TEXT)
            WHERE t.title_key=d.key AND COALESCE(p.status,'pending') NOT IN ('done','empty'))""").fetchone()[0]
    return {"schema_version": VERSION, "updated_at": datetime.now(timezone.utc).isoformat(),
            "titles": total, "fully_checked_titles": complete, "stages": stages,
            "profiles_stored": db.execute("SELECT COUNT(*) FROM characters").fetchone()[0],
            "title_character_links": db.execute("SELECT COUNT(*) FROM title_characters").fetchone()[0],
            "pending_vector_updates": db.execute("SELECT COUNT(*) FROM documents WHERE upserted=0").fetchone()[0],
            "published": False}


@contextmanager
def exclusive_worker(database):
    # OS-held locks recover after crashes; a stale PID file cannot block resumption.
    path = database.with_suffix(".enrichment.lock")
    handle = path.open("a+b")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another full-enrichment worker already owns this catalog") from None
    try:
        yield
    finally:
        handle.close()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--database", type=Path, default=Path("rag/ingestion/catalog.sqlite"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--interval", type=float, default=1.1)
    parser.add_argument("--max-cycles", type=int, default=0, help="Optional verification bound; 0 processes the full catalog")
    parser.add_argument("--top-titles", type=int, default=10000, help="Balanced MAL ranking scope; existing data outside this scope is retained")
    parser.add_argument("--stages", nargs="+", choices=list(HANDLERS), default=["roster", "profile"], help="Default fills character gaps without refreshing the entire metadata catalog")
    args = parser.parse_args()
    if args.interval < 1.0 or args.batch_size < 1 or args.max_cycles < 0 or args.top_titles < 1:
        parser.error("Use interval >= 1 second, positive batch size, and nonnegative max cycles")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        with open_db(args.database) as db:
            print(json.dumps(coverage(db), indent=2))
        return
    with exclusive_worker(args.database), open_db(args.database) as db:
        setup(db, args.top_titles)
        client = MalClient(interval=args.interval)
        progress_path = args.database.with_name("full_enrichment_progress.json")
        write_json(progress_path, {**coverage(db), "worker_status": "starting", "pid": os.getpid()})
        cycle = 0
        try:
            verify_robots(client)
            while True:
                processed = 0
                for kind in args.stages:
                    write_json(progress_path, {**coverage(db), "worker_status": "running", "stage": kind, "pid": os.getpid()})
                    processed += run_batch(db, client, kind, args.batch_size)
                    write_json(progress_path, {**coverage(db), "worker_status": "running", "stage": kind, "pid": os.getpid()})
                cycle += 1
                if args.max_cycles and cycle >= args.max_cycles:
                    break
                if not processed:
                    # Leave delayed retries checkpointed; cached/out-of-scope
                    # tasks must not keep a completed scoped run alive forever.
                    break
        except BaseException as exc:
            write_json(progress_path, {**coverage(db), "worker_status": "paused", "error": f"{type(exc).__name__}: {str(exc)[:200]}", "pid": os.getpid()})
            raise
        finally:
            client.api.close()
            client.web.close()
        result = {**coverage(db), "worker_status": "checkpointed" if args.max_cycles else "finished", "pid": os.getpid()}
        write_json(progress_path, result)
        LOG.info("Collection checkpoint: %s", json.dumps(result))
        LOG.info("Raw catalog collection is separate from Pinecone upsert and deployable bundles")


if __name__ == "__main__":
    main()
