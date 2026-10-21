"""Validated, version-pinned MAL bulk snapshots. Offline ingestion only."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit
import zipfile

import httpx

from rag.catalog_policy import excluded_genres
from rag.catalog_sync import open_db, store, write_json
from rag.character_data import schema, update_title

LOG = logging.getLogger(__name__)
ROOT = Path("rag/ingestion/bulk")
SOURCES = {
    "profiles": ("syahrulapriansyah2/anime-characters", 1, "MIT", {"mal_characters_detailed.csv": {"character_id", "full_name", "url", "attributes", "description"}}),
    "casts": ("syahrulapriansyah2/myanimelist-2025", 1, "MIT", {"mal_anime.csv": {"myanimelist_id", "source_url", "characters"}}),
    "metadata": ("hamzaashfaque1999/myanimelist-scraped-data", 1, "MIT", {"anime_entries.csv": {"id", "link", "title_name", "themes"}, "manga_entries.csv": {"id", "link", "title_name", "themes"}}),
    "additional_profiles": ("andreuvallhernndez/myanimelist-jikan", 1, "CC0: Public Domain", {"characters.csv": {"mal_id", "name", "about", "url"}}),
}


def listed(value):
    if not value or value in {"null", "None", "nan"}:
        return []
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        result = ast.literal_eval(value)
    if not isinstance(result, list):
        raise ValueError("Expected structured list")
    return result


def text_values(value):
    # Some publisher columns are scalar text, not serialized lists.
    return listed(value) if value and value.lstrip().startswith("[") else [value] if value else []


def identity(url, medium, identifier):
    parsed = urlsplit(url)
    match = re.match(r"^/(anime|manga|character)/(\d+)(?:/|$)", parsed.path)
    return parsed.scheme == "https" and parsed.hostname == "myanimelist.net" and bool(match and match[1] == medium and int(match[2]) == int(identifier))


def normalized(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKD", value).casefold()))


def rows(archive, filename):
    csv.field_size_limit(8 * 1024 * 1024)
    with zipfile.ZipFile(archive) as zipped, zipped.open(filename) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))


def download(name):
    ref, version, license_name, expected = SOURCES[name]
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / f"{name}-v{version}.zip"
    manifest = target.with_suffix(".json")
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        response = client.get("https://www.kaggle.com/api/v1/datasets/view/" + ref)
        response.raise_for_status()
        info = response.json()
        if info.get("licenseName") != license_name:
            raise ValueError(f"Source license changed for {name}; review required")
        if not target.exists():
            staging = target.with_suffix(".downloading")
            size = 0
            with client.stream("GET", "https://www.kaggle.com/api/v1/datasets/download/" + ref,
                               params={"datasetVersionNumber": version}) as result:
                result.raise_for_status()
                with staging.open("wb") as handle:
                    for block in result.iter_bytes(1024 * 1024):
                        size += len(block)
                        if size > 180 * 1024 * 1024:
                            raise ValueError("Unexpected bulk archive size; stopped")
                        handle.write(block)
            staging.replace(target)
    with zipfile.ZipFile(target) as zipped:
        for filename, required in expected.items():
            with zipped.open(filename) as raw:
                fields = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")).fieldnames or []
                if not required <= set(fields):
                    raise ValueError(f"Unexpected {name}/{filename} schema: {fields}")
        bad = zipped.testzip()
        if bad:
            raise ValueError(f"Corrupt archive member: {bad}")
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != digest:
        raise ValueError(f"Cached snapshot changed for {name}")
    write_json(manifest, {"source": ref, "version": version, "license": license_name,
                          "url": "https://www.kaggle.com/datasets/" + ref,
                          "last_updated": info.get("lastUpdated"), "sha256": digest,
                          "bytes": target.stat().st_size})
    LOG.info("Downloaded and CRC/schema-verified %s bytes=%d", name, target.stat().st_size)
    return name, target


def profile_from_row(row, source):
    identifier = int(row.get("character_id") or row["mal_id"])
    if identifier < 1 or not identity(row["url"], "character", identifier):
        raise ValueError("Character ID/URL mismatch")
    name = row.get("full_name") or row.get("name") or ""
    aliases = list(filter(None, [name, row.get("name"), row.get("alternate_name"), row.get("name_kanji")]))
    aliases.extend(listed(row.get("nicknames", "")))
    attributes = listed(row.get("attributes", ""))
    if not all(isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("value"), str) for a in attributes):
        raise ValueError("Invalid character attributes")
    description = row.get("description") or row.get("about") or ""
    if re.fullmatch(r"\s*No biography (?:written|available)\.?\s*", description, re.I):
        description = ""
    about = "\n".join([*(f"{a['name']}: {a['value']}" for a in attributes), description]).strip()
    if not name:
        raise ValueError("Missing character name")
    return {"mal_id": identifier, "heading": name, "name": name, "about": about,
            "aliases": list(dict.fromkeys(aliases)), "attributes": attributes,
            "image_url": row.get("main_picture"), "source_url": row["url"],
            "source_dataset": SOURCES[source][0], "profile_schema_version": 2}


def audit(paths):
    report = {"sources": {}, "invalid_rows": [], "valid": True}
    for source in ("profiles", "additional_profiles"):
        filename = next(iter(SOURCES[source][3]))
        seen, duplicate, biographies, count = set(), 0, 0, 0
        for number, row in enumerate(rows(paths[source], filename), 2):
            count += 1
            try:
                profile = profile_from_row(row, source)
                duplicate += int(profile["mal_id"] in seen)
                seen.add(profile["mal_id"])
                biographies += int(bool(profile["about"]))
            except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                report["invalid_rows"].append({"source": source, "line": number, "error": str(exc)[:120]})
        report["sources"][source] = {"rows": count, "unique_ids": len(seen), "duplicates": duplicate, "with_biographies": biographies}
    for source in ("casts", "metadata"):
        for filename in SOURCES[source][3]:
            medium = "manga" if filename.startswith("manga") else "anime"
            count, links = 0, 0
            for number, row in enumerate(rows(paths[source], filename), 2):
                count += 1
                try:
                    identifier = int(row.get("myanimelist_id") or row["id"])
                    if not identity(row.get("source_url") or row["link"], medium, identifier):
                        raise ValueError("Title ID/URL mismatch")
                    if source == "casts":
                        for character in listed(row["characters"]):
                            if not identity(character["url"], "character", int(character["id"])) or not character.get("name"):
                                raise ValueError("Invalid cast identity")
                            links += 1
                    else:
                        for field in ("themes", "genres", "producers", "licensors", "studios", "authors", "serialization", "demographic"):
                            if row.get(field):
                                text_values(row[field])
                except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                    report["invalid_rows"].append({"source": source, "file": filename, "line": number, "error": str(exc)[:120]})
            report["sources"][source + "/" + filename] = {"rows": count, "cast_links": links}
    # Invalid records are quarantined by identity validation during import, never guessed.
    report["valid"] = len(report["invalid_rows"]) == 0
    write_json(ROOT / "audit.json", report)
    return report


def import_all(db, paths):
    schema(db)
    db.execute("CREATE TABLE IF NOT EXISTS bulk_rows (source TEXT, key TEXT, payload_json TEXT, PRIMARY KEY(source,key))")
    touched = set()
    stats = {"profiles_imported": 0, "live_profiles_preserved": 0, "cast_titles_added": 0, "cast_links_added": 0,
             "metadata_titles_enriched": 0, "excluded_titles_skipped": 0, "identity_conflicts": 0}
    # Prefer the newer MIT profile over the older supplementary snapshot, but
    # never replace a directly fetched current MAL profile with an old dump.
    for source in ("additional_profiles", "profiles"):
        filename = next(iter(SOURCES[source][3]))
        for row in rows(paths[source], filename):
            profile = profile_from_row(row, source)
            identifier = profile["mal_id"]
            existing = db.execute("SELECT row_json FROM characters WHERE mal_id=?", (identifier,)).fetchone()
            old = json.loads(existing[0]) if existing else {}
            if old.get("source_url") and not old.get("source_dataset"):
                stats["live_profiles_preserved"] += 1
                continue
            if old.get("about") and not profile["about"]:
                continue
            profile["aliases"] = list(dict.fromkeys(old.get("aliases", []) + profile["aliases"]))
            profile["image_url"] = profile.get("image_url") or old.get("image_url")
            db.execute("INSERT OR REPLACE INTO characters VALUES(?,?)", (identifier, json.dumps(profile, ensure_ascii=True)))
            stats["profiles_imported"] += 1
            if stats["profiles_imported"] % 1000 == 0:
                db.commit()
        db.commit()
        LOG.info("Profiles imported from %s", source)
    for row in rows(paths["casts"], "mal_anime.csv"):
        key = f"anime:{int(row['myanimelist_id'])}"
        existing = db.execute("SELECT row_json FROM documents WHERE key=?", (key,)).fetchone()
        if not existing:
            continue
        title = json.loads(existing[0])
        if excluded_genres(title):
            stats["excluded_titles_skipped"] += 1
            continue
        count = 0
        for character in listed(row["characters"]):
            db.execute("INSERT OR IGNORE INTO title_characters VALUES(?,?,?,?)", (key, int(character["id"]), character["name"], "Unknown"))
            count += db.execute("SELECT changes()").fetchone()[0]
        if count:
            stats["cast_titles_added"] += 1
            stats["cast_links_added"] += count
            touched.add(key)
        if len(touched) % 500 == 0:
            db.commit()
    db.commit()
    LOG.info("Imported explicit anime/character links; no anime-to-manga inference")
    for filename in SOURCES["metadata"][3]:
        medium = "manga" if filename.startswith("manga") else "anime"
        for row in rows(paths["metadata"], filename):
            key = f"{medium}:{int(row['id'])}"
            existing = db.execute("SELECT row_json,enriched,upserted FROM documents WHERE key=?", (key,)).fetchone()
            if not existing:
                continue
            title = json.loads(existing[0])
            if excluded_genres(title) or excluded_genres({"genres": listed(row.get("genres", ""))}):
                stats["excluded_titles_skipped"] += 1
                continue
            aliases = [title["title"], *title.get("alternative_titles", {}).get("synonyms", []), *[v for v in title.get("alternative_titles", {}).values() if isinstance(v, str)]]
            source_aliases = [row["title_name"], row.get("english_name", "")]
            if not {normalized(a) for a in aliases} & {normalized(a) for a in source_aliases if a}:
                stats["identity_conflicts"] += 1
                continue
            db.execute("INSERT OR REPLACE INTO bulk_rows VALUES(?,?,?)", ("metadata", key, json.dumps(row, ensure_ascii=True)))
            details = title.setdefault("search_details", {})
            before = json.dumps(details, sort_keys=True)
            for field, label in (("themes", "Themes"), ("demographic", "Demographic"), ("producers", "Producers"), ("licensors", "Licensors"), ("studios", "Studios"), ("authors", "Authors"), ("serialization", "Serialization")):
                if not details.get(label) and row.get(field):
                    values = text_values(row[field])
                    details[label] = ", ".join(v if isinstance(v, str) else v.get("name", "") for v in values)
            if not details.get("Background") and row.get("background"):
                details["Background"] = row["background"]
            if before != json.dumps(details, sort_keys=True):
                store(db, title, enriched=bool(existing[1]), upserted=bool(existing[2]))
                touched.add(key)
                stats["metadata_titles_enriched"] += 1
            if stats["metadata_titles_enriched"] % 500 == 0:
                db.commit()
        db.commit()
        LOG.info("Missing metadata filled from %s", filename)
    touched.update(key for key, in db.execute("SELECT DISTINCT title_key FROM title_characters"))
    for count, key in enumerate(sorted(touched), 1):
        body = db.execute("SELECT row_json FROM documents WHERE key=?", (key,)).fetchone()
        if body and not excluded_genres(json.loads(body[0])):
            update_title(db, key, complete=True)
        if count % 100 == 0:
            db.commit()
            LOG.info("Recomposed full enriched texts %d/%d", count, len(touched))
    db.commit()
    stats["profiles_stored"] = db.execute("SELECT COUNT(*) FROM characters").fetchone()[0]
    stats["rosters_stored"] = db.execute("SELECT COUNT(DISTINCT title_key) FROM title_characters").fetchone()[0]
    stats["biographies_available"] = db.execute("SELECT COUNT(*) FROM characters WHERE length(json_extract(row_json,'$.about'))>0").fetchone()[0]
    stats["pending_vector_updates"] = db.execute("SELECT COUNT(*) FROM documents WHERE upserted=0").fetchone()[0]
    write_json(ROOT / "import.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "audit", "import"])
    parser.add_argument("--database", type=Path, default=Path("rag/ingestion/catalog.sqlite"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "download":
        with ThreadPoolExecutor(max_workers=4) as pool:
            print(json.dumps({name: str(path) for name, path in pool.map(download, SOURCES)}))
        return
    paths = {name: ROOT / f"{name}-v{source[1]}.zip" for name, source in SOURCES.items()}
    report = audit(paths)
    print(json.dumps(report))
    if args.command == "import":
        if not report["valid"]:
            raise RuntimeError("Bulk validation failed; no catalog import performed")
        with open_db(args.database) as db:
            print(json.dumps(import_all(db, paths)))


if __name__ == "__main__":
    main()
