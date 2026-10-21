"""Offline MAL character rosters and deduplicated biographies; never a serving dependency."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv

from rag.catalog_policy import extract_section
from rag.catalog_sync import MalClient, compose, name_aliases, open_db, store, write_json

LOG = logging.getLogger(__name__)


class RosterParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.records = {}
        self.link = None
        self.name = None
        self.parts = []
        self.pending = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            match = re.match(r"(?:https://myanimelist\.net)?/character/(\d+)/[^?#]*", attrs.get("href") or "")
            self.link = (int(match[1]), attrs["href"]) if match else None
        if tag == "h3" and "h3_character_name" in (attrs.get("class") or "") and self.link:
            self.name, self.parts = self.link, []

    def handle_data(self, data):
        if self.name:
            self.parts.append(data)
        elif self.pending and data.strip() in {"Main", "Supporting"}:
            self.records[self.pending]["role"] = data.strip()
            self.pending = None

    def handle_endtag(self, tag):
        if tag == "h3" and self.name:
            identifier, url = self.name
            name = " ".join("".join(self.parts).split())
            if name:
                self.records[identifier] = {"mal_id": identifier, "name": name, "url": url, "role": "Unknown"}
                self.pending = identifier
            self.name = None
        if tag == "a":
            self.link = None


class ProfileParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.header = False
        self.active = False
        self.ignore = 0
        self.names = []
        self.parts = []
        self.image_url = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img" and "portrait-" in (attrs.get("class") or ""):
            self.image_url = attrs.get("data-src") or attrs.get("src")
        if tag in {"script", "style"}:
            self.ignore += 1
        if "normal_header" in (attrs.get("class") or ""):
            self.active = False
            self.header = tag == "h2"
        if self.active and tag in {"br", "p", "div"} and not self.ignore:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.ignore:
            return
        if self.header:
            self.names.append(data)
        elif self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignore = max(0, self.ignore - 1)
        if tag == "h2" and self.header:
            self.header = False
            self.active = True

    def result(self):
        about = "\n".join(line.strip() for line in "".join(self.parts).splitlines() if line.strip())
        heading = " ".join("".join(self.names).split())
        aliases = [heading] if heading else []
        aliases.extend(re.findall(r'"([^"\n]+)"', heading))
        if heading:
            aliases.extend(filter(None, [re.sub(r"\s*\([^)]*\)\s*", " ", heading).strip(), *re.findall(r"\(([^)]+)\)", heading)]))
        return {"heading": heading, "aliases": list(dict.fromkeys(aliases)), "about": about, "image_url": self.image_url}


def schema(db):
    db.execute("CREATE TABLE IF NOT EXISTS characters (mal_id INTEGER PRIMARY KEY, row_json TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS title_characters (title_key TEXT NOT NULL, character_id INTEGER NOT NULL, name TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(title_key, character_id))")
    db.execute("CREATE INDEX IF NOT EXISTS character_titles ON title_characters(character_id)")
    db.commit()


def update_title(db, key, *, complete=False):
    body, checked, indexed = db.execute("SELECT row_json,enriched,upserted FROM documents WHERE key=?", (key,)).fetchone()
    row = json.loads(body)
    records = []
    for identifier, name, role, profile in db.execute("SELECT t.character_id,t.name,t.role,c.row_json FROM title_characters t LEFT JOIN characters c ON c.mal_id=t.character_id WHERE t.title_key=? ORDER BY t.role='Main' DESC,t.character_id", (key,)):
        character = {"mal_id": identifier, "name": name, "role": role}
        if profile:
            character.update(json.loads(profile))
        records.append(character)
    # Keep legacy roster names until the entire page has been refreshed.
    names = list(dict.fromkeys([r["name"] for r in records] + [n.strip() for n in extract_section(row.get("text", ""), "Characters").split(";") if n.strip() and n.strip() != "Unknown"]))
    text = compose(row, row.get("text", ""), names)
    if complete:
        text = text.rsplit("Character Search Aliases:\n", 1)[0] + "Character Search Aliases:\n" + ("; ".join(name_aliases(names, None)) or "Unknown")
    extra_aliases = [a for r in records for a in r.get("aliases", [])]
    if extra_aliases:
        text += "; " + "; ".join(name_aliases(extra_aliases, None if complete else 140))
    if complete:
        biographies = [f"{r['name']} ({r['role']}): {r['about']}" for r in records if r.get("about")]
    else:
        biographies = [f"{r['name']} ({r['role']}): {r['about'][:1800]}" for r in records if r.get("about") and r["role"] == "Main"][:4]
    if biographies:
        text += "\nCharacter Details:\n" + "\n".join(biographies)
    unchanged = row.get("text") == text
    row.update(text=text, characters=records)
    if not records and not checked:
        row.pop("character_schema_version", None)
        row.pop("character_source", None)
    if records or row.get("character_schema_version"):
        row["character_source"] = row.get("character_source", "mal_public_pages") if checked else "mal_bulk_snapshot"
        row["character_schema_version"] = 2 if complete else 1
    store(db, row, enriched=bool(checked), upserted=bool(indexed and unchanged))


def rosters(db, client, limit, refresh):
    where = "" if refresh else "AND COALESCE(json_extract(row_json,'$.character_schema_version'),0)<1"
    rows = db.execute(f"SELECT key,row_json FROM documents WHERE COALESCE(json_extract(row_json,'$.character_page_missing'),0)=0 {where} ORDER BY COALESCE(json_extract(row_json,'$.score'),0) DESC,COALESCE(json_extract(row_json,'$.catalog_rank'),999999),rowid LIMIT ?", (limit,)).fetchall()
    for count, (key, body) in enumerate(rows, 1):
        row = json.loads(body)
        response = client.get(f"https://myanimelist.net/{row['type']}/{row['mal_id']}/_/characters", page=True)
        parser = RosterParser()
        parser.feed(response.text)
        if not parser.records and "No characters" not in response.text:
            raise RuntimeError(f"Unrecognized roster {key}; existing data retained")
        db.execute("DELETE FROM title_characters WHERE title_key=?", (key,))
        db.executemany("INSERT INTO title_characters VALUES(?,?,?,?)", [(key, r["mal_id"], r["name"], r["role"]) for r in parser.records.values()])
        db.execute("UPDATE documents SET enriched=1 WHERE key=?", (key,))
        update_title(db, key)
        db.commit()
        LOG.info("Roster %d/%d %s characters=%d", count, len(rows), key, len(parser.records))


def profiles(db, client, limit):
    rows = db.execute("SELECT t.character_id,MIN(COALESCE(json_extract(d.row_json,'$.catalog_rank'),999999)) AS priority FROM title_characters t JOIN documents d ON d.key=t.title_key LEFT JOIN characters c ON c.mal_id=t.character_id WHERE c.mal_id IS NULL AND t.role='Main' GROUP BY t.character_id ORDER BY priority,t.character_id LIMIT ?", (limit,)).fetchall()
    for count, (identifier, _) in enumerate(rows, 1):
        url = f"https://myanimelist.net/character/{identifier}/_"
        response = client.get(url, page=True)
        parser = ProfileParser()
        parser.feed(response.text)
        profile = parser.result()
        if not profile["heading"]:
            raise RuntimeError(f"Unrecognized character {identifier}; checkpoint retained")
        profile.update(mal_id=identifier, source_url=url)
        db.execute("INSERT INTO characters VALUES(?,?)", (identifier, json.dumps(profile, ensure_ascii=True)))
        for key, in db.execute("SELECT title_key FROM title_characters WHERE character_id=?", (identifier,)).fetchall():
            update_title(db, key)
        db.commit()
        LOG.info("Profile %d/%d id=%d biography_chars=%d", count, len(rows), identifier, len(profile["about"]))


def export(db):
    path = Path("rag/ingestion/characters.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for identifier, body in db.execute("SELECT mal_id,row_json FROM characters ORDER BY mal_id"):
            row = json.loads(body)
            row["appearances"] = [{"title_key": key, "role": role} for key, role in db.execute("SELECT title_key,role FROM title_characters WHERE character_id=? ORDER BY title_key", (identifier,))]
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    stats = {"profiles": db.execute("SELECT COUNT(*) FROM characters").fetchone()[0], "title_character_links": db.execute("SELECT COUNT(*) FROM title_characters").fetchone()[0], "rosters": db.execute("SELECT COUNT(DISTINCT title_key) FROM title_characters").fetchone()[0], "path": str(path)}
    write_json(path.with_name("characters_manifest.json"), stats)
    print(json.dumps(stats))


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rosters", "profiles", "export"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--database", type=Path, default=Path("rag/ingestion/catalog.sqlite"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    with open_db(args.database) as db:
        schema(db)
        if args.command == "export":
            export(db)
        else:
            client = MalClient(interval=1.1)
            if args.command == "rosters":
                rosters(db, client, args.limit, args.refresh)
            else:
                profiles(db, client, args.limit)


if __name__ == "__main__":
    main()
