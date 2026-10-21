"""Offline-built, disk-backed field search. No additional service or model."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sqlite3
import hashlib
import shutil
import tempfile
import threading
import unicodedata
import zlib
from functools import lru_cache
from pathlib import Path

from rag.catalog_policy import excluded_genres, extract_section
from rag.character_lexicon import alias_words, character_aliases

STOP = set("a an the and or i im m s t ve d my me you your they their he his him she her it its we our this that these those of for from to in on at by with as is are was were be been being has have had do does did can could would should will not no but so then than just very where when who whose which what how about into out up all some one someone something trying looking remember forgot know think story series show title name anime manga manhwa manhua please find help tell started around named character".split())


def normalized(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKD", text).casefold(), re.UNICODE))


def terms(text):
    return list(dict.fromkeys(t for t in normalized(text).split() if len(t) > 1 and t not in STOP))[:60]


def quoted(text):
    return '"' + text.replace('"', '""') + '"'


def creator_key(text):
    return re.sub(r"([aeio])u", r"\1", normalized(text))


def explicit_creator(query, name):
    text = creator_key(query)
    name = creator_key(name)
    if text == name:
        return True
    if text.startswith(name + " "):
        remainder = text[len(name):].strip().split()
        if " " in name or (remainder and remainder[0] in {"anime", "manga", "manhwa", "series", "show", "movie", "novel", "studio", "animation"}):
            return True
    for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
        prefix = text[:match.start()].split()
        if prefix and prefix[-1] in {"by", "author", "artist", "studio"}:
            return True
    return False


def build(corpus: Path, output: Path, *, source_db=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".building")
    if temp.exists():
        temp.unlink()
    db = sqlite3.connect(temp)
    db.executescript("""
        PRAGMA journal_mode=OFF;
        CREATE TABLE works (id INTEGER PRIMARY KEY, medium TEXT, mal_id INTEGER, year INTEGER, catalog_rank INTEGER, genre TEXT, format TEXT);
        CREATE TABLE creators (name TEXT, work_id INTEGER);
        CREATE INDEX creator_lookup ON creators(name);
        CREATE TABLE genres (name TEXT PRIMARY KEY);
        CREATE TABLE studios (name TEXT PRIMARY KEY);
        CREATE TABLE character_names (alias TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE character_words (alias TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE character_works (alias TEXT, work_id INTEGER, PRIMARY KEY(alias,work_id)) WITHOUT ROWID;
        CREATE TABLE serving_snippets (medium TEXT, mal_id INTEGER, snippet TEXT, PRIMARY KEY(medium,mal_id)) WITHOUT ROWID;
        CREATE VIRTUAL TABLE evidence USING fts5(title, plot, names, creator, genre, year, format, content='', tokenize='porter unicode61 remove_diacritics 2');
        CREATE VIRTUAL TABLE evidence_vocab USING fts5vocab(evidence, 'row');
    """)
    from contextlib import nullcontext
    source = nullcontext((body for body, in source_db.execute("SELECT row_json FROM documents WHERE upserted=1 OR json_extract(row_json,'$.vector_exists')=1 ORDER BY key"))) if source_db is not None else gzip.open(corpus, "rt", encoding="utf-8")
    with source as handle:
        for identifier, line in enumerate(handle, 1):
            row = json.loads(line)
            if excluded_genres(row):
                continue
            detail = row.get("search_details", {})
            maker = ", ".join(filter(None, (detail.get("Authors"), detail.get("Studios"))))
            year = str(detail.get("Started") or "")[:4]
            genres = ", ".join(row.get("genres", []))
            names = extract_section(row["text"], "Character Search Aliases")
            character_names = character_aliases(row["text"])
            db.executemany("INSERT OR IGNORE INTO character_names VALUES(?)", [(alias,) for alias in sorted(character_names)])
            db.executemany("INSERT OR IGNORE INTO character_words VALUES(?)", [(token,) for token in sorted(alias_words(character_names))])
            db.executemany("INSERT OR IGNORE INTO character_works VALUES(?,?)", [(alias, identifier) for alias in sorted(character_names)])
            synopsis = row.get("synopsis") or extract_section(row["text"], "Synopsis")
            plot = synopsis + "\n" + extract_section(row["text"], "Character Details")
            aliases = row.get("alternative_titles", {})
            titles = " ; ".join(filter(None, [row["title"], aliases.get("en"), *aliases.get("synonyms", [])]))
            db.execute("INSERT INTO works VALUES(?,?,?,?,?,?,?)", (identifier, row["type"], row["mal_id"], int(year) if year.isdigit() else 0, row.get("catalog_rank") or 999999, normalized(genres), row.get("display_type") or ""))
            db.execute("INSERT INTO evidence(rowid,title,plot,names,creator,genre,year,format) VALUES(?,?,?,?,?,?,?,?)", (identifier, titles, plot, names, creator_key(maker), genres, year, row.get("display_type") or ""))
            db.executemany("INSERT INTO creators VALUES(?,?)", [(creator_key(name), identifier) for name in maker.split(", ") if name.strip()])
            db.executemany("INSERT OR IGNORE INTO genres VALUES(?)", [(normalized(g),) for g in row.get("genres", [])])
            db.executemany("INSERT OR IGNORE INTO studios VALUES(?)", [(creator_key(s),) for s in (detail.get("Studios") or "").split(", ") if s])
    # Serving text has compact casts; FTS above keeps the full biographies.
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not excluded_genres(row):
                db.execute("INSERT OR REPLACE INTO serving_snippets VALUES(?,?,?)",
                    (row["type"], row["mal_id"], row["text"]))
    db.execute("INSERT INTO evidence(evidence) VALUES('optimize')")
    db.commit()
    db.close()
    temp.replace(output)


class DiskLexicon:
    def __init__(self, index, table):
        if table not in {"character_names", "character_words"}:
            raise ValueError("Unknown character lexicon")
        self.index, self.table = index, table
        with index.lock:
            self.size = index.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        self.contains = lru_cache(maxsize=4096)(self.lookup)

    def lookup(self, value):
        with self.index.lock:
            return bool(self.index.db.execute(f"SELECT 1 FROM {self.table} WHERE alias=?", (value,)).fetchone())

    def __contains__(self, value):
        return isinstance(value, str) and self.contains(value)

    def __len__(self):
        return self.size


class EvidenceIndex:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.db.execute("PRAGMA cache_size=-512")
        self.db.execute("PRAGMA mmap_size=0")
        self.has_vocab = bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='evidence_vocab'").fetchone())
        self.has_character_lexicons = bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='character_names'").fetchone())
        self.has_character_works = bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='character_works'").fetchone())
        self.has_serving_snippets = bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='serving_snippets'").fetchone())
        from nltk.stem import PorterStemmer
        self.stemmer = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)

    def character_lexicons(self):
        return DiskLexicon(self, "character_names"), DiskLexicon(self, "character_words")

    def serving_snippet(self, medium, identifier):
        with self.lock:
            row = self.db.execute("SELECT snippet FROM serving_snippets WHERE medium=? AND mal_id=?", (medium, identifier)).fetchone()
        if not row:
            return ""
        text = zlib.decompress(row[0]).decode("utf-8") if isinstance(row[0], bytes) else row[0]
        return " ".join(text.split())

    def character_search(self, alias, medium=None, limit=80):
        if not self.has_character_works:
            return []
        with self.lock:
            rows = self.db.execute("""SELECT w.medium,w.mal_id FROM character_works c
                JOIN works w ON w.id=c.work_id WHERE c.alias=? AND (? IS NULL OR w.medium=?)
                ORDER BY w.catalog_rank,w.mal_id LIMIT ?""", (alias, medium, medium, limit)).fetchall()
            return [{"media_type": medium, "mal_id": identifier} for medium, identifier in rows]

    @classmethod
    def from_bundle(cls, path):
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()[:20]
        owner = str(os.getuid()) if hasattr(os, "getuid") else hashlib.sha256(str(Path.home()).encode()).hexdigest()[:12]
        directory = Path(tempfile.gettempdir()) / f"ragnarok-evidence-{owner}"
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or (hasattr(os, "getuid") and directory.stat().st_uid != os.getuid()):
            raise RuntimeError("Evidence cache is not owned by the service user")
        target = directory / f"{digest}.sqlite"
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temp:
                staging = Path(temp.name)
                with gzip.open(path, "rb") as source:
                    shutil.copyfileobj(source, temp)
            try:
                staging.replace(target)
            except PermissionError:
                # Another Windows worker may already have opened the same
                # immutable generation while this worker was unpacking it.
                if not target.exists():
                    raise
                staging.unlink(missing_ok=True)
        return cls(target)

    def connection(self):
        return self.db

    def search(self, query, medium=None, limit=80, *, creators_only=False):
        with self.lock:
            return self._search(query, medium, limit, creators_only=creators_only)

    def _search(self, query, medium, limit, *, creators_only):
        tokens = terms(query)
        if not tokens:
            return []
        db = self.connection()
        words = normalized(query).split()
        phrases = {" ".join(words[i:i+n]) for n in range(1, 6) for i in range(len(words)-n+1)}
        creator_phrases = {creator_key(p) for p in phrases}
        # Only creator names explicitly present in the request constrain results.
        creators = [name for name, in db.execute("SELECT DISTINCT name FROM creators WHERE name IN (%s)" % ",".join("?" for _ in creator_phrases), list(creator_phrases)) if len(name) > 3 and name not in STOP and explicit_creator(query, name)]
        creators = [name for name in creators if not any(name != other and name in other for other in creators)]
        if any(" " in name for name in creators):
            creators = [name for name in creators if " " in name]
        if creators_only and not creators:
            return []
        expression = " OR ".join(map(quoted, tokens))
        if creators:
            guard = "creator : (" + " OR ".join(map(quoted, creators)) + ")"
            expression = f"({guard}) AND (({expression}) OR ({guard}))"
        elif self.has_vocab and creators_only is False:
            stems = {self.stemmer.stem(t) for t in tokens}
            counts = db.execute("SELECT term,doc FROM evidence_vocab WHERE term IN (%s) ORDER BY doc,term LIMIT 2" % ",".join("?" for _ in stems), list(stems)).fetchall()
            if len(counts) == 2:
                if counts[0][1] > 2000:
                    return []
                # Conjunctive rare clues avoid scoring nearly the entire catalog
                # for a conversational OR query on a shared-core VM.
                guard = " AND ".join(quoted(term) for term, _ in counts)
                expression = f"({guard}) AND ({expression})"
        year_match = re.search(r"\b((?:18|19|20)\d{2})\b", query)
        year = int(year_match[1]) if year_match else None
        params = [expression]
        where = ""
        if medium:
            where += " AND w.medium=?"
            params.append(medium)
        if creators:
            where += " AND w.id IN (SELECT work_id FROM creators WHERE name IN (%s))" % ",".join("?" for _ in creators)
            params.extend(creators)
        if creators and year:
            where += " AND w.year=?"
            params.append(year)
            genres = [name for name, in db.execute("SELECT name FROM genres WHERE name IN (%s)" % ",".join("?" for _ in phrases), list(phrases))]
            for genre in genres:
                where += " AND instr(' ' || w.genre || ' ', ?) > 0"
                params.append(" " + genre + " ")
        params.append(limit)
        rows = db.execute(f"SELECT w.medium,w.mal_id,bm25(evidence,0.1,1,1,0,0.5,0,0),w.catalog_rank,w.genre,w.format FROM evidence JOIN works w ON w.id=evidence.rowid WHERE evidence MATCH ? {where} ORDER BY bm25(evidence,0.1,1,1,0,0.5,0,0),w.catalog_rank LIMIT ?", params).fetchall()
        return [{"media_type": r[0], "mal_id": r[1], "score": -r[2], "catalog_rank": r[3], "genres": r[4], "display_type": r[5], "creator_match": bool(creators)} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("rag/data/catalog.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("rag/ingestion/evidence.sqlite"))
    args = parser.parse_args()
    build(args.corpus, args.output)
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
