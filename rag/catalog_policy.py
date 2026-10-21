"""Shared catalog exclusions; apply before indexing and before serving results."""
from __future__ import annotations

import re
from typing import Any

EXCLUDED_GENRES = frozenset({"boys love", "girls love", "erotica", "hentai"})
GENRE_ALIASES = {"yaoi": "boys love", "shounen ai": "boys love", "shonen ai": "boys love",
                 "yuri": "girls love", "shoujo ai": "girls love", "shojo ai": "girls love"}


def normalize_genre(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return GENRE_ALIASES.get(normalized, normalized)


def extract_section(text: str, name: str) -> str:
    match = re.search(rf"(?:^|\n|\s){re.escape(name)}:\s*(.*?)(?=\s+(?:Type|Title|Title Aliases|Genres|Synopsis|Characters|Character Search Aliases|Character Details):|$)", text, re.I | re.S)
    return match.group(1).strip() if match else ""


def excluded_genres(row: dict[str, Any]) -> list[str]:
    genres = row.get("genres", [])
    if isinstance(genres, str):
        genres = re.split(r"[,;|]", genres)
    names = [g.get("name", "") if isinstance(g, dict) else str(g) for g in genres]
    text = str(row.get("text") or row.get("chunk_text") or row.get("snippet") or "")
    names.extend(re.split(r"[,;|]", extract_section(text, "Genres")))
    return sorted({normalize_genre(g) for g in names} & EXCLUDED_GENRES)
