"""Shared character-name normalization for offline and serving indexes."""
import re
from rag.catalog_policy import extract_section


def normalized_alias(value):
    value = value.lower().replace("\u2019", "'")
    value = value.translate(str.maketrans({c: " " for c in "-/:,.()"}))
    return " ".join(value.split())


def character_aliases(text):
    result = set()
    for section in ("Characters", "Character Search Aliases"):
        for value in extract_section(text, section).split(";"):
            value = " ".join(value.split("(", 1)[0].split())
            if not value or value.lower() == "unknown":
                continue
            result.add(normalized_alias(value))
            if "," in value:
                last, first = [part.strip() for part in value.split(",", 1)]
                if first and last:
                    result.add(normalized_alias(f"{first} {last}"))
                    result.add(normalized_alias(f"{last} {first}"))
    return result - {""}


def alias_words(aliases):
    return {word for alias in aliases for word in re.findall(r"[a-z0-9]+", alias)}
