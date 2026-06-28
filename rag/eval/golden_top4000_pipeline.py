from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
SECTION_PREFIX_RE = re.compile(r"^[A-Za-z ]+:\s*")
ALPHA_TOKEN_RE = re.compile(r"[a-z]{3,}")


@dataclass(frozen=True)
class CorpusRow:
    medium: str
    mal_id: int
    title: str
    synopsis: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    medium: str
    query_type: str
    query: str
    target_title: str
    target_mal_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Golden evaluation pipeline for top_4000 RAG. "
            "Builds semantic + keyword + title queries across anime and manga."
        )
    )
    parser.add_argument(
        "--docs-jsonl-path",
        default="rag/ingestion/top_4000_documents_character_enriched.jsonl",
        help="Path to enriched top_4000 jsonl corpus.",
    )
    parser.add_argument(
        "--cases-json-path",
        default="",
        help="Optional prebuilt golden cases JSON. When set, case generation from docs-jsonl is skipped.",
    )
    parser.add_argument("--rag-endpoint", default="http://127.0.0.1:8090/rag/search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-medium", type=int, default=100, help="Cases per medium (anime, manga).")
    parser.add_argument("--semantic-ratio", type=float, default=0.75, help="Semantic query ratio per medium.")
    parser.add_argument(
        "--title-per-medium",
        type=int,
        default=40,
        help="Additional direct-title cases per medium (anime, manga).",
    )
    parser.add_argument(
        "--title-alias-json-path",
        default="rag/ingestion/top_4000_title_aliases.json",
        help="Optional JSON map containing alternative English/Romaji/Japanese titles.",
    )
    parser.add_argument("--top-k-success", type=int, default=10, choices=[1, 3, 5, 10])
    parser.add_argument(
        "--max-keyword-alias-frequency",
        type=int,
        default=3,
        help="Maximum corpus frequency allowed for selected keyword aliases (controls ambiguity).",
    )
    parser.add_argument("--min-hit-rate", type=float, default=0.90)
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--output-path", default="rag/eval/golden_top4000_results.json")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for quick checks.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\n", " ").split()).strip()


def load_cases(path: Path) -> list[GoldenCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Invalid golden cases file: {path}")

    cases: list[GoldenCase] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid golden case at index {idx}: expected object")
        case_id = clean_text(row.get("case_id"))
        medium = clean_text(row.get("medium")).lower()
        query_type = clean_text(row.get("query_type")).lower()
        query = clean_text(row.get("query"))
        target_title = clean_text(row.get("target_title"))
        target_mal_id = row.get("target_mal_id")
        if (
            not case_id
            or medium not in {"anime", "manga"}
            or query_type not in {"semantic", "keyword", "title"}
            or not query
            or not target_title
            or not isinstance(target_mal_id, int)
        ):
            raise ValueError(f"Invalid golden case at index {idx}: {row!r}")
        cases.append(
            GoldenCase(
                case_id=case_id,
                medium=medium,
                query_type=query_type,
                query=query,
                target_title=target_title,
                target_mal_id=target_mal_id,
            )
        )
    return cases


def normalize_title(value: str) -> str:
    return TITLE_NORMALIZE_RE.sub(" ", value.lower()).strip()


def extract_section(text: str, marker: str) -> str:
    lowered = text.lower()
    idx = lowered.find(marker.lower())
    if idx < 0:
        return ""
    tail = text[idx + len(marker) :].strip()
    if not tail:
        return ""

    stop_markers = ("character search aliases:", "characters:", "synopsis:", "genres:", "type:", "title:")
    stop = len(tail)
    for marker_name in stop_markers:
        next_idx = tail.lower().find(marker_name)
        if next_idx > 0:
            stop = min(stop, next_idx)
    return clean_text(tail[:stop])


def parse_aliases(text: str) -> tuple[str, ...]:
    section = extract_section(text, "Character Search Aliases:")
    aliases: list[str] = []
    seen: set[str] = set()

    if section:
        for raw in section.split(";"):
            alias = clean_text(SECTION_PREFIX_RE.sub("", raw))
            if not alias:
                continue
            if alias.lower() == "unknown":
                continue
            key = alias.lower()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
        if aliases:
            return tuple(aliases)

    chars = extract_section(text, "Characters:")
    for raw in chars.split(";"):
        name = clean_text(raw.split("(", 1)[0])
        if not name or name.lower() == "unknown":
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(name)

    return tuple(aliases)


def parse_synopsis(text: str) -> str:
    synopsis = extract_section(text, "Synopsis:")
    return synopsis or "No synopsis available."


def load_corpus(path: Path) -> list[CorpusRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")

    deduped: dict[tuple[str, int], CorpusRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            medium = row.get("type")
            mal_id = row.get("mal_id")
            title = clean_text(row.get("title"))
            text = row.get("text")
            if medium not in {"anime", "manga"} or not isinstance(mal_id, int) or not title or not isinstance(text, str):
                continue

            corpus_row = CorpusRow(
                medium=medium,
                mal_id=mal_id,
                title=title,
                synopsis=parse_synopsis(text),
                aliases=parse_aliases(text),
            )
            key = (medium, mal_id)
            existing = deduped.get(key)
            if existing is None or len(text) > len(existing.synopsis):
                deduped[key] = corpus_row

    return list(deduped.values())


def collect_title_alias_values(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        alias = clean_text(value)
        if alias and alias.lower() != "unknown":
            output.append(alias)
        return
    if isinstance(value, list):
        for item in value:
            collect_title_alias_values(item, output)
        return
    if isinstance(value, dict):
        for key in (
            "title",
            "title_english",
            "title_japanese",
            "title_synonyms",
            "synonyms",
            "titles",
            "aliases",
            "alternative_titles",
        ):
            if key in value:
                collect_title_alias_values(value.get(key), output)


def dedupe_aliases(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = clean_text(value)
        key = normalize_title(alias)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(alias)
    return tuple(deduped)


def load_title_alias_map(path: Path) -> dict[tuple[str, int], tuple[str, ...]]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    alias_map: dict[tuple[str, int], tuple[str, ...]] = {}

    def append_aliases(key: tuple[str, int], value: Any) -> None:
        aliases_raw: list[str] = []
        collect_title_alias_values(value, aliases_raw)
        aliases = dedupe_aliases(aliases_raw)
        if aliases:
            alias_map[key] = aliases

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            mal_id = row.get("mal_id")
            if not isinstance(mal_id, int):
                continue
            media_type = clean_text(row.get("type")).lower()
            if media_type not in {"anime", "manga"}:
                media_type = "*"
            append_aliases((media_type, mal_id), row)
    elif isinstance(payload, dict):
        for raw_key, row in payload.items():
            media_type = "*"
            mal_id: int | None = None
            key = clean_text(raw_key).lower()
            parsed = re.match(r"^(anime|manga):(\d+)$", key)
            if parsed:
                media_type = parsed.group(1)
                mal_id = int(parsed.group(2))
            elif key.isdigit():
                mal_id = int(key)
            elif isinstance(row, dict) and isinstance(row.get("mal_id"), int):
                mal_id = int(row["mal_id"])
                nested_type = clean_text(row.get("type")).lower()
                if nested_type in {"anime", "manga"}:
                    media_type = nested_type
            if mal_id is None:
                continue
            append_aliases((media_type, mal_id), row)

    return alias_map


def title_query_variants(title: str, title_aliases: tuple[str, ...]) -> list[str]:
    variants: list[str] = []

    def add(value: str) -> None:
        cleaned = clean_text(value)
        if not cleaned:
            return
        key = normalize_title(cleaned)
        if not key:
            return
        if key in {normalize_title(existing) for existing in variants}:
            return
        variants.append(cleaned)

    add(title)
    for alias in title_aliases:
        add(alias)

    bases = list(variants)
    for base in bases:
        add(base.lower())
        compact = re.sub(r"[^A-Za-z0-9 ]+", " ", base)
        add(" ".join(compact.split()))
        if ":" in base:
            add(base.split(":", 1)[0])
        if "-" in base:
            add(base.split("-", 1)[0])
    return variants


def is_specific_title_variant(query: str, canonical_title: str) -> bool:
    query_norm = normalize_title(query)
    canonical_norm = normalize_title(canonical_title)
    if not query_norm or not canonical_norm:
        return False

    query_tokens = query_norm.split()
    canonical_tokens = canonical_norm.split()
    if len(query_tokens) >= 2:
        return True
    # One-token title queries are only specific enough when the canonical title itself is one token.
    return len(query_tokens) == 1 and len(canonical_tokens) == 1 and query_tokens[0] == canonical_tokens[0]


SEMANTIC_TEMPLATES = [
    "manga or anime about {clue}",
    "series where {clue}",
    "story with this vibe: {clue}",
    "what title is about {clue}",
]

KEYWORD_TEMPLATES = [
    "{name}",
    "series with character {name}",
    "find title for {name}",
]


def synopsis_clue(synopsis: str, target_title: str) -> str:
    source = clean_text(synopsis)
    if not source:
        return ""

    text = source
    # Remove only longer title tokens to avoid over-stripping short glue words.
    title_tokens = [token for token in normalize_title(target_title).split() if len(token) >= 5][:6]
    for token in title_tokens:
        text = re.sub(rf"\b{re.escape(token)}\b", "", text, flags=re.IGNORECASE)

    words = [word for word in clean_text(text).split() if word]
    stripped = " ".join(words[:28]) if words else source[:140]
    if not is_semantic_clue(stripped):
        return " ".join(source.split()[:28])
    return stripped


def is_semantic_clue(clue: str) -> bool:
    cleaned = clean_text(clue)
    if not cleaned or cleaned.lower().startswith("no synopsis available"):
        return False
    tokens = ALPHA_TOKEN_RE.findall(cleaned.lower())
    if len(tokens) < 6:
        return False
    return len(set(tokens)) >= 4


def build_alias_frequency(corpus: list[CorpusRow]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for row in corpus:
        seen_for_row: set[str] = set()
        for alias in row.aliases:
            key = normalize_title(alias)
            if not key or key in seen_for_row:
                continue
            seen_for_row.add(key)
            frequencies[key] = frequencies.get(key, 0) + 1
    return frequencies


def select_keyword_alias(aliases: tuple[str, ...], title: str, alias_frequency: dict[str, int]) -> str:
    candidates = [alias for alias in aliases if len(alias.split()) >= 2]
    if not candidates:
        candidates = list(aliases)
    if not candidates:
        return title

    def sort_key(value: str) -> tuple[int, int]:
        key = normalize_title(value)
        frequency = alias_frequency.get(key, 9999)
        return (frequency, -len(value))

    candidates.sort(key=sort_key)
    return candidates[0]


def has_meaningful_synopsis(row: CorpusRow) -> bool:
    clue = synopsis_clue(row.synopsis, row.title)
    return is_semantic_clue(clue)


def has_keyword_alias(row: CorpusRow, alias_frequency: dict[str, int], max_alias_frequency: int) -> bool:
    if not row.aliases:
        return False
    alias = select_keyword_alias(row.aliases, row.title, alias_frequency)
    alias_clean = clean_text(alias)
    if not alias_clean:
        return False
    if alias_clean.lower() == "unknown":
        return False
    if len(alias_clean.split()) < 2:
        return False
    return alias_frequency.get(normalize_title(alias_clean), 9999) <= max_alias_frequency


def build_cases(
    corpus: list[CorpusRow],
    *,
    per_medium: int,
    semantic_ratio: float,
    title_per_medium: int,
    seed: int,
    title_alias_map: dict[tuple[str, int], tuple[str, ...]],
    alias_frequency: dict[str, int],
    max_keyword_alias_frequency: int,
) -> list[GoldenCase]:
    rng = random.Random(seed)
    cases: list[GoldenCase] = []
    per_medium_semantic = int(round(per_medium * semantic_ratio))
    per_medium_keyword = per_medium - per_medium_semantic

    for medium in ("anime", "manga"):
        rows = [row for row in corpus if row.medium == medium and row.title]
        semantic_pool = [row for row in rows if has_meaningful_synopsis(row)]
        keyword_pool = [
            row
            for row in rows
            if has_keyword_alias(
                row,
                alias_frequency=alias_frequency,
                max_alias_frequency=max_keyword_alias_frequency,
            )
        ]

        if len(semantic_pool) < per_medium_semantic:
            raise RuntimeError(
                f"Not enough {medium} semantic rows for target count={per_medium_semantic}. found={len(semantic_pool)}"
            )
        if len(keyword_pool) < per_medium_keyword:
            raise RuntimeError(
                f"Not enough {medium} keyword rows for target count={per_medium_keyword}. found={len(keyword_pool)}"
            )

        rng.shuffle(semantic_pool)
        semantic_rows = semantic_pool[:per_medium_semantic]

        semantic_ids = {(row.medium, row.mal_id) for row in semantic_rows}
        keyword_candidates = [row for row in keyword_pool if (row.medium, row.mal_id) not in semantic_ids]
        if len(keyword_candidates) < per_medium_keyword:
            keyword_candidates = keyword_pool
        rng.shuffle(keyword_candidates)
        keyword_rows = keyword_candidates[:per_medium_keyword]

        for row in semantic_rows:
            clue = synopsis_clue(row.synopsis, row.title)
            template = rng.choice(SEMANTIC_TEMPLATES)
            query = template.format(clue=clue)
            cases.append(
                GoldenCase(
                    case_id=f"{medium}-semantic-{len(cases)+1:04d}",
                    medium=medium,
                    query_type="semantic",
                    query=query,
                    target_title=row.title,
                    target_mal_id=row.mal_id,
                )
            )

        for row in keyword_rows:
            name = select_keyword_alias(row.aliases, row.title, alias_frequency)
            template = rng.choice(KEYWORD_TEMPLATES)
            query = template.format(name=name)
            cases.append(
                GoldenCase(
                    case_id=f"{medium}-keyword-{len(cases)+1:04d}",
                    medium=medium,
                    query_type="keyword",
                    query=query,
                    target_title=row.title,
                    target_mal_id=row.mal_id,
                )
            )

        if title_per_medium > 0:
            used_ids = {(row.medium, row.mal_id) for row in semantic_rows}
            used_ids.update((row.medium, row.mal_id) for row in keyword_rows)
            title_candidates = [row for row in rows if (row.medium, row.mal_id) not in used_ids]
            if len(title_candidates) < title_per_medium:
                title_candidates = rows
            if len(title_candidates) < title_per_medium:
                raise RuntimeError(
                    f"Not enough {medium} rows for title test count={title_per_medium}. found={len(title_candidates)}"
                )

            rng.shuffle(title_candidates)
            title_rows = title_candidates[:title_per_medium]
            for row in title_rows:
                aliases = list(title_alias_map.get((row.medium, row.mal_id), ()))
                aliases.extend(title_alias_map.get(("*", row.mal_id), ()))
                variants = title_query_variants(row.title, dedupe_aliases(aliases))
                if not variants:
                    variants = [row.title]
                specific_variants = [item for item in variants if is_specific_title_variant(item, row.title)]
                if not specific_variants:
                    specific_variants = [row.title]

                # Favor true alternative titles when available, but keep queries specific.
                alt_variants = [
                    item
                    for item in specific_variants
                    if normalize_title(item) != normalize_title(row.title)
                ]
                if alt_variants and rng.random() < 0.45:
                    query = rng.choice(alt_variants)
                elif rng.random() < 0.55:
                    query = row.title
                else:
                    query = rng.choice(specific_variants)
                cases.append(
                    GoldenCase(
                        case_id=f"{medium}-title-{len(cases)+1:04d}",
                        medium=medium,
                        query_type="title",
                        query=query,
                        target_title=row.title,
                        target_mal_id=row.mal_id,
                    )
                )

    rng.shuffle(cases)
    return cases


def is_match(target_title: str, target_mal_id: int, result_title: str, result_mal_id: int | None) -> bool:
    if isinstance(result_mal_id, int) and result_mal_id == target_mal_id:
        return True
    target_norm = normalize_title(target_title)
    result_norm = normalize_title(result_title)
    if not target_norm or not result_norm:
        return False
    return target_norm == result_norm or target_norm in result_norm or result_norm in target_norm


def evaluate_case(
    client: httpx.Client,
    endpoint: str,
    case: GoldenCase,
    top_k_success: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(endpoint, json={"query": case.query})
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("top_results", [])
        if not isinstance(results, list):
            results = []
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "case_id": case.case_id,
            "medium": case.medium,
            "query_type": case.query_type,
            "query": case.query,
            "target_title": case.target_title,
            "target_mal_id": case.target_mal_id,
            "hit": False,
            "hit_rank": None,
            "elapsed_ms": elapsed_ms,
            "top_titles": [],
            "error": clean_text(str(exc)) or "request_failed",
        }

    hit_rank: int | None = None
    for idx, row in enumerate(results[:top_k_success], start=1):
        title = clean_text(row.get("title")) if isinstance(row, dict) else ""
        mal_id_raw = row.get("mal_id") if isinstance(row, dict) else None
        mal_id = mal_id_raw if isinstance(mal_id_raw, int) else None
        if is_match(case.target_title, case.target_mal_id, title, mal_id):
            hit_rank = idx
            break

    top_titles = []
    for row in results[:10]:
        if isinstance(row, dict):
            top_titles.append(clean_text(row.get("title")))

    return {
        "case_id": case.case_id,
        "medium": case.medium,
        "query_type": case.query_type,
        "query": case.query,
        "target_title": case.target_title,
        "target_mal_id": case.target_mal_id,
        "hit": hit_rank is not None,
        "hit_rank": hit_rank,
        "elapsed_ms": elapsed_ms,
        "top_titles": top_titles,
        "error": None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    hits = sum(1 for row in results if row.get("hit"))
    errors = sum(1 for row in results if row.get("error"))
    avg_ms = round(sum(int(row.get("elapsed_ms", 0)) for row in results) / max(1, total), 2)

    by_group: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        buckets[f"medium:{row['medium']}"].append(row)
        buckets[f"type:{row['query_type']}"].append(row)
        buckets[f"medium:{row['medium']}|type:{row['query_type']}"].append(row)

    for key, rows in buckets.items():
        group_total = len(rows)
        group_hits = sum(1 for row in rows if row.get("hit"))
        by_group[key] = {
            "cases": group_total,
            "hit_rate": round(group_hits / max(1, group_total), 4),
        }

    return {
        "cases": total,
        "hits": hits,
        "errors": errors,
        "hit_rate": round(hits / max(1, total), 4),
        "avg_latency_ms": avg_ms,
        "groups": by_group,
    }


def main() -> None:
    args = parse_args()
    if args.cases_json_path:
        cases = load_cases(Path(args.cases_json_path))
    else:
        corpus = load_corpus(Path(args.docs_jsonl_path))
        alias_frequency = build_alias_frequency(corpus)
        title_alias_map = load_title_alias_map(Path(args.title_alias_json_path))
        cases = build_cases(
            corpus,
            per_medium=args.per_medium,
            semantic_ratio=args.semantic_ratio,
            title_per_medium=args.title_per_medium,
            seed=args.seed,
            title_alias_map=title_alias_map,
            alias_frequency=alias_frequency,
            max_keyword_alias_frequency=args.max_keyword_alias_frequency,
        )

    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    timeout = httpx.Timeout(args.timeout_seconds)
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout) as client:
        for case in cases:
            row = evaluate_case(client, args.rag_endpoint, case, args.top_k_success)
            results.append(row)
            error = clean_text(row.get("error"))
            print(
                f"[{case.case_id}] {case.query_type:<8} {case.medium:<5} "
                f"hit={row['hit']} rank={row['hit_rank']} ms={row['elapsed_ms']}"
                + (f" error={error[:120]}" if error else "")
            )

    summary = summarize(results)
    print("\nGolden Summary")
    print(f"- Cases: {summary['cases']}")
    print(f"- Hit@{args.top_k_success}: {summary['hits']}/{summary['cases']} ({summary['hit_rate']:.2%})")
    print(f"- Request errors: {summary['errors']}")
    print(f"- Avg latency: {summary['avg_latency_ms']} ms")
    for group, values in sorted(summary["groups"].items()):
        print(f"- {group}: {values['hit_rate']:.2%} ({int(values['cases'])} cases)")

    output = {
        "config": {
            "docs_jsonl_path": args.docs_jsonl_path,
            "cases_json_path": args.cases_json_path,
            "rag_endpoint": args.rag_endpoint,
            "seed": args.seed,
            "per_medium": args.per_medium,
            "semantic_ratio": args.semantic_ratio,
            "title_per_medium": args.title_per_medium,
            "title_alias_json_path": args.title_alias_json_path,
            "max_keyword_alias_frequency": args.max_keyword_alias_frequency,
            "top_k_success": args.top_k_success,
            "min_hit_rate": args.min_hit_rate,
            "limit": args.limit,
        },
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"- Wrote: {out_path}")

    if summary["hit_rate"] < args.min_hit_rate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
