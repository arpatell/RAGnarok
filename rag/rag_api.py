from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_cerebras import ChatCerebras
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field, PrivateAttr
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

LOGGER = logging.getLogger("rag_api")
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
TOKEN_RE = re.compile(r"[a-z0-9]+")
SPINOFF_TITLE_MARKERS = (
    "recap",
    "rewrite",
    "ova",
    "special",
    "movie",
    "season",
    "part",
    "ii",
    "iii",
    "iv",
    "2nd",
    "3rd",
    "4th",
    "zero",
)
QUERY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "does",
    "finds",
    "for",
    "from",
    "guy",
    "guys",
    "how",
    "in",
    "into",
    "is",
    "it",
    "keep",
    "keeps",
    "manga",
    "anime",
    "of",
    "on",
    "or",
    "people",
    "should",
    "that",
    "the",
    "their",
    "them",
    "they",
    "thrown",
    "to",
    "who",
    "with",
}
CHARACTER_QUERY_HINT_TOKENS = {"character", "characters", "named", "name", "alias", "aliases"}
QUERY_SCaffold_TOKENS = {"series", "title", "find"}
TITLE_QUERY_HINT_TOKENS = {
    "about",
    "where",
    "with",
    "who",
    "whose",
    "vibe",
    "plot",
    "story",
    "finds",
    "looking",
    "searching",
    "recommend",
    "recommendation",
    "guy",
    "girl",
    "boy",
    "woman",
    "man",
}
load_dotenv()

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a manga/anime search assistant.\n"
    "Given the user query, Pinecone retrieval chunks, and live Jikan stats for the top match, "
    "write a concise 2-3 sentence justification for why the top result fits the query.\n"
    "Use only the provided information, never invent facts.\n"
    "You must end with exactly one citation in this format: [Source: MAL-ID {mal_id} - {title}]."
)

JIKAN_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)


class RagResult(BaseModel):
    _character_aliases: str = PrivateAttr(default="")
    rank: int
    mal_id: int | None
    title: str
    snippet: str
    score: float
    citation: str
    read_now_title: str


class HighlightCard(BaseModel):
    mal_id: int | None
    title: str
    justification: str
    citation: str
    live_media_type: str | None = None
    live_status: str | None = None
    live_score: float | None = None
    image_url: str | None = None


class RagSearchResponse(BaseModel):
    query: str
    top_results: list[RagResult]
    highlight: HighlightCard | None


@dataclass
class RagApiConfig:
    pinecone_index: str
    pinecone_namespaces: tuple[str, ...]
    bm25_values_path: str
    text_key: str
    top_k: int
    candidate_pool_size: int
    hybrid_dense_alpha: float
    cerebras_model: str
    jikan_timeout_seconds: float


@dataclass(frozen=True)
class TitleIndexEntry:
    mal_id: int
    media_type: str
    title: str
    snippet: str
    aliases: tuple[str, ...]


def _parse_namespaces() -> tuple[str, ...]:
    raw_namespaces = os.getenv("PINECONE_NAMESPACES", "").strip()
    if raw_namespaces:
        parsed = [part.strip() for part in raw_namespaces.split(",") if part.strip()]
        if parsed:
            return tuple(dict.fromkeys(parsed))

    primary = os.getenv("PINECONE_NAMESPACE", "top-4000").strip() or "top-4000"
    return (primary,)


def load_config() -> RagApiConfig:
    index = os.getenv("PINECONE_INDEX", "").strip()
    if not index:
        raise RuntimeError("PINECONE_INDEX is required.")

    top_k = max(1, int(os.getenv("RAG_TOP_K", "10")))
    candidate_pool_default = max(top_k * 3, 30)
    candidate_pool_size = max(top_k, int(os.getenv("RAG_CANDIDATE_POOL_SIZE", str(candidate_pool_default))))

    return RagApiConfig(
        pinecone_index=index,
        pinecone_namespaces=_parse_namespaces(),
        bm25_values_path=os.getenv("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"),
        text_key=os.getenv("PINECONE_TEXT_KEY", "chunk_text"),
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        hybrid_dense_alpha=float(os.getenv("HYBRID_DENSE_ALPHA", "0.35")),
        cerebras_model=os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
        jikan_timeout_seconds=float(os.getenv("JIKAN_TIMEOUT_SECONDS", "8")),
    )


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\n", " ").split()).strip()


def _extract_image_url(item: dict[str, Any]) -> str | None:
    images = item.get("images")
    if not isinstance(images, dict):
        return None

    for key in ("jpg", "webp"):
        variant = images.get(key)
        if not isinstance(variant, dict):
            continue
        for image_key in ("large_image_url", "image_url", "small_image_url"):
            value = clean_text(variant.get(image_key))
            if value:
                return value
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        mapped = value.to_dict()
        if isinstance(mapped, dict):
            return mapped
    return {}


def _match_attr(match: Any, key: str, default: Any) -> Any:
    if isinstance(match, dict):
        return match.get(key, default)
    return getattr(match, key, default)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalized_text(value: str) -> str:
    lowered = clean_text(value).lower().replace("’", "'")
    lowered = lowered.replace("-", " ").replace("/", " ").replace(":", " ").replace(",", " ")
    lowered = lowered.replace(".", " ").replace("(", " ").replace(")", " ")
    return " ".join(lowered.split())


def _loose_romanization(value: str) -> str:
    text = value
    replacements = (
        ("ou", "o"),
        ("oo", "o"),
        ("uu", "u"),
        ("aa", "a"),
        ("ii", "i"),
        ("oh", "o"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def _tokens(value: str) -> list[str]:
    normalized = _normalized_text(value)
    if not normalized:
        return []
    return TOKEN_RE.findall(normalized)


def _alias_token_set(value: str) -> set[str]:
    base_tokens = _tokens(value)
    if not base_tokens:
        return set()

    variants: set[str] = set(base_tokens)
    collapsed = [_loose_romanization(token) for token in base_tokens]
    variants.update(collapsed)
    return variants


def _extract_character_aliases(snippet: str) -> set[str]:
    aliases: set[str] = set()
    lowered = snippet.lower()
    markers = ("character search aliases:", "characters:")
    for marker in markers:
        idx = lowered.find(marker)
        if idx < 0:
            continue

        tail = snippet[idx + len(marker) :].strip()
        if not tail:
            continue

        # Stop at the next section header if present.
        for stop_marker in ("character search aliases:", "synopsis:", "genres:", "type:", "title:"):
            stop_idx = tail.lower().find(stop_marker)
            if stop_idx > 0:
                tail = tail[:stop_idx].strip()

        for entry in tail.split(";"):
            entry_base = clean_text(entry.split("(", 1)[0])
            if not entry_base or entry_base.lower() == "unknown":
                continue

            aliases.add(_normalized_text(entry_base))
            if "," in entry_base:
                last, first = [clean_text(part) for part in entry_base.split(",", 1)]
                if first and last:
                    aliases.add(_normalized_text(f"{first} {last}"))
                    aliases.add(_normalized_text(f"{last} {first}"))

    return {alias for alias in aliases if alias}


def _aliases_from_metadata(value: str) -> set[str]:
    aliases: set[str] = set()
    for part in value.split(";"):
        alias = _normalized_text(part)
        if alias:
            aliases.add(alias)
    return aliases


def _load_alias_lexicon(docs_jsonl_path: str) -> tuple[set[str], set[str]]:
    path = Path(docs_jsonl_path)
    if not path.exists():
        return set(), set()

    alias_phrases: set[str] = set()
    alias_tokens: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            for alias in _extract_character_aliases(text):
                alias_phrases.add(alias)
                alias_tokens.update(_tokens(alias))

    return alias_phrases, alias_tokens


def _collect_title_alias_values(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        alias = clean_text(value)
        if alias and alias.lower() != "unknown":
            output.append(alias)
        return
    if isinstance(value, list):
        for item in value:
            _collect_title_alias_values(item, output)
        return
    if isinstance(value, dict):
        for key in (
            "title",
            "title_english",
            "title_japanese",
            "title_synonyms",
            "synonyms",
            "aliases",
            "titles",
            "alternative_titles",
        ):
            if key in value:
                _collect_title_alias_values(value.get(key), output)


def _dedupe_aliases(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        alias = clean_text(raw)
        alias_norm = _normalized_text(alias)
        if not alias_norm or alias_norm in seen:
            continue
        seen.add(alias_norm)
        deduped.append(alias)
    return tuple(deduped)


def _load_title_alias_map(path_value: str) -> dict[tuple[str, int], tuple[str, ...]]:
    path = Path(path_value)
    if not path_value or not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        LOGGER.warning("Title alias map could not be parsed: %s", path)
        return {}

    alias_map: dict[tuple[str, int], tuple[str, ...]] = {}

    def append_aliases(key: tuple[str, int], value: Any) -> None:
        raw_aliases: list[str] = []
        _collect_title_alias_values(value, raw_aliases)
        aliases = _dedupe_aliases(raw_aliases)
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
        for key, row in payload.items():
            media_type = "*"
            mal_id: int | None = None
            match = re.match(r"^(anime|manga):(\d+)$", clean_text(key).lower())
            if match:
                media_type = match.group(1)
                mal_id = int(match.group(2))
            elif clean_text(key).isdigit():
                mal_id = int(clean_text(key))
            elif isinstance(row, dict) and isinstance(row.get("mal_id"), int):
                mal_id = int(row["mal_id"])
                nested_type = clean_text(row.get("type")).lower()
                if nested_type in {"anime", "manga"}:
                    media_type = nested_type
            if mal_id is None:
                continue
            append_aliases((media_type, mal_id), row)

    return alias_map


def _load_title_index(docs_jsonl_path: str, title_alias_path: str) -> dict[tuple[str, int], TitleIndexEntry]:
    path = Path(docs_jsonl_path)
    if not path.exists():
        return {}

    alias_map = _load_title_alias_map(title_alias_path)
    records: dict[tuple[str, int], TitleIndexEntry] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            media_type = clean_text(row.get("type")).lower()
            mal_id = row.get("mal_id")
            title = clean_text(row.get("title"))
            snippet = clean_text(row.get("text"))

            if media_type not in {"anime", "manga"} or not isinstance(mal_id, int) or not title:
                continue

            aliases_raw = [title]
            aliases_raw.extend(alias_map.get((media_type, mal_id), ()))
            aliases_raw.extend(alias_map.get(("*", mal_id), ()))
            aliases = _dedupe_aliases(aliases_raw)
            key = (media_type, mal_id)
            existing = records.get(key)
            if existing is None or len(snippet) > len(existing.snippet):
                records[key] = TitleIndexEntry(
                    mal_id=mal_id,
                    media_type=media_type,
                    title=title,
                    snippet=snippet,
                    aliases=aliases,
                )

    return records


def _is_character_query(
    query: str,
    query_tokens: list[str],
    alias_phrase_set: set[str] | None = None,
    alias_token_set: set[str] | None = None,
) -> bool:
    raw_tokens = _tokens(query)
    if len(raw_tokens) < 1:
        return False

    normalized_query = _normalized_text(query)
    phrase_set = alias_phrase_set or set()
    token_set = alias_token_set or set()
    if normalized_query and normalized_query in phrase_set:
        return True

    raw_set = set(raw_tokens)
    if raw_set.intersection(CHARACTER_QUERY_HINT_TOKENS):
        candidate_tokens = [token for token in raw_tokens if token not in QUERY_STOPWORDS and token not in CHARACTER_QUERY_HINT_TOKENS]
        if not candidate_tokens:
            return False
        if not all(token.isalpha() for token in candidate_tokens):
            return False
        matched = sum(1 for token in candidate_tokens if token in token_set)
        return matched >= max(1, len(candidate_tokens) - 1)

    if len(raw_tokens) > 3:
        return False
    if any(token in QUERY_STOPWORDS for token in raw_tokens):
        return False
    if not all(token.isalpha() for token in raw_tokens):
        return False

    if not query_tokens:
        return False
    matched = sum(1 for token in raw_tokens if token in token_set)
    return matched >= len(raw_tokens)


def _hybrid_alpha_for_query(*, query_tokens: list[str], query_is_character: bool, default_alpha: float) -> float:
    if query_is_character:
        return float(os.getenv("HYBRID_DENSE_ALPHA_CHARACTER", "0.12"))
    if len(query_tokens) <= 2:
        return float(os.getenv("HYBRID_DENSE_ALPHA_SHORT_QUERY", "0.60"))
    return default_alpha


def _character_match_stats(
    *,
    query_tokens: list[str],
    snippet: str,
    metadata_aliases: str = "",
) -> tuple[bool, int]:
    query_phrase = " ".join(query_tokens).strip()
    query_token_set = set(query_tokens)
    query_alias_tokens = set(_loose_romanization(token) for token in query_tokens)
    aliases = _aliases_from_metadata(clean_text(metadata_aliases))
    if not aliases:
        aliases = _extract_character_aliases(snippet)
    snippet_token_set = set(_tokens(snippet))

    if query_phrase and query_phrase in aliases:
        return True, len(query_token_set)

    char_tokens: set[str] = set()
    char_alias_tokens: set[str] = set()
    for alias in aliases:
        alias_tokens = _tokens(alias)
        char_tokens.update(alias_tokens)
        char_alias_tokens.update(_loose_romanization(tok) for tok in alias_tokens)

    direct_hits = len(query_token_set.intersection(char_tokens))
    alias_hits = len(query_alias_tokens.intersection(char_alias_tokens))
    snippet_hits = len(query_token_set.intersection(snippet_token_set))
    return False, max(direct_hits, alias_hits, snippet_hits)


def _query_tokens(query: str) -> list[str]:
    raw_tokens = _tokens(query)
    result: list[str] = []
    for token in raw_tokens:
        if token in QUERY_STOPWORDS or token in CHARACTER_QUERY_HINT_TOKENS or token in QUERY_SCaffold_TOKENS:
            continue
        result.append(token)
    if result:
        return result

    relaxed: list[str] = []
    for token in raw_tokens:
        if token in QUERY_STOPWORDS:
            continue
        relaxed.append(token)
    if relaxed:
        return relaxed
    return raw_tokens


def _title_is_spinoff(title: str) -> bool:
    title_norm = _normalized_text(title)
    return any(marker in title_norm for marker in SPINOFF_TITLE_MARKERS)


def _query_mentions_spinoff(query_tokens: list[str]) -> bool:
    return any(token in SPINOFF_TITLE_MARKERS for token in query_tokens)


def _canonical_root_title(title: str) -> str:
    clean_title = clean_text(title)
    if ":" in clean_title:
        return clean_text(clean_title.split(":", 1)[0])
    if " - " in clean_title:
        return clean_text(clean_title.split(" - ", 1)[0])
    return clean_title


def _preferred_media_type_from_snippet(snippet: str) -> str | None:
    lowered = snippet.lower()
    if "type: manga" in lowered:
        return "manga"
    if "type: anime" in lowered:
        return "anime"
    return None


class HybridRagService:
    def __init__(self, config: RagApiConfig) -> None:
        self.config = config

        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required.")

        self._embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._pc = Pinecone(api_key=pinecone_api_key)
        self._index = self._pc.Index(config.pinecone_index)
        # Required LangChain primitive initialization.
        self._vector_store = PineconeVectorStore(
            index=self._index,
            embedding=self._embeddings,
            namespace=config.pinecone_namespaces[0],
            text_key=config.text_key,
        )

        bm25 = BM25Encoder()
        loaded_bm25 = bm25.load(config.bm25_values_path)
        self._bm25 = loaded_bm25 if isinstance(loaded_bm25, BM25Encoder) else bm25
        self._llm_rerank_enabled = os.getenv("RAG_LLM_RERANK_ENABLED", "0").strip() == "1"
        self._llm = None
        if self._llm_rerank_enabled:
            cerebras_api_key = os.getenv("CEREBRAS_API_KEY", "").strip()
            if not cerebras_api_key:
                raise RuntimeError("CEREBRAS_API_KEY is required when RAG_LLM_RERANK_ENABLED=1.")
            llm_timeout_seconds = float(os.getenv("CEREBRAS_TIMEOUT_SECONDS", "4"))
            llm_max_retries = int(os.getenv("CEREBRAS_MAX_RETRIES", "0"))
            self._llm = ChatCerebras(
                model=config.cerebras_model,
                temperature=0,
                api_key=cerebras_api_key,
                timeout=llm_timeout_seconds,
                max_retries=llm_max_retries,
            )
        self._llm_backoff_until = 0.0
        self._http = httpx.Client(
            timeout=config.jikan_timeout_seconds,
            headers={"Accept": "application/json", "User-Agent": "manga-rag-api/1.0"},
        )
        aliases_path = os.getenv(
            "CHARACTER_ALIAS_DOCS_PATH",
            os.getenv("TOP4000_ENRICHED_JSONL_PATH", "rag/ingestion/top_4000_documents_character_enriched.jsonl"),
        )
        title_docs_path = os.getenv("TITLE_INDEX_DOCS_PATH", aliases_path)
        title_alias_path = os.getenv("TITLE_ALIAS_JSON_PATH", "rag/ingestion/top_4000_title_aliases.json")
        self._alias_phrase_set, self._alias_token_set = _load_alias_lexicon(aliases_path)
        self._title_index = _load_title_index(title_docs_path, title_alias_path)
        self._title_exact_lookup: dict[str, list[TitleIndexEntry]] = {}
        self._title_by_mal_id: dict[int, list[TitleIndexEntry]] = {}
        for entry in self._title_index.values():
            self._title_by_mal_id.setdefault(entry.mal_id, []).append(entry)
            for alias in entry.aliases:
                alias_norm = _normalized_text(alias)
                if alias_norm:
                    self._title_exact_lookup.setdefault(alias_norm, []).append(entry)
                loose_norm = _loose_romanization(alias_norm)
                if loose_norm:
                    self._title_exact_lookup.setdefault(loose_norm, []).append(entry)
        self._title_match_threshold = max(0.80, min(0.99, float(os.getenv("RAG_TITLE_MATCH_THRESHOLD", "0.88"))))
        self._title_near_threshold = max(0.85, min(0.995, float(os.getenv("RAG_TITLE_NEAR_THRESHOLD", "0.935"))))
        self._title_jikan_fallback_enabled = os.getenv("RAG_TITLE_JIKAN_FALLBACK", "0").strip() == "1"
        self._title_jikan_search_limit = max(2, int(os.getenv("RAG_TITLE_JIKAN_SEARCH_LIMIT", "6")))
        self._title_query_cache_ttl = max(300.0, float(os.getenv("RAG_TITLE_QUERY_CACHE_TTL_SECONDS", "1800")))
        self._title_query_cache: dict[str, tuple[float, list[tuple[float, TitleIndexEntry]]]] = {}
        self._jikan_cache_ttl = max(30.0, float(os.getenv("JIKAN_CACHE_TTL_SECONDS", "300")))
        self._jikan_min_interval = max(0.4, float(os.getenv("JIKAN_MIN_INTERVAL_SECONDS", "0.8")))
        self._jikan_max_retries = max(1, int(os.getenv("JIKAN_MAX_RETRIES", "2")))
        self._jikan_request_budget_seconds = max(2.0, float(os.getenv("JIKAN_REQUEST_BUDGET_SECONDS", "6.0")))
        self._next_jikan_request_at = 0.0
        self._jikan_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        self._jikan_lock = threading.Lock()
        self._response_cache_ttl = max(0.0, float(os.getenv("RAG_RESPONSE_CACHE_TTL_SECONDS", "300")))
        self._response_cache_max = max(0, int(os.getenv("RAG_RESPONSE_CACHE_MAX", "256")))
        self._response_cache: dict[str, tuple[float, RagSearchResponse]] = {}
        self._inflight_searches: dict[str, threading.Event] = {}
        self._search_cache_lock = threading.Lock()
        LOGGER.info(
            "Hybrid RAG ready. index=%s namespaces=%s top_k=%d pool=%d alpha=%.2f aliases=%d titles=%d llm=%s title_jikan=%s",
            config.pinecone_index,
            ",".join(config.pinecone_namespaces),
            config.top_k,
            config.candidate_pool_size,
            config.hybrid_dense_alpha,
            len(self._alias_phrase_set),
            len(self._title_index),
            self._llm_rerank_enabled,
            self._title_jikan_fallback_enabled,
        )

    @staticmethod
    def _scale_hybrid(
        dense_vector: list[float],
        sparse_vector: dict[str, list[int] | list[float]],
        dense_alpha: float,
    ) -> tuple[list[float], dict[str, list[int] | list[float]]]:
        alpha = max(0.0, min(1.0, dense_alpha))
        scaled_dense = [value * alpha for value in dense_vector]
        sparse_values = sparse_vector.get("values", [])
        sparse_indices = sparse_vector.get("indices", [])
        scaled_sparse = {
            "indices": sparse_indices,
            "values": [float(value) * (1.0 - alpha) for value in sparse_values],  # type: ignore[arg-type]
        }
        return scaled_dense, scaled_sparse

    @staticmethod
    def _build_citation(mal_id: int | None, title: str) -> str:
        if mal_id is None:
            return "[Source: MAL-ID unknown - Unknown]"
        return f"[Source: MAL-ID {mal_id} - {title}]"

    @staticmethod
    def _title_similarity(query_norm: str, alias_norm: str) -> float:
        if not query_norm or not alias_norm:
            return 0.0
        if query_norm == alias_norm:
            return 1.0

        query_loose = _loose_romanization(query_norm)
        alias_loose = _loose_romanization(alias_norm)
        if query_loose == alias_loose:
            return 0.995

        ratio = SequenceMatcher(None, query_norm, alias_norm).ratio()
        loose_ratio = SequenceMatcher(None, query_loose, alias_loose).ratio()
        best_ratio = max(ratio, loose_ratio)

        query_tokens = set(_tokens(query_norm))
        alias_tokens = set(_tokens(alias_norm))
        overlap = len(query_tokens.intersection(alias_tokens))
        token_union = len(query_tokens.union(alias_tokens))
        precision = overlap / max(1, len(query_tokens))
        recall = overlap / max(1, len(alias_tokens))
        jaccard = overlap / token_union if token_union > 0 else 0.0

        contains = query_norm in alias_norm or alias_norm in query_norm
        containment_ratio = (
            min(len(query_norm), len(alias_norm)) / max(len(query_norm), len(alias_norm))
            if contains and max(len(query_norm), len(alias_norm)) > 0
            else 0.0
        )

        score = (best_ratio * 0.60) + (precision * 0.20) + (recall * 0.10) + (jaccard * 0.10)
        if query_tokens and query_tokens.issubset(alias_tokens):
            score += 0.05
        if contains:
            score = max(score, 0.86 + (0.12 * containment_ratio))
        return min(1.0, score)

    def _title_candidates_from_jikan(self, query: str, query_norm: str) -> list[tuple[float, TitleIndexEntry]]:
        if not self._title_jikan_fallback_enabled:
            return []

        with self._jikan_lock:
            cached = self._title_query_cache.get(query_norm)
            now = time.monotonic()
            if cached and cached[0] > now:
                return cached[1]

        ranked: dict[tuple[str, int], tuple[float, TitleIndexEntry]] = {}
        for media_type in ("manga", "anime"):
            try:
                with self._jikan_lock:
                    wait_for = self._next_jikan_request_at - time.monotonic()
                    if wait_for > 0:
                        time.sleep(wait_for)
                    self._next_jikan_request_at = time.monotonic() + self._jikan_min_interval

                response = self._http.get(
                    f"{JIKAN_BASE_URL}/{media_type}",
                    params={"q": query, "limit": self._title_jikan_search_limit},
                )
                if response.status_code == 429:
                    continue
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Title fallback Jikan search failed for %s query='%s': %s", media_type, query, exc)
                continue

            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                continue

            for item in rows:
                if not isinstance(item, dict):
                    continue
                mal_id = item.get("mal_id")
                if not isinstance(mal_id, int):
                    continue

                candidates = self._title_by_mal_id.get(mal_id, [])
                if not candidates:
                    continue

                jikan_titles_raw: list[str] = []
                for key in ("title", "title_english", "title_japanese"):
                    value = clean_text(item.get(key))
                    if value:
                        jikan_titles_raw.append(value)
                synonyms = item.get("title_synonyms")
                if isinstance(synonyms, list):
                    for value in synonyms:
                        cleaned = clean_text(value)
                        if cleaned:
                            jikan_titles_raw.append(cleaned)
                jikan_titles = [title for title in jikan_titles_raw if _normalized_text(title)]
                if not jikan_titles:
                    continue

                for entry in candidates:
                    best_score = ranked.get((entry.media_type, entry.mal_id), (0.0, entry))[0]
                    for title in jikan_titles:
                        score = self._title_similarity(query_norm, _normalized_text(title))
                        if score > best_score:
                            best_score = score
                    if best_score >= (self._title_match_threshold - 0.03):
                        ranked[(entry.media_type, entry.mal_id)] = (best_score, entry)

        results = sorted(
            ranked.values(),
            key=lambda pair: (-pair[0], pair[1].title.lower()),
        )
        with self._jikan_lock:
            self._title_query_cache[query_norm] = (time.monotonic() + self._title_query_cache_ttl, results)
        return results

    def direct_title_retrieve(self, query: str) -> list[RagResult]:
        if not self._title_index:
            return []

        query_norm = _normalized_text(query)
        if not query_norm:
            return []

        raw_tokens = _tokens(query)
        if len(raw_tokens) > 12:
            return []
        query_number_tokens = {token for token in raw_tokens if token.isdigit()}

        query_mentions_spinoff = _query_mentions_spinoff(raw_tokens)
        semantic_hint = bool(set(raw_tokens).intersection(TITLE_QUERY_HINT_TOKENS))
        exact_key = _loose_romanization(query_norm)
        exact_entries = list(self._title_exact_lookup.get(query_norm, []))
        if exact_key != query_norm:
            exact_entries.extend(self._title_exact_lookup.get(exact_key, []))

        by_key: dict[tuple[str, int], tuple[float, TitleIndexEntry]] = {}
        for entry in exact_entries:
            key = (entry.media_type, entry.mal_id)
            prev = by_key.get(key)
            if prev is None or prev[0] < 1.0:
                by_key[key] = (1.0, entry)

        for entry in self._title_index.values():
            best_score = by_key.get((entry.media_type, entry.mal_id), (0.0, entry))[0]
            for alias in entry.aliases:
                alias_norm = _normalized_text(alias)
                if not alias_norm:
                    continue
                score = self._title_similarity(query_norm, alias_norm)
                if score > best_score:
                    best_score = score
            if best_score >= self._title_match_threshold:
                by_key[(entry.media_type, entry.mal_id)] = (best_score, entry)

        max_score = max((score for score, _entry in by_key.values()), default=0.0)
        if len(raw_tokens) <= 8 and (not by_key or (max_score < self._title_near_threshold and not semantic_hint)):
            jikan_candidates = self._title_candidates_from_jikan(query, query_norm)
            for score, entry in jikan_candidates:
                key = (entry.media_type, entry.mal_id)
                current = by_key.get(key)
                if current is None or score > current[0]:
                    by_key[key] = (score, entry)

        if not by_key:
            return []

        max_score = max(score for score, _entry in by_key.values())
        if max_score < self._title_near_threshold:
            if semantic_hint:
                return []
            if len(raw_tokens) >= 6:
                return []
            if len(raw_tokens) >= 4 and max_score < (self._title_near_threshold + 0.02):
                return []

        ranked_pairs = sorted(
            by_key.values(),
            key=lambda pair: (
                -pair[0],
                len(query_number_tokens.difference({token for token in _tokens(pair[1].title) if token.isdigit()})),
                _title_is_spinoff(pair[1].title) and not query_mentions_spinoff,
                abs(len(_normalized_text(pair[1].title)) - len(query_norm)),
                pair[1].title.lower(),
            ),
        )

        results: list[RagResult] = []
        top_n = max(self.config.top_k, 20)
        for rank, (score, entry) in enumerate(ranked_pairs[:top_n], start=1):
            citation = self._build_citation(entry.mal_id, entry.title)
            snippet = entry.snippet or f"Type: {entry.media_type}\nTitle: {entry.title}\nSynopsis: Unknown"
            results.append(
                RagResult(
                    rank=rank,
                    mal_id=entry.mal_id,
                    title=entry.title,
                    snippet=snippet,
                    score=float(score),
                    citation=citation,
                    read_now_title=entry.title,
                )
            )

        return results

    def _run_namespace_query(
        self,
        *,
        namespace: str,
        dense_query: list[float],
        sparse_query: dict[str, list[int] | list[float]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        response = self._index.query(
            namespace=namespace,
            vector=dense_query,
            sparse_vector=sparse_query,
            top_k=top_k,
            include_metadata=True,
        )

        matches = _as_dict(response).get("matches")
        if matches is None:
            matches = getattr(response, "matches", [])
        if not isinstance(matches, list):
            matches = []

        rows: list[dict[str, Any]] = []
        for match in matches:
            metadata_raw = _match_attr(match, "metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            title = clean_text(metadata.get("title")) or "Unknown"
            mal_id_raw = metadata.get("mal_id")
            mal_id = int(mal_id_raw) if isinstance(mal_id_raw, (int, float)) else None
            snippet = clean_text(metadata.get(self.config.text_key))
            score = float(_match_attr(match, "score", 0.0))
            rows.append(
                {
                    "namespace": namespace,
                    "mal_id": mal_id,
                    "title": title,
                    "snippet": snippet,
                    "character_aliases": clean_text(metadata.get("character_aliases")),
                    "score": score,
                }
            )
        return rows

    def _dedupe_candidates(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            mal_id = row.get("mal_id")
            title = clean_text(row.get("title"))
            if isinstance(mal_id, int):
                key = f"mal:{mal_id}"
            else:
                key = f"title:{_normalized_text(title)}"

            existing = best_by_key.get(key)
            if existing is None or float(row.get("score", 0.0)) > float(existing.get("score", 0.0)):
                best_by_key[key] = row
        return list(best_by_key.values())

    def _rerank(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = _query_tokens(query)
        if not _is_character_query(query, query_tokens, self._alias_phrase_set, self._alias_token_set):
            ranked = list(rows)
            ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            return ranked

        ranked = list(rows)
        for row in ranked:
            snippet = clean_text(row.get("snippet"))
            score = float(row.get("score", 0.0))
            alias_exact, best_hits = _character_match_stats(
                query_tokens=query_tokens,
                snippet=snippet,
                metadata_aliases=clean_text(row.get("character_aliases", "")),
            )
            required_hits = len(set(query_tokens))
            if alias_exact:
                score += 0.8
                row["_alias_hit"] = 2
            else:
                if required_hits > 0 and best_hits >= required_hits:
                    score += 0.75
                elif best_hits >= 2:
                    score += 0.28
                elif best_hits == 1:
                    score -= 0.16
                else:
                    score -= 0.32
                row["_alias_hit"] = best_hits

            row["score"] = score

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in ranked:
            root = _normalized_text(_canonical_root_title(clean_text(row.get("title"))))
            if not root:
                continue
            groups.setdefault(root, []).append(row)

        for root, group_rows in groups.items():
            if len(group_rows) < 2:
                continue
            canonical: dict[str, Any] | None = None
            has_alias_hit = any(int(row.get("_alias_hit", 0)) > 0 for row in group_rows)
            if not has_alias_hit:
                continue
            for row in group_rows:
                title_norm = _normalized_text(clean_text(row.get("title")))
                if title_norm == root:
                    canonical = row
                    break
            if canonical is None:
                continue
            canonical["score"] = float(canonical.get("score", 0.0)) + 0.22
            for row in group_rows:
                if row is canonical:
                    continue
                row["score"] = float(row.get("score", 0.0)) - 0.06

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked

    def hybrid_retrieve(self, query: str) -> list[RagResult]:
        started = time.perf_counter()
        query_tokens = _query_tokens(query)
        query_is_character = _is_character_query(query, query_tokens, self._alias_phrase_set, self._alias_token_set)
        candidate_pool = self.config.candidate_pool_size
        if query_is_character:
            character_pool = int(os.getenv("RAG_CHARACTER_CANDIDATE_POOL", "80"))
            candidate_pool = max(candidate_pool, character_pool)

        embed_started = time.perf_counter()
        dense_query = self._embeddings.embed_query(query)
        embed_ms = int((time.perf_counter() - embed_started) * 1000)
        sparse_started = time.perf_counter()
        sparse_query = self._bm25.encode_queries(query)
        sparse_ms = int((time.perf_counter() - sparse_started) * 1000)
        dense_alpha = _hybrid_alpha_for_query(
            query_tokens=query_tokens,
            query_is_character=query_is_character,
            default_alpha=self.config.hybrid_dense_alpha,
        )
        dense_query, sparse_query = self._scale_hybrid(
            dense_vector=dense_query,
            sparse_vector=sparse_query,
            dense_alpha=dense_alpha,
        )

        collected: list[dict[str, Any]] = []
        pinecone_started = time.perf_counter()
        for namespace in self.config.pinecone_namespaces:
            try:
                collected.extend(
                    self._run_namespace_query(
                        namespace=namespace,
                        dense_query=dense_query,
                        sparse_query=sparse_query,
                        top_k=candidate_pool,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Namespace query failed for '%s': %s", namespace, exc)
        pinecone_ms = int((time.perf_counter() - pinecone_started) * 1000)

        rank_started = time.perf_counter()
        deduped = self._dedupe_candidates(collected)
        reranked = self._rerank(query, deduped)
        top_rows = reranked[:candidate_pool]
        rank_ms = int((time.perf_counter() - rank_started) * 1000)

        results: list[RagResult] = []
        for rank, row in enumerate(top_rows, start=1):
            title = clean_text(row.get("title")) or "Unknown"
            mal_id_raw = row.get("mal_id")
            mal_id = int(mal_id_raw) if isinstance(mal_id_raw, int) else None
            snippet = clean_text(row.get("snippet"))
            character_aliases = clean_text(row.get("character_aliases"))
            score = float(row.get("score", 0.0))
            citation = self._build_citation(mal_id, title)

            result = RagResult(
                rank=rank,
                mal_id=mal_id,
                title=title,
                snippet=snippet,
                score=score,
                citation=citation,
                read_now_title=title,
            )
            result._character_aliases = character_aliases
            results.append(result)
        total_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "RAG hybrid query ms total=%d embed=%d sparse=%d pinecone=%d rerank=%d collected=%d deduped=%d pool=%d character=%s query=%r",
            total_ms,
            embed_ms,
            sparse_ms,
            pinecone_ms,
            rank_ms,
            len(collected),
            len(deduped),
            candidate_pool,
            query_is_character,
            query[:120],
        )
        return results

    def fetch_live_jikan_details(self, mal_id: int, preferred_media_type: str | None = None) -> dict[str, Any]:
        with self._jikan_lock:
            cached = self._jikan_cache.get(mal_id)
            now = time.monotonic()
            if cached and cached[0] > now:
                return cached[1]
        deadline = time.monotonic() + self._jikan_request_budget_seconds

        media_types: tuple[str, str] | tuple[str, ...]
        if preferred_media_type in {"manga", "anime"}:
            alternate = "anime" if preferred_media_type == "manga" else "manga"
            media_types = (preferred_media_type, alternate)
        else:
            media_types = ("manga", "anime")

        for media_type in media_types:
            if time.monotonic() >= deadline:
                return {}
            url = f"{JIKAN_BASE_URL}/{media_type}/{mal_id}/full"
            retry_delay = 0.5
            for attempt in range(self._jikan_max_retries):
                if time.monotonic() >= deadline:
                    return {}
                try:
                    with self._jikan_lock:
                        wait_for = self._next_jikan_request_at - time.monotonic()
                        if wait_for > 0:
                            remaining = max(0.0, deadline - time.monotonic())
                            if remaining <= 0:
                                return {}
                            time.sleep(min(wait_for, remaining))
                        self._next_jikan_request_at = time.monotonic() + self._jikan_min_interval

                    response = self._http.get(url)
                    if response.status_code == 404:
                        break
                    if response.status_code == 429:
                        return {}
                    if response.status_code in JIKAN_RETRYABLE_STATUS and attempt < self._jikan_max_retries - 1:
                        retry_after = response.headers.get("retry-after")
                        retry_after_seconds = 0.0
                        if retry_after and retry_after.isdigit():
                            retry_after_seconds = float(retry_after)
                        remaining = max(0.0, deadline - time.monotonic())
                        if remaining <= 0:
                            return {}
                        time.sleep(min(max(retry_delay, retry_after_seconds), remaining))
                        retry_delay = min(retry_delay * 2.0, 4.0)
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        payload["media_type"] = media_type
                        with self._jikan_lock:
                            self._jikan_cache[mal_id] = (time.monotonic() + self._jikan_cache_ttl, payload)
                        return payload
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt >= self._jikan_max_retries - 1:
                        LOGGER.warning("Live Jikan fetch failed for %s/%s: %s", media_type, mal_id, exc)
                        break
                    remaining = max(0.0, deadline - time.monotonic())
                    if remaining <= 0:
                        return {}
                    time.sleep(min(retry_delay, remaining))
                    retry_delay = min(retry_delay * 2.0, 4.0)
        return {}

    def synthesize_justification(
        self,
        *,
        query: str,
        top_results: list[RagResult],
        live_jikan_payload: dict[str, Any],
        llm_reasoning: str | None = None,
    ) -> str:
        if not top_results:
            return ""

        top = top_results[0]
        payload_data = live_jikan_payload.get("data", {}) if isinstance(live_jikan_payload, dict) else {}
        status = clean_text(payload_data.get("status")) if isinstance(payload_data, dict) else ""
        status_note = f" Status: {status}." if status else ""
        reasoning = clean_text(llm_reasoning or "")
        reasoning = re.sub(r"\[Source:[^\]]+\]", "", reasoning).strip()

        if reasoning:
            return f"Top match for '{query}' is {top.title}. {reasoning}{status_note} {top.citation}"

        return f"Top match for '{query}' is {top.title}.{status_note} {top.citation}"

    def llm_rank_candidates(self, query: str, candidates: list[RagResult]) -> tuple[list[RagResult], str | None]:
        if len(candidates) <= 1:
            return candidates, None
        if not self._llm_rerank_enabled or self._llm is None:
            return candidates, None
        if time.monotonic() < self._llm_backoff_until:
            return candidates, None

        query_tokens = _query_tokens(query)
        query_is_character = _is_character_query(query, query_tokens, self._alias_phrase_set, self._alias_token_set)
        if query_is_character:
            top_exact, _ = _character_match_stats(
                query_tokens=query_tokens,
                snippet=candidates[0].snippet,
                metadata_aliases=candidates[0]._character_aliases,
            )
            if top_exact:
                return candidates, None
        default_cap = "18" if query_is_character else "14"
        llm_candidate_cap = int(os.getenv("RAG_LLM_RERANK_CANDIDATES", default_cap))
        llm_candidate_cap = max(self.config.top_k, llm_candidate_cap)
        pool = candidates[:llm_candidate_cap]

        candidate_lines: list[str] = []
        # Semantic plot queries need enough synopsis text for relation-level judging.
        # Too-short snippets make the LLM overvalue incidental keyword overlap.
        snippet_limit = 240 if query_is_character else 360
        for idx, row in enumerate(pool, start=1):
            snippet = clean_text(row.snippet)[:snippet_limit]
            snippet_lower = row.snippet.lower()
            if "type: manga" in snippet_lower:
                media_type = "manga"
            elif "type: anime" in snippet_lower:
                media_type = "anime"
            else:
                media_type = "unknown"
            alias_exact, alias_hits = _character_match_stats(
                query_tokens=query_tokens,
                snippet=row.snippet,
                metadata_aliases=row._character_aliases,
            )
            candidate_lines.append(
                f"id={idx} | title={row.title} | media_type={media_type} | mal_id={row.mal_id} | score={row.score:.5f} | "
                f"alias_exact={alias_exact} | alias_hits={alias_hits} | snippet={snippet}"
            )

        system_prompt = (
            "You rank retrieval candidates for a manga/anime search query.\n"
            "Use user intent, plot semantics, character/name clues, and ambiguity handling.\n"
            "The Pinecone score is a retrieval hint, not final truth.\n"
            "Choose the closest concrete match to the query details, not just broad thematic overlap.\n"
            "First infer the user's requested relationship/event, then judge whether each candidate's synopsis supports that relationship as a central premise.\n"
            "Do not rank a candidate first merely because it contains query words; incidental mentions, background religion, or unrelated third-party conflicts are weak evidence.\n"
            "For 'X vs Y', 'X against Y', or similar opposition queries, prefer candidates where X and Y are active opposing sides in a contest, war, judgment, rivalry, or direct conflict.\n"
            "If another candidate explicitly satisfies the requested relationship, it must outrank a candidate with only partial keyword overlap.\n"
            "If the query names a character/person and several franchise entries match, rank the core/canonical series entry above recaps/movies/seasons unless the user asked for those.\n"
            "For character queries, exact alias matches (alias_exact=true) are highly authoritative and should stay above non-matches.\n"
            "If the query includes explicit nouns or key terms, prefer candidates whose title/snippet includes those terms only when the surrounding evidence matches the query intent.\n"
            "When a query is ambiguous, put the best match first and keep close contenders after it.\n"
            "In reasoning, briefly cite the decisive evidence from the winning synopsis and why weaker contenders are less exact if relevant.\n"
            "Return strict JSON only with this schema:\n"
            '{"ordered_candidate_ids":[int,...],"close_contenders":[int,...],"reasoning":"short text"}'
        )
        human_prompt = (
            f"User query:\n{query}\n\n"
            f"Query tokens: {query_tokens}\n"
            f"Character-name style query: {query_is_character}\n\n"
            f"Candidates:\n" + "\n".join(candidate_lines) + "\n\n"
            "Return JSON now."
        )

        ordered_ids: list[int] = []
        llm_reasoning: str | None = None
        try:
            response = self._llm.invoke([("system", system_prompt), ("human", human_prompt)])
            content = clean_text(getattr(response, "content", ""))
            parsed = _extract_first_json_object(content)
            if isinstance(parsed, dict):
                raw_ids = parsed.get("ordered_candidate_ids")
                if isinstance(raw_ids, list):
                    for item in raw_ids:
                        if isinstance(item, int):
                            ordered_ids.append(item)
                reasoning = clean_text(parsed.get("reasoning"))
                if reasoning:
                    llm_reasoning = reasoning
        except Exception as exc:  # noqa: BLE001
            exc_text = clean_text(str(exc)).lower()
            if "429" in exc_text or "quota" in exc_text or "queue_exceeded" in exc_text:
                cooldown = max(10.0, float(os.getenv("RAG_LLM_BACKOFF_SECONDS", "60")))
                self._llm_backoff_until = time.monotonic() + cooldown
            LOGGER.warning("LLM ranking fallback to retrieval order: %s", exc)
            return candidates, None

        if not ordered_ids:
            return candidates, llm_reasoning

        by_id = {idx: row for idx, row in enumerate(pool, start=1)}
        reranked: list[RagResult] = []
        seen: set[int] = set()
        for candidate_id in ordered_ids:
            if candidate_id in seen:
                continue
            row = by_id.get(candidate_id)
            if row is None:
                continue
            seen.add(candidate_id)
            reranked.append(row)

        for idx, row in enumerate(pool, start=1):
            if idx in seen:
                continue
            reranked.append(row)

        tail = candidates[len(pool) :]
        return reranked + tail, llm_reasoning

    def normalize_ranked_results(self, query: str, ranked: list[RagResult]) -> list[RagResult]:
        if not ranked:
            return ranked

        items = list(ranked)
        query_tokens = _query_tokens(query)
        query_is_character = _is_character_query(query, query_tokens, self._alias_phrase_set, self._alias_token_set)

        if query_is_character:
            best_exact_idx: int | None = None
            best_exact_score = float("-inf")
            for idx, row in enumerate(items[:20]):
                alias_exact, _ = _character_match_stats(
                    query_tokens=query_tokens,
                    snippet=row.snippet,
                    metadata_aliases=row._character_aliases,
                )
                if not alias_exact:
                    continue
                row_score = float(row.score)
                if row_score > best_exact_score:
                    best_exact_score = row_score
                    best_exact_idx = idx
            if best_exact_idx is not None and best_exact_idx > 0:
                exact_match = items.pop(best_exact_idx)
                items.insert(0, exact_match)

            top_root = _normalized_text(_canonical_root_title(items[0].title))
            if top_root:
                canonical_idx: int | None = None
                for idx, row in enumerate(items[:20]):
                    if _normalized_text(row.title) == top_root:
                        canonical_idx = idx
                        break
                if canonical_idx is not None and canonical_idx > 0:
                    canonical = items.pop(canonical_idx)
                    items.insert(0, canonical)

        return items

    @staticmethod
    def _copy_response(response: RagSearchResponse) -> RagSearchResponse:
        return response.model_copy(deep=True)

    def _get_cached_response(self, cache_key: str) -> RagSearchResponse | None:
        if self._response_cache_ttl <= 0 or not cache_key:
            return None
        with self._search_cache_lock:
            cached = self._response_cache.get(cache_key)
            now = time.monotonic()
            if cached and cached[0] > now:
                return self._copy_response(cached[1])
            if cached:
                self._response_cache.pop(cache_key, None)
        return None

    def _store_cached_response(self, cache_key: str, response: RagSearchResponse) -> None:
        if self._response_cache_ttl <= 0 or self._response_cache_max <= 0 or not cache_key:
            return
        with self._search_cache_lock:
            while len(self._response_cache) >= self._response_cache_max:
                oldest_key = next(iter(self._response_cache), None)
                if oldest_key is None:
                    break
                self._response_cache.pop(oldest_key, None)
            self._response_cache[cache_key] = (
                time.monotonic() + self._response_cache_ttl,
                self._copy_response(response),
            )

    def _search_uncached(self, query: str) -> RagSearchResponse:
        started = time.perf_counter()
        direct_started = time.perf_counter()
        title_first = self.direct_title_retrieve(query)
        direct_ms = int((time.perf_counter() - direct_started) * 1000)
        retrieved = title_first if title_first else self.hybrid_retrieve(query)
        if not retrieved:
            return RagSearchResponse(query=query, top_results=[], highlight=None)

        llm_reasoning: str | None = None
        llm_ms = 0
        if title_first:
            ranked = retrieved
        else:
            llm_started = time.perf_counter()
            ranked, llm_reasoning = self.llm_rank_candidates(query, retrieved)
            llm_ms = int((time.perf_counter() - llm_started) * 1000)
        normalized_ranked = ranked if title_first else self.normalize_ranked_results(query, ranked)
        top_results_raw = normalized_ranked[: self.config.top_k]
        top_results: list[RagResult] = []
        for idx, row in enumerate(top_results_raw, start=1):
            top_results.append(
                RagResult(
                    rank=idx,
                    mal_id=row.mal_id,
                    title=row.title,
                    snippet=row.snippet,
                    score=row.score,
                    citation=row.citation,
                    read_now_title=row.read_now_title,
                )
            )
        if not top_results:
            return RagSearchResponse(query=query, top_results=[], highlight=None)

        top_result = top_results[0]

        justification = self.synthesize_justification(
            query=query,
            top_results=top_results,
            live_jikan_payload={},
            llm_reasoning=llm_reasoning,
        )

        highlight = HighlightCard(
            mal_id=top_result.mal_id,
            title=top_result.title,
            justification=justification,
            citation=top_result.citation,
            live_media_type=_preferred_media_type_from_snippet(top_result.snippet),
            live_status=None,
            live_score=None,
            image_url=None,
        )

        response = RagSearchResponse(query=query, top_results=top_results, highlight=highlight)
        total_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "RAG search ms total=%d direct_title=%d llm=%d mode=%s results=%d query=%r",
            total_ms,
            direct_ms,
            llm_ms,
            "title" if title_first else "hybrid",
            len(top_results),
            query[:120],
        )
        return response

    def search(self, query: str) -> RagSearchResponse:
        cache_key = _normalized_text(query)
        cached = self._get_cached_response(cache_key)
        if cached:
            LOGGER.info("RAG search cache hit query=%r", query[:120])
            return cached

        owner = False
        wait_event: threading.Event | None = None
        if cache_key:
            with self._search_cache_lock:
                wait_event = self._inflight_searches.get(cache_key)
                if wait_event is None:
                    wait_event = threading.Event()
                    self._inflight_searches[cache_key] = wait_event
                    owner = True

        if wait_event is not None and not owner:
            wait_event.wait(max(0.1, float(os.getenv("RAG_INFLIGHT_WAIT_SECONDS", "15"))))
            cached_after_wait = self._get_cached_response(cache_key)
            if cached_after_wait:
                LOGGER.info("RAG search in-flight shared query=%r", query[:120])
                return cached_after_wait

        try:
            response = self._search_uncached(query)
            self._store_cached_response(cache_key, response)
            return self._copy_response(response)
        finally:
            if owner and wait_event is not None:
                with self._search_cache_lock:
                    wait_event.set()
                    self._inflight_searches.pop(cache_key, None)


app = FastAPI(title="Manga RAG API", version="2.1.0")
_service: HybridRagService | None = None
_startup_error: str | None = None


def get_service() -> HybridRagService:
    global _service
    if _service is None:
        _service = HybridRagService(load_config())
    return _service


@app.get("/health")
def health() -> dict[str, str]:
    if _startup_error:
        return {"status": "error", "detail": _startup_error}
    try:
        get_service()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "initialized": "true"}


@app.on_event("startup")
def warm_start_service() -> None:
    global _startup_error
    started = time.perf_counter()
    try:
        get_service()
        _startup_error = None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info("RAG service warm-start complete ms=%d.", elapsed_ms)
    except Exception as exc:  # noqa: BLE001
        _startup_error = str(exc)
        LOGGER.exception("RAG warm-start failed.")


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(payload: RagSearchRequest) -> RagSearchResponse:
    query = clean_text(payload.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        return get_service().search(query)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("RAG search failed.")
        raise HTTPException(status_code=500, detail=f"RAG search failed: {exc}") from exc
