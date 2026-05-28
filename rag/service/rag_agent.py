from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote_plus

import httpx
from langchain_cerebras import ChatCerebras
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .models import RagSearchResponse, RagSearchResult

JIKAN_BASE_URL = "https://api.jikan.moe/v4"
MAL_LINK_RE = re.compile(r"myanimelist\.net/manga/\d+/([^/?#\s]+)", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

STRICT_RESOLVE_FETCH_SYSTEM_PROMPT = (
    "You are a manga resolver and fetch agent.\n\n"
    "Follow this logic exactly:\n"
    "1) If the user provides an exact manga title, use the `fetch_from_jikan_by_exact_title` tool immediately.\n"
    "2) If the user provides a plot description, vibe, character hint, or any vague query, you MUST NOT use Jikan first.\n"
    "   You MUST first call `web_search_title_resolver` to discover the canonical exact manga title.\n"
    "   Only AFTER you have the exact title (example: Death Note), call `fetch_from_jikan_by_exact_title`.\n\n"
    "Never pass plot descriptions, full sentences, or vague keywords into the Jikan tool.\n"
    "Final response must cite MAL IDs as [Source: MAL-ID <id>] when available."
)


class WebSearchTitleResolverInput(BaseModel):
    description: str = Field(
        min_length=2,
        description=(
            "Vague user description, vibe, or semantic query to resolve into an exact manga title."
        ),
    )


class FetchFromJikanExactTitleInput(BaseModel):
    exact_title: str = Field(
        min_length=1,
        description=(
            "ONLY pass exact, canonical manga titles to this tool. "
            "Never pass plot descriptions, full sentences, or vague keywords."
        ),
    )


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _looks_like_exact_title(query: str) -> bool:
    cleaned = _clean_text(query)
    if not cleaned:
        return False

    words = WORD_RE.findall(cleaned)
    if len(words) == 0 or len(words) > 8:
        return False

    lowered = cleaned.lower()
    vague_signals = [
        "about",
        "where",
        "who",
        "guy",
        "girl",
        "story",
        "plot",
        "vibe",
        "find",
        "looking for",
        "something like",
    ]
    return not any(signal in lowered for signal in vague_signals)


def _title_match_score(query_title: str, candidate_title: str) -> float:
    query_norm = _normalize_title(query_title)
    candidate_norm = _normalize_title(candidate_title)
    if not query_norm or not candidate_norm:
        return float("inf")
    if query_norm == candidate_norm:
        return 0.0
    if candidate_norm.startswith(query_norm) or query_norm.startswith(candidate_norm):
        return 1.0
    if candidate_norm in query_norm or query_norm in candidate_norm:
        return 2.0

    query_tokens = set(query_norm.split(" "))
    candidate_tokens = set(candidate_norm.split(" "))
    overlap = len(query_tokens.intersection(candidate_tokens))
    if overlap == 0:
        return 10.0

    return 3.0 - min(overlap * 0.4, 1.8)


def _extract_image_url(item: dict[str, Any]) -> str | None:
    images = item.get("images")
    if not isinstance(images, dict):
        return None

    jpg = images.get("jpg")
    if isinstance(jpg, dict):
        for key in ("large_image_url", "image_url", "small_image_url"):
            value = _clean_text(jpg.get(key))
            if value:
                return value

    webp = images.get("webp")
    if isinstance(webp, dict):
        for key in ("large_image_url", "image_url", "small_image_url"):
            value = _clean_text(webp.get(key))
            if value:
                return value

    return None


def _build_read_options(title: str) -> dict[str, str]:
    clean = _clean_text(title)
    underscored = re.sub(r"\s+", "_", clean)
    encoded = quote_plus(clean)
    return {
        "mangakatana": f"https://mangakatana.com/?search={encoded}",
        "weebcentral": f"https://weebcentral.com/search?text={encoded}",
        "mangakakalot": f"https://mangakakalot.com/search/story/{underscored}",
        "assortedscans": f"https://assortedscans.com/?s={encoded}",
    }


class MangaRagAgent:
    def __init__(
        self,
        *,
        pinecone_index: str,
        pinecone_namespace: str = "manga-rag",
        bm25_path: str = "rag/ingestion/bm25_values.json",
        top_k: int = 10,
        score_threshold: float = 0.75,
        recursion_limit: int = 8,
        cerebras_model: str = "gpt-oss-120b",
    ) -> None:
        # Kept for backward compatibility with previous constructor signature.
        self._pinecone_index = pinecone_index
        self._pinecone_namespace = pinecone_namespace
        self._bm25_path = bm25_path
        self._score_threshold = score_threshold

        self._top_k = top_k
        self._recursion_limit = recursion_limit
        self._jikan_cache: dict[str, dict[str, Any]] = {}

        self._llm = ChatCerebras(
            model=cerebras_model,
            api_key=os.environ["CEREBRAS_API_KEY"],
            temperature=0,
        )
        self._ddg = DuckDuckGoSearchRun()
        self._http = httpx.Client(
            timeout=20.0,
            headers={"User-Agent": "PanelFlow-RAG-ResolveFetch/1.0", "Accept": "application/json"},
        )

        self._resolver_tool = self._build_web_search_title_resolver_tool()
        self._jikan_tool = self._build_fetch_from_jikan_by_exact_title_tool()
        self._graph = create_react_agent(
            self._llm,
            tools=[self._resolver_tool, self._jikan_tool],
            prompt=STRICT_RESOLVE_FETCH_SYSTEM_PROMPT,
        )

    def _build_web_search_title_resolver_tool(self):
        @tool("web_search_title_resolver", args_schema=WebSearchTitleResolverInput)
        def web_search_title_resolver(description: str) -> str:
            """Resolve a vague manga description to an exact canonical title via web search."""
            cleaned = _clean_text(description)
            if not cleaned:
                return json.dumps({"resolved_title": "", "search_query": "", "evidence": ""}, ensure_ascii=True)

            forced_query = f"myanimelist manga {cleaned}"
            raw_results = self._ddg.run(forced_query)

            candidate_titles: list[str] = []
            for match in MAL_LINK_RE.findall(raw_results):
                title_from_slug = _clean_text(match.replace("_", " "))
                if title_from_slug and title_from_slug not in candidate_titles:
                    candidate_titles.append(title_from_slug)

            if not candidate_titles:
                # Fallback heuristic: most DDG snippets start with "Title - ..."
                for line in raw_results.split("\n"):
                    left = _clean_text(line.split(" - ")[0])
                    if left and len(left.split(" ")) <= 8:
                        candidate_titles.append(left)
                    if len(candidate_titles) >= 5:
                        break

            resolved_title = candidate_titles[0] if candidate_titles else ""
            payload = {
                "search_query": forced_query,
                "resolved_title": resolved_title,
                "candidate_titles": candidate_titles[:5],
                "evidence": raw_results[:1200],
            }
            return json.dumps(payload, ensure_ascii=True)

        return web_search_title_resolver

    def _fetch_jikan_exact_title_payload(self, exact_title: str) -> dict[str, Any]:
        cleaned = _clean_text(exact_title)
        if not cleaned:
            raise ValueError("Exact title is empty.")

        normalized_key = _normalize_title(cleaned)
        if normalized_key in self._jikan_cache:
            cached = dict(self._jikan_cache[normalized_key])
            cached["retrieval_mode"] = "cache_hit"
            return cached

        response = self._http.get(
            f"{JIKAN_BASE_URL}/manga",
            params={"q": cleaned, "limit": self._top_k},
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("data", [])
        if not isinstance(candidates, list):
            candidates = []

        ranked = sorted(
            candidates,
            key=lambda item: _title_match_score(
                cleaned,
                _clean_text(
                    item.get("title_english") or item.get("title") or item.get("title_japanese") or ""
                ),
            ),
        )

        results: list[dict[str, Any]] = []
        for item in ranked[: self._top_k]:
            mal_id = item.get("mal_id")
            title = _clean_text(item.get("title_english") or item.get("title") or item.get("title_japanese") or "Unknown")
            synopsis = _clean_text(item.get("synopsis") or "")
            genres_raw = item.get("genres")
            genres: list[str] = []
            if isinstance(genres_raw, list):
                for entry in genres_raw:
                    if isinstance(entry, dict):
                        name = _clean_text(entry.get("name"))
                        if name:
                            genres.append(name)

            citation = f"[Source: MAL-ID {mal_id}]" if isinstance(mal_id, int) else "[Source: MAL-ID unknown]"
            image_url = _extract_image_url(item)

            results.append(
                {
                    "title": title,
                    "media_type": _clean_text(item.get("type") or "manga").lower() or "manga",
                    "mal_id": mal_id if isinstance(mal_id, int) else None,
                    "synopsis": synopsis,
                    "genres": genres,
                    "citations": [citation],
                    "score": None,
                    "image_url": image_url,
                    "read_options": _build_read_options(title),
                }
            )

        resolved_title = results[0]["title"] if results else cleaned
        payload = {
            "resolved_title": resolved_title,
            "retrieval_mode": "live_jikan",
            "results": results,
        }
        self._jikan_cache[normalized_key] = payload
        return payload

    def _build_fetch_from_jikan_by_exact_title_tool(self):
        @tool("fetch_from_jikan_by_exact_title", args_schema=FetchFromJikanExactTitleInput)
        def fetch_from_jikan_by_exact_title(exact_title: str) -> str:
            """ONLY pass exact, canonical manga titles to this tool. Never pass plot descriptions, full sentences, or vague keywords."""
            cleaned = _clean_text(exact_title)
            if not cleaned:
                return json.dumps({"resolved_title": "", "retrieval_mode": "live_jikan", "results": []}, ensure_ascii=True)

            if not _looks_like_exact_title(cleaned):
                raise ValueError(
                    "Rejected: input does not look like an exact canonical title. "
                    "Resolve with web_search_title_resolver first."
                )

            payload = self._fetch_jikan_exact_title_payload(cleaned)
            return json.dumps(payload, ensure_ascii=True)

        return fetch_from_jikan_by_exact_title

    @staticmethod
    def _extract_answer(graph_output: dict[str, Any]) -> str:
        messages = graph_output.get("messages", [])
        for message in reversed(messages):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""

    @staticmethod
    def _extract_fetch_payload(graph_output: dict[str, Any]) -> dict[str, Any] | None:
        messages = graph_output.get("messages", [])
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                continue
            if getattr(message, "name", "") != "fetch_from_jikan_by_exact_title":
                continue

            content = getattr(message, "content", "")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _deterministic_resolve_then_fetch(self, query: str) -> dict[str, Any]:
        if _looks_like_exact_title(query):
            return self._fetch_jikan_exact_title_payload(query)

        resolver_output_raw = self._resolver_tool.invoke({"description": query})
        resolver_output: dict[str, Any]
        try:
            resolver_output = json.loads(resolver_output_raw)
        except json.JSONDecodeError:
            resolver_output = {}

        resolved_title = _clean_text(resolver_output.get("resolved_title"))
        if not resolved_title:
            resolved_title = query

        return self._fetch_jikan_exact_title_payload(resolved_title)

    def query(self, query: str) -> RagSearchResponse:
        cleaned_query = _clean_text(query)
        if not cleaned_query:
            return RagSearchResponse(
                query=query,
                answer="Query was empty.",
                results=[],
                retrieval_mode="resolve_then_fetch",
                resolved_title=None,
            )

        graph_output = self._graph.invoke(
            {"messages": [("user", cleaned_query)]},
            config={"recursion_limit": self._recursion_limit},
        )
        answer = self._extract_answer(graph_output)
        tool_payload = self._extract_fetch_payload(graph_output)
        if tool_payload is None:
            tool_payload = self._deterministic_resolve_then_fetch(cleaned_query)

        raw_results = tool_payload.get("results", [])
        parsed_results: list[RagSearchResult] = []
        if isinstance(raw_results, list):
            for row in raw_results[: self._top_k]:
                if not isinstance(row, dict):
                    continue
                parsed_results.append(
                    RagSearchResult(
                        title=_clean_text(row.get("title")) or "Unknown",
                        media_type=_clean_text(row.get("media_type") or "manga") or "manga",
                        mal_id=row.get("mal_id") if isinstance(row.get("mal_id"), int) else None,
                        synopsis=_clean_text(row.get("synopsis")),
                        genres=[_clean_text(item) for item in row.get("genres", []) if _clean_text(item)],
                        citations=[_clean_text(item) for item in row.get("citations", []) if _clean_text(item)],
                        score=float(row.get("score")) if isinstance(row.get("score"), (float, int)) else None,
                        image_url=_clean_text(row.get("image_url")) or None,
                        read_options=row.get("read_options") if isinstance(row.get("read_options"), dict) else {},
                    )
                )

        resolved_title = _clean_text(tool_payload.get("resolved_title")) or None
        retrieval_mode = _clean_text(tool_payload.get("retrieval_mode")) or "resolve_then_fetch"
        final_answer = (
            answer
            if answer
            else (
                f"Resolved '{cleaned_query}' to '{resolved_title}' and fetched top {len(parsed_results)} results."
                if resolved_title
                else "Fetched results from Jikan."
            )
        )

        return RagSearchResponse(
            query=cleaned_query,
            answer=final_answer,
            results=parsed_results,
            retrieval_mode=retrieval_mode,
            resolved_title=resolved_title,
        )
