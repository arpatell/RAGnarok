from __future__ import annotations

from pydantic import BaseModel, Field


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)


class RagSearchResult(BaseModel):
    title: str
    media_type: str
    mal_id: int | None
    synopsis: str
    genres: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    score: float | None = None
    image_url: str | None = None
    read_options: dict[str, str] = Field(default_factory=dict)


class RagSearchResponse(BaseModel):
    query: str
    answer: str
    results: list[RagSearchResult]
    retrieval_mode: str = "resolve_then_fetch"
    resolved_title: str | None = None
