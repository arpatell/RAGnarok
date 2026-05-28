# RAGnarok Reader

RAGnarok Reader is an AI-powered manga and anime discovery app with a universal manga/manhwa reader. It is built for searches that are hard to express as exact titles: plot fragments, character names, abilities, settings, rivalries, or vague memories.

Instead of relying on static keyword matching, RAGnarok combines hybrid retrieval, character-aware metadata, and LLM reranking to reason over likely matches before presenting a clean result list and a reader-first experience.

## Product Overview

RAGnarok is designed around two connected user flows:

- Discover a series by describing what you remember.
- Open a readable chapter experience from a supported or public chapter URL.

Search examples the system is designed to handle:

- `guy finds notebook that can kill people` -> Death Note
- `satoru gojo` -> Jujustu Kaisen
- `satoko kirigaya` -> Firefly Wedding
- `gods vs humans` -> Record of Ragnarok
- `girl can hear ghosts` -> Phantom Whispers

The app supports plot descriptions, title searches, character aliases, and direct URL imports without requiring users to know database IDs or exact romanized titles.

## Core Features

- Natural-language manga and anime discovery.
- Hybrid semantic + keyword retrieval over a curated top-series corpus.
- Character-aware matching using Jikan character metadata and normalized aliases.
- LLM reranking that treats Pinecone results as context, not final ranking.
- Direct title matching for English and romanized titles.
- Clean result cards with synopsis, characters, genres, type, and citations.
- Browser-side Jikan enrichment for images, media subtype, and chapter counts.
- Universal reader with paginated manga mode and continuous manhwa mode.
- Direct browser-side image loading to avoid backend bandwidth-heavy image proxying.
- Reader history, favorites, bookmarks, progress, and read/unread tracking.
- URL paste flow for public chapter pages that expose panel images.

## Search Philosophy

RAGnarok follows a retrieval-first design.

It does not hardcode fixes for individual titles. If a query fails, the intended solution is to improve retrieval quality, metadata coverage, reranking logic, or evidence presentation. The system should remain general across anime and manga, not tuned around one-off examples.

The top Pinecone matches are treated as candidate evidence. The first vector result is not assumed to be the answer. The LLM layer is responsible for comparing candidates against user intent and selecting the best match, with close contenders preserved in the result list.

## RAG Architecture

The RAG system uses a character-enriched corpus built from top anime and manga records. Each document includes title, type, genres, synopsis, character names, and character search aliases.

Retrieval combines:

- Dense semantic search for plot, vibe, premise, and conceptual queries.
- Sparse/BM25-style search for exact character names, aliases, titles, and niche terms.
- Title-first matching for exact or near-exact title queries.
- Query-aware weighting so character/title searches do not behave like vague plot searches.

This lets the system support both:

- Semantic intent: `blind swordsman wandering feudal japan`
- Keyword intent: `Satoru Gojo`

## LLM Reranking

The reranker receives the candidate set from retrieval and judges which result best satisfies the query. Its job is not to invent facts, but to compare the supplied snippets and metadata.

The reranking prompt emphasizes:

- User intent over raw retrieval score.
- Relation-level matching over incidental keyword overlap.
- Character alias authority for exact character-name queries.
- Canonical/core entries over recaps, specials, or unrelated franchise entries.
- Explicit evidence from synopsis and metadata.

For example, a query like `gods vs humans` should favor a series where gods and humans are active opposing sides in a central conflict, rather than a result that merely mentions God or humans in unrelated context.

## Reader Architecture

The reader separates chapter extraction from image delivery.

The backend extracts chapter metadata and panel URLs from public HTML. The frontend then loads panel images directly from the user browser, reducing backend bandwidth usage and avoiding unnecessary image relay costs.

Reader capabilities include:

- Paginated mode for traditional manga.
- Scroll mode for manhwa/webtoon-style reading.
- 0px panel gap support for connected vertical strips.
- Adjacent-panel preloading around the current page.
- Previous/next chapter navigation.
- Sparse chapter-list expansion when chapter count metadata is available.
- Browser history handling for readable back navigation.

## URL Import

The paste feature works best with public chapter pages that expose image panel URLs in HTML or script data. It can also use AI-assisted HTML analysis when a source is not covered by a dedicated adapter.

Some sites intentionally hide, authenticate, hotlink-protect, or dynamically render image URLs. Those sources may not be readable without source-specific support.

## Technology Stack

Frontend:

- React
- TypeScript
- Vite
- Browser-side Jikan API calls
- LocalStorage persistence

Backend:

- Express
- TypeScript
- Cheerio
- Zod
- AI-assisted HTML extraction through Cerebras or OpenAI-compatible APIs

RAG service:

- Python
- FastAPI
- Pinecone
- LangChain
- Cerebras
- BM25 sparse retrieval
- Dense embeddings
- Character-enriched Jikan/MAL metadata

Data sources:

- Jikan API for anime/manga metadata, character rosters, images, types, and chapter counts.
- Pinecone for hybrid vector retrieval.
- Public manga chapter pages for reader panel extraction.

## Theoretical Applications

RAGnarok’s architecture is useful beyond manga/anime search.

The same approach can support:

- Media discovery from vague memory.
- Character-aware search for games, books, comics, or film.
- Customer support retrieval where exact keyword matches are insufficient.
- Legal or medical document triage where user phrasing differs from source terminology.
- Product catalogs with aliases, nicknames, specs, and subjective descriptions.
- Research assistants that need hybrid retrieval plus final LLM judgment.

The key pattern is separating candidate retrieval from final judgment. Retrieval maximizes recall; reranking reasons over intent, ambiguity, and evidence.

## Repository Layout

- `apps/frontend` contains the React reader and discovery UI.
- `apps/backend` contains the Express API, URL ingestion, AI extraction, and RAG proxy.
- `rag` contains the Python RAG service, ingestion scripts, repair tools, and evaluation code.
- `scripts/gcp` contains VM helper scripts for the current backend/RAG hosting path.
- `packages` is reserved for shared workspace packages.

Generated ingestion files, local caches, virtual environments, logs, and build outputs are intentionally ignored.

