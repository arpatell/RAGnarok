import { IngestError } from "../errors.js";

const DEFAULT_RAG_API_URL = "http://127.0.0.1:8090/rag/search";
const DEFAULT_TOP_K = 10;
const REQUEST_TIMEOUT_MS = Number.parseInt(process.env.RAG_REQUEST_TIMEOUT_MS ?? "30000", 10);
const JIKAN_TIMEOUT_MS = Math.max(3_000, Number.parseInt(process.env.JIKAN_TIMEOUT_MS ?? "6500", 10));
const JIKAN_BASE_URL = "https://api.jikan.moe/v4";
const JIKAN_MIN_INTERVAL_MS = Math.max(450, Number.parseInt(process.env.JIKAN_MIN_INTERVAL_MS ?? "700", 10));
const JIKAN_MAX_RETRIES = Math.max(1, Number.parseInt(process.env.JIKAN_MAX_RETRIES ?? "3", 10));
const JIKAN_CACHE_TTL_MS = Math.max(60_000, Number.parseInt(process.env.JIKAN_CACHE_TTL_MS ?? "300000", 10));
const JIKAN_RETRYABLE_STATUS = new Set([408, 425, 500, 502, 503, 504, 520, 522, 524]);
const JIKAN_ENRICH_TOP_N = Math.max(0, Number.parseInt(process.env.RAG_JIKAN_ENRICH_TOP_N ?? "2", 10));
const JIKAN_ANIME_COMPANION_TOP_N = Math.max(0, Number.parseInt(process.env.RAG_JIKAN_ANIME_COMPANION_TOP_N ?? "0", 10));

let nextJikanRequestAt = 0;
const jikanJsonCache = new Map<string, { expiresAt: number; value: unknown }>();

type ResultSource = "pinecone";

type JikanMediaType = "manga" | "anime";

interface RemoteRagResult {
  rank?: number;
  mal_id?: number | null;
  title?: string;
  snippet?: string;
  score?: number;
  citation?: string;
}

interface RemoteRagPayload {
  query?: string;
  answer?: string;
  retrieval_mode?: string;
  top_results?: RemoteRagResult[];
  results?: RemoteRagResult[];
  highlight?: {
    title?: string;
    mal_id?: number | null;
    justification?: string;
    citation?: string;
    image_url?: string | null;
  } | null;
}

interface JikanImageVariant {
  image_url?: string | null;
  large_image_url?: string | null;
  small_image_url?: string | null;
}

interface JikanRelationEntry {
  mal_id?: number;
  type?: string;
  name?: string;
}

interface JikanRelation {
  relation?: string;
  entry?: JikanRelationEntry[];
}

interface JikanMediaData {
  mal_id?: number;
  title?: string | null;
  title_english?: string | null;
  title_japanese?: string | null;
  synopsis?: string | null;
  type?: string | null;
  status?: string | null;
  chapters?: number | null;
  volumes?: number | null;
  episodes?: number | null;
  score?: number | null;
  genres?: Array<{ name?: string | null }>;
  images?: {
    jpg?: JikanImageVariant;
    webp?: JikanImageVariant;
  };
  relations?: JikanRelation[];
}

interface JikanFullPayload {
  data?: JikanMediaData;
}

interface AnimeCompanion {
  mal_id: number | null;
  title: string;
  status: string | null;
  episodes: number | null;
  score: number | null;
  image_url: string | null;
  watch_url: string;
  citation: string;
}

interface JikanEnrichment {
  mediaType: JikanMediaType;
  typeLabel: string | null;
  title: string;
  synopsis: string;
  genres: string[];
  status: string | null;
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  score: number | null;
  imageUrl: string | null;
  animeCompanion: AnimeCompanion | null;
}

export interface RagSearchResult {
  title: string;
  media_type: string;
  display_type: string | null;
  mal_id: number | null;
  synopsis: string;
  characters: string[];
  genres: string[];
  citations: string[];
  score: number | null;
  source: ResultSource;
  source_label: string;
  image_url: string | null;
  status: string | null;
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  jikan_score: number | null;
  read_options: Record<string, string>;
  watch_options: Record<string, string>;
  anime_companion: AnimeCompanion | null;
}

export interface RagSearchResponse {
  query: string;
  answer: string;
  retrieval_mode: string;
  results: RagSearchResult[];
  highlight: {
    title: string;
    mal_id: number | null;
    justification: string;
    citation: string;
  } | null;
}

export interface RagResultLiveMetadata {
  mal_id: number;
  media_type: string;
  display_type: string | null;
  title: string;
  image_url: string | null;
  status: string | null;
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  jikan_score: number | null;
}

function cleanText(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }
  return value
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function safeNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function encodeTitle(value: string): string {
  return encodeURIComponent(value.trim());
}

function encodeTitleForPath(value: string): string {
  return value
    .trim()
    .replace(/\s+/g, "_")
    .replace(/_{2,}/g, "_");
}

function buildReadOptions(title: string): Record<string, string> {
  const encoded = encodeTitle(title);
  const underscored = encodeTitleForPath(title);
  return {
    mangakatana: `https://mangakatana.com/?search=${encoded}`,
    weebcentral: `https://weebcentral.com/search?text=${encoded}`,
    mangakakalot: `https://mangakakalot.com/search/story/${underscored}`,
    assortedscans: `https://assortedscans.com/?s=${encoded}`
  };
}

function buildWatchOptions(title: string): Record<string, string> {
  const keyword = title
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .join("+");

  if (!keyword) {
    return {};
  }

  return {
    anikaitv: `https://anikaitv.to/filter?keyword=${keyword}`
  };
}

function buildCitation(malId: number | null, title: string): string {
  if (malId !== null) {
    return `[Source: MAL-ID ${malId} - ${title}]`;
  }
  return `[Source: MAL-ID unknown - ${title}]`;
}

function normalizeCitationList(citation: string | undefined, fallback: string): string[] {
  const normalized = cleanText(citation || "");
  if (!normalized) {
    return [fallback];
  }

  if (normalized.startsWith("[Source:")) {
    return [normalized];
  }

  return [fallback, normalized];
}

function extractImageUrl(images: JikanMediaData["images"]): string | null {
  const candidates: Array<string | null | undefined> = [
    images?.jpg?.large_image_url,
    images?.jpg?.image_url,
    images?.jpg?.small_image_url,
    images?.webp?.large_image_url,
    images?.webp?.image_url,
    images?.webp?.small_image_url
  ];

  for (const candidate of candidates) {
    const clean = cleanText(candidate);
    if (clean) {
      return clean;
    }
  }

  return null;
}

function extractGenres(data: JikanMediaData): string[] {
  if (!Array.isArray(data.genres)) {
    return [];
  }

  return data.genres
    .map((genre) => cleanText(genre?.name))
    .filter((genre) => genre.length > 0)
    .slice(0, 8);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeRetryDelayMs(attempt: number, retryAfterHeader: string | null): number {
  if (retryAfterHeader) {
    const retryAfterSeconds = Number.parseInt(retryAfterHeader, 10);
    if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0) {
      return Math.min(8_000, retryAfterSeconds * 1000);
    }

    const retryDateMs = Date.parse(retryAfterHeader);
    if (Number.isFinite(retryDateMs)) {
      const delta = retryDateMs - Date.now();
      if (delta > 0) {
        return Math.min(8_000, delta);
      }
    }
  }

  const exponential = Math.min(8_000, 300 * 2 ** attempt);
  const jitter = Math.floor(Math.random() * 250);
  return exponential + jitter;
}

async function waitForJikanSlot(): Promise<void> {
  const now = Date.now();
  const waitFor = nextJikanRequestAt - now;
  if (waitFor > 0) {
    await delay(waitFor);
  }
  nextJikanRequestAt = Date.now() + JIKAN_MIN_INTERVAL_MS;
}

async function fetchJikanJson<T>(url: string, timeoutMs: number): Promise<T> {
  const cached = jikanJsonCache.get(url);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.value as T;
  }

  let lastError: Error | null = null;

  for (let attempt = 0; attempt < JIKAN_MAX_RETRIES; attempt += 1) {
    await waitForJikanSlot();

    let response: Response;
    try {
      response = await fetch(url, {
        signal: AbortSignal.timeout(timeoutMs),
        headers: {
          Accept: "application/json",
          "Cache-Control": "no-cache",
          "User-Agent": "PanelFlow-RAG/2.1"
        }
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      lastError = new Error(`Request failed: ${message}`);
      if (attempt < JIKAN_MAX_RETRIES - 1) {
        await delay(computeRetryDelayMs(attempt, null));
        continue;
      }
      throw lastError;
    }

    if (response.ok) {
      const payload = (await response.json()) as T;
      jikanJsonCache.set(url, {
        expiresAt: Date.now() + JIKAN_CACHE_TTL_MS,
        value: payload
      });
      return payload;
    }

    const body = await response.text().catch(() => "");
    const message = `HTTP ${response.status}: ${body.slice(0, 220)}`;
    lastError = new Error(message);

    if (response.status === 429) {
      throw lastError;
    }

    if (!JIKAN_RETRYABLE_STATUS.has(response.status) || attempt >= JIKAN_MAX_RETRIES - 1) {
      throw lastError;
    }

    await delay(computeRetryDelayMs(attempt, response.headers.get("retry-after")));
  }

  throw lastError ?? new Error("Jikan request failed after retries.");
}

function parseMediaTypeFromSnippet(snippet: string): JikanMediaType | null {
  const normalized = snippet.toLowerCase();
  if (normalized.includes("type: manga")) {
    return "manga";
  }
  if (normalized.includes("type: anime")) {
    return "anime";
  }
  return null;
}

function parseDisplayTypeFromSnippet(snippet: string): string | null {
  const extracted = extractSnippetSection(snippet, "type:", [
    "title:",
    "synopsis:",
    "characters:",
    "character search aliases:",
    "genres:"
  ]);
  if (!extracted) {
    return null;
  }

  const normalized = extracted.split(/[|;]/)[0]?.trim() ?? "";
  return normalized.length > 0 ? normalized : null;
}

function parseMediaHint(value: string | null | undefined): JikanMediaType | null {
  const normalized = cleanText(value).toLowerCase();
  if (!normalized) {
    return null;
  }

  if (normalized.includes("anime")) {
    return "anime";
  }

  if (
    normalized.includes("manga") ||
    normalized.includes("manhwa") ||
    normalized.includes("manhua") ||
    normalized.includes("novel") ||
    normalized.includes("doujin")
  ) {
    return "manga";
  }

  return null;
}

function extractSnippetSection(snippet: string, marker: string, stopMarkers: string[]): string {
  const normalized = cleanText(snippet);
  if (!normalized) {
    return "";
  }

  const lower = normalized.toLowerCase();
  const markerLower = marker.toLowerCase();
  const markerIdx = lower.indexOf(markerLower);
  if (markerIdx < 0) {
    return "";
  }

  const startIdx = markerIdx + markerLower.length;
  let endIdx = normalized.length;
  for (const stopMarker of stopMarkers) {
    const stopLower = stopMarker.toLowerCase();
    if (stopLower === markerLower) {
      continue;
    }
    const idx = lower.indexOf(stopLower, startIdx);
    if (idx >= 0 && idx < endIdx) {
      endIdx = idx;
    }
  }

  return normalized
    .slice(startIdx, endIdx)
    .replace(/^[-:|]+/, "")
    .trim();
}

function extractSynopsisFromSnippet(snippet: string): string {
  const extracted = extractSnippetSection(snippet, "synopsis:", [
    "characters:",
    "character search aliases:",
    "genres:",
    "type:",
    "title:"
  ]);

  if (!extracted || /^unknown$/i.test(extracted)) {
    return "";
  }
  return extracted;
}

function extractGenresFromSnippet(snippet: string): string[] {
  const extracted = extractSnippetSection(snippet, "genres:", [
    "characters:",
    "character search aliases:",
    "synopsis:",
    "type:",
    "title:"
  ]);
  if (!extracted || /^unknown$/i.test(extracted)) {
    return [];
  }

  const seen = new Set<string>();
  const genres: string[] = [];
  for (const token of extracted.split(/[;,|]+/)) {
    const value = cleanText(token);
    if (!value) {
      continue;
    }
    const key = value.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    genres.push(value);
    if (genres.length >= 8) {
      break;
    }
  }

  return genres;
}

function extractCharactersFromSnippet(snippet: string): string[] {
  const extracted = extractSnippetSection(snippet, "characters:", [
    "character search aliases:",
    "genres:",
    "synopsis:",
    "type:",
    "title:"
  ]);
  if (!extracted || /^(unknown|none)$/i.test(extracted)) {
    return [];
  }

  const roughParts = extracted.split(/[;|]+/);
  const tokens = roughParts.length > 1 ? roughParts : extracted.split(/,\s+/);
  const seen = new Set<string>();
  const characters: string[] = [];

  for (const token of tokens) {
    const value = cleanText(token).replace(/\s*-\s*.*$/, "").trim();
    if (!value || value.length < 2) {
      continue;
    }
    const key = value.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    characters.push(value);
    if (characters.length >= 24) {
      break;
    }
  }

  return characters;
}

async function fetchAnimeCompanion(
  mangaTitle: string,
  mangaFull: JikanMediaData
): Promise<AnimeCompanion | null> {
  const relations = Array.isArray(mangaFull.relations) ? mangaFull.relations : [];
  const adaptation = relations.find((row) => cleanText(row.relation).toLowerCase() === "adaptation");
  const animeEntry = adaptation?.entry?.find((entry) => cleanText(entry.type).toLowerCase() === "anime");

  let animeData: JikanMediaData | null = null;

  if (animeEntry?.mal_id && Number.isFinite(animeEntry.mal_id)) {
    try {
      const payload = await fetchJikanJson<JikanFullPayload>(
        `${JIKAN_BASE_URL}/anime/${animeEntry.mal_id}/full`,
        JIKAN_TIMEOUT_MS
      );
      animeData = payload.data ?? null;
    } catch {
      animeData = null;
    }
  }

  if (!animeData) {
    try {
      const searchUrl = new URL(`${JIKAN_BASE_URL}/anime`);
      searchUrl.searchParams.set("q", mangaTitle);
      searchUrl.searchParams.set("limit", "1");
      const payload = await fetchJikanJson<{ data?: JikanMediaData[] }>(searchUrl.toString(), JIKAN_TIMEOUT_MS);
      animeData = Array.isArray(payload.data) && payload.data.length > 0 ? payload.data[0] ?? null : null;
    } catch {
      animeData = null;
    }
  }

  if (!animeData) {
    return null;
  }

  const title =
    cleanText(animeData.title_english) ||
    cleanText(animeData.title) ||
    cleanText(animeData.title_japanese) ||
    cleanText(mangaTitle);
  const malId = safeNumber(animeData.mal_id);

  return {
    mal_id: malId,
    title,
    status: cleanText(animeData.status) || null,
    episodes: safeNumber(animeData.episodes),
    score: safeNumber(animeData.score),
    image_url: extractImageUrl(animeData.images),
    watch_url: buildWatchOptions(title).anikaitv ?? "",
    citation: buildCitation(malId, title)
  };
}

async function fetchJikanDetails(
  malId: number,
  fallbackTitle: string,
  preferredMediaType: JikanMediaType | null,
  includeAnimeCompanion: boolean
): Promise<JikanEnrichment | null> {
  const mediaTypeOrder: JikanMediaType[] = preferredMediaType
    ? [preferredMediaType, preferredMediaType === "manga" ? "anime" : "manga"]
    : ["manga", "anime"];

  for (const mediaType of mediaTypeOrder) {
    try {
      const payload = await fetchJikanJson<JikanFullPayload>(`${JIKAN_BASE_URL}/${mediaType}/${malId}/full`, JIKAN_TIMEOUT_MS);
      const data = payload.data;
      if (!data) {
        continue;
      }

      const canonicalTitle =
        cleanText(data.title_english) ||
        cleanText(data.title) ||
        cleanText(data.title_japanese) ||
        fallbackTitle;

      const enrichment: JikanEnrichment = {
        mediaType,
        typeLabel: cleanText(data.type) || null,
        title: canonicalTitle,
        synopsis: cleanText(data.synopsis),
        genres: extractGenres(data),
        status: cleanText(data.status) || null,
        chapters: safeNumber(data.chapters),
        volumes: safeNumber(data.volumes),
        episodes: safeNumber(data.episodes),
        score: safeNumber(data.score),
        imageUrl: extractImageUrl(data.images),
        animeCompanion:
          includeAnimeCompanion && mediaType === "manga"
            ? await fetchAnimeCompanion(canonicalTitle, data)
            : null
      };

      return enrichment;
    } catch {
      // Try next media type.
    }
  }

  return null;
}

async function callHybridRagService(query: string): Promise<RemoteRagPayload> {
  const raw = process.env.RAG_API_URL?.trim();
  const target = raw && raw.length > 0 ? raw : DEFAULT_RAG_API_URL;

  let response: Response;
  try {
    response = await fetch(target, {
      method: "POST",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      body: JSON.stringify({ query })
    });
  } catch (error) {
    const base = error instanceof Error ? error.message : String(error);
    const cause =
      error && typeof error === "object" && "cause" in error
        ? (error as { cause?: unknown }).cause
        : undefined;
    const causeMessage =
      cause instanceof Error
        ? cause.message
        : typeof cause === "string"
          ? cause
          : "";
    const message = causeMessage ? `${base} (${causeMessage})` : base;
    throw new IngestError(
      "INGEST_FAILED",
      `Smart Search backend is unavailable (${message}). Start the Python RAG service and retry.`,
      502
    );
  }

  if (!response.ok) {
    const rawBody = await response.text().catch(() => "");
    throw new IngestError(
      "INGEST_FAILED",
      `Smart Search service failed (${response.status}). ${rawBody.slice(0, 180)}`,
      502
    );
  }

  return (await response.json()) as RemoteRagPayload;
}

function toSourceLabel(): string {
  return "Smart Match";
}

function normalizeRemoteResults(payload: RemoteRagPayload): RemoteRagResult[] {
  if (Array.isArray(payload.top_results)) {
    return payload.top_results;
  }

  if (Array.isArray(payload.results)) {
    return payload.results;
  }

  return [];
}

function buildFallbackHighlight(result: RagSearchResult, query: string): RagSearchResponse["highlight"] {
  const citation = result.citations[0] ?? buildCitation(result.mal_id, result.title);
  return {
    title: result.title,
    mal_id: result.mal_id,
    justification: `Top match for "${query}" is ${result.title}. ${citation}`,
    citation
  };
}

export async function searchMangaRag(query: string): Promise<RagSearchResponse> {
  const normalized = cleanText(query);
  if (!normalized) {
    return {
      query,
      answer: "",
      retrieval_mode: "hybrid_cache",
      results: [],
      highlight: null
    };
  }

  const remote = await callHybridRagService(normalized);
  const remoteAnswer = cleanText(remote.answer);
  const remoteResults = normalizeRemoteResults(remote).slice(0, DEFAULT_TOP_K);
  const enrichedResults: RagSearchResult[] = [];

  for (const row of remoteResults) {
    const fallbackTitle = cleanText(row.title) || "Unknown";
    const malId = safeNumber(row.mal_id);
    const snippet = cleanText(row.snippet);
    const preferredMediaType = parseMediaTypeFromSnippet(snippet);
    const title = fallbackTitle;
    const citation = buildCitation(malId, title);
    const mediaType = preferredMediaType ?? "manga";
    const displayType = parseDisplayTypeFromSnippet(snippet) || mediaType;
    const synopsis = extractSynopsisFromSnippet(snippet) || snippet || "No synopsis available.";
    const characters = extractCharactersFromSnippet(snippet);
    const genres = extractGenresFromSnippet(snippet);
    const source: ResultSource = "pinecone";

    enrichedResults.push({
      title,
      media_type: mediaType,
      display_type: displayType,
      mal_id: malId,
      synopsis,
      characters,
      genres,
      citations: normalizeCitationList(cleanText(row.citation), citation),
      score: safeNumber(row.score),
      source,
      source_label: toSourceLabel(),
      image_url: null,
      status: null,
      chapters: null,
      volumes: null,
      episodes: null,
      jikan_score: null,
      read_options: buildReadOptions(title),
      watch_options: buildWatchOptions(title),
      anime_companion: null
    } satisfies RagSearchResult);
  }

  const first = enrichedResults[0] ?? null;
  const highlightImageUrl = cleanText(remote.highlight?.image_url) || null;
  if (first && !first.image_url && highlightImageUrl) {
    first.image_url = highlightImageUrl;
  }
  const highlight = first
    ? (() => {
        const citation =
          cleanText(remote.highlight?.citation) || first.citations[0] || buildCitation(first.mal_id, first.title);
        const justification =
          cleanText(remote.highlight?.justification) ||
          remoteAnswer ||
          buildFallbackHighlight(first, normalized)?.justification ||
          "Top result selected from hybrid retrieval.";

        return {
          title: cleanText(remote.highlight?.title) || first.title,
          mal_id: safeNumber(remote.highlight?.mal_id) ?? first.mal_id,
          justification,
          citation
        };
      })()
    : null;

  return {
    query: normalized,
    answer:
      remoteAnswer ||
      (enrichedResults.length > 0
        ? `Found ${enrichedResults.length} strong matches.`
        : "No Smart Search matches found."),
    retrieval_mode: cleanText(remote.retrieval_mode) || "hybrid_cache",
    results: enrichedResults,
    highlight
  };
}

export async function fetchRagResultLiveMetadata(
  malId: number,
  fallbackTitle: string,
  mediaHint: string | null | undefined
): Promise<RagResultLiveMetadata | null> {
  if (!Number.isFinite(malId) || malId <= 0) {
    return null;
  }

  const preferredMediaType = parseMediaHint(mediaHint);
  const normalizedTitle = cleanText(fallbackTitle) || "Unknown";
  let enrichment: JikanEnrichment | null = null;

  for (let attempt = 0; attempt < 3; attempt += 1) {
    enrichment = await fetchJikanDetails(
      malId,
      normalizedTitle,
      preferredMediaType,
      false
    );

    if (enrichment) {
      break;
    }

    if (attempt < 2) {
      await delay(750 + attempt * 900);
    }
  }

  if (!enrichment) {
    return null;
  }

  return {
    mal_id: malId,
    media_type: enrichment.mediaType,
    display_type: enrichment.typeLabel || enrichment.mediaType,
    title: enrichment.title,
    image_url: enrichment.imageUrl,
    status: enrichment.status,
    chapters: enrichment.chapters,
    volumes: enrichment.volumes,
    episodes: enrichment.episodes,
    jikan_score: enrichment.score
  };
}
