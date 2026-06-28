import type {
  IngestResponse,
  RagResultLiveMetaPayload,
  ReadNowPayload,
  RagSearchPayload,
  SearchResult,
  SuggestionsPayload,
  SupportedAdapter
} from "../types";

interface ErrorResponse {
  error?: string;
  code?: string;
}

export class ApiRequestError extends Error {
  public readonly status: number;
  public readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
  }
}

type JikanMediaType = "anime" | "manga";

interface JikanImageVariant {
  image_url?: string | null;
  large_image_url?: string | null;
  small_image_url?: string | null;
}

interface JikanFullData {
  mal_id?: number;
  title?: string | null;
  title_english?: string | null;
  title_japanese?: string | null;
  title_synonyms?: string[] | null;
  type?: string | null;
  status?: string | null;
  chapters?: number | null;
  volumes?: number | null;
  episodes?: number | null;
  score?: number | null;
  images?: {
    jpg?: JikanImageVariant;
    webp?: JikanImageVariant;
  };
}

interface JikanFullPayload {
  data?: JikanFullData;
}

interface JikanSearchPayload {
  data?: JikanFullData[];
}

interface JikanRecommendationEntry {
  mal_id?: number;
  title?: string | null;
  url?: string | null;
  images?: JikanFullData["images"];
}

interface JikanRecommendationRow {
  entry?: JikanRecommendationEntry;
  votes?: number | null;
}

interface JikanRecommendationsPayload {
  data?: JikanRecommendationRow[];
}

const JIKAN_BASE_URL = "https://api.jikan.moe/v4";
const JIKAN_MAX_RETRIES = 5;
const JIKAN_MIN_INTERVAL_MS = 1_100;
const JIKAN_CACHE_TTL_MS = 15 * 60 * 1000;
const JIKAN_RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504, 520, 522, 524]);
const jikanFullCache = new Map<string, { expiresAt: number; value: JikanFullData | null }>();
const jikanChapterCountCache = new Map<string, { expiresAt: number; value: number | null }>();
const jikanRecommendationsCache = new Map<string, { expiresAt: number; value: SuggestionsPayload }>();
let nextJikanRequestAt = 0;
const API_BASE_URL = cleanText(import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

function apiPath(pathname: string): string {
  if (!pathname.startsWith("/")) {
    return pathname;
  }
  return API_BASE_URL ? `${API_BASE_URL}${pathname}` : pathname;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorBody = (await response.json().catch(() => ({}))) as ErrorResponse;
    throw new ApiRequestError(
      errorBody.error ?? `Request failed with status ${response.status}`,
      response.status,
      errorBody.code
    );
  }

  return (await response.json()) as T;
}

function cleanText(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }

  return value
    .replace(/\s+/g, " ")
    .trim();
}

function safeNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function extractJikanImageUrl(images: JikanFullData["images"]): string | null {
  const candidates: Array<string | null | undefined> = [
    images?.jpg?.large_image_url,
    images?.jpg?.image_url,
    images?.jpg?.small_image_url,
    images?.webp?.large_image_url,
    images?.webp?.image_url,
    images?.webp?.small_image_url
  ];

  for (const candidate of candidates) {
    const value = cleanText(candidate);
    if (value) {
      return value;
    }
  }

  return null;
}

function normalizeMediaTypeHint(mediaType: string): JikanMediaType {
  const normalized = mediaType.trim().toLowerCase();
  if (normalized.includes("anime")) {
    return "anime";
  }
  return "manga";
}

function mediaTypeOrder(hint: JikanMediaType): JikanMediaType[] {
  return hint === "anime" ? ["anime", "manga"] : ["manga", "anime"];
}

function normalizeTitleForMatch(value: string): string {
  return cleanText(value)
    .toLowerCase()
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function stripChapterNoise(value: string): string {
  return cleanText(value)
    .replace(/^(?:chapter|chap|ch\.?|episode|ep\.?)\s*[-#: ]?\d+(?:\.\d+)?\s*[-–—:|]\s*/i, "")
    .replace(/\s*[-–—:|]\s*(?:chapter|chap|ch\.?|episode|ep\.?)\s*[-#: ]?\d+(?:\.\d+)?.*$/i, "")
    .replace(/\b(?:chapter|chap|ch\.?|episode|ep\.?)\s*[-#: ]?\d+(?:\.\d+)?.*$/i, "")
    .trim();
}

function collectJikanTitleCandidates(data: JikanFullData): string[] {
  return [
    data.title,
    data.title_english,
    data.title_japanese,
    ...(Array.isArray(data.title_synonyms) ? data.title_synonyms : [])
  ]
    .map((title) => cleanText(title))
    .filter(Boolean);
}

function scoreJikanMangaCandidate(candidate: JikanFullData, query: string): number {
  const normalizedQuery = normalizeTitleForMatch(query);
  if (!normalizedQuery) {
    return Number.POSITIVE_INFINITY;
  }

  const queryTokens = normalizedQuery.split(/\s+/).filter((token) => token.length >= 2);
  let best = Number.POSITIVE_INFINITY;

  for (const title of collectJikanTitleCandidates(candidate)) {
    const normalizedTitle = normalizeTitleForMatch(title);
    if (!normalizedTitle) {
      continue;
    }

    if (normalizedTitle === normalizedQuery) {
      return 0;
    }

    if (normalizedTitle.startsWith(normalizedQuery) || normalizedQuery.startsWith(normalizedTitle)) {
      best = Math.min(best, 1);
      continue;
    }

    const titleTokens = normalizedTitle.split(/\s+/).filter((token) => token.length >= 2);
    const matched = queryTokens.filter((token) => titleTokens.includes(token)).length;
    if (matched > 0 && matched >= Math.min(queryTokens.length, titleTokens.length)) {
      best = Math.min(best, 1.5);
      continue;
    }

    if (normalizedTitle.includes(normalizedQuery) || normalizedQuery.includes(normalizedTitle)) {
      best = Math.min(best, 2);
    }
  }

  return best;
}

function buildJikanMangaQueries(options: {
  seriesTitle: string;
  chapterTitle?: string | null;
  chapterUrl?: string;
}): string[] {
  const queries = new Set<string>();

  for (const value of [options.seriesTitle, options.chapterTitle ?? ""]) {
    const cleaned = stripChapterNoise(value);
    if (cleaned) {
      queries.add(cleaned);
    }

    const leading = cleaned.split(/[-–—:|([\\]/)[0]?.trim();
    if (leading) {
      queries.add(leading);
    }
  }

  try {
    const hostLabel = new URL(options.chapterUrl ?? "").hostname
      .replace(/^www\./i, "")
      .split(".")[0]
      ?.replace(/[-_]+/g, " ")
      .trim();
    if (hostLabel && !/^(read|manga|manhwa|chapter|chapters)$/i.test(hostLabel)) {
      queries.add(hostLabel);
    }
  } catch {
    // URL-derived query is best effort only.
  }

  return Array.from(queries).filter((query) => query.length >= 2);
}

function computeRetryDelayMs(attempt: number, retryAfterHeader: string | null, status?: number): number {
  if (retryAfterHeader) {
    const retryAfterSeconds = Number.parseInt(retryAfterHeader, 10);
    if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0) {
      return Math.min(65_000, retryAfterSeconds * 1000);
    }
  }

  const base = status === 429 ? 2_500 : 500;
  const cap = status === 429 ? 20_000 : 8_000;
  const exponential = Math.min(cap, base * 2 ** attempt);
  const jitter = Math.floor(Math.random() * 250);
  return exponential + jitter;
}

function cacheKey(mediaType: JikanMediaType, malId: number): string {
  return `${mediaType}:${malId}`;
}

async function waitFor(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }

  await new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
      resolve();
    }, ms);

    const onAbort = () => {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      reject(new DOMException("Aborted", "AbortError"));
    };

    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function waitForJikanSlot(signal?: AbortSignal): Promise<void> {
  const now = Date.now();
  const waitMs = nextJikanRequestAt - now;
  if (waitMs > 0) {
    await waitFor(waitMs, signal);
  }

  nextJikanRequestAt = Date.now() + JIKAN_MIN_INTERVAL_MS;
}

async function fetchJikanFullByType(
  mediaType: JikanMediaType,
  malId: number,
  signal?: AbortSignal
): Promise<JikanFullData | null> {
  const url = `${JIKAN_BASE_URL}/${mediaType}/${malId}/full`;
  const key = cacheKey(mediaType, malId);
  const cached = jikanFullCache.get(key);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < JIKAN_MAX_RETRIES; attempt += 1) {
    await waitForJikanSlot(signal);

    let response: Response;
    try {
      response = await fetch(url, {
        signal,
        cache: "no-store",
        headers: {
          Accept: "application/json",
          "Cache-Control": "no-cache"
        }
      });
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw error;
      }

      lastError = error instanceof Error ? error : new Error(String(error));
      if (attempt < JIKAN_MAX_RETRIES - 1) {
        await waitFor(computeRetryDelayMs(attempt, null), signal);
        continue;
      }
      throw lastError;
    }

    if (response.ok) {
      const payload = (await response.json()) as JikanFullPayload;
      const value = payload.data ?? null;
      jikanFullCache.set(key, {
        expiresAt: Date.now() + JIKAN_CACHE_TTL_MS,
        value
      });
      return value;
    }

    if (response.status === 404) {
      jikanFullCache.set(key, {
        expiresAt: Date.now() + JIKAN_CACHE_TTL_MS,
        value: null
      });
      return null;
    }

    const body = await response.text().catch(() => "");
    lastError = new Error(`Jikan ${mediaType}/${malId}/full failed (${response.status}): ${body.slice(0, 180)}`);

    if (!JIKAN_RETRYABLE_STATUS.has(response.status) || attempt >= JIKAN_MAX_RETRIES - 1) {
      throw lastError;
    }

    const retryAfter = response.headers.get("retry-after");
    const retryDelayMs = computeRetryDelayMs(attempt, retryAfter, response.status);
    nextJikanRequestAt = Math.max(nextJikanRequestAt, Date.now() + retryDelayMs);
    await waitFor(retryDelayMs, signal);
  }

  throw lastError ?? new Error(`Jikan ${mediaType}/${malId}/full failed.`);
}

async function fetchJikanMangaSearch(query: string, signal?: AbortSignal): Promise<JikanFullData[]> {
  await waitForJikanSlot(signal);

  const url = new URL(`${JIKAN_BASE_URL}/manga`);
  url.searchParams.set("q", query);
  url.searchParams.set("limit", "10");

  const response = await fetch(url.toString(), {
    signal,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      "Cache-Control": "no-cache"
    }
  });

  if (!response.ok) {
    if (JIKAN_RETRYABLE_STATUS.has(response.status)) {
      const retryDelayMs = computeRetryDelayMs(0, response.headers.get("retry-after"), response.status);
      nextJikanRequestAt = Math.max(nextJikanRequestAt, Date.now() + retryDelayMs);
    }
    return [];
  }

  const payload = (await response.json()) as JikanSearchPayload;
  return Array.isArray(payload.data) ? payload.data : [];
}

export function relayImageUrl(url: string): string {
  return url;
}

export async function ingestChapter(url: string): Promise<IngestResponse> {
  const response = await fetch(apiPath("/api/ingest"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ url })
  });

  return parseJson<IngestResponse>(response);
}

export async function fetchSupportedAdapters(): Promise<SupportedAdapter[]> {
  const response = await fetch(apiPath("/adapters"));
  const body = await parseJson<{ adapters: SupportedAdapter[] }>(response);
  return body.adapters;
}

export async function searchManga(query: string, signal?: AbortSignal): Promise<SearchResult[]> {
  const baseUrl = apiPath(`/api/search?q=${encodeURIComponent(query)}`);
  let response = await fetch(baseUrl, {
    signal,
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache"
    }
  });

  if (response.status === 304) {
    response = await fetch(`${baseUrl}&_ts=${Date.now()}`, {
      signal,
      cache: "reload",
      headers: {
        "Cache-Control": "no-cache"
      }
    });
  }

  const body = await parseJson<{ results: SearchResult[] }>(response);
  return body.results;
}

export async function searchMangaRag(query: string, signal?: AbortSignal): Promise<RagSearchPayload> {
  const baseUrl = apiPath(`/api/rag/search?q=${encodeURIComponent(query)}`);
  let response = await fetch(baseUrl, {
    signal,
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache"
    }
  });

  if (response.status === 304) {
    response = await fetch(`${baseUrl}&_ts=${Date.now()}`, {
      signal,
      cache: "reload",
      headers: {
        "Cache-Control": "no-cache"
      }
    });
  }

  return parseJson<RagSearchPayload>(response);
}

export async function searchMangaRagStream(
  query: string,
  onUpdate: (payload: RagSearchPayload) => void,
  signal?: AbortSignal
): Promise<RagSearchPayload> {
  const baseUrl = apiPath(`/api/rag/search/stream?q=${encodeURIComponent(query)}`);
  const response = await fetch(baseUrl, {
    signal,
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache",
      Accept: "application/x-ndjson"
    }
  });

  if (!response.ok) {
    return parseJson<RagSearchPayload>(response);
  }

  if (!response.body) {
    const payload = await searchMangaRag(query, signal);
    onUpdate(payload);
    return payload;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let latest: RagSearchPayload | null = null;

  const consumeLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }

    const event = JSON.parse(trimmed) as {
      type?: string;
      payload?: RagSearchPayload;
      error?: string;
      code?: string;
    };

    if (event.type === "result" || event.type === "done") {
      if (event.payload) {
        latest = event.payload;
        onUpdate(event.payload);
      }
      return;
    }

    if (event.type === "error") {
      throw new ApiRequestError(event.error ?? "Smart Search failed.", response.status, event.code);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      consumeLine(line);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    consumeLine(buffer);
  }

  if (!latest) {
    throw new ApiRequestError("Smart Search returned no results.", response.status);
  }

  return latest;
}

export async function fetchRagResultLiveMeta(
  malId: number,
  title: string,
  mediaType: string,
  signal?: AbortSignal
): Promise<RagResultLiveMetaPayload> {
  const hint = normalizeMediaTypeHint(mediaType);
  const candidateTypes = mediaTypeOrder(hint);
  let lastError: Error | null = null;

  for (const candidateType of candidateTypes) {
    try {
      const data = await fetchJikanFullByType(candidateType, malId, signal);
      if (!data) {
        continue;
      }

      const displayType = cleanText(data.type) || candidateType;
      const normalizedTitle =
        cleanText(data.title_english) ||
        cleanText(data.title) ||
        cleanText(data.title_japanese) ||
        cleanText(title) ||
        "Unknown";

      return {
        mal_id: safeNumber(data.mal_id) ?? malId,
        media_type: candidateType,
        display_type: displayType,
        title: normalizedTitle,
        title_candidates: collectJikanTitleCandidates(data),
        image_url: extractJikanImageUrl(data.images),
        status: cleanText(data.status) || null,
        chapters: safeNumber(data.chapters),
        volumes: safeNumber(data.volumes),
        episodes: safeNumber(data.episodes),
        jikan_score: safeNumber(data.score)
      };
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw error;
      }
      lastError = error instanceof Error ? error : new Error(String(error));
    }
  }

  throw lastError ?? new Error(`No Jikan metadata found for MAL-ID ${malId}.`);
}

export async function fetchJikanMangaChapterCount(
  options: {
    seriesTitle: string;
    chapterTitle?: string | null;
    chapterUrl?: string;
  },
  signal?: AbortSignal
): Promise<number | null> {
  const queries = buildJikanMangaQueries(options);
  const cacheKey = queries.map((query) => normalizeTitleForMatch(query)).join("|");
  const cached = jikanChapterCountCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  let best: { data: JikanFullData; score: number } | null = null;

  for (const query of queries) {
    const candidates = await fetchJikanMangaSearch(query, signal);
    for (const candidate of candidates) {
      const score = scoreJikanMangaCandidate(candidate, query);
      if (!Number.isFinite(score) || score > 2.5) {
        continue;
      }

      if (
        !best ||
        score < best.score ||
        (score === best.score && (safeNumber(candidate.chapters) ?? 0) > (safeNumber(best.data.chapters) ?? 0))
      ) {
        best = { data: candidate, score };
      }
    }

    if (best?.score === 0 && safeNumber(best.data.chapters)) {
      break;
    }
  }

  let chapterCount = best ? safeNumber(best.data.chapters) : null;
  const malId = best ? safeNumber(best.data.mal_id) : null;
  if ((!chapterCount || chapterCount <= 0) && malId !== null) {
    const full = await fetchJikanFullByType("manga", malId, signal);
    chapterCount = full ? safeNumber(full.chapters) : null;
  }

  const normalizedCount =
    chapterCount !== null && chapterCount > 0 ? Math.min(Math.floor(chapterCount), 1500) : null;
  if (cacheKey) {
    jikanChapterCountCache.set(cacheKey, {
      expiresAt: Date.now() + JIKAN_CACHE_TTL_MS,
      value: normalizedCount
    });
  }

  return normalizedCount;
}

async function resolveJikanMangaIdByTitle(seriesTitle: string, signal?: AbortSignal): Promise<number | null> {
  const queries = buildJikanMangaQueries({ seriesTitle });
  let best: { data: JikanFullData; score: number } | null = null;

  for (const query of queries) {
    const candidates = await fetchJikanMangaSearch(query, signal);
    for (const candidate of candidates) {
      const score = scoreJikanMangaCandidate(candidate, query);
      if (!Number.isFinite(score) || score > 2.5) {
        continue;
      }

      if (!best || score < best.score) {
        best = { data: candidate, score };
      }
    }

    if (best?.score === 0) {
      break;
    }
  }

  return best ? safeNumber(best.data.mal_id) : null;
}

export async function fetchJikanMangaRecommendations(
  seriesTitle: string,
  signal?: AbortSignal
): Promise<SuggestionsPayload> {
  const normalizedTitle = normalizeTitleForMatch(seriesTitle);
  const cached = jikanRecommendationsCache.get(normalizedTitle);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  const malId = await resolveJikanMangaIdByTitle(seriesTitle, signal);
  if (malId === null) {
    const empty = { alternatives: [], similarSeries: [], trending: [] };
    jikanRecommendationsCache.set(normalizedTitle, {
      expiresAt: Date.now() + JIKAN_CACHE_TTL_MS,
      value: empty
    });
    return empty;
  }

  await waitForJikanSlot(signal);
  const response = await fetch(`${JIKAN_BASE_URL}/manga/${malId}/recommendations`, {
    signal,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      "Cache-Control": "no-cache"
    }
  });

  if (!response.ok) {
    if (JIKAN_RETRYABLE_STATUS.has(response.status)) {
      const retryDelayMs = computeRetryDelayMs(0, response.headers.get("retry-after"), response.status);
      nextJikanRequestAt = Math.max(nextJikanRequestAt, Date.now() + retryDelayMs);
    }
    throw new Error(`Jikan manga/${malId}/recommendations failed (${response.status}).`);
  }

  const payload = (await response.json()) as JikanRecommendationsPayload;
  const seen = new Set<string>();
  const similarSeries = (Array.isArray(payload.data) ? payload.data : [])
    .map((row) => {
      const title = cleanText(row.entry?.title);
      const key = normalizeTitleForMatch(title);
      if (!title || !key || seen.has(key)) {
        return null;
      }
      seen.add(key);
      return {
        title,
        coverUrl: extractJikanImageUrl(row.entry?.images),
        chapterCount: 0,
        votes: safeNumber(row.votes) ?? undefined,
        malId: safeNumber(row.entry?.mal_id) ?? undefined,
        url: cleanText(row.entry?.url) || undefined
      };
    })
    .filter((row): row is NonNullable<typeof row> => Boolean(row))
    .slice(0, 5);

  const result: SuggestionsPayload = {
    alternatives: [],
    similarSeries,
    trending: []
  };

  jikanRecommendationsCache.set(normalizedTitle, {
    expiresAt: Date.now() + JIKAN_CACHE_TTL_MS,
    value: result
  });
  return result;
}

function isLatinReadableTitle(value: string): boolean {
  const cleaned = cleanText(value);
  if (!cleaned || cleaned.length < 2) {
    return false;
  }
  if (/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0600-\u06ff]/.test(cleaned)) {
    return false;
  }
  return /[a-zA-Z]/.test(cleaned);
}

export async function resolveReadNowByTitle(
  title: string,
  titleCandidates: string[] = [],
  signal?: AbortSignal,
  mediaType?: string
): Promise<ReadNowPayload> {
  const latinCandidates = [title, ...titleCandidates].filter(isLatinReadableTitle);
  const params = new URLSearchParams({ title: latinCandidates[0] ?? title });
  if (latinCandidates.length > 0) {
    params.set("titles", JSON.stringify(latinCandidates));
  }
  if (mediaType?.trim()) {
    params.set("media_type", mediaType.trim());
  }
  const baseUrl = apiPath(`/api/read-now?${params.toString()}`);
  let response = await fetch(baseUrl, {
    signal,
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache"
    }
  });

  if (response.status === 304) {
    response = await fetch(`${baseUrl}&_ts=${Date.now()}`, {
      signal,
      cache: "reload",
      headers: {
        "Cache-Control": "no-cache"
      }
    });
  }

  return parseJson<ReadNowPayload>(response);
}

export async function fetchSuggestions(
  seriesTitle: string,
  genres: string[],
  currentAdapter: string
): Promise<SuggestionsPayload> {
  const response = await fetch(apiPath("/api/suggestions"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      seriesTitle,
      genres,
      currentAdapter
    })
  });

  return parseJson<SuggestionsPayload>(response);
}
