import type {
  IngestResponse,
  RagResultLiveMetaPayload,
  ReadNowPayload,
  RagSearchPayload,
  SearchResult,
  SuggestionsPayload,
  SupportedAdapter,
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

/**
 * Metadata enrichment is served by the backend via /api/meta endpoints,
 * backed by MAL API v2 (primary) and AniList GraphQL (fallback) with
 * server-side caching. Jikan is no longer called browser-side; the outage
 * event plumbing is kept temporarily so existing UI listeners keep working.
 */
export const JIKAN_OUTAGE_EVENT = "panelflow:jikan-outage";

export function notifyJikanOutage(status: number | null, url = ""): void {
  if (
    (status !== null && (status < 500 || status > 599)) ||
    typeof window === "undefined"
  ) {
    return;
  }

  window.dispatchEvent(
    new CustomEvent(JIKAN_OUTAGE_EVENT, {
      detail: {
        status,
        url,
      },
    }),
  );
}

const API_BASE_URL = cleanText(import.meta.env.VITE_API_BASE_URL ?? "").replace(
  /\/+$/,
  "",
);

function apiPath(pathname: string): string {
  if (!pathname.startsWith("/")) {
    return pathname;
  }
  return API_BASE_URL ? `${API_BASE_URL}${pathname}` : pathname;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorBody = (await response
      .json()
      .catch(() => ({}))) as ErrorResponse;
    throw new ApiRequestError(
      errorBody.error ?? `Request failed with status ${response.status}`,
      response.status,
      errorBody.code,
    );
  }

  return (await response.json()) as T;
}

function cleanText(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }

  return value.replace(/\s+/g, " ").trim();
}

// ---------------------------------------------------------------------------
// Core API calls
// ---------------------------------------------------------------------------

export function relayImageUrl(url: string): string {
  return url;
}

export async function ingestChapter(url: string): Promise<IngestResponse> {
  const response = await fetch(apiPath("/api/ingest"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });

  return parseJson<IngestResponse>(response);
}

export async function fetchSupportedAdapters(): Promise<SupportedAdapter[]> {
  const response = await fetch(apiPath("/adapters"));
  const body = await parseJson<{ adapters: SupportedAdapter[] }>(response);
  return body.adapters;
}

async function fetchWith304Bust(
  pathWithQuery: string,
  signal?: AbortSignal,
): Promise<Response> {
  const baseUrl = apiPath(pathWithQuery);
  let response = await fetch(baseUrl, {
    signal,
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache",
    },
  });

  if (response.status === 304) {
    response = await fetch(`${baseUrl}&_ts=${Date.now()}`, {
      signal,
      cache: "reload",
      headers: {
        "Cache-Control": "no-cache",
      },
    });
  }

  return response;
}

export async function searchManga(
  query: string,
  signal?: AbortSignal,
): Promise<SearchResult[]> {
  const response = await fetchWith304Bust(
    `/api/search?q=${encodeURIComponent(query)}`,
    signal,
  );
  const body = await parseJson<{ results: SearchResult[] }>(response);
  return body.results;
}

export async function searchMangaRag(
  query: string,
  signal?: AbortSignal,
): Promise<RagSearchPayload> {
  const response = await fetchWith304Bust(
    `/api/rag/search?q=${encodeURIComponent(query)}`,
    signal,
  );
  return parseJson<RagSearchPayload>(response);
}

export async function searchMangaRagStream(
  query: string,
  onUpdate: (payload: RagSearchPayload) => void,
  signal?: AbortSignal,
): Promise<RagSearchPayload> {
  const baseUrl = apiPath(
    `/api/rag/search/stream?q=${encodeURIComponent(query)}`,
  );
  const response = await fetch(baseUrl, {
    signal,
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache",
      Accept: "application/x-ndjson",
    },
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

    let event: {
      type?: string;
      payload?: RagSearchPayload;
      error?: string;
      code?: string;
    };
    try {
      event = JSON.parse(trimmed);
    } catch {
      // Skip malformed NDJSON lines instead of failing the whole stream.
      return;
    }

    if (event.type === "result" || event.type === "done") {
      if (event.payload) {
        latest = event.payload;
        onUpdate(event.payload);
      }
      return;
    }

    if (event.type === "error") {
      throw new ApiRequestError(
        event.error ?? "Smart Search failed.",
        response.status,
        event.code,
      );
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
    throw new ApiRequestError(
      "Smart Search returned no results.",
      response.status,
    );
  }

  return latest;
}

// ---------------------------------------------------------------------------
// Metadata enrichment (backend /api/meta — MAL v2 primary, AniList fallback)
// ---------------------------------------------------------------------------

interface BackendMetaPayload {
  mal_id: number | null;
  media_type: string | null;
  display_type: string | null;
  title: string;
  title_candidates: string[];
  image_url: string | null;
  status: string | null;
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  score: number | null;
  source: "mal" | "anilist" | null;
}

function safeNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeMediaTypeHint(mediaType: string): "anime" | "manga" {
  return mediaType.trim().toLowerCase().includes("anime") ? "anime" : "manga";
}

/**
 * Builds a validated absolute API URL. Only http(s) requests to the
 * configured API origin (or the current page origin) are allowed.
 */
function buildValidatedApiUrl(
  pathname: string,
  params: Record<string, string>,
): string {
  let url: URL;
  try {
    url = new URL(pathname, API_BASE_URL || window.location.origin);
  } catch {
    throw new Error("Invalid API URL.");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Blocked non-http API request.");
  }
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}

export async function fetchRagResultLiveMeta(
  malId: number,
  title: string,
  mediaType: string,
  signal?: AbortSignal,
  _strictMediaType = false,
): Promise<RagResultLiveMetaPayload> {
  const url = buildValidatedApiUrl("/api/meta", {
    type: normalizeMediaTypeHint(mediaType),
    id: String(malId),
    title,
  });
  // _strictMediaType only constrains preferred order on the backend;
  // the hint type is passed as primary either way.

  const response = await fetch(url, {
    signal,
    cache: "default",
  });

  if (!response.ok) {
    throw new ApiRequestError(
      `Metadata lookup failed for MAL-ID ${malId} (${response.status}).`,
      response.status,
      "META_UNAVAILABLE",
    );
  }

  const data = (await response.json()) as BackendMetaPayload | null;
  if (!data || !data.title || data.title === "Unknown") {
    throw new ApiRequestError(
      `No metadata found for MAL-ID ${malId}.`,
      404,
      "META_NOT_FOUND",
    );
  }

  return {
    mal_id: safeNumber(data.mal_id) ?? malId,
    media_type: cleanText(data.media_type) || normalizeMediaTypeHint(mediaType),
    display_type:
      cleanText(data.display_type) ||
      cleanText(data.media_type) ||
      normalizeMediaTypeHint(mediaType),
    title: cleanText(data.title),
    title_candidates: Array.isArray(data.title_candidates)
      ? data.title_candidates.filter(Boolean)
      : [],
    image_url: cleanText(data.image_url) || null,
    status: cleanText(data.status) || null,
    chapters: safeNumber(data.chapters),
    volumes: safeNumber(data.volumes),
    episodes: safeNumber(data.episodes),
    jikan_score: safeNumber(data.score),
  };
}

/** Kept for call-site compatibility; now backed by /api/meta/chapter-count. */
export async function fetchJikanMangaChapterCount(
  options: {
    seriesTitle: string;
    chapterTitle?: string | null;
    chapterUrl?: string;
  },
  signal?: AbortSignal,
): Promise<number | null> {
  try {
    const chapterCountUrl = buildValidatedApiUrl("/api/meta/chapter-count", {
      title: options.seriesTitle,
      ...(options.chapterTitle?.trim()
        ? { chapterTitle: options.chapterTitle.trim() }
        : {}),
      ...(options.chapterUrl?.trim() ? { url: options.chapterUrl.trim() } : {}),
    });
    const body = await parseJson<{ chapters: number | null }>(
      await fetch(chapterCountUrl, {
        signal,
        cache: "default",
      }),
    );
    const count = safeNumber(body.chapters);
    return count !== null && count > 0
      ? Math.min(Math.floor(count), 1500)
      : null;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }
    return null;
  }
}

/** Kept for call-site compatibility; now backed by /api/meta/recommendations. */
export async function fetchJikanMangaRecommendations(
  seriesTitle: string,
  signal?: AbortSignal,
): Promise<SuggestionsPayload> {
  const recommendationsUrl = buildValidatedApiUrl("/api/meta/recommendations", {
    title: seriesTitle,
  });
  const response = await fetch(recommendationsUrl, {
    signal,
    cache: "default",
  });

  if (!response.ok) {
    throw new Error(`Recommendations lookup failed (${response.status}).`);
  }

  const payload = (await response.json()) as SuggestionsPayload;
  return {
    alternatives: Array.isArray(payload.alternatives)
      ? payload.alternatives
      : [],
    similarSeries: Array.isArray(payload.similarSeries)
      ? payload.similarSeries
      : [],
    trending: Array.isArray(payload.trending) ? payload.trending : [],
  };
}

// ---------------------------------------------------------------------------
// Reader + suggestions
// ---------------------------------------------------------------------------

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
  mediaType?: string,
): Promise<ReadNowPayload> {
  const latinCandidates = [title, ...titleCandidates].filter(
    isLatinReadableTitle,
  );
  const params = new URLSearchParams({ title: latinCandidates[0] ?? title });
  if (latinCandidates.length > 0) {
    params.set("titles", JSON.stringify(latinCandidates));
  }
  if (mediaType?.trim()) {
    params.set("media_type", mediaType.trim());
  }

  const response = await fetchWith304Bust(
    `/api/read-now?${params.toString()}`,
    signal,
  );
  return parseJson<ReadNowPayload>(response);
}

export async function fetchSuggestions(
  seriesTitle: string,
  genres: string[],
  currentAdapter: string,
): Promise<SuggestionsPayload> {
  const response = await fetch(apiPath("/api/suggestions"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      seriesTitle,
      genres,
      currentAdapter,
    }),
  });

  return parseJson<SuggestionsPayload>(response);
}
