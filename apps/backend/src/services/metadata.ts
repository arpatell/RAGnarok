import { LRUCache } from "lru-cache";
import { resolveMangaChapterCount } from "./mal.js";

/**
 * Unified media metadata provider.
 *
 * Primary source: MAL official API v2 (X-MAL-CLIENT-ID).
 * Fallback source: AniList GraphQL (keyless, generous rate limits).
 *
 * Design constraints (e2-micro friendly):
 * - Single attempt per upstream call, short timeout, no retry storms.
 * - Negative caching so an outage never re-stalls every request.
 * - Small outbound concurrency cap shared by all provider calls.
 */

const MAL_BASE_URL = "https://api.myanimelist.net/v2";
const ANILIST_GRAPHQL_URL = "https://graphql.anilist.co";

/**
 * Outbound allowlist for every upstream metadata request in this module.
 * All URLs are constructed from these constants plus numeric ids / encoded
 * query strings, but we validate anyway so no user-controlled host can ever
 * reach fetch().
 */
const OUTBOUND_HOST_ALLOWLIST = new Set([
  "api.myanimelist.net",
  "graphql.anilist.co",
]);

function assertAllowedOutboundUrl(url: string): void {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("Blocked outbound metadata request: invalid URL");
  }
  if (
    parsed.protocol !== "https:" ||
    !OUTBOUND_HOST_ALLOWLIST.has(parsed.hostname)
  ) {
    throw new Error(`Blocked outbound metadata request to ${parsed.hostname}`);
  }
}

const REQUEST_TIMEOUT_MS = Math.max(
  1_000,
  Number.parseInt(process.env.META_REQUEST_TIMEOUT_MS ?? "3000", 10),
);
const POSITIVE_CACHE_TTL_MS = Math.max(
  60_000,
  Number.parseInt(process.env.META_CACHE_TTL_MS ?? "1_800_000", 10),
);
const NEGATIVE_CACHE_TTL_MS = Math.max(
  30_000,
  Number.parseInt(process.env.META_NEGATIVE_CACHE_TTL_MS ?? "600_000", 10),
);
const MAX_CONCURRENT_FETCHES = Math.max(
  1,
  Number.parseInt(process.env.META_MAX_CONCURRENCY ?? "4", 10),
);

export type MetaMediaType = "manga" | "anime";

export interface MetaPayload {
  mal_id: number | null;
  media_type: MetaMediaType;
  display_type: string | null;
  title: string;
  title_candidates: string[];
  image_url: string | null;
  status: string | null;
  synopsis: string | null;
  genres: string[];
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  score: number | null;
  source: "mal" | "anilist" | null;
}

export interface AnimeCompanionInfo {
  mal_id: number | null;
  title: string;
  status: string | null;
  episodes: number | null;
  score: number | null;
  image_url: string | null;
  watch_url: string;
  citation: string;
}

interface MetaFetchOptions {
  malId: number;
  fallbackTitle?: string;
  preferredType?: MetaMediaType | null;
  strictMediaType?: boolean;
}

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

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
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

let activeFetches = 0;
const fetchWaiters: Array<() => void> = [];

async function withFetchSlot<T>(operation: () => Promise<T>): Promise<T> {
  if (activeFetches >= MAX_CONCURRENT_FETCHES) {
    await new Promise<void>((resolve) => {
      fetchWaiters.push(resolve);
    });
  }
  activeFetches += 1;
  try {
    return await operation();
  } finally {
    activeFetches = Math.max(0, activeFetches - 1);
    const next = fetchWaiters.shift();
    if (next) {
      next();
    }
  }
}

interface CacheEntry<T> {
  expiresAt: number;
  value: T;
}

const metaCache = new LRUCache<string, CacheEntry<MetaPayload | null>>({
  max: 2048,
});

function readMetaCache(key: string): MetaPayload | null | undefined {
  const entry = metaCache.get(key);
  if (!entry) {
    return undefined;
  }
  if (entry.expiresAt <= Date.now()) {
    metaCache.delete(key);
    return undefined;
  }
  return entry.value;
}

function writeMetaCache(key: string, value: MetaPayload | null): void {
  metaCache.set(key, {
    expiresAt:
      Date.now() + (value ? POSITIVE_CACHE_TTL_MS : NEGATIVE_CACHE_TTL_MS),
    value,
  });
}

function normalizeTitle(value: string): string {
  return cleanText(value)
    .toLowerCase()
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

/** Loose containment/token match used to guard against wrong-title enrichment. */
export function metaTitleMatches(
  payloadTitleCandidates: string[],
  resultTitle: string,
): boolean {
  const normalizedResult = normalizeTitle(resultTitle);
  if (!normalizedResult) {
    return true;
  }

  const resultTokens = normalizedResult
    .split(" ")
    .filter((token) => token.length >= 2);
  for (const candidate of payloadTitleCandidates) {
    const normalizedCandidate = normalizeTitle(candidate);
    if (!normalizedCandidate) {
      continue;
    }
    if (
      normalizedCandidate === normalizedResult ||
      normalizedCandidate.includes(normalizedResult) ||
      normalizedResult.includes(normalizedCandidate)
    ) {
      return true;
    }

    const candidateTokens = normalizedCandidate
      .split(" ")
      .filter((token) => token.length >= 2);
    const matched = resultTokens.filter((token) =>
      candidateTokens.includes(token),
    ).length;
    if (resultTokens.length >= 2 && matched / resultTokens.length >= 0.75) {
      return true;
    }
  }
  return false;
}

function getMalClientId(): string {
  return cleanText(
    process.env.MAL_CLIENT_ID ?? process.env.MYANIMELIST_CLIENT_ID,
  );
}

// ---------------------------------------------------------------------------
// MAL API v2
// ---------------------------------------------------------------------------

interface MalMainPicture {
  large?: string | null;
  medium?: string | null;
}

interface MalRelatedNode {
  id?: number;
  title?: string | null;
  media_type?: string | null;
  main_picture?: MalMainPicture | null;
}

interface MalDetailsNode {
  id?: number;
  title?: string | null;
  alternative_titles?: {
    synonyms?: string[] | null;
    en?: string | null;
    ja?: string | null;
  } | null;
  main_picture?: MalMainPicture | null;
  synopsis?: string | null;
  genres?: Array<{ name?: string | null }> | null;
  type?: string | null;
  status?: string | null;
  num_chapters?: number | null;
  num_volumes?: number | null;
  num_episodes?: number | null;
  mean?: number | null;
  related_anime?: Array<{ node?: MalRelatedNode | null }> | null;
  related_manga?: Array<{ node?: MalRelatedNode | null }> | null;
}

const MAL_FIELDS =
  "id,title,type,status,num_chapters,num_volumes,num_episodes,mean,main_picture,synopsis,genres,alternative_titles,related_anime{node{id,title,media_type,main_picture}}";

async function fetchMalDetails(
  mediaType: MetaMediaType,
  malId: number,
): Promise<MalDetailsNode | null> {
  const clientId = getMalClientId();
  if (!clientId) {
    return null;
  }

  let url: URL;
  try {
    url = new URL(`${MAL_BASE_URL}/${mediaType}/${malId}`);
  } catch {
    return null;
  }
  url.searchParams.set("fields", MAL_FIELDS);
  assertAllowedOutboundUrl(url.toString());

  try {
    return await withFetchSlot(async () => {
      const response = await fetch(url.toString(), {
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        headers: {
          Accept: "application/json",
          "X-MAL-CLIENT-ID": clientId,
        },
      });
      if (!response.ok) {
        return null;
      }
      const payload = (await response.json()) as MalDetailsNode;
      return typeof payload?.id === "number" ? payload : null;
    });
  } catch {
    return null;
  }
}

function collectMalTitleCandidates(node: MalDetailsNode): string[] {
  const values = [
    node.title,
    node.alternative_titles?.en,
    node.alternative_titles?.ja,
    ...(Array.isArray(node.alternative_titles?.synonyms)
      ? node.alternative_titles.synonyms
      : []),
  ];
  const seen = new Set<string>();
  const titles: string[] = [];
  for (const value of values) {
    const cleaned = cleanText(value);
    const key = normalizeTitle(cleaned);
    if (!cleaned || !key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    titles.push(cleaned);
  }
  return titles;
}

function malImageUrl(node: MalDetailsNode | MalRelatedNode): string | null {
  return (
    cleanText(node.main_picture?.large) ||
    cleanText(node.main_picture?.medium) ||
    null
  );
}

function mapMetaFromMal(
  node: MalDetailsNode,
  mediaType: MetaMediaType,
): MetaPayload {
  return {
    mal_id: safeNumber(node.id),
    media_type: mediaType,
    display_type: cleanText(node.type) || mediaType,
    title:
      cleanText(node.title) ||
      cleanText(node.alternative_titles?.en) ||
      cleanText(node.alternative_titles?.ja) ||
      "Unknown",
    title_candidates: collectMalTitleCandidates(node),
    image_url: malImageUrl(node),
    status: cleanText(node.status) || null,
    synopsis: cleanText(node.synopsis) || null,
    genres: (Array.isArray(node.genres) ? node.genres : [])
      .map((genre) => cleanText(genre?.name))
      .filter(Boolean)
      .slice(0, 10),
    chapters: safeNumber(node.num_chapters),
    volumes: safeNumber(node.num_volumes),
    episodes: safeNumber(node.num_episodes),
    score: safeNumber(node.mean),
    source: "mal",
  };
}

// ---------------------------------------------------------------------------
// AniList GraphQL fallback
// ---------------------------------------------------------------------------

interface AniListTitle {
  romaji?: string | null;
  english?: string | null;
  native?: string | null;
}

interface AniListMedia {
  id?: number;
  idMal?: number | null;
  title?: AniListTitle | null;
  synonyms?: string[] | null;
  format?: string | null;
  status?: string | null;
  chapters?: number | null;
  volumes?: number | null;
  episodes?: number | null;
  averageScore?: number | null;
  coverImage?: {
    extraLarge?: string | null;
    large?: string | null;
    medium?: string | null;
  } | null;
  genres?: string[] | null;
  description?: string | null;
  recommendations?: {
    nodes?: Array<{
      mediaRecommendation?: {
        idMal?: number | null;
        title?: AniListTitle | null;
        coverImage?: {
          large?: string | null;
          extraLarge?: string | null;
        } | null;
        format?: string | null;
        chapters?: number | null;
        averageScore?: number | null;
      } | null;
    }> | null;
  } | null;
}

const ANILIST_MEDIA_FRAGMENT = `
  fragment MediaCore on Media {
    id
    idMal
    title { romaji english native }
    synonyms
    format
    status
    chapters
    volumes
    episodes
    averageScore
    coverImage { extraLarge large medium }
    genres
    description(asHtml: false)
  }
`;

async function runAniListQuery<T>(
  query: string,
  variables: Record<string, unknown>,
): Promise<T | null> {
  assertAllowedOutboundUrl(ANILIST_GRAPHQL_URL);

  try {
    return await withFetchSlot(async () => {
      const response = await fetch(ANILIST_GRAPHQL_URL, {
        method: "POST",
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ query, variables }),
      });
      if (!response.ok) {
        return null;
      }
      const payload = (await response.json()) as { data?: T };
      return payload.data ?? null;
    });
  } catch {
    return null;
  }
}

const ANILIST_BY_MAL_QUERY = `
  query ($idMal: Int, $type: MediaType) {
    Media(id_mal: $idMal, type: $type) {
      ...MediaCore
    }
  }
  ${ANILIST_MEDIA_FRAGMENT}
`;

const ANILIST_SEARCH_QUERY = `
  query ($search: String, $type: MediaType) {
    Page(perPage: 5) {
      media(search: $search, type: $type) {
        ...MediaCore
      }
    }
  }
  ${ANILIST_MEDIA_FRAGMENT}
`;

const ANILIST_RECOMMENDATIONS_QUERY = `
  query ($idMal: Int, $type: MediaType) {
    Media(id_mal: $idMal, type: $type) {
      recommendations(perPage: 6, sort: RATING_DESC) {
        nodes {
          mediaRecommendation {
            idMal
            title { romaji english }
            coverImage { large extraLarge }
            format
            chapters
            averageScore
          }
        }
      }
    }
  }
`;

function aniListStatusLabel(status: string | null | undefined): string | null {
  const cleaned = cleanText(status);
  switch (cleaned.toUpperCase()) {
    case "FINISHED":
      return "Finished";
    case "RELEASING":
      return "Currently Publishing";
    case "NOT_YET_RELEASED":
      return "Not yet released";
    case "CANCELLED":
      return "Cancelled";
    case "HIATUS":
      return "On hiatus";
    case "CANCELLED_AND_DROPPED":
      return "Cancelled";
    default:
      return cleaned || null;
  }
}

function aniListDisplayType(format: string | null | undefined): string | null {
  const cleaned = cleanText(format).toUpperCase();
  switch (cleaned) {
    case "TV":
      return "TV";
    case "TV_SHORT":
      return "TV Short";
    case "MOVIE":
      return "Movie";
    case "SPECIAL":
      return "Special";
    case "OVA":
      return "OVA";
    case "ONA":
      return "ONA";
    case "MUSIC":
      return "Music";
    case "MANGA":
      return "Manga";
    case "NOVEL":
      return "Novel";
    case "LIGHT_NOVEL":
      return "Light Novel";
    case "ONE_SHOT":
      return "One-shot";
    case "DOUJIN":
      return "Doujin";
    default:
      return cleaned || null;
  }
}

function collectAniListTitleCandidates(media: AniListMedia): string[] {
  return [
    media.title?.english,
    media.title?.romaji,
    media.title?.native,
    ...(Array.isArray(media.synonyms) ? media.synonyms : []),
  ]
    .map((value) => cleanText(value))
    .filter(Boolean);
}

function mapMetaFromAniList(
  media: AniListMedia,
  mediaType: MetaMediaType,
): MetaPayload {
  const candidates = collectAniListTitleCandidates(media);
  return {
    mal_id: safeNumber(media.idMal),
    media_type: mediaType,
    display_type: aniListDisplayType(media.format) || mediaType,
    title:
      cleanText(media.title?.english) ||
      cleanText(media.title?.romaji) ||
      cleanText(media.title?.native) ||
      "Unknown",
    title_candidates: candidates,
    image_url:
      cleanText(media.coverImage?.extraLarge) ||
      cleanText(media.coverImage?.large) ||
      cleanText(media.coverImage?.medium) ||
      null,
    status: aniListStatusLabel(media.status),
    synopsis: cleanText(media.description) || null,
    genres: (Array.isArray(media.genres) ? media.genres : [])
      .filter(Boolean)
      .slice(0, 10),
    chapters: safeNumber(media.chapters),
    volumes: safeNumber(media.volumes),
    episodes: safeNumber(media.episodes),
    // AniList averageScore is 0-100; MAL mean is 0-10. Normalize to the MAL scale.
    score:
      safeNumber(media.averageScore) === null
        ? null
        : (safeNumber(media.averageScore) ?? 0) / 10,
    source: "anilist",
  };
}

async function fetchAniListByMalId(
  malId: number,
  mediaType: MetaMediaType,
): Promise<AniListMedia | null> {
  const data = await runAniListQuery<{ Media?: AniListMedia | null }>(
    ANILIST_BY_MAL_QUERY,
    {
      idMal: malId,
      type: mediaType.toUpperCase(),
    },
  );
  const media = data?.Media ?? null;
  return media && typeof media.id === "number" ? media : null;
}

async function fetchAniListBySearch(
  query: string,
  mediaType: MetaMediaType,
): Promise<AniListMedia[]> {
  const data = await runAniListQuery<{
    Page?: { media?: AniListMedia[] | null } | null;
  }>(ANILIST_SEARCH_QUERY, {
    search: query,
    type: mediaType.toUpperCase(),
  });
  const rows = data?.Page?.media;
  return Array.isArray(rows)
    ? rows.filter((row): row is AniListMedia =>
        Boolean(row && typeof row.id === "number"),
      )
    : [];
}

// ---------------------------------------------------------------------------
// Public provider API
// ---------------------------------------------------------------------------

function mediaTypeOrder(preferred: MetaMediaType | null): MetaMediaType[] {
  if (!preferred) {
    return ["manga", "anime"];
  }
  return preferred === "anime" ? ["anime", "manga"] : ["manga", "anime"];
}

export async function fetchMediaMeta(
  options: MetaFetchOptions,
): Promise<MetaPayload | null> {
  const malId = options.malId;
  if (!Number.isFinite(malId) || malId <= 0) {
    return null;
  }

  const preferred = options.preferredType ?? null;
  const types =
    options.strictMediaType && preferred
      ? [preferred]
      : mediaTypeOrder(preferred);
  const fallbackTitle = cleanText(options.fallbackTitle);

  for (const mediaType of types) {
    const cacheKey = `mal:${mediaType}:${malId}`;
    const cached = readMetaCache(cacheKey);
    if (cached !== undefined) {
      if (
        cached &&
        (!fallbackTitle ||
          !options.strictMediaType ||
          metaTitleMatches(cached.title_candidates, fallbackTitle))
      ) {
        return cached;
      }
      if (cached === null) {
        continue;
      }
    }

    const malNode = await fetchMalDetails(mediaType, malId);
    if (malNode) {
      const mapped = mapMetaFromMal(malNode, mediaType);
      writeMetaCache(cacheKey, mapped);
      if (metaTitleMatches(mapped.title_candidates, fallbackTitle)) {
        return mapped;
      }
      if (!fallbackTitle) {
        return mapped;
      }
      // Wrong-title enrichment guard matched nothing; keep cached and fall through.
      continue;
    }

    writeMetaCache(cacheKey, null);
  }

  for (const mediaType of types) {
    const cacheKey = `anilist:${mediaType}:${malId}`;
    const cached = readMetaCache(cacheKey);
    if (cached !== undefined) {
      if (cached && metaTitleMatches(cached.title_candidates, fallbackTitle)) {
        return cached;
      }
      if (cached === null) {
        continue;
      }
    }

    const aniMedia = await fetchAniListByMalId(malId, mediaType);
    if (aniMedia) {
      const mapped = mapMetaFromAniList(aniMedia, mediaType);
      writeMetaCache(cacheKey, mapped);
      if (
        !fallbackTitle ||
        metaTitleMatches(mapped.title_candidates, fallbackTitle)
      ) {
        return mapped;
      }
      continue;
    }

    writeMetaCache(cacheKey, null);
  }

  return null;
}

export async function resolveAnimeCompanion(
  mangaMalId: number,
  mangaTitle: string,
): Promise<AnimeCompanionInfo | null> {
  // 1) MAL manga payload already carries related_anime — one call covers both.
  const mangaNode = await fetchMalDetails("manga", mangaMalId);
  if (mangaNode && Array.isArray(mangaNode.related_anime)) {
    const animeEntry = mangaNode.related_anime
      .map((row) => row.node)
      .find((node) => node && typeof node.id === "number");
    if (animeEntry?.id) {
      const animeMeta = await fetchMediaMeta({
        malId: animeEntry.id,
        preferredType: "anime",
        strictMediaType: true,
      });
      if (animeMeta) {
        return companionFromMeta(animeMeta, mangaTitle);
      }
    }
  }

  // 2) AniList by MAL id.
  const animeMeta = await fetchMediaMeta({
    malId: mangaMalId,
    preferredType: "anime",
    strictMediaType: true,
    fallbackTitle: mangaTitle,
  });
  if (animeMeta) {
    return companionFromMeta(animeMeta, mangaTitle);
  }

  return null;
}

function companionFromMeta(
  meta: MetaPayload,
  mangaTitle: string,
): AnimeCompanionInfo {
  const keyword = meta.title.trim().split(/\s+/).filter(Boolean).join("+");
  return {
    mal_id: meta.mal_id,
    title: meta.title || mangaTitle,
    status: meta.status,
    episodes: meta.episodes,
    score: meta.score,
    image_url: meta.image_url,
    watch_url: keyword ? `https://anikaitv.to/filter?keyword=${keyword}` : "",
    citation:
      meta.mal_id === null
        ? `[Source: MAL-ID unknown - ${meta.title}]`
        : `[Source: MAL-ID ${meta.mal_id} - ${meta.title}]`,
  };
}

export async function resolveRecommendations(seriesTitle: string): Promise<{
  alternatives: unknown[];
  similarSeries: Array<{
    title: string;
    coverUrl: string | null;
    chapterCount: number;
    votes?: number;
    malId?: number;
    url?: string;
  }>;
  trending: unknown[];
}> {
  const empty = { alternatives: [], similarSeries: [], trending: [] };
  const cleanedTitle = cleanText(seriesTitle);
  if (!cleanedTitle) {
    return empty;
  }

  // Resolve a MAL id first (best-effort, single attempt each).
  let anchorMalId: number | null = null;
  const clientId = getMalClientId();
  if (clientId) {
    try {
      const url = new URL(`${MAL_BASE_URL}/manga`);
      url.searchParams.set("q", cleanedTitle);
      url.searchParams.set("limit", "3");
      url.searchParams.set("fields", "id,title");
      assertAllowedOutboundUrl(url.toString());
      const response = await withFetchSlot(() =>
        fetch(url.toString(), {
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
          headers: { Accept: "application/json", "X-MAL-CLIENT-ID": clientId },
        }),
      );
      if (response.ok) {
        const payload = (await response.json()) as {
          data?: Array<{ node?: { id?: number } }>;
        };
        anchorMalId = payload.data?.[0]?.node?.id ?? null;
      }
    } catch {
      anchorMalId = null;
    }
  }

  if (anchorMalId !== null) {
    const data = await runAniListQuery<{
      Media?:
        | (AniListMedia & { recommendations?: AniListMedia["recommendations"] })
        | null;
    }>(ANILIST_RECOMMENDATIONS_QUERY, { idMal: anchorMalId, type: "MANGA" });

    const nodes = data?.Media?.recommendations?.nodes ?? [];
    const similarSeries = nodes
      .map((row) => row.mediaRecommendation)
      .filter((rec): rec is NonNullable<typeof rec> =>
        Boolean(rec && rec.title),
      )
      .map((rec) => ({
        title:
          cleanText(rec.title?.english) ||
          cleanText(rec.title?.romaji) ||
          "Unknown",
        coverUrl:
          cleanText(rec.coverImage?.large) ||
          cleanText(rec.coverImage?.extraLarge) ||
          null,
        chapterCount: safeNumber(rec.chapters) ?? 0,
        malId: safeNumber(rec.idMal) ?? undefined,
      }))
      .filter((row) => row.title !== "Unknown")
      .slice(0, 5);

    if (similarSeries.length > 0) {
      return { ...empty, similarSeries };
    }
  }

  // Final fallback: treat top AniList search hits (minus the exact series) as similar.
  const searchHits = await fetchAniListBySearch(cleanedTitle, "manga");
  const exactKey = normalizeTitle(cleanedTitle);
  const similarSeries = searchHits
    .filter(
      (hit) =>
        normalizeTitle(hit.title?.english ?? hit.title?.romaji ?? "") !==
        exactKey,
    )
    .slice(0, 5)
    .map((hit) => ({
      title:
        cleanText(hit.title?.english) ||
        cleanText(hit.title?.romaji) ||
        "Unknown",
      coverUrl:
        cleanText(hit.coverImage?.large) ||
        cleanText(hit.coverImage?.extraLarge) ||
        null,
      chapterCount: safeNumber(hit.chapters) ?? 0,
      malId: safeNumber(hit.idMal) ?? undefined,
    }))
    .filter((row) => row.title !== "Unknown");

  return { ...empty, similarSeries };
}

export async function resolveChapterCount(options: {
  seriesTitle: string;
  chapterTitle?: string | null;
  chapterUrl?: string;
}): Promise<number | null> {
  const malCount = await resolveMangaChapterCount({
    seriesTitle: options.seriesTitle,
    chapterTitle: options.chapterTitle,
    chapterUrl: options.chapterUrl,
  });
  if (malCount && malCount > 0) {
    return Math.min(Math.floor(malCount), 1500);
  }

  // AniList fallback keyed by title search.
  const queries = [
    cleanText(options.seriesTitle),
    cleanText(options.chapterTitle ?? ""),
  ].filter(Boolean);
  for (const query of queries) {
    const hits = await fetchAniListBySearch(query, "manga");
    if (hits.length > 0) {
      const chapters = safeNumber(hits[0]?.chapters);
      if (chapters && chapters > 0) {
        return Math.min(Math.floor(chapters), 1500);
      }
    }
  }

  return null;
}
