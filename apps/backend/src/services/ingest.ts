import * as cheerio from "cheerio";
import { LRUCache } from "lru-cache";
import { z } from "zod";
import { resolveAdapterForUrl } from "../adapters/index.js";
import { IngestError } from "../errors.js";
import type { Adapter, ChapterListItem, ExtractedChapterData, IngestResponse } from "../types.js";
import { extractChapterData } from "./aiExtractor.js";
import { sortChapterList } from "./chapterSort.js";
import { fetchChapterHtml } from "./fetchProxy.js";
import { extractChapterLinks } from "./html.js";
import { isLikelyLoginWall } from "./loginDetection.js";
import { detectReadingMode } from "./mode.js";

const urlSchema = z.string().url();

const parseCache = new LRUCache<string, IngestResponse>({
  max: 500,
  ttl: 15 * 60 * 1000
});

const CHAPTER_LINK_SELECTORS = [
  ".chapters a",
  ".chapter-list a",
  ".wp-manga-chapter a",
  ".row-content-chapter a",
  "#_listUl a",
  ".episode_lst a",
  "a[href*='/chapter']",
  "a[href*='/c']",
  "a[href*='/episode']"
];

const MIN_FULL_CHAPTER_LIST_SIZE = 8;
const MAX_INDEX_FETCH_ATTEMPTS = 3;
const SERIES_ROOT_SEGMENTS = new Set(["manga", "manhwa", "manhua", "comic", "series", "title", "titles"]);
const CHAPTER_ROOT_SEGMENTS = new Set(["chapter", "chapters", "viewer", "read", "episode", "episodes"]);
const SERIES_QUERY_KEYS = ["title_no", "series", "manga", "comic", "id"] as const;
const CHAPTER_IN_SEGMENT_PATTERN =
  /(?:-|_| )?(?:chapter|chap|ch|episode|ep|c)[-_ ]*\d+(?:\.\d+)?(?:[-_ ]?(?:v|p)\d+)?$/i;
const WEEBCENTRAL_IMAGE_RE =
  /https?:\/\/[^"'\s<>]+\.(?:png|jpg|jpeg|webp|avif)(?:\?[^"'\s<>]*)?/gi;

function looksLikeChapterUrl(url: string): boolean {
  return /(?:\/|-)chapter[-_/ ]?\d+|\bc\d+(?:\.\d+)?\b|(?:\/|-)ep(?:isode)?[-_/ ]?\d+/i.test(url);
}

function looksLikeChapterText(value: string | null | undefined): boolean {
  return /\b(chapter|ch\.?|episode|ep\.?|prologue|oneshot|one-shot)\b/i.test(value ?? "");
}

function parseChapterNumber(value: string | null | undefined): number | null {
  const match = (value ?? "").match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function hasPanelEvidenceInHtml(html: string, panelUrl: string): boolean {
  const normalizedHtml = html.replace(/\\\//g, "/").replace(/&amp;/g, "&");
  const normalizedUrl = panelUrl.replace(/&amp;/g, "&");

  if (normalizedHtml.includes(normalizedUrl)) {
    return true;
  }

  const noQuery = normalizedUrl.split("?")[0] ?? normalizedUrl;
  if (noQuery && normalizedHtml.includes(noQuery)) {
    return true;
  }

  try {
    const parsed = new URL(normalizedUrl);
    const pathname = parsed.pathname;
    if (pathname && pathname !== "/" && normalizedHtml.includes(pathname)) {
      return true;
    }
  } catch {
    // Ignore URL parsing failures because raw matching already ran.
  }

  return false;
}

function normalizeComparableUrl(url: string): string {
  try {
    const parsed = new URL(url);
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return url;
  }
}

function normalizeHost(hostname: string): string {
  return hostname.toLowerCase().replace(/^(www\d*|www|m|mobile|read)\./i, "");
}

function splitLowerPathSegments(pathname: string): string[] {
  return pathname
    .split("/")
    .map((segment) => segment.trim().toLowerCase())
    .filter(Boolean);
}

function normalizeSeriesSegment(segment: string): string {
  const cleaned = segment.trim().toLowerCase();
  if (!cleaned) {
    return "";
  }

  const stripped = cleaned.replace(CHAPTER_IN_SEGMENT_PATTERN, "").replace(/[-_]+$/g, "").trim();
  return stripped || cleaned;
}

function buildSeriesKey(url: string): string | null {
  try {
    const parsed = new URL(url);
    const host = normalizeHost(parsed.hostname);
    if (!host) {
      return null;
    }

    for (const key of SERIES_QUERY_KEYS) {
      const value = parsed.searchParams.get(key)?.trim().toLowerCase();
      if (value) {
        return `${host}?${key}=${value}`;
      }
    }

    const segments = splitLowerPathSegments(parsed.pathname);
    if (segments.length === 0) {
      return host;
    }

    const first = segments[0] ?? "";
    const second = segments[1] ?? "";
    const normalizedSecond = normalizeSeriesSegment(second) || second;

    if (segments.length >= 2 && SERIES_ROOT_SEGMENTS.has(first) && second) {
      return `${host}/${first}/${normalizedSecond}`;
    }

    if (segments.length >= 3 && first.length === 2) {
      const third = segments[2] ?? "";
      const normalizedThird = normalizeSeriesSegment(third) || third;
      if (third) {
        return `${host}/${first}/${normalizedSecond}/${normalizedThird}`;
      }
    }

    if (segments.length >= 2 && !CHAPTER_ROOT_SEGMENTS.has(first) && second) {
      return `${host}/${first}/${normalizedSecond}`;
    }

    return host;
  } catch {
    return null;
  }
}

function isSameSeriesUrl(baseUrl: string, candidateUrl: string): boolean {
  const baseKey = buildSeriesKey(baseUrl);
  const candidateKey = buildSeriesKey(candidateUrl);

  if (!baseKey || !candidateKey) {
    return false;
  }

  return baseKey === candidateKey;
}

function filterChapterListToSeries(baseUrl: string, chapters: ChapterListItem[]): ChapterListItem[] {
  const filtered = chapters.filter((chapter) => isSameSeriesUrl(baseUrl, chapter.url));
  if (filtered.length === 0) {
    return chapters;
  }

  return filtered;
}

function mergeUniqueChapters(chapters: ChapterListItem[]): ChapterListItem[] {
  const unique = new Map<string, ChapterListItem>();

  for (const chapter of chapters) {
    const normalizedUrl = normalizeComparableUrl(chapter.url);
    const existing = unique.get(normalizedUrl);
    if (!existing) {
      unique.set(normalizedUrl, {
        ...chapter,
        url: normalizedUrl
      });
      continue;
    }

    unique.set(normalizedUrl, {
      ...existing,
      number: existing.number || chapter.number,
      title: existing.title || chapter.title,
      special: existing.special ?? chapter.special,
      url: normalizedUrl
    });
  }

  return Array.from(unique.values());
}

function inferSeriesCandidateUrls(chapterUrl: string): string[] {
  try {
    const parsed = new URL(chapterUrl);
    parsed.hash = "";
    parsed.search = "";

    const pathParts = parsed.pathname.split("/").filter(Boolean);
    const candidates = new Set<string>();

    const addCandidate = (parts: string[]) => {
      const candidate = new URL(parsed.toString());
      candidate.pathname = parts.length > 0 ? `/${parts.join("/")}/` : "/";
      candidates.add(candidate.toString());
    };

    if (pathParts.length > 1) {
      addCandidate(pathParts.slice(0, -1));
    }

    if (pathParts.length > 2) {
      addCandidate(pathParts.slice(0, -2));
    }

    if (pathParts.length > 0) {
      const last = pathParts[pathParts.length - 1] ?? "";
      const cleanedLast = last
        .replace(/(?:chapter|chap|ch|episode|ep)[-_ ]*\d+(?:\.\d+)?/i, "")
        .replace(/[-_]+$/, "")
        .trim();

      if (cleanedLast && cleanedLast !== last) {
        const replaced = [...pathParts];
        replaced[replaced.length - 1] = cleanedLast;
        addCandidate(replaced);
      }
    }

    return Array.from(candidates).filter(
      (candidate) => normalizeComparableUrl(candidate) !== normalizeComparableUrl(chapterUrl)
    );
  } catch {
    return [];
  }
}

async function enrichSparseChapterList(
  chapterUrl: string,
  _seriesTitle: string,
  _chapterUrlPattern: string | null,
  chapterList: ChapterListItem[]
): Promise<ChapterListItem[]> {
  let merged = mergeUniqueChapters(chapterList);

  const candidates = inferSeriesCandidateUrls(chapterUrl).slice(0, MAX_INDEX_FETCH_ATTEMPTS);

  for (const candidateUrl of candidates) {
    try {
      const fetched = await fetchChapterHtml(candidateUrl);
      if (fetched.status >= 400 || !fetched.html) {
        continue;
      }

      const $ = cheerio.load(fetched.html);
      const discovered = extractChapterLinks($, CHAPTER_LINK_SELECTORS, fetched.finalUrl);
      if (discovered.length === 0) {
        continue;
      }

      const nextMerged = mergeUniqueChapters([...merged, ...discovered]);
      if (nextMerged.length > merged.length) {
        merged = nextMerged;
      }

      if (merged.length >= MIN_FULL_CHAPTER_LIST_SIZE) {
        break;
      }
    } catch {
      // Best-effort enrichment only.
    }
  }

  return merged;
}

function toAbsoluteUrl(baseUrl: string, value: string | null | undefined): string | null {
  if (!value) {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed || trimmed.startsWith("#") || /^(javascript|about|data):/i.test(trimmed)) {
    return null;
  }

  try {
    return new URL(trimmed, baseUrl).toString();
  } catch {
    return null;
  }
}

function extractWeebcentralChapterId(url: string): string {
  try {
    const parsed = new URL(url);
    if (!normalizeHost(parsed.hostname).endsWith("weebcentral.com")) {
      return "";
    }

    const segments = splitLowerPathSegments(parsed.pathname);
    const chapterIndex = segments.findIndex((segment) => segment === "chapters");
    if (chapterIndex < 0) {
      return "";
    }

    const id = parsed.pathname.split("/").filter(Boolean)[chapterIndex + 1] ?? "";
    return id.trim();
  } catch {
    return "";
  }
}

async function fetchWeebcentralLongStripPanels(chapterUrl: string): Promise<string[]> {
  const chapterId = extractWeebcentralChapterId(chapterUrl);
  if (!chapterId) {
    return [];
  }

  let endpoint: string;
  try {
    const parsed = new URL(chapterUrl);
    endpoint = `${parsed.origin}/chapters/${chapterId}/images?is_prev=False&current_page=1&reading_style=long_strip`;
  } catch {
    return [];
  }

  try {
    const fetched = await fetchChapterHtml(endpoint);
    if (fetched.status >= 400 || !fetched.html) {
      return [];
    }

    const matches = fetched.html.match(WEEBCENTRAL_IMAGE_RE) ?? [];
    const unique = Array.from(new Set(matches));
    if (unique.length < 2) {
      return [];
    }

    return unique;
  } catch {
    return [];
  }
}

function resolveFirstChapterUrlFromSeriesPage(
  seriesUrl: string,
  extracted: ExtractedChapterData
): string | null {
  const current = normalizeComparableUrl(seriesUrl);
  const suggested = toAbsoluteUrl(seriesUrl, extracted.suggestedFirstChapterUrl);

  if (suggested && normalizeComparableUrl(suggested) !== current) {
    return suggested;
  }

  const candidates = mergeUniqueChapters(extracted.chapterList)
    .map((chapter) => ({
      ...chapter,
      url: toAbsoluteUrl(seriesUrl, chapter.url) ?? chapter.url
    }))
    .filter((chapter) => {
      const normalized = normalizeComparableUrl(chapter.url);
      if (normalized === current) {
        return false;
      }

      return (
        looksLikeChapterText(chapter.title) ||
        looksLikeChapterText(chapter.number) ||
        looksLikeChapterUrl(chapter.url)
      );
    });

  if (candidates.length === 0) {
    return null;
  }

  const sorted = sortChapterList(candidates);
  const firstNumeric =
    sorted.find((chapter) => {
      const parsed = parseChapterNumber(chapter.number || chapter.title);
      return parsed !== null && parsed >= 0 && chapter.special !== true;
    }) ?? sorted.find((chapter) => chapter.special !== true);

  return firstNumeric?.url ?? sorted[0]?.url ?? null;
}

async function extractFromUrl(chapterUrl: string): Promise<{
  adapter: Adapter | null;
  fetched: Awaited<ReturnType<typeof fetchChapterHtml>>;
  extracted: ExtractedChapterData;
  panelUrls: string[];
  hasChapterSignal: boolean;
}> {
  const adapter = resolveAdapterForUrl(chapterUrl);
  const fetched = await fetchChapterHtml(chapterUrl);
  const extracted = await extractChapterData({
    html: fetched.html,
    chapterUrl: fetched.finalUrl,
    adapter
  });

  const candidatePanelUrls = Array.from(
    new Set(extracted.chapter.panelCandidates.map((candidate) => candidate.url))
  );
  let panelUrls = candidatePanelUrls.filter((url) => hasPanelEvidenceInHtml(fetched.html, url));
  if (panelUrls.length < 2) {
    const weebcentralPanels = await fetchWeebcentralLongStripPanels(fetched.finalUrl);
    if (weebcentralPanels.length >= 2) {
      panelUrls = weebcentralPanels;
    }
  }

  const hasChapterSignal =
    looksLikeChapterUrl(fetched.finalUrl) ||
    looksLikeChapterText(extracted.chapter.title) ||
    looksLikeChapterText(extracted.chapter.number);

  return {
    adapter,
    fetched,
    extracted,
    panelUrls,
    hasChapterSignal
  };
}

async function ingestUrlInternal(rawUrl: string, depth: number): Promise<IngestResponse> {
  const parseResult = urlSchema.safeParse(rawUrl);
  if (!parseResult.success) {
    throw new IngestError("INVALID_URL", "Please provide a valid chapter URL.", 400);
  }

  const chapterUrl = parseResult.data;

  const cached = parseCache.get(chapterUrl);
  if (cached) {
    return cached;
  }

  const { adapter, fetched, extracted, panelUrls, hasChapterSignal } = await extractFromUrl(chapterUrl);

  const shouldResolveSeriesPage =
    depth < 1 &&
    panelUrls.length < 4 &&
    (extracted.pageType === "series" || (!hasChapterSignal && extracted.chapterList.length > 0));

  if (shouldResolveSeriesPage) {
    const firstChapterUrl = resolveFirstChapterUrlFromSeriesPage(fetched.finalUrl, extracted);
    if (firstChapterUrl && normalizeComparableUrl(firstChapterUrl) !== normalizeComparableUrl(fetched.finalUrl)) {
      const resolved = await ingestUrlInternal(firstChapterUrl, depth + 1);
      parseCache.set(chapterUrl, resolved);
      return resolved;
    }
  }

  if (!hasChapterSignal && panelUrls.length < 4) {
    throw new IngestError(
      "NOT_CHAPTER",
      "This looks like a series or info page, but a readable first chapter could not be found.",
      400
    );
  }

  if (
    isLikelyLoginWall({
      finalUrl: fetched.finalUrl,
      status: fetched.status,
      html: fetched.html,
      panelCount: panelUrls.length,
      treatLowPanelCountAsLoginWall: looksLikeChapterUrl(fetched.finalUrl)
    })
  ) {
    throw new IngestError(
      "NOT_PUBLIC",
      "This source does not expose images externally. Try another source.",
      403
    );
  }

  if (panelUrls.length < 2) {
    if (!looksLikeChapterUrl(fetched.finalUrl) && extracted.chapterList.length > 0) {
      throw new IngestError(
        "NOT_CHAPTER",
        "This looks like a series page. Select a chapter and paste that chapter link.",
        400
      );
    }

    throw new IngestError(
      "NOT_CHAPTER",
      "This page does not expose panel image links in HTML. Open this chapter on the source site.",
      400
    );
  }

  const chapterListInput = mergeUniqueChapters([
    ...extracted.chapterList,
    {
      number: extracted.chapter.number,
      title: extracted.chapter.title,
      url: fetched.finalUrl
    }
  ]);

  const enrichedChapterList = await enrichSparseChapterList(
    fetched.finalUrl,
    extracted.series.title,
    extracted.chapterUrlPattern,
    chapterListInput
  );
  const seriesConstrainedChapterList = filterChapterListToSeries(fetched.finalUrl, enrichedChapterList);

  const detectedMode =
    extracted.detectedModeHint ??
    detectReadingMode(adapter?.defaultMode ?? "paginated", extracted.chapter.panelCandidates);

  const result: IngestResponse = {
    series: {
      title: extracted.series.title,
      coverUrl: extracted.series.coverUrl,
      genres: extracted.series.genres,
      status: extracted.series.status
    },
    chapter: {
      number: extracted.chapter.number,
      title: extracted.chapter.title,
      panelUrls,
      totalPages: panelUrls.length
    },
    chapterList: sortChapterList(seriesConstrainedChapterList),
    detectedMode,
    sourceAdapter: adapter?.id ?? extracted.source,
    extractionSource: extracted.source
  };

  parseCache.set(chapterUrl, result);
  return result;
}

export async function ingestUrl(rawUrl: string): Promise<IngestResponse> {
  return ingestUrlInternal(rawUrl, 0);
}
