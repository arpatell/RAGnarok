import { LRUCache } from "lru-cache";
import type { ChapterListItem } from "../types.js";

const MAL_BASE_URL = "https://api.myanimelist.net/v2";
const REQUEST_TIMEOUT_MS = 12_000;
const SEARCH_LIMIT = 10;

const chapterCountCache = new LRUCache<string, number>({
  max: 250,
  ttl: 30 * 60 * 1000
});

const chapterCountMissCache = new LRUCache<string, true>({
  max: 250,
  ttl: 30 * 60 * 1000
});

interface MalSearchResult {
  id: number;
  title?: string | null;
  alternative_titles?: unknown;
  num_chapters?: number | null;
}

interface MalSearchEntry {
  node?: MalSearchResult;
}

interface MalSearchResponse {
  data?: Array<MalSearchResult | MalSearchEntry>;
}

interface MalDetailsResponse {
  data?: MalSearchResult;
  id?: number;
  title?: string | null;
  alternative_titles?: unknown;
  num_chapters?: number | null;
}

interface ScoredCandidate {
  manga: MalSearchResult;
  score: number;
}

function cleanText(value: string | null | undefined): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function normalizeText(value: string | null | undefined): string {
  return cleanText(value).toLowerCase();
}

function collectStrings(value: unknown, values: Set<string>): void {
  if (typeof value === "string") {
    const cleaned = normalizeText(value);
    if (cleaned) {
      values.add(cleaned);
    }
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectStrings(item, values);
    }
    return;
  }

  if (value && typeof value === "object") {
    for (const nested of Object.values(value as Record<string, unknown>)) {
      collectStrings(nested, values);
    }
  }
}

function collectMangaTitles(manga: MalSearchResult): string[] {
  const titles = new Set<string>();
  collectStrings(manga.title, titles);
  collectStrings(manga.alternative_titles, titles);
  return Array.from(titles);
}

function normalizeSearchResult(candidate: MalSearchResult | MalSearchEntry): MalSearchResult | null {
  const node = (candidate as MalSearchEntry).node ?? (candidate as MalSearchResult);
  if (!node || typeof node !== "object") {
    return null;
  }

  if (typeof node.id !== "number" || !Number.isFinite(node.id)) {
    return null;
  }

  return {
    id: node.id,
    title: typeof node.title === "string" ? node.title : null,
    alternative_titles: node.alternative_titles,
    num_chapters: typeof node.num_chapters === "number" ? node.num_chapters : null
  };
}

function normalizeDetailsResult(payload: MalDetailsResponse | null): MalSearchResult | null {
  if (!payload) {
    return null;
  }

  const raw = payload.data ?? payload;
  if (typeof raw.id !== "number" || !Number.isFinite(raw.id)) {
    return null;
  }

  return {
    id: raw.id,
    title: typeof raw.title === "string" ? raw.title : null,
    alternative_titles: raw.alternative_titles,
    num_chapters: typeof raw.num_chapters === "number" ? raw.num_chapters : null
  };
}

function scoreCandidate(manga: MalSearchResult, query: string): number {
  const titles = collectMangaTitles(manga);
  if (titles.length === 0) {
    return Number.POSITIVE_INFINITY;
  }

  const normalizedQuery = normalizeText(query);
  const queryTokens = normalizedQuery.split(/\s+/).filter((token) => token.length >= 3);
  let bestScore = Number.POSITIVE_INFINITY;

  for (const title of titles) {
    if (title === normalizedQuery) {
      return 0;
    }

    if (title.startsWith(normalizedQuery)) {
      bestScore = Math.min(bestScore, 1);
      continue;
    }

    const titleTokens = title.split(/\s+/).filter((token) => token.length >= 3);
    const tokenMatches = queryTokens.filter((token) => titleTokens.includes(token)).length;

    if (tokenMatches > 0 && tokenMatches >= Math.min(queryTokens.length, titleTokens.length)) {
      bestScore = Math.min(bestScore, 1.5);
      continue;
    }

    if (title.includes(normalizedQuery) || normalizedQuery.includes(title)) {
      bestScore = Math.min(bestScore, 2);
    }
  }

  return bestScore;
}

function buildSearchQueries(seriesTitle: string): string[] {
  const cleaned = cleanText(seriesTitle);
  if (!cleaned) {
    return [];
  }

  const variants = new Set<string>([cleaned]);

  const stripped = cleaned
    .replace(/[\(\)\[\]\{\}:|]/g, " ")
    .replace(/[\u2013\u2014\-_/]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (stripped) {
    variants.add(stripped);
  }

  const leadingTitle = cleaned.split(/[\-:|\(\[]/)[0]?.trim();
  if (leadingTitle) {
    variants.add(leadingTitle);
  }

  const lower = cleaned.toLowerCase();
  if (lower.includes(" - ")) {
    variants.add(cleaned.split(" - ")[0]?.trim() ?? "");
  }

  const withoutChapterNoise = cleaned
    .replace(/\b(chapter|chap|ch|episode|ep)\b.*$/i, "")
    .replace(/\bvol(?:ume)?\b.*$/i, "")
    .replace(/\b\d+(?:\.\d+)?\b.*$/i, "")
    .replace(/[\s\-_:|]+$/g, "")
    .trim();
  if (withoutChapterNoise) {
    variants.add(withoutChapterNoise);
  }

  return Array.from(variants).filter(Boolean);
}

function formatLikeTemplate(template: string, value: number): string {
  const decimalPlaces = template.includes(".") ? (template.split(".")[1] ?? "").length : 0;
  const sign = value < 0 ? "-" : "";
  const absolute = Math.abs(value);

  let formatted = decimalPlaces > 0 ? absolute.toFixed(decimalPlaces) : String(Math.round(absolute));

  const integerTemplate = template.replace(/^-/, "").split(".")[0] ?? template;
  const leadingZeros = integerTemplate.match(/^0+/)?.[0].length ?? 0;
  if (leadingZeros > 0) {
    const [intPart, decimalPart] = formatted.split(".");
    const safeIntPart = intPart ?? "0";
    const paddedInt = safeIntPart.padStart(Math.max(safeIntPart.length, leadingZeros + 1), "0");
    formatted = decimalPart ? `${paddedInt}.${decimalPart}` : paddedInt;
  }

  return `${sign}${formatted}`;
}

function parseChapterNumber(value: string): string {
  const cleaned = cleanText(value);
  const match = cleaned.match(/-?\d+(?:\.\d+)?/);
  if (match) {
    return match[0] ?? cleaned;
  }

  return cleaned;
}

function findChapterToken(pathname: string): { start: number; end: number; raw: string } | null {
  const keywordPattern = /(chapter|chap|ch|episode|ep|c)([-_ ]*)(\d+(?:\.\d+)?)/gi;
  let selected: { start: number; end: number; raw: string } | null = null;

  for (const match of pathname.matchAll(keywordPattern)) {
    const fullMatch = match[0] ?? "";
    const numericRaw = match[3] ?? "";
    const fullIndex = match.index ?? -1;
    if (!fullMatch || !numericRaw || fullIndex < 0) {
      continue;
    }

    const numericStart = fullIndex + fullMatch.lastIndexOf(numericRaw);
    selected = {
      start: numericStart,
      end: numericStart + numericRaw.length,
      raw: numericRaw
    };
  }

  if (selected) {
    return selected;
  }

  const genericPattern = /(\d+(?:\.\d+)?)/g;
  for (const match of pathname.matchAll(genericPattern)) {
    const raw = match[1] ?? "";
    const index = match.index ?? -1;
    if (!raw || index < 0) {
      continue;
    }

    selected = {
      start: index,
      end: index + raw.length,
      raw
    };
  }

  return selected;
}

function renderChapterUrlFromTemplate(chapterUrl: string, chapterNumber: string): string | null {
  const cleanedNumber = cleanText(chapterNumber);
  if (!cleanedNumber) {
    return null;
  }

  try {
    const parsed = new URL(chapterUrl);
    const token = findChapterToken(parsed.pathname);
    if (token) {
      const numeric = Number.parseFloat(parseChapterNumber(cleanedNumber));
      const replacement = Number.isFinite(numeric)
        ? formatLikeTemplate(token.raw, numeric)
        : cleanedNumber;

      parsed.pathname = `${parsed.pathname.slice(0, token.start)}${replacement}${parsed.pathname.slice(token.end)}`;
      return parsed.toString();
    }

    for (const [key, value] of parsed.searchParams.entries()) {
      if (!/(chapter|chap|ch|episode|ep|c)/i.test(key)) {
        continue;
      }

      const numericMatch = value.match(/-?\d+(?:\.\d+)?/);
      if (numericMatch) {
        const numeric = Number.parseFloat(parseChapterNumber(cleanedNumber));
        const replacement = Number.isFinite(numeric)
          ? formatLikeTemplate(numericMatch[0] ?? "", numeric)
          : cleanedNumber;
        parsed.searchParams.set(key, value.replace(numericMatch[0] ?? "", replacement));
        return parsed.toString();
      }

      parsed.searchParams.set(key, cleanedNumber);
      return parsed.toString();
    }
  } catch {
    return null;
  }

  return null;
}

function renderChapterUrlFromPattern(
  chapterUrl: string,
  chapterUrlPattern: string | null | undefined,
  chapterNumber: string
): string | null {
  const cleanedNumber = cleanText(chapterNumber);
  const pattern = cleanText(chapterUrlPattern);
  if (!cleanedNumber || !pattern || !pattern.includes("{chapter}")) {
    return null;
  }

  const renderedPattern = pattern.replaceAll("{chapter}", encodeURIComponent(cleanedNumber));

  try {
    return new URL(renderedPattern, chapterUrl).toString();
  } catch {
    return null;
  }
}

function getClientId(): string | null {
  const value = cleanText(process.env.MAL_CLIENT_ID ?? process.env.MYANIMELIST_CLIENT_ID);
  return value || null;
}

async function fetchJson<T>(url: string): Promise<T | null> {
  const clientId = getClientId();
  if (!clientId) {
    return null;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        "X-MAL-CLIENT-ID": clientId
      }
    });

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function resolveMALChapterCount(seriesTitle: string): Promise<number | null> {
  const cacheKey = normalizeText(seriesTitle);
  const cached = chapterCountCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  if (chapterCountMissCache.get(cacheKey) !== undefined) {
    return null;
  }

  if (!cacheKey) {
    chapterCountMissCache.set(cacheKey, true);
    return null;
  }

  const scoredCandidates: ScoredCandidate[] = [];

  for (const query of buildSearchQueries(seriesTitle)) {
    const queryUrl = new URL(`${MAL_BASE_URL}/manga`);
    queryUrl.searchParams.set("q", query);
    queryUrl.searchParams.set("limit", String(SEARCH_LIMIT));
    queryUrl.searchParams.set("fields", "id,title,alternative_titles,num_chapters");

    const payload = await fetchJson<MalSearchResponse>(queryUrl.toString());
    const candidates = (payload?.data ?? [])
      .map((candidate) => normalizeSearchResult(candidate))
      .filter((candidate): candidate is MalSearchResult => candidate !== null);

    if (candidates.length === 0) {
      continue;
    }

    for (const manga of candidates) {
      scoredCandidates.push({
        manga,
        score: scoreCandidate(manga, seriesTitle)
      });
    }
  }

  if (scoredCandidates.length === 0) {
    chapterCountMissCache.set(cacheKey, true);
    return null;
  }

  const dedupedById = new Map<number, ScoredCandidate>();
  for (const candidate of scoredCandidates) {
    const existing = dedupedById.get(candidate.manga.id);
    if (!existing || candidate.score < existing.score) {
      dedupedById.set(candidate.manga.id, candidate);
    }
  }

  const ranked = Array.from(dedupedById.values())
    .filter((candidate) => Number.isFinite(candidate.score) && candidate.score <= 2.5)
    .sort((a, b) => {
      if (a.score !== b.score) {
        return a.score - b.score;
      }

      const aCount = a.manga.num_chapters ?? 0;
      const bCount = b.manga.num_chapters ?? 0;
      return bCount - aCount;
    });

  for (const candidate of ranked) {
    const chosen = candidate.manga;

    if (typeof chosen.id !== "number" || !Number.isFinite(chosen.id)) {
      continue;
    }

    const detailsCountRaw = chosen.num_chapters ?? null;
    const detailsCount =
      typeof detailsCountRaw === "number"
        ? detailsCountRaw
        : Number.parseInt(String(detailsCountRaw), 10);

    if (Number.isFinite(detailsCount) && detailsCount > 0) {
      chapterCountCache.set(cacheKey, detailsCount);
      return detailsCount;
    }

    const detailsUrl = new URL(`${MAL_BASE_URL}/manga/${chosen.id}`);
    detailsUrl.searchParams.set("fields", "id,title,alternative_titles,num_chapters");
    const detailsPayload = await fetchJson<MalDetailsResponse>(detailsUrl.toString());
    const details = normalizeDetailsResult(detailsPayload);
    const resolvedCountRaw = details?.num_chapters ?? null;
    const resolvedCount =
      typeof resolvedCountRaw === "number" ? resolvedCountRaw : Number.parseInt(String(resolvedCountRaw), 10);

    if (Number.isFinite(resolvedCount) && resolvedCount > 0) {
      chapterCountCache.set(cacheKey, resolvedCount);
      return resolvedCount;
    }
  }

  chapterCountMissCache.set(cacheKey, true);
  return null;
}

function parseExistingChapterNumber(value: string): number | null {
  const cleaned = cleanText(value);
  const match = cleaned.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildChapterPlaceholders(options: {
  chapterUrl: string;
  chapterUrlPattern?: string | null;
  existingChapters: ChapterListItem[];
  maxChapterNumber: number;
}): ChapterListItem[] {
  const maxChapter = Math.floor(options.maxChapterNumber);
  if (!Number.isFinite(maxChapter) || maxChapter <= 0) {
    return [];
  }

  const existingNumbers = new Set<number>();
  for (const chapter of options.existingChapters) {
    const parsed = parseExistingChapterNumber(chapter.number);
    if (parsed !== null) {
      existingNumbers.add(parsed);
    }
  }

  const placeholders: ChapterListItem[] = [];
  for (let chapterNumber = 1; chapterNumber <= maxChapter; chapterNumber += 1) {
    if (existingNumbers.has(chapterNumber)) {
      continue;
    }

    const renderedUrl =
      renderChapterUrlFromPattern(options.chapterUrl, options.chapterUrlPattern, String(chapterNumber)) ??
      renderChapterUrlFromTemplate(options.chapterUrl, String(chapterNumber));
    if (!renderedUrl) {
      continue;
    }

    placeholders.push({
      number: String(chapterNumber),
      title: null,
      url: renderedUrl
    });
  }

  return placeholders;
}

export async function buildMALChapterPlaceholders(options: {
  seriesTitle: string;
  chapterUrl: string;
  chapterUrlPattern?: string | null;
  existingChapters: ChapterListItem[];
}): Promise<ChapterListItem[]> {
  const chapterCount = await resolveMALChapterCount(options.seriesTitle);
  if (!chapterCount || chapterCount <= 0) {
    return [];
  }

  const highestExisting = options.existingChapters.reduce((maximum, chapter) => {
    const parsed = parseExistingChapterNumber(chapter.number);
    return parsed !== null ? Math.max(maximum, parsed) : maximum;
  }, 0);

  const maxChapterNumber = Math.max(highestExisting, chapterCount);
  return buildChapterPlaceholders({
    chapterUrl: options.chapterUrl,
    chapterUrlPattern: options.chapterUrlPattern,
    existingChapters: options.existingChapters,
    maxChapterNumber
  });
}