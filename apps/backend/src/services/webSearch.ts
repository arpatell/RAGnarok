import * as cheerio from "cheerio";
import { LRUCache } from "lru-cache";
import { resolveAdapterForUrl } from "../adapters/index.js";
import { sortChapterList } from "./chapterSort.js";
import { fetchChapterHtml } from "./fetchProxy.js";

const MAX_CANDIDATE_URLS = 12;
const MAX_RESULTS = 6;
const SEARCH_CONCURRENCY = 3;
const SEARCH_DEADLINE_MS = 38_000;

const BROWSER_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

// Domains that never host free, directly-readable manga/manhwa chapter pages
const SKIP_DOMAINS = new Set([
  "wikipedia.org",
  "reddit.com",
  "myanimelist.net",
  "anilist.co",
  "fandom.com",
  "wikia.com",
  "amazon.com",
  "amazon.co.jp",
  "crunchyroll.com",
  "viz.com",
  "shonenjump.com",
  "youtube.com",
  "youtu.be",
  "twitter.com",
  "x.com",
  "instagram.com",
  "facebook.com",
  "tiktok.com",
  "goodreads.com",
  "imdb.com",
  "cbr.com",
  "screenrant.com",
  "sportskeeda.com",
  "epicstream.com",
  "fictionhorizon.com",
  "ranker.com",
  "tvtropes.org",
  "anime-planet.com",
  "kitsu.io",
  "livechart.me",
  "mangaupdates.com",
  "novelupdates.com",
  // Document / paywall hosting — not free readers
  "scribd.com",
  "patreon.com",
  "kofi.com",
  // Article / "where to read" blogs — not reading sites
  "mangashed.com",
  "otakusnotes.com",
  "secondmin.co.kr",
  "cbr.com",
  "gamerant.com",
  "9anime.to",
  "9anime.id"
]);


export interface MangaSearchResult {
  seriesTitle: string;
  coverUrl: string | null;
  firstChapterUrl: string;
  chapterCount: number;
  genres: string[];
  adapter: string;
  sourceDomain: string;
}

export interface ReadNowResolution {
  requestedTitle: string;
  matchedTitle: string;
  seriesUrl: string;
  chapterUrl: string;
  sourceDomain: string;
}

interface SourceReadNowMatch {
  queryTitle: string;
  candidate: SourceSeriesCandidate;
  score: number;
}

const WEEBCENTRAL_BASE_URL = "https://weebcentral.com";
const WEEBCENTRAL_DOMAIN = "weebcentral.com";
const WEEBCENTRAL_SEARCH_TIMEOUT_MS = Math.max(1_000, Number.parseInt(process.env.WEEBCENTRAL_SEARCH_TIMEOUT_MS ?? "2500", 10));
const WEEBCENTRAL_SEARCH_LIMIT = 24;
const MANHWAZONE_BASE_URL = "https://manhwazone.com";
const MANHWAZONE_DOMAIN = "manhwazone.com";
const MANHWAZONE_SEARCH_TIMEOUT_MS = Math.max(1_000, Number.parseInt(process.env.MANHWAZONE_SEARCH_TIMEOUT_MS ?? "2500", 10));
const MANHWAZONE_SEARCH_LIMIT = 24;
const MIN_READ_NOW_TITLE_SCORE = 0.58;
const READ_NOW_TIMEOUT_MS = Math.max(1_500, Number.parseInt(process.env.READ_NOW_TIMEOUT_MS ?? "4000", 10));
const READ_NOW_SERIES_RESOLVE_TIMEOUT_MS = Math.max(
  800,
  Number.parseInt(process.env.READ_NOW_SERIES_RESOLVE_TIMEOUT_MS ?? "1500", 10)
);
const READ_NOW_TITLE_VARIANT_LIMIT = Math.max(1, Number.parseInt(process.env.READ_NOW_TITLE_VARIANT_LIMIT ?? "4", 10));
const READ_NOW_CACHE_TTL_MS = Math.max(0, Number.parseInt(process.env.READ_NOW_CACHE_TTL_MS ?? "600000", 10));

interface SourceSeriesCandidate {
  url: string;
  title: string;
  firstChapterUrl?: string;
}

export interface ReadNowResolveOptions {
  preferManhwa?: boolean;
}

const readNowCache = new LRUCache<string, ReadNowResolution>({
  max: 500,
  ttl: READ_NOW_CACHE_TTL_MS
});
const inFlightReadNowResolutions = new Map<string, Promise<ReadNowResolution | null>>();

function isLatinReadableTitle(value: string): boolean {
  const cleaned = cleanTitle(value);
  if (!cleaned || cleaned.length < 2) {
    return false;
  }

  if (/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0600-\u06ff]/.test(cleaned)) {
    return false;
  }

  return /[a-zA-Z]/.test(cleaned);
}

function normalizeSeriesSlug(slug: string): string {
  return slug
    .replace(/\.\d+$/, "")
    .replace(/[-_][a-z0-9]{4,10}$/i, "")
    .replace(/[-_]+/g, " ")
    .trim();
}

function normalizeTitle(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseChapterNumber(value: string | null | undefined): number | null {
  const match = (value ?? "").match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function isChapterOneLabel(value: string | null | undefined): boolean {
  return /\b(?:chapter|ch\.?|episode|ep\.?)\s*0*1(?:\.0+)?\b/i.test(value ?? "");
}

function chooseChapterOne(chapterList: Array<{ number: string; title: string | null; url: string; special?: boolean }>) {
  return (
    chapterList.find((chapter) => isChapterOneLabel(chapter.title) || isChapterOneLabel(chapter.number)) ??
    chapterList.find((chapter) => {
      const parsed = parseChapterNumber(chapter.number || chapter.title);
      return parsed === 1 && chapter.special !== true;
    }) ??
    chapterList.find((chapter) => chapter.special !== true) ??
    chapterList[0] ??
    null
  );
}

const TITLE_STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "at",
  "by",
  "for",
  "from",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "the",
  "to",
  "who",
  "with"
]);

function tokenizeTitle(value: string, includeStopwords = false): string[] {
  return normalizeTitle(value)
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token.length >= 2)
    .filter((token) => includeStopwords || !TITLE_STOPWORDS.has(token));
}

function seriesNameFromSeriesUrl(seriesUrl: string): string {
  try {
    const parsed = new URL(seriesUrl);
    const segments = parsed.pathname.split("/").filter(Boolean);
    const slug = segments[1] ?? segments[0] ?? "";
    if (!slug) {
      return "";
    }

    return normalizeSeriesSlug(slug);
  } catch {
    return "";
  }
}

function titleSimilarityScore(queryTitle: string, candidateTitle: string): number {
  const queryNorm = normalizeTitle(queryTitle);
  const candidateNorm = normalizeTitle(candidateTitle);
  if (!queryNorm || !candidateNorm) {
    return 0;
  }
  if (queryNorm === candidateNorm) {
    return 1;
  }

  const queryTokens = tokenizeTitle(queryTitle);
  const candidateTokens = tokenizeTitle(candidateTitle);
  if (queryTokens.length === 0 || candidateTokens.length === 0) {
    return 0;
  }

  const querySet = new Set(queryTokens);
  const candidateSet = new Set(candidateTokens);
  let intersectionCount = 0;
  for (const token of querySet) {
    if (candidateSet.has(token)) {
      intersectionCount += 1;
    }
  }

  const queryCoverage = intersectionCount / querySet.size;
  const candidateCoverage = intersectionCount / candidateSet.size;
  const dice = (2 * intersectionCount) / (querySet.size + candidateSet.size);
  let score = Math.max(dice, (queryCoverage * 0.7) + (candidateCoverage * 0.3));

  if (candidateNorm.includes(queryNorm) || queryNorm.includes(candidateNorm)) {
    score = Math.max(score, 0.82);
  }

  // Loose title matching should tolerate subtitles and scan-site suffixes, but not unrelated first results.
  if (queryCoverage === 1 && querySet.size >= 2) {
    score = Math.max(score, 0.74);
  }

  const queryPhrase = queryTokens.join(" ");
  const candidatePhrase = candidateTokens.join(" ");
  if (queryPhrase.length >= 8 && candidatePhrase.includes(queryPhrase)) {
    score = 1;
  } else if (candidatePhrase.length >= 8 && queryPhrase.includes(candidatePhrase)) {
    score = Math.max(score, 0.86);
  }

  if (intersectionCount >= 3 && querySet.size >= 3) {
    score = Math.max(score, 0.68);
  }

  if (intersectionCount >= 2 && querySet.size >= 3 && candidateSet.size >= 2) {
    score = Math.max(score, 0.62);
  }

  return Math.min(1, score);
}

function isWeebCentralHost(url: string): boolean {
  const domain = extractDomain(url);
  return domain === WEEBCENTRAL_DOMAIN || domain.endsWith(`.${WEEBCENTRAL_DOMAIN}`);
}

function isWeebCentralSeriesUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    if (!isWeebCentralHost(parsed.toString())) {
      return false;
    }

    const segments = parsed.pathname.split("/").filter(Boolean);
    return segments[0] === "series" && Boolean(segments[1]);
  } catch {
    return false;
  }
}

function isWeebCentralChapterUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    if (!isWeebCentralHost(parsed.toString())) {
      return false;
    }

    const segments = parsed.pathname.split("/").filter(Boolean);
    return segments[0] === "chapters" && Boolean(segments[1]);
  } catch {
    return false;
  }
}

function isManhwaZoneHost(url: string): boolean {
  const domain = extractDomain(url);
  return domain === MANHWAZONE_DOMAIN || domain.endsWith(`.${MANHWAZONE_DOMAIN}`);
}

function isManhwaZonePreviewUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return isManhwaZoneHost(parsed.toString()) && parsed.pathname.startsWith("/preview/");
  } catch {
    return false;
  }
}

function isManhwaZoneSeriesUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    if (!isManhwaZoneHost(parsed.toString()) || isManhwaZonePreviewUrl(parsed.toString())) {
      return false;
    }

    const segments = parsed.pathname.split("/").filter(Boolean);
    const first = segments[0]?.toLowerCase() ?? "";
    if (segments.length === 0 || first === "search") {
      return false;
    }

    return ![
      "account",
      "auth",
      "cdn-cgi",
      "contact",
      "dmca",
      "login",
      "privacy",
      "register",
      "terms"
    ].includes(first);
  } catch {
    return false;
  }
}

function dedupeUrls(urls: string[]): string[] {
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const url of urls) {
    const trimmed = url.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    deduped.push(trimmed);
  }
  return deduped;
}

function cleanCandidateText(value: string | undefined | null): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function extractAnchorCandidateTitle(anchor: cheerio.Cheerio<any>): string {
  const textLines = (anchor.text() ?? "")
    .split(/\r?\n/)
    .map((line) => cleanCandidateText(line))
    .filter(Boolean);

  const values = [
    anchor.attr("title"),
    anchor.attr("aria-label"),
    anchor.find("img").first().attr("alt"),
    anchor.find("h1,h2,h3,h4,.truncate,.line-clamp-1,.line-clamp-2,[class*='title']").first().text(),
    ...textLines,
    anchor.text()
  ];

  for (const value of values) {
    const cleaned = cleanCandidateText(value);
    if (cleaned && cleaned.length <= 180) {
      return cleanTitle(cleaned);
    }
  }

  return "";
}

function buildWeebCentralSearchUrl(title: string): string {
  const query = new URLSearchParams({ text: title }).toString().replace(/^text=/, "");
  return `${WEEBCENTRAL_BASE_URL}/search/data?text=${query}&sort=Best+Match&order=Descending&official=Any&anime=Any&adult=Any&display_mode=Full+Display`;
}

function extractWeebCentralCandidateTitle(anchor: cheerio.Cheerio<any>, seriesUrl: string): string {
  const titleCandidates = [
    anchor.attr("title"),
    anchor.attr("aria-label"),
    anchor.find("img").first().attr("alt"),
    anchor.find("h1,h2,h3,h4,[class*='title'],[class*='font']").first().text(),
    anchor.text()
  ];

  const container = anchor.closest("article,li,section,div").first();
  if (container.length > 0) {
    titleCandidates.push(
      container.find("h1,h2,h3,h4,[class*='title'],[class*='font']").first().text(),
      container.find("img").first().attr("alt"),
      container.text()
    );
  }

  for (const value of titleCandidates) {
    const cleaned = cleanTitle(cleanCandidateText(value));
    if (cleaned && cleaned.length <= 180 && !/^(official|subscribe|show details)$/i.test(cleaned)) {
      return cleaned;
    }
  }

  return seriesNameFromSeriesUrl(seriesUrl);
}

async function fetchWeebCentralSearchSeriesCandidates(title: string): Promise<SourceSeriesCandidate[]> {
  const searchUrl = buildWeebCentralSearchUrl(title);

  try {
    const response = await fetch(searchUrl, {
      headers: {
        "User-Agent": BROWSER_UA,
        Accept: "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
      },
      signal: AbortSignal.timeout(WEEBCENTRAL_SEARCH_TIMEOUT_MS)
    });

    // eslint-disable-next-line no-console
    console.log(`[read-now] weebcentral status=${response.status} ok=${response.ok} query="${title.substring(0, 60)}"`);

    if (!response.ok) {
      return [];
    }

    const html = await response.text();
    const $ = cheerio.load(html);
    const candidates = new Map<string, SourceSeriesCandidate>();

    $("a[href]").each((_, el) => {
      const anchor = $(el);
      const resolved = resolveUrl(WEEBCENTRAL_BASE_URL, anchor.attr("href") ?? "");
      if (!isWeebCentralSeriesUrl(resolved)) {
        return;
      }

      const candidateTitle = extractWeebCentralCandidateTitle(anchor, resolved);
      if (!candidateTitle) {
        return;
      }

      const existing = candidates.get(resolved);
      if (!existing || candidateTitle.length > existing.title.length) {
        candidates.set(resolved, { url: resolved, title: candidateTitle });
      }
    });

    const sorted = Array.from(candidates.values())
      .sort((a, b) => {
        const scoreA = scoreSourceCandidate(title, a);
        const scoreB = scoreSourceCandidate(title, b);
        return scoreB - scoreA;
      })
      .slice(0, WEEBCENTRAL_SEARCH_LIMIT);

    // eslint-disable-next-line no-console
    console.log(`[read-now] weebcentral candidates=${sorted.length} query="${title.substring(0, 60)}"`);

    return sorted;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(`[read-now] weebcentral fetch error query="${title.substring(0, 60)}": ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

function isGenericManhwaZoneLabel(value: string): boolean {
  const normalized = normalizeTitle(value);
  return normalized === "manhwa" || normalized === "manhua" || normalized === "manga" || normalized === "read";
}

function upsertManhwaZoneCandidate(
  candidates: Map<string, SourceSeriesCandidate>,
  seriesUrl: string,
  title: string,
  firstChapterUrl?: string
): void {
  const candidateTitle = cleanTitle(title) || seriesNameFromSeriesUrl(seriesUrl);
  if (!candidateTitle) {
    return;
  }

  const existing = candidates.get(seriesUrl);
  candidates.set(seriesUrl, {
    url: seriesUrl,
    title: existing && existing.title.length > candidateTitle.length ? existing.title : candidateTitle,
    firstChapterUrl: firstChapterUrl ?? existing?.firstChapterUrl
  });
}

async function fetchManhwaZoneSearchSeriesCandidates(title: string): Promise<SourceSeriesCandidate[]> {
  const searchUrl = `${MANHWAZONE_BASE_URL}/search?keyword=${encodeURIComponent(title)}`;

  try {
    const response = await fetch(searchUrl, {
      headers: {
        "User-Agent": BROWSER_UA,
        Accept: "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
      },
      signal: AbortSignal.timeout(MANHWAZONE_SEARCH_TIMEOUT_MS)
    });

    // eslint-disable-next-line no-console
    console.log(`[read-now] manhwazone status=${response.status} ok=${response.ok} query="${title.substring(0, 60)}"`);

    if (!response.ok) {
      return [];
    }

    const html = await response.text();
    const $ = cheerio.load(html);
    const candidates = new Map<string, SourceSeriesCandidate>();

    $("a[href]").each((_, el) => {
      const anchor = $(el);
      const href = anchor.attr("href") ?? "";
      const resolved = resolveUrl(MANHWAZONE_BASE_URL, href);
      if (!isManhwaZonePreviewUrl(resolved)) {
        return;
      }

      const container = anchor.parent().parent();
      container.find("a[href*='/series/']").each((__, seriesEl) => {
        const seriesAnchor = $(seriesEl);
        const seriesHref = seriesAnchor.attr("href") ?? "";
        const seriesUrl = resolveUrl(MANHWAZONE_BASE_URL, seriesHref);
        if (!isManhwaZoneSeriesUrl(seriesUrl)) {
          return;
        }

        const anchorTitle = extractAnchorCandidateTitle(seriesAnchor);
        const seriesTitle = isGenericManhwaZoneLabel(anchorTitle)
          ? seriesNameFromSeriesUrl(seriesUrl)
          : anchorTitle || seriesNameFromSeriesUrl(seriesUrl);
        upsertManhwaZoneCandidate(candidates, seriesUrl, seriesTitle, resolved);
      });
    });

    $("a[href]").each((_, el) => {
      const anchor = $(el);
      const href = anchor.attr("href") ?? "";
      const resolved = resolveUrl(MANHWAZONE_BASE_URL, href);
      if (!isManhwaZoneSeriesUrl(resolved)) {
        return;
      }

      const anchorTitle = extractAnchorCandidateTitle(anchor);
      const seriesTitle = isGenericManhwaZoneLabel(anchorTitle)
        ? seriesNameFromSeriesUrl(resolved)
        : anchorTitle || seriesNameFromSeriesUrl(resolved);
      upsertManhwaZoneCandidate(candidates, resolved, seriesTitle);
    });

    const sorted = Array.from(candidates.values())
      .sort((a, b) => {
        const scoreA = Math.max(
          titleSimilarityScore(title, a.title),
          titleSimilarityScore(title, seriesNameFromSeriesUrl(a.url))
        );
        const scoreB = Math.max(
          titleSimilarityScore(title, b.title),
          titleSimilarityScore(title, seriesNameFromSeriesUrl(b.url))
        );
        return scoreB - scoreA;
      })
      .slice(0, MANHWAZONE_SEARCH_LIMIT);

    // eslint-disable-next-line no-console
    console.log(`[read-now] manhwazone candidates=${sorted.length} query="${title.substring(0, 60)}"`);

    return sorted;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(`[read-now] manhwazone fetch error query="${title.substring(0, 60)}": ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

type DirectSourceChapter = { number: string; title: string | null; url: string; special?: boolean };

function cleanChapterLabel(value: string | null | undefined): string {
  const cleaned = cleanCandidateText(value);
  const chapterMatch = cleaned.match(/\b(?:chapter|ch\.?|episode|ep\.?)\s*-?\d+(?:\.\d+)?\b/i);
  return chapterMatch?.[0]?.trim() || cleaned;
}

function inferChapterLabel(title: string, url: string): string {
  const cleanedTitle = cleanChapterLabel(title);
  if (cleanedTitle) {
    return cleanedTitle;
  }

  const urlMatch = url.match(/(?:chapter|\/c|episode|ep)[-_/]?(\d+(?:\.\d+)?)/i);
  if (urlMatch?.[1]) {
    return urlMatch[1];
  }

  return url.split("/").filter(Boolean).at(-1)?.slice(0, 64) || "Unknown";
}

function getWeebCentralChapterGrid($: cheerio.CheerioAPI): cheerio.Cheerio<any> {
  return $("div.grid.grid-cols-2.gap-3.justify-items-center").filter((_, node) =>
    ($(node).attr("class") ?? "").split(/\s+/).includes("lg:grid-cols-3")
  );
}

function getWeebCentralButtonChapterUrl(button: cheerio.Cheerio<any>, baseUrl: string): string {
  for (const attr of ["href", "data-url", "data-href", "hx-get", "formaction"]) {
    const value = button.attr(attr);
    if (!value) {
      continue;
    }

    const resolved = resolveUrl(baseUrl, value);
    if (isWeebCentralChapterUrl(resolved)) {
      return resolved;
    }
  }

  return baseUrl;
}

function parseWeebCentralSeriesChapterList($: cheerio.CheerioAPI, baseUrl: string): DirectSourceChapter[] {
  const unique = new Map<string, DirectSourceChapter>();
  const chapterGrid = getWeebCentralChapterGrid($);

  chapterGrid.find("a[href]").each((_, node) => {
    const anchor = $(node);
    const url = resolveUrl(baseUrl, anchor.attr("href") ?? "");
    if (!url) {
      return;
    }

    const title = cleanChapterLabel(anchor.text());
    unique.set(url, {
      number: inferChapterLabel(title, url),
      title: title || null,
      url
    });
  });

  chapterGrid.find("button#selected_chapter").each((_, node) => {
    const button = $(node);
    const title = cleanChapterLabel(button.text());
    if (!title) {
      return;
    }

    const url = getWeebCentralButtonChapterUrl(button, baseUrl);
    unique.set(url, {
      number: inferChapterLabel(title, url),
      title,
      url
    });
  });

  $("#chapter-list a[href*='/chapters/']").each((_, node) => {
    const anchor = $(node);
    const url = resolveUrl(baseUrl, anchor.attr("href") ?? "");
    if (!isWeebCentralChapterUrl(url)) {
      return;
    }

    const title = cleanChapterLabel(anchor.text());
    unique.set(url, {
      number: inferChapterLabel(title, url),
      title: title || null,
      url
    });
  });

  return Array.from(unique.values());
}

function parseManhwaZoneSeriesChapterList($: cheerio.CheerioAPI, baseUrl: string): DirectSourceChapter[] {
  const unique = new Map<string, DirectSourceChapter>();
  const chapterList = $("ol[role='list'][aria-label='List of chapters'].space-y-3");

  chapterList.find("li[role='listitem']").each((_, node) => {
    const anchor = $(node).find("a[href]").first();
    const href = anchor.attr("href") ?? "";
    if (!href) {
      return;
    }

    const url = resolveUrl(baseUrl, href);
    const title = cleanChapterLabel(anchor.find("span.truncate.flex-1").first().text());
    unique.set(url, {
      number: inferChapterLabel(title, url),
      title: title || null,
      url
    });
  });

  return Array.from(unique.values());
}

async function resolveFirstWeebCentralChapterFromSeriesUrl(
  seriesUrl: string,
  fallbackTitle: string
): Promise<{ chapterUrl: string; matchedTitle: string } | null> {
  const fetched = await fetchChapterHtml(seriesUrl);
  if (fetched.status >= 400 || !fetched.html) {
    return null;
  }

  const $ = cheerio.load(fetched.html);
  const chapterList = sortChapterList(parseWeebCentralSeriesChapterList($, fetched.finalUrl));
  const firstChapter = chooseChapterOne(chapterList);
  if (!firstChapter?.url) {
    return null;
  }

  const rawTitle =
    getMetaAttr($, "meta[property='og:title']", "content") ||
    getFirstText($, TITLE_SELECTORS) ||
    fallbackTitle ||
    seriesNameFromSeriesUrl(seriesUrl);
  const matchedTitle = cleanTitle(rawTitle) || fallbackTitle || seriesNameFromSeriesUrl(seriesUrl);
  return {
    chapterUrl: firstChapter.url,
    matchedTitle
  };
}

async function resolveFirstManhwaZoneChapterFromSeriesUrl(
  seriesUrl: string,
  fallbackTitle: string
): Promise<{ chapterUrl: string; matchedTitle: string } | null> {
  const fetched = await fetchChapterHtml(seriesUrl);
  if (fetched.status >= 400 || !fetched.html) {
    return null;
  }

  const $ = cheerio.load(fetched.html);
  const chapterList = parseManhwaZoneSeriesChapterList($, fetched.finalUrl);

  const sortedChapters = sortChapterList(chapterList);
  const firstChapter = chooseChapterOne(sortedChapters);
  if (!firstChapter?.url) {
    return null;
  }

  const rawTitle =
    getMetaAttr($, "meta[property='og:title']", "content") ||
    getFirstText($, TITLE_SELECTORS) ||
    fallbackTitle;
  const matchedTitle = cleanTitle(rawTitle) || fallbackTitle;

  return {
    chapterUrl: firstChapter.url,
    matchedTitle
  };
}

function buildReadNowTitleCandidates(titles: string[] | string): string[] {
  const rawTitles = Array.isArray(titles) ? titles : [titles];
  const candidates: string[] = [];
  const seen = new Set<string>();

  for (const raw of rawTitles) {
    const cleaned = cleanTitle(raw);
    const variants = [
      cleaned,
      cleaned.replace(/\s*\([^)]*\)\s*/g, " "),
      cleaned.split(/\s*[-–—:|]\s*/)[0] ?? ""
    ];

    for (const variant of variants) {
      const value = cleanTitle(variant);
      if (!isLatinReadableTitle(value)) {
        continue;
      }
      const key = normalizeTitle(value);
      if (!value || value.length < 2 || seen.has(key)) {
        continue;
      }
      seen.add(key);
      candidates.push(value);
    }
  }

  return candidates.slice(0, 12);
}

function scoreSourceCandidate(queryTitle: string, candidate: SourceSeriesCandidate): number {
  const slugTitle = seriesNameFromSeriesUrl(candidate.url);
  return Math.max(
    titleSimilarityScore(queryTitle, candidate.title),
    titleSimilarityScore(queryTitle, slugTitle),
    slugTitle ? titleSimilarityScore(queryTitle, slugTitle.replace(/\s+it$/, "").trim()) : 0
  );
}

function chooseBestSourceMatch(
  queryTitles: string[],
  candidates: SourceSeriesCandidate[],
  minScore = MIN_READ_NOW_TITLE_SCORE
): SourceReadNowMatch | null {
  const scored = queryTitles.flatMap((queryTitle) =>
    candidates.map((candidate) => ({
      queryTitle,
      candidate,
      score: scoreSourceCandidate(queryTitle, candidate)
    }))
  ).sort((a, b) => b.score - a.score);

  const best = scored[0];
  return best && best.score >= minScore ? best : null;
}

async function withTimeoutValue<T>(promise: Promise<T>, timeoutMs: number, fallback: T): Promise<T> {
  let timer: NodeJS.Timeout | null = null;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((resolve) => {
        timer = setTimeout(() => resolve(fallback), timeoutMs);
      })
    ]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

function firstNonNull<T>(promises: Array<Promise<T | null>>): Promise<T | null> {
  if (promises.length === 0) {
    return Promise.resolve(null);
  }

  return new Promise((resolve) => {
    let remaining = promises.length;
    let settled = false;

    for (const promise of promises) {
      promise
        .then((value) => {
          if (settled) {
            return;
          }
          if (value) {
            settled = true;
            resolve(value);
            return;
          }
          remaining -= 1;
          if (remaining === 0) {
            settled = true;
            resolve(null);
          }
        })
        .catch(() => {
          if (settled) {
            return;
          }
          remaining -= 1;
          if (remaining === 0) {
            settled = true;
            resolve(null);
          }
        });
    }
  });
}

async function resolveWeebCentralReadNowFromTitles(queryTitles: string[]): Promise<ReadNowResolution | null> {
  const limitedTitles = queryTitles.slice(0, READ_NOW_TITLE_VARIANT_LIMIT);
  const candidateLists = await Promise.all(limitedTitles.map((queryTitle) => fetchWeebCentralSearchSeriesCandidates(queryTitle)));
  const candidates = candidateLists.flat();

  const selectedMatch = chooseBestSourceMatch(limitedTitles, candidates);
  if (!selectedMatch) {
    // eslint-disable-next-line no-console
    console.log(`[read-now] weebcentral no loose match candidates=${candidates.length}`);
    return null;
  }

  // eslint-disable-next-line no-console
  console.log(`[read-now] weebcentral selected title="${selectedMatch.candidate.title}" url=${selectedMatch.candidate.url} score=${selectedMatch.score.toFixed(2)}`);

  const resolved = await withTimeoutValue(
    resolveFirstWeebCentralChapterFromSeriesUrl(
      selectedMatch.candidate.url,
      selectedMatch.candidate.title
    ),
    READ_NOW_SERIES_RESOLVE_TIMEOUT_MS,
    null
  );
  if (!resolved) {
    return null;
  }

  return {
    requestedTitle: queryTitles[0] ?? selectedMatch.queryTitle,
    matchedTitle: resolved.matchedTitle,
    seriesUrl: selectedMatch.candidate.url,
    chapterUrl: resolved.chapterUrl,
    sourceDomain: extractDomain(resolved.chapterUrl)
  };
}

async function resolveManhwaZoneReadNowFromTitles(queryTitles: string[]): Promise<ReadNowResolution | null> {
  const limitedTitles = queryTitles.slice(0, READ_NOW_TITLE_VARIANT_LIMIT);
  const candidateLists = await Promise.all(limitedTitles.map((queryTitle) => fetchManhwaZoneSearchSeriesCandidates(queryTitle)));
  const candidates = candidateLists.flat();

  const selectedMatch = chooseBestSourceMatch(limitedTitles, candidates);
  if (!selectedMatch) {
    // eslint-disable-next-line no-console
    console.log(`[read-now] manhwazone no loose match candidates=${candidates.length}`);
    return null;
  }

  // eslint-disable-next-line no-console
  console.log(`[read-now] manhwazone selected title="${selectedMatch.candidate.title}" url=${selectedMatch.candidate.url} score=${selectedMatch.score.toFixed(2)}`);

  let resolved: { chapterUrl: string; matchedTitle: string } | null = null;
  if (selectedMatch.candidate.firstChapterUrl) {
    // eslint-disable-next-line no-console
    console.log(
      `[read-now] manhwazone using search preview chapter url=${selectedMatch.candidate.firstChapterUrl}`
    );
    resolved = {
      chapterUrl: selectedMatch.candidate.firstChapterUrl,
      matchedTitle: selectedMatch.candidate.title
    };
  }
  if (!resolved?.chapterUrl) {
    resolved = await withTimeoutValue(
      resolveFirstManhwaZoneChapterFromSeriesUrl(
        selectedMatch.candidate.url,
        selectedMatch.candidate.title
      ),
      READ_NOW_SERIES_RESOLVE_TIMEOUT_MS,
      null
    );
  }
  if (!resolved) {
    return null;
  }

  return {
    requestedTitle: queryTitles[0] ?? selectedMatch.queryTitle,
    matchedTitle: resolved.matchedTitle,
    seriesUrl: selectedMatch.candidate.url,
    chapterUrl: resolved.chapterUrl,
    sourceDomain: extractDomain(resolved.chapterUrl)
  };
}

export async function resolveDirectReadNow(
  titles: string[] | string,
  options: ReadNowResolveOptions = {}
): Promise<ReadNowResolution | null> {
  const queryTitles = buildReadNowTitleCandidates(titles);
  if (queryTitles.length === 0) {
    return null;
  }

  const cacheKey = `${options.preferManhwa ? "manhwa" : "any"}:${queryTitles
    .slice(0, READ_NOW_TITLE_VARIANT_LIMIT)
    .map((title) => normalizeTitle(title))
    .join("|")}`;
  const cached = readNowCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const inFlight = inFlightReadNowResolutions.get(cacheKey);
  if (inFlight) {
    return inFlight;
  }

  const sourceTasks = options.preferManhwa
    ? [
        resolveManhwaZoneReadNowFromTitles(queryTitles),
        resolveWeebCentralReadNowFromTitles(queryTitles)
      ]
    : [
        resolveWeebCentralReadNowFromTitles(queryTitles),
        resolveManhwaZoneReadNowFromTitles(queryTitles)
      ];

  const resolution = withTimeoutValue(firstNonNull(sourceTasks), READ_NOW_TIMEOUT_MS, null);
  inFlightReadNowResolutions.set(cacheKey, resolution);

  try {
    const result = await resolution;
    if (result) {
      readNowCache.set(cacheKey, result);
    }
    return result;
  } finally {
    inFlightReadNowResolutions.delete(cacheKey);
  }
}

function extractDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

// URLs that look like series/list pages rather than chapter pages — skip during search
// to avoid the expensive series-page crawl (fetch + AI + MAL) that would run in background
// even after a Promise.race timeout, starving the event loop.
// A URL is a strong chapter candidate if its path includes a chapter/episode keyword
const CHAPTER_PATH_RE = /\/(chapter[s]?|episode[s]?|ep)[-_/]?\d/i;
// Weak chapter indicator: a hyphenated slug ending in a chapter number
const CHAPTER_SLUG_RE = /-chapter-\d|\/c\d{1,4}(?:[._-]|$)/i;
// Series/list pages are low-priority — only try them after all chapter URLs
const SERIES_PATH_RE = /\/(manga|manhwa|manhua|comic|series|titles?|title)\//i;

function looksLikeChapterUrlForSearch(url: string): boolean {
  try {
    const { pathname } = new URL(url);
    if (CHAPTER_PATH_RE.test(pathname)) return true;
    if (CHAPTER_SLUG_RE.test(pathname)) return true;
    if (SERIES_PATH_RE.test(pathname)) return false;
    return true;
  } catch {
    return false;
  }
}

interface LightIngestResult {
  title: string;
  coverUrl: string | null;
  genres: string[];
  chapterCount: number;
  sourceAdapter: string;
}

const TITLE_SELECTORS = [
  "meta[property='og:title']",
  "meta[name='twitter:title']",
  "h1.manga-title",
  "h1.series-title",
  ".series-header h1",
  ".entry-title",
  "h1",
  "title"
];

const COVER_SELECTORS = [
  "meta[property='og:image']",
  "meta[name='twitter:image']",
  ".cover img",
  ".info-image img",
  ".summary_image img",
  "img[alt*='cover' i]",
  ".series-cover img"
];

const GENRE_SELECTORS = [
  "a[href*='/genre/']",
  "a[href*='/tag/']",
  ".genres a",
  ".genres-content a",
  ".series-genres a"
];

const CHAPTER_COUNT_SELECTORS = [
  ".chapter-count",
  ".total-chapter"
];

// Signals that a page is likely a chapter-reader rather than a series/index page
const CHAPTER_READER_SIGNALS = [
  "#imgs",
  "#reader",
  ".reading-content",
  ".chapter-content",
  ".container-chapter-reader",
  ".page-break img",
  ".wp-manga-chapter-img",
  "img[class*='chapter-img']",
  "img[id*='image']"
];

function getMetaAttr($: cheerio.CheerioAPI, selector: string, attr: string): string {
  const el = $(selector).first();
  return (el.attr(attr) ?? el.attr("content") ?? "").trim();
}

function getFirstText($: cheerio.CheerioAPI, selectors: string[]): string {
  for (const sel of selectors) {
    const el = $(sel).first();
    const text = (el.attr("content") ?? el.text()).trim();
    if (text) return text;
  }
  return "";
}

function getFirstAttr($: cheerio.CheerioAPI, selectors: string[], attr: string): string {
  for (const sel of selectors) {
    const el = $(sel).first();
    const val = (el.attr(attr) ?? "").trim();
    if (val) return val;
  }
  return "";
}

function resolveUrl(base: string, href: string): string {
  try {
    return new URL(href, base).toString();
  } catch {
    return href;
  }
}

function cleanTitle(raw: string): string {
  let t = raw.trim();

  t = t.replace(/\s+cover\s*$/i, "").trim();

  // Remove everything after | (site name separator, e.g. "One Piece | MangaDex")
  t = t.replace(/\s*\|.*$/, "").trim();

  // Remove common action-word prefixes
  t = t.replace(/^(?:read|where\s+to\s+read|free\s+reading|free\s+read|every\s+chapter\s+of|all\s+chapters?\s+of|how\s+to\s+read)\s+/i, "").trim();

  // Remove trailing "- [Word(s)] Scans" / "- [Word(s)] Scan" / "– Asura Scans" etc.
  t = t.replace(/\s*[-–—]\s*[\w\s]+scans?\s*$/i, "").trim();

  // Remove everything from a dash/em-dash if what follows looks like a descriptor or site name
  t = t.replace(/\s*[-–—]\s*(?:read|free|online|officially\s+licensed|ad\s+free|chapter\s*[\d.]+|vol\.?\s*\d+|manga|manhwa|manhua|comic|scan|scans|raw|official|licensed|era|velvet|asura).*$/i, "").trim();

  // Remove trailing " on [Site]" or " from [Site]"
  t = t.replace(/\s+(?:on|from|at|via)\s+\S+\s*$/i, "").trim();

  // Remove parenthesized or bracketed qualifiers like "(Volume)", "(Color)", "[OFFICIAL]"
  t = t.replace(/\s*[\[(](?:volume|vol\.?|color(?:ed)?|colou?red|official|licensed|english|raw|chapter\s*[\d.]+)[^\])]*[\])]/gi, "").trim();

  // Remove trailing chapter/volume numbers and reading descriptors
  t = t.replace(/\s*(?:chapter|ch\.?|episode|ep\.?|vol\.?)\s*[\d.]+.*$/i, "").trim();
  t = t.replace(/\s+(?:manga|manhwa|manhua|comic)\s*(?:online\s*(?:for\s*free)?)?$/i, "").trim();
  t = t.replace(/\s+online\s*(?:for\s*free)?$/i, "").trim();

  // Remove trailing punctuation (commas, colons, dashes)
  t = t.replace(/[,.:;!?\-–—]+$/, "").trim();

  return t;
}

async function lightIngest(url: string): Promise<{ chapterUrl: string; result: LightIngestResult } | null> {
  const fetched = await fetchChapterHtml(url);
  if (fetched.status >= 400 || !fetched.html) return null;

  const $ = cheerio.load(fetched.html);

  const pageHost = (() => {
    try { return new URL(fetched.finalUrl).hostname; } catch { return ""; }
  })();

  const isChapterPage = CHAPTER_READER_SIGNALS.some((sel) => $(sel).length > 0);
  // Count same-host chapter-like links to determine if this is a series listing
  const chapterLinkCount = $("a").filter((_, el) => {
    const href = $(el).attr("href") ?? "";
    if (!href) return false;
    try { if (new URL(resolveUrl(fetched.finalUrl, href)).hostname !== pageHost) return false; } catch { return false; }
    return /\/chapters?\//i.test(href) || /chapter[-_/]?\d/i.test(href) || /episode[-_/]?\d/i.test(href);
  }).length;
  const isSeriesPage = !isChapterPage && chapterLinkCount >= 2;

  const rawTitle =
    getMetaAttr($, "meta[property='og:title']", "content") ||
    getFirstText($, TITLE_SELECTORS);

  const title = cleanTitle(rawTitle);
  if (!title || title.length < 2) return null;

  const rawCover =
    getMetaAttr($, "meta[property='og:image']", "content") ||
    getFirstAttr($, COVER_SELECTORS, "src");
  const coverUrl = rawCover ? resolveUrl(fetched.finalUrl, rawCover) : null;

  const genres: string[] = [];
  for (const sel of GENRE_SELECTORS) {
    $(sel).each((_, el) => {
      const g = $(el).text().trim();
      if (g && !genres.includes(g)) genres.push(g);
    });
    if (genres.length > 0) break;
  }

  let chapterCount = 0;
  for (const sel of CHAPTER_COUNT_SELECTORS) {
    const text = $(sel).first().text().trim();
    const n = parseInt(text, 10);
    if (n > 0) { chapterCount = n; break; }
  }

  const chapterUrl = isSeriesPage
    ? (() => {
        // Find the earliest chapter link on a series page — same-host only
        const links: Array<{ num: number; url: string }> = [];
        $("a").each((_, el) => {
          const href = $(el).attr("href") ?? "";
          if (!href) return;
          const resolved = resolveUrl(fetched.finalUrl, href);
          try {
            const linkHost = new URL(resolved).hostname;
            if (linkHost !== pageHost) return;
          } catch { return; }

          // Try to extract a chapter number from the href:
          // 1. Explicit pattern: /chapter-1, /chapter/1, -c1
          const explicit = href.match(/(?:chapter|\/c|episode)[-_/]?(\d+(?:\.\d+)?)/i);
          if (explicit) {
            links.push({ num: parseFloat(explicit[1] ?? "9999"), url: resolved });
            return;
          }
          // 2. UUID-style: /chapters/UUID/00001 — last segment is chapter number
          const uuidChapter = href.match(/\/chapters?\/[^/]+\/(\d{1,6})(?:[?#]|$)/i);
          if (uuidChapter) {
            links.push({ num: parseFloat(uuidChapter[1] ?? "9999"), url: resolved });
            return;
          }
          // 3. Trailing numeric slug with chapter keyword in parent path
          if (/\/chapters?\//i.test(href)) {
            const lastSeg = href.split("/").filter(Boolean).pop() ?? "";
            const n = parseFloat(lastSeg);
            if (!Number.isNaN(n) && n < 10000) {
              links.push({ num: n, url: resolved });
            }
          }
        });
        if (links.length === 0) return null;
        links.sort((a, b) => a.num - b.num);
        return links[0]?.url ?? null;
      })()
    : fetched.finalUrl;

  if (!chapterUrl) return null;

  return {
    chapterUrl,
    result: {
      title,
      coverUrl,
      genres: genres.slice(0, 6),
      chapterCount,
      sourceAdapter: "search"
    }
  };
}

// URL path patterns that indicate blog/article/guide pages rather than reading pages
const ARTICLE_PATH_RE = /\/(blogs?|posts?|articles?|news|guide|review|wiki|about|faq|help)\//i;

function isSafeToIngest(url: string): boolean {
  const domain = extractDomain(url);
  if (!domain) return false;
  if (Array.from(SKIP_DOMAINS).some((skip) => domain === skip || domain.endsWith(`.${skip}`))) {
    return false;
  }
  // Reject blog/article paths even on otherwise legitimate domains
  try {
    const { pathname } = new URL(url);
    if (ARTICLE_PATH_RE.test(pathname)) return false;
  } catch { /* ignore */ }
  return true;
}

const PER_URL_TIMEOUT_MS = 18_000;

async function lightIngestWithTimeout(
  url: string
): Promise<{ chapterUrl: string; result: LightIngestResult } | null> {
  return Promise.race([
    lightIngest(url),
    new Promise<null>((resolve) => setTimeout(() => resolve(null), PER_URL_TIMEOUT_MS))
  ]);
}

// Titles that look like article headlines rather than manga series names
const ARTICLE_HEADLINE_RE = /^(?:best\s+|where\s+|how\s+|top\s+\d+|the\s+best\s+|official\s+|a\s+(?:fan|complete|beginner|full|quick|simple)\s+(?:'s\s+)?guide|guide\s+to|complete\s+guide)/i;

function isTitleRelevant(title: string, query: string): boolean {
  const titleLow = title.toLowerCase();
  const queryLow = query.toLowerCase();
  // Accept if the title contains the full query (most common case)
  if (titleLow.includes(queryLow)) return true;
  // Accept if title and query share at least half the significant query words
  const words = queryLow.split(/\s+/).filter((w) => w.length > 2);
  if (words.length === 0) return true;
  const matches = words.filter((w) => titleLow.includes(w)).length;
  return matches >= Math.ceil(words.length / 2);
}

async function buildDirectSearchResult(
  candidate: SourceSeriesCandidate,
  sourceAdapter: "weebcentral" | "manhwazone"
): Promise<MangaSearchResult | null> {
  const resolved =
    sourceAdapter === "weebcentral"
      ? await resolveFirstWeebCentralChapterFromSeriesUrl(candidate.url, candidate.title)
      : await resolveFirstManhwaZoneChapterFromSeriesUrl(candidate.url, candidate.title);

  if (!resolved) {
    return null;
  }

  let coverUrl: string | null = null;
  let genres: string[] = [];
  let chapterCount = 0;

  try {
    const fetched = await fetchChapterHtml(candidate.url);
    const adapter = resolveAdapterForUrl(candidate.url);
    if (fetched.status < 400 && fetched.html && adapter) {
      const parsed = adapter.parse(fetched.html, fetched.finalUrl);
      coverUrl = parsed.series.coverUrl ?? null;
      genres = parsed.series.genres ?? [];
      chapterCount = parsed.chapterList.length;
    }
  } catch {
    // Metadata is optional for search cards; the resolved chapter URL is required.
  }

  return {
    seriesTitle: resolved.matchedTitle || candidate.title,
    coverUrl,
    firstChapterUrl: resolved.chapterUrl,
    chapterCount,
    genres: genres.slice(0, 6),
    adapter: sourceAdapter,
    sourceDomain: extractDomain(resolved.chapterUrl)
  };
}

async function fetchStrictDirectSearchMatches(
  query: string,
  sourceAdapter: "weebcentral" | "manhwazone"
): Promise<SourceSeriesCandidate[]> {
  const queryTitles = buildReadNowTitleCandidates(query);
  const candidates: SourceSeriesCandidate[] = [];

  for (const queryTitle of queryTitles) {
    candidates.push(
      ...(sourceAdapter === "weebcentral"
        ? await fetchWeebCentralSearchSeriesCandidates(queryTitle)
        : await fetchManhwaZoneSearchSeriesCandidates(queryTitle))
    );
  }

  const seen = new Set<string>();
  return queryTitles
    .flatMap((queryTitle) =>
      candidates.map((candidate) => ({
        candidate,
        score: scoreSourceCandidate(queryTitle, candidate)
      }))
    )
    .filter((entry) => entry.score >= MIN_READ_NOW_TITLE_SCORE)
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.candidate)
    .filter((candidate) => {
      if (seen.has(candidate.url)) {
        return false;
      }
      seen.add(candidate.url);
      return true;
    })
    .slice(0, MAX_CANDIDATE_URLS);
}

export async function searchManga(query: string): Promise<MangaSearchResult[]> {
  let sourceAdapter: "weebcentral" | "manhwazone" = "weebcentral";
  let candidates = await fetchStrictDirectSearchMatches(query, "weebcentral");
  if (candidates.length === 0) {
    sourceAdapter = "manhwazone";
    candidates = await fetchStrictDirectSearchMatches(query, "manhwazone");
  }

  if (candidates.length === 0) {
    return [];
  }

  // eslint-disable-next-line no-console
  console.log(`[search] source=${sourceAdapter} candidates=${candidates.length}`);

  const deadline = Date.now() + SEARCH_DEADLINE_MS;
  const seenTitles = new Set<string>();
  const results: MangaSearchResult[] = [];

  for (let i = 0; i < candidates.length; i += SEARCH_CONCURRENCY) {
    if (results.length >= MAX_RESULTS || Date.now() >= deadline) {
      break;
    }

    const batch = candidates.slice(i, i + SEARCH_CONCURRENCY);
    const settled = await Promise.allSettled(batch.map((candidate) => buildDirectSearchResult(candidate, sourceAdapter)));

    for (const outcome of settled) {
      if (outcome.status !== "fulfilled" || !outcome.value) {
        continue;
      }

      const result = outcome.value;
      const title = result.seriesTitle;

      if (!title) {
        continue;
      }

      if (!isTitleRelevant(title, query) || ARTICLE_HEADLINE_RE.test(title)) {
        continue;
      }

      // Strip all punctuation/whitespace for robust deduplication across apostrophe/dash variants
      const titleKey = title
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "");
      if (seenTitles.has(titleKey)) {
        continue;
      }

      seenTitles.add(titleKey);

      results.push(result);

      if (results.length >= MAX_RESULTS) {
        break;
      }
    }
  }

  // eslint-disable-next-line no-console
  console.log(`[search] done results=${results.length}`);

  return results;
}
