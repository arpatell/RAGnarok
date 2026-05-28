import * as cheerio from "cheerio";
import { resolveAdapterForUrl } from "../adapters/index.js";
import { sortChapterList } from "./chapterSort.js";
import { fetchChapterHtml } from "./fetchProxy.js";

const DDG_SEARCH_URL = "https://html.duckduckgo.com/html/";
const DDG_TIMEOUT_MS = 12_000;
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

const MANGAKATANA_BASE_URL = "https://mangakatana.com";
const MANGAKATANA_SEARCH_TIMEOUT_MS = 12_000;
const MANGAKATANA_SEARCH_LIMIT = 8;
const MANGAKATANA_DOMAIN = "mangakatana.com";

function normalizeTitle(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenizeTitle(value: string): string[] {
  return normalizeTitle(value)
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);
}

function seriesNameFromSeriesUrl(seriesUrl: string): string {
  try {
    const parsed = new URL(seriesUrl);
    const segments = parsed.pathname.split("/").filter(Boolean);
    const slug = segments[1] ?? "";
    if (!slug) {
      return "";
    }

    return slug
      .replace(/\.\d+$/, "")
      .replace(/[-_]+/g, " ")
      .trim();
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
  const precision = intersectionCount / querySet.size;
  const recall = intersectionCount / candidateSet.size;
  const base = (precision * 0.65) + (recall * 0.35);

  if (candidateNorm.includes(queryNorm) || queryNorm.includes(candidateNorm)) {
    return Math.min(1, base + 0.12);
  }
  return base;
}

function isMangaKatanaHost(url: string): boolean {
  const domain = extractDomain(url);
  return domain === MANGAKATANA_DOMAIN || domain.endsWith(`.${MANGAKATANA_DOMAIN}`);
}

function isMangaKatanaSeriesUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    if (!isMangaKatanaHost(parsed.toString())) {
      return false;
    }
    const segments = parsed.pathname.split("/").filter(Boolean);
    if (segments.length !== 2) {
      return false;
    }
    return segments[0] === "manga" && !segments[1]?.startsWith("page");
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

async function fetchMangaKatanaSearchSeriesUrls(title: string): Promise<string[]> {
  const encoded = encodeURIComponent(title);
  const urlsToTry = [
    `${MANGAKATANA_BASE_URL}/?search=${encoded}&search_by=book_name`,
    `${MANGAKATANA_BASE_URL}/?search=${encoded}`,
    `${MANGAKATANA_BASE_URL}/?s=${encoded}`
  ];

  for (const url of urlsToTry) {
    try {
      const response = await fetch(url, {
        headers: {
          "User-Agent": BROWSER_UA,
          Accept: "text/html,application/xhtml+xml,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.9"
        },
        signal: AbortSignal.timeout(MANGAKATANA_SEARCH_TIMEOUT_MS)
      });
      if (!response.ok) {
        continue;
      }

      const html = await response.text();
      const $ = cheerio.load(html);
      const links: string[] = [];
      $("a[href]").each((_, el) => {
        const href = $(el).attr("href") ?? "";
        const resolved = resolveUrl(MANGAKATANA_BASE_URL, href);
        if (!isMangaKatanaSeriesUrl(resolved)) {
          return;
        }
        links.push(resolved);
      });

      const deduped = dedupeUrls(links);
      if (deduped.length > 0) {
        return deduped.slice(0, MANGAKATANA_SEARCH_LIMIT);
      }
    } catch {
      // Try next search URL shape.
    }
  }

  return [];
}

async function fetchMangaKatanaSeriesCandidatesViaDdg(title: string): Promise<string[]> {
  const queries = [
    `site:mangakatana.com/manga "${title}"`,
    `${title} site:mangakatana.com/manga`
  ];
  for (const query of queries) {
    const urls = await fetchDuckDuckGoUrls(query);
    const deduped = dedupeUrls(urls.filter(isMangaKatanaSeriesUrl));
    if (deduped.length > 0) {
      return deduped.slice(0, MANGAKATANA_SEARCH_LIMIT);
    }
  }
  return [];
}

async function resolveFirstChapterFromSeriesUrl(seriesUrl: string): Promise<{ chapterUrl: string; matchedTitle: string } | null> {
  const fetched = await fetchChapterHtml(seriesUrl);
  if (fetched.status >= 400 || !fetched.html) {
    return null;
  }

  const adapter = resolveAdapterForUrl(seriesUrl);
  if (!adapter) {
    return null;
  }
  const parsed = adapter.parse(fetched.html, seriesUrl);
  const chapterList = sortChapterList(parsed.chapterList);
  if (chapterList.length === 0) {
    return null;
  }

  const firstNumeric = chapterList.find((chapter) => chapter.special !== true) ?? chapterList[0];
  if (!firstNumeric?.url) {
    return null;
  }

  const matchedTitle = cleanTitle(parsed.series.title || seriesNameFromSeriesUrl(seriesUrl) || "Unknown Series");
  return {
    chapterUrl: firstNumeric.url,
    matchedTitle
  };
}

function scoreSeriesCandidate(queryTitle: string, seriesUrl: string): number {
  const urlTitle = seriesNameFromSeriesUrl(seriesUrl);
  return titleSimilarityScore(queryTitle, urlTitle);
}

function chooseBestSeriesUrl(queryTitle: string, seriesUrls: string[]): string | null {
  const scored = seriesUrls
    .map((url) => ({ url, score: scoreSeriesCandidate(queryTitle, url) }))
    .sort((a, b) => b.score - a.score);
  return scored[0]?.url ?? null;
}

export async function resolveMangaKatanaReadNow(title: string): Promise<ReadNowResolution | null> {
  const requestedTitle = cleanTitle(title);
  if (!requestedTitle) {
    return null;
  }

  let seriesCandidates = await fetchMangaKatanaSearchSeriesUrls(requestedTitle);
  if (seriesCandidates.length === 0) {
    seriesCandidates = await fetchMangaKatanaSeriesCandidatesViaDdg(requestedTitle);
  }
  if (seriesCandidates.length === 0) {
    return null;
  }

  const selectedSeriesUrl = chooseBestSeriesUrl(requestedTitle, seriesCandidates);
  if (!selectedSeriesUrl) {
    return null;
  }

  const resolved = await resolveFirstChapterFromSeriesUrl(selectedSeriesUrl);
  if (!resolved) {
    return null;
  }

  return {
    requestedTitle,
    matchedTitle: resolved.matchedTitle,
    seriesUrl: selectedSeriesUrl,
    chapterUrl: resolved.chapterUrl,
    sourceDomain: extractDomain(resolved.chapterUrl)
  };
}

async function fetchDuckDuckGoUrls(query: string): Promise<string[]> {
  const params = new URLSearchParams({ q: query, b: "", kl: "us-en" });

  let html: string;
  try {
    const response = await fetch(DDG_SEARCH_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        Accept: "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
      },
      body: params.toString(),
      signal: AbortSignal.timeout(DDG_TIMEOUT_MS)
    });

    // eslint-disable-next-line no-console
    console.log(`[search] ddg status=${response.status} ok=${response.ok} query="${query.substring(0, 60)}"`);

    if (!response.ok) {
      return [];
    }

    html = await response.text();
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(`[search] ddg fetch error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }

  const $ = cheerio.load(html);
  const urls: string[] = [];

  $("a.result__a").each((_, el) => {
    const href = $(el).attr("href") ?? "";
    if (href.startsWith("http")) {
      urls.push(href);
    }
  });

  // eslint-disable-next-line no-console
  console.log(`[search] ddg extracted ${urls.length} urls`);

  return urls;
}

async function fetchMangaBuddyUrls(query: string): Promise<string[]> {
  const apiUrl = `https://mangabuddy.com/api/manga/search?q=${encodeURIComponent(query)}&limit=8`;
  let html: string;
  try {
    const response = await fetch(apiUrl, {
      headers: { "User-Agent": BROWSER_UA, Accept: "text/html" },
      signal: AbortSignal.timeout(12_000)
    });
    if (!response.ok) return [];
    html = await response.text();
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(`[search] mangabuddy fetch error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }

  const $ = cheerio.load(html);
  const seriesUrls: string[] = [];

  $("a[title][href]").each((_, el) => {
    const href = $(el).attr("href") ?? "";
    // Series slugs are single-segment paths like /solo-leveling (no nested slashes, no chapter keyword)
    if (!href.startsWith("/") || href.slice(1).includes("/") || href.includes("chapter") || href.length < 4) {
      return;
    }
    const resolved = `https://mangabuddy.com${href}`;
    if (!seriesUrls.includes(resolved)) seriesUrls.push(resolved);
  });

  // eslint-disable-next-line no-console
  console.log(`[search] mangabuddy extracted ${seriesUrls.length} urls`);
  return seriesUrls.slice(0, MAX_CANDIDATE_URLS);
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

export async function searchManga(query: string): Promise<MangaSearchResult[]> {
  // Primary query: exact phrase match for precision
  let rawUrls = await fetchDuckDuckGoUrls(`"${query}" read manga manhwa online chapter 1`);

  // Fallback 1: unquoted DDG query (helps with off-hours throttling / rare titles)
  if (rawUrls.length === 0) {
    // eslint-disable-next-line no-console
    console.log(`[search] primary returned 0 urls, trying fallback for "${query}"`);
    rawUrls = await fetchDuckDuckGoUrls(`${query} read manga manhwa online chapter 1`);
  }

  // Fallback 2: MangaBuddy direct search (when DDG is rate-limited / returning 202)
  if (rawUrls.length === 0) {
    // eslint-disable-next-line no-console
    console.log(`[search] ddg yielded 0 urls, trying MangaBuddy for "${query}"`);
    rawUrls = await fetchMangaBuddyUrls(query);
  }

  // Prefer chapter-page URLs; still include series pages as fallback
  const safeUrls = rawUrls.filter(isSafeToIngest);
  const chapterUrls = safeUrls.filter(looksLikeChapterUrlForSearch);
  const otherUrls = safeUrls.filter((u) => !looksLikeChapterUrlForSearch(u));
  const candidates = [...chapterUrls, ...otherUrls].slice(0, MAX_CANDIDATE_URLS);
  if (candidates.length === 0) {
    return [];
  }

  // eslint-disable-next-line no-console
  console.log(`[search] candidates=${candidates.length} chapter=${chapterUrls.length} other=${otherUrls.length}`);

  const deadline = Date.now() + SEARCH_DEADLINE_MS;
  const seenTitles = new Set<string>();
  const results: MangaSearchResult[] = [];

  for (let i = 0; i < candidates.length; i += SEARCH_CONCURRENCY) {
    if (results.length >= MAX_RESULTS || Date.now() >= deadline) {
      break;
    }

    const batch = candidates.slice(i, i + SEARCH_CONCURRENCY);
    const settled = await Promise.allSettled(batch.map(lightIngestWithTimeout));

    for (const outcome of settled) {
      if (outcome.status !== "fulfilled" || !outcome.value) {
        continue;
      }

      const { chapterUrl, result } = outcome.value;
      const title = result.title;

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

      results.push({
        seriesTitle: title,
        coverUrl: result.coverUrl,
        firstChapterUrl: chapterUrl,
        chapterCount: result.chapterCount,
        genres: result.genres,
        adapter: result.sourceAdapter,
        sourceDomain: extractDomain(chapterUrl)
      });

      if (results.length >= MAX_RESULTS) {
        break;
      }
    }
  }

  // eslint-disable-next-line no-console
  console.log(`[search] done results=${results.length}`);

  return results;
}
