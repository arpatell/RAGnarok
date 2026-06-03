import * as cheerio from "cheerio";
import { z } from "zod";
import type {
  Adapter,
  ChapterListItem,
  ExtractedChapterData,
  PanelCandidate,
  ReaderMode,
  SeriesStatus
} from "../types.js";
import {
  extractChapterLinks,
  extractPanelCandidates,
  extractTexts,
  getFirstAttr,
  getFirstText,
  inferChapterNumber,
  normalizeStatus,
  sanitizeHtml
} from "./html.js";

const AI_TIMEOUT_MS = Number.parseInt(process.env.AI_EXTRACTOR_TIMEOUT_MS ?? "25000", 10);
const MAX_HTML_SNIPPET = 80_000;
const MAX_CANDIDATE_PANELS = 180;
const MAX_CANDIDATE_LINKS = 200;
const MAX_CANDIDATE_PAGE_LINKS = 260;

const GENERIC_PANEL_SELECTORS = [
  "#imgs img",
  ".chapter-content img",
  ".reading-content img",
  ".container-chapter-reader img",
  "#reader img",
  ".reader img",
  ".viewer img",
  "article img",
  "main img"
];

const GENERIC_CHAPTER_LINK_SELECTORS = [
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

const GENERIC_TITLE_SELECTORS = [
  "meta[property='og:title']",
  "h1",
  "h2",
  "title"
];

const GENERIC_CHAPTER_TITLE_SELECTORS = [
  ".chapter-heading",
  ".entry-title",
  "h1",
  "title"
];

const GENERIC_COVER_SELECTORS = [
  "meta[property='og:image']",
  ".cover img",
  ".info-image img",
  ".summary_image img",
  "img[alt*='cover' i]"
];

const GENERIC_GENRE_SELECTORS = [
  "a[href*='/genre/']",
  "a[href*='/tag/']",
  ".genres a",
  ".genres-content a"
];

const GENERIC_STATUS_SELECTORS = [
  ".status",
  ".series-status",
  ".post-status .summary-content",
  ".summary-content",
  ".story-info-right .table-value"
];

const READER_CONTAINER_SELECTORS = [
  "#imgs",
  "#reader",
  ".reading-content",
  ".chapter-content",
  ".container-chapter-reader",
  "main"
];

const flexibleString = z.preprocess((value) => (value == null ? "" : String(value)), z.string());
const flexibleNullableString = z.preprocess(
  (value) => (value == null ? null : String(value)),
  z.string().nullable()
);
const flexibleStringArray = z.preprocess((value) => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean);
  }

  if (typeof value === "string") {
    return value
      .split(/[\n,;|]+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  return [];
}, z.array(z.string()));

const aiOutputSchema = z.object({
  pageType: z.preprocess((value) => {
    const normalized = String(value ?? "unknown").toLowerCase();
    return ["chapter", "series", "unknown"].includes(normalized) ? normalized : "unknown";
  }, z.enum(["chapter", "series", "unknown"]).default("unknown")),
  firstChapterUrl: flexibleNullableString.default(null),
  seriesTitle: flexibleString.default(""),
  chapterNumber: flexibleString.default(""),
  chapterTitle: flexibleNullableString.default(null),
  coverUrl: flexibleNullableString.default(null),
  genres: flexibleStringArray.default([]),
  status: z.preprocess((value) => {
    const normalized = String(value ?? "unknown").toLowerCase();
    if (normalized.includes("ongoing") || normalized.includes("publishing")) {
      return "ongoing";
    }
    if (normalized.includes("complete") || normalized.includes("finished")) {
      return "completed";
    }
    return "unknown";
  }, z.enum(["ongoing", "completed", "unknown"]).default("unknown")),
  panelUrls: flexibleStringArray.default([]),
  chapterList: z
    .array(
      z.object({
        number: flexibleString.default(""),
        title: flexibleNullableString.default(null),
        url: flexibleString.default("")
      })
    )
    .default([]),
  detectedMode: z.enum(["paginated", "scroll", "unknown"]).default("unknown"),
  navigation: z
    .object({
      previousChapterUrl: flexibleNullableString.default(null),
      nextChapterUrl: flexibleNullableString.default(null),
      chapterUrlPattern: flexibleNullableString.default(null)
    })
    .default({
      previousChapterUrl: null,
      nextChapterUrl: null,
      chapterUrlPattern: null
    })
});

type AIOutput = z.infer<typeof aiOutputSchema>;

interface AIProviderConfig {
  name: "cerebras" | "openai";
  apiKey: string;
  model: string;
  baseUrl: string;
  supportsResponseFormat: boolean;
}

interface AIExtractionFailure {
  reason: string;
  status?: number;
  detail?: string;
}

type AIExtractionAttempt =
  | { ok: true; output: AIOutput; source: string }
  | { ok: false; failure: AIExtractionFailure; source: string | null };

function trimEnv(value: string | undefined): string {
  return (value ?? "").trim();
}

function resolveApiKey(raw: string | undefined): string {
  const value = trimEnv(raw);
  if (!value) {
    return "";
  }

  if (/replace_with|your_.*key|changeme|example/i.test(value)) {
    return "";
  }

  return value;
}

function resolveAIProvider(): AIProviderConfig | null {
  const cerebrasApiKey = resolveApiKey(process.env.CEREBRAS_API_KEY);
  if (cerebrasApiKey) {
    return {
      name: "cerebras",
      apiKey: cerebrasApiKey,
      model: trimEnv(process.env.CEREBRAS_MODEL) || "gpt-oss-120b",
      baseUrl: (trimEnv(process.env.CEREBRAS_BASE_URL) || "https://api.cerebras.ai/v1").replace(
        /\/+$/,
        ""
      ),
      supportsResponseFormat: false
    };
  }

  const openAiApiKey = resolveApiKey(process.env.OPENAI_API_KEY);
  if (openAiApiKey) {
    return {
      name: "openai",
      apiKey: openAiApiKey,
      model: trimEnv(process.env.OPENAI_MODEL) || "gpt-4o-mini",
      baseUrl: (trimEnv(process.env.OPENAI_BASE_URL) || "https://api.openai.com/v1").replace(
        /\/+$/,
        ""
      ),
      supportsResponseFormat: true
    };
  }

  return null;
}

function cleanText(value: string | null | undefined): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
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

function dedupePanelCandidates(candidates: PanelCandidate[]): PanelCandidate[] {
  const map = new Map<string, PanelCandidate>();
  for (const candidate of candidates) {
    if (!candidate.url) {
      continue;
    }

    map.set(candidate.url, candidate);
  }

  return Array.from(map.values());
}

function dedupeStrings(values: string[]): string[] {
  const set = new Set<string>();
  for (const value of values) {
    const cleaned = cleanText(value);
    if (cleaned) {
      set.add(cleaned);
    }
  }

  return Array.from(set);
}

interface NumericTokenMatch {
  start: number;
  end: number;
  raw: string;
  numericValue: number;
}

interface AdjacentChapterUrls {
  previousChapterUrl: string | null;
  nextChapterUrl: string | null;
  chapterUrlPattern: string | null;
}

function normalizeChapterTitle(value: string | null | undefined): string | null {
  const cleaned = cleanText(value);
  return cleaned || null;
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

function findChapterToken(pathname: string): NumericTokenMatch | null {
  let selected: NumericTokenMatch | null = null;

  const keywordPattern = /(chapter|chap|ch|episode|ep|c)([-_ ]*)(\d+(?:\.\d+)?)/gi;
  for (const match of pathname.matchAll(keywordPattern)) {
    const fullMatch = match[0] ?? "";
    const numericRaw = match[3] ?? "";
    const numericValue = Number.parseFloat(numericRaw);

    if (!fullMatch || !numericRaw || !Number.isFinite(numericValue)) {
      continue;
    }

    const fullIndex = match.index ?? -1;
    if (fullIndex < 0) {
      continue;
    }

    const numericStart = fullIndex + fullMatch.lastIndexOf(numericRaw);
    selected = {
      start: numericStart,
      end: numericStart + numericRaw.length,
      raw: numericRaw,
      numericValue
    };
  }

  if (selected) {
    return selected;
  }

  const genericPattern = /(\d+(?:\.\d+)?)/g;
  for (const match of pathname.matchAll(genericPattern)) {
    const numericRaw = match[1] ?? "";
    const numericValue = Number.parseFloat(numericRaw);
    if (!numericRaw || !Number.isFinite(numericValue)) {
      continue;
    }

    const numericStart = match.index ?? -1;
    if (numericStart < 0) {
      continue;
    }

    selected = {
      start: numericStart,
      end: numericStart + numericRaw.length,
      raw: numericRaw,
      numericValue
    };
  }

  return selected;
}

function inferAdjacentFromQueryParams(targetUrl: URL, offset: number): string | null {
  const cloned = new URL(targetUrl.toString());

  for (const [key, value] of cloned.searchParams.entries()) {
    if (!/(chapter|chap|ch|episode|ep|c)/i.test(key)) {
      continue;
    }

    const numericMatch = value.match(/\d+(?:\.\d+)?/);
    if (!numericMatch) {
      continue;
    }

    const numericValue = Number.parseFloat(numericMatch[0]);
    if (!Number.isFinite(numericValue)) {
      continue;
    }

    const nextValue = numericValue + offset;
    if (nextValue < 0) {
      return null;
    }

    const replacement = formatLikeTemplate(numericMatch[0], nextValue);
    cloned.searchParams.set(key, value.replace(numericMatch[0], replacement));
    return cloned.toString();
  }

  return null;
}

function inferAdjacentFromPathname(targetUrl: URL, offset: number): string | null {
  const token = findChapterToken(targetUrl.pathname);
  if (!token) {
    return null;
  }

  const nextValue = token.numericValue + offset;
  if (nextValue < 0) {
    return null;
  }

  const replacement = formatLikeTemplate(token.raw, nextValue);
  const cloned = new URL(targetUrl.toString());
  cloned.pathname = `${targetUrl.pathname.slice(0, token.start)}${replacement}${targetUrl.pathname.slice(token.end)}`;
  return cloned.toString();
}

function inferAdjacentChapterUrls(chapterUrl: string): AdjacentChapterUrls {
  try {
    const parsed = new URL(chapterUrl);
    const previousChapterUrl =
      inferAdjacentFromPathname(parsed, -1) ?? inferAdjacentFromQueryParams(parsed, -1);
    const nextChapterUrl =
      inferAdjacentFromPathname(parsed, 1) ?? inferAdjacentFromQueryParams(parsed, 1);

    const chapterToken = findChapterToken(parsed.pathname);
    const chapterUrlPattern = chapterToken
      ? `${parsed.pathname.slice(0, chapterToken.start)}{chapter}${parsed.pathname.slice(chapterToken.end)}`
      : null;

    return {
      previousChapterUrl,
      nextChapterUrl,
      chapterUrlPattern
    };
  } catch {
    return {
      previousChapterUrl: null,
      nextChapterUrl: null,
      chapterUrlPattern: null
    };
  }
}

function shiftChapterNumber(number: string, delta: number): string {
  const cleaned = cleanText(number);
  const match = cleaned.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return "";
  }

  const current = Number.parseFloat(match[0]);
  if (!Number.isFinite(current)) {
    return "";
  }

  const next = current + delta;
  if (next < 0) {
    return "";
  }

  return formatLikeTemplate(match[0], next);
}

function extractCandidatePageLinks(
  $: cheerio.CheerioAPI,
  chapterUrl: string
): Array<{ text: string; url: string }> {
  const links: Array<{ text: string; url: string }> = [];
  const seen = new Set<string>();

  let chapterOrigin = "";
  try {
    chapterOrigin = new URL(chapterUrl).origin;
  } catch {
    chapterOrigin = "";
  }

  $("a[href]").each((_, node) => {
    if (links.length >= MAX_CANDIDATE_PAGE_LINKS) {
      return false;
    }

    const anchor = $(node);
    const absoluteUrl = toAbsoluteUrl(chapterUrl, anchor.attr("href"));
    if (!absoluteUrl || seen.has(absoluteUrl)) {
      return;
    }

    try {
      const parsed = new URL(absoluteUrl);
      if (chapterOrigin && parsed.origin !== chapterOrigin) {
        return;
      }

      if (!/^https?:$/i.test(parsed.protocol)) {
        return;
      }
    } catch {
      return;
    }

    const text = cleanText(anchor.text()) || cleanText(anchor.attr("title"));
    const interesting =
      /chapter|chap|ch\.?|episode|ep\.?|next|prev|read/i.test(text) ||
      /chapter|episode|\/c\d+/i.test(absoluteUrl);

    if (!interesting && links.length >= Math.floor(MAX_CANDIDATE_PAGE_LINKS / 2)) {
      return;
    }

    seen.add(absoluteUrl);
    links.push({
      text,
      url: absoluteUrl
    });
  });

  return links;
}

function augmentChapterListWithInferredNavigation(options: {
  chapterList: ChapterListItem[];
  chapterUrl: string;
  chapterNumber: string;
  chapterTitle: string | null;
  navigation: AdjacentChapterUrls;
}): ChapterListItem[] {
  const map = new Map<string, ChapterListItem>();
  const addChapter = (item: { number: string; title: string | null; url: string }) => {
    const absoluteUrl = toAbsoluteUrl(options.chapterUrl, item.url);
    if (!absoluteUrl) {
      return;
    }

    const inferredNumber =
      cleanText(item.number) || inferChapterNumber(cleanText(item.title), absoluteUrl);

    map.set(absoluteUrl, {
      number: inferredNumber,
      title: normalizeChapterTitle(item.title),
      url: absoluteUrl
    });
  };

  for (const chapter of options.chapterList) {
    addChapter(chapter);
  }

  addChapter({
    number: options.chapterNumber,
    title: options.chapterTitle,
    url: options.chapterUrl
  });

  const inferredPreviousNumber = shiftChapterNumber(options.chapterNumber, -1);
  const inferredNextNumber = shiftChapterNumber(options.chapterNumber, 1);

  if (options.navigation.previousChapterUrl) {
    addChapter({
      number: inferredPreviousNumber,
      title: null,
      url: options.navigation.previousChapterUrl
    });
  }

  if (options.navigation.nextChapterUrl) {
    addChapter({
      number: inferredNextNumber,
      title: null,
      url: options.navigation.nextChapterUrl
    });
  }

  return Array.from(map.values());
}

function mergeChapterLists(primary: ChapterListItem[], secondary: ChapterListItem[]): ChapterListItem[] {
  const first = primary.length >= secondary.length ? primary : secondary;
  const second = first === primary ? secondary : primary;

  const map = new Map<string, ChapterListItem>();

  for (const chapter of first) {
    const absolute = toAbsoluteUrl(chapter.url, chapter.url);
    if (!absolute) {
      continue;
    }

    map.set(absolute, {
      number: cleanText(chapter.number),
      title: cleanText(chapter.title),
      url: absolute,
      special: chapter.special
    });
  }

  for (const chapter of second) {
    const absolute = toAbsoluteUrl(chapter.url, chapter.url);
    if (!absolute || map.has(absolute)) {
      continue;
    }

    map.set(absolute, {
      number: cleanText(chapter.number),
      title: cleanText(chapter.title),
      url: absolute,
      special: chapter.special
    });
  }

  return Array.from(map.values());
}

function normalizeSeriesTitle(title: string, chapterTitle: string | null): string {
  const cleaned = cleanText(title);
  if (!cleaned) {
    return "Unknown Series";
  }

  const withoutTrailingChapter = cleanText(
    cleaned.replace(/\s*[,|:\-–]\s*(chapter|ch\.?|episode|ep\.?)\s*[-#: ]?\d+(?:\.\d+)?\s*$/i, "")
  );
  if (withoutTrailingChapter && withoutTrailingChapter.length >= 2) {
    return withoutTrailingChapter;
  }

  const splitOnDash = cleaned.split(/\s+-\s+/);
  if (splitOnDash.length > 1 && /chapter|ch\.?|episode|ep\.?/i.test(splitOnDash[0] ?? "")) {
    const withoutChapterPrefix = cleanText(splitOnDash.slice(1).join(" - "));
    if (withoutChapterPrefix) {
      return withoutChapterPrefix;
    }
  }

  if (chapterTitle) {
    const cleanedChapter = cleanText(chapterTitle);
    if (cleanedChapter && cleaned !== cleanedChapter && cleaned.startsWith(cleanedChapter)) {
      const remainder = cleanText(cleaned.replace(cleanedChapter, ""));
      if (remainder) {
        return remainder;
      }
    }
  }

  return cleaned;
}

function safeAdapterParse(adapter: Adapter, sanitizedHtml: string, chapterUrl: string) {
  try {
    return adapter.parse(sanitizedHtml, chapterUrl);
  } catch {
    return null;
  }
}

function buildHeuristicExtraction(html: string, chapterUrl: string, adapter: Adapter | null): ExtractedChapterData {
  const sanitized = sanitizeHtml(html);
  const $ = cheerio.load(sanitized);
  const adapterParsed = adapter ? safeAdapterParse(adapter, sanitized, chapterUrl) : null;

  const genericPanelCandidates = extractPanelCandidates($, GENERIC_PANEL_SELECTORS, chapterUrl);
  const adapterPanelCandidates = adapterParsed?.chapter.panelCandidates ?? [];

  const panelCandidates = dedupePanelCandidates([
    ...(adapterPanelCandidates.length >= genericPanelCandidates.length
      ? adapterPanelCandidates
      : genericPanelCandidates),
    ...(adapterPanelCandidates.length >= genericPanelCandidates.length
      ? genericPanelCandidates
      : adapterPanelCandidates)
  ]);

  const genericChapterList = extractChapterLinks($, GENERIC_CHAPTER_LINK_SELECTORS, chapterUrl);
  const adapterChapterList = adapterParsed?.chapterList ?? [];
  const chapterList = mergeChapterLists(adapterChapterList, genericChapterList);

  const chapterTitle =
    cleanText(adapterParsed?.chapter.title) || cleanText(getFirstText($, GENERIC_CHAPTER_TITLE_SELECTORS)) || null;

  const inferredSeriesTitle =
    cleanText(adapterParsed?.series.title) || cleanText(getFirstText($, GENERIC_TITLE_SELECTORS));

  const chapterNumber =
    cleanText(adapterParsed?.chapter.number) || inferChapterNumber(chapterTitle ?? inferredSeriesTitle, chapterUrl);

  const coverUrl =
    toAbsoluteUrl(chapterUrl, adapterParsed?.series.coverUrl ?? null) ||
    getFirstAttr($, GENERIC_COVER_SELECTORS, "src", chapterUrl);

  const genres = dedupeStrings([
    ...(adapterParsed?.series.genres ?? []),
    ...extractTexts($, GENERIC_GENRE_SELECTORS, 20)
  ]).slice(0, 20);

  const statusText = getFirstText($, GENERIC_STATUS_SELECTORS);
  const status = ((adapterParsed?.series.status as SeriesStatus | undefined) ??
    normalizeStatus(statusText)) as SeriesStatus;

  return {
    series: {
      title: normalizeSeriesTitle(inferredSeriesTitle, chapterTitle),
      coverUrl,
      genres,
      status
    },
    chapter: {
      number: chapterNumber,
      title: chapterTitle,
      panelCandidates,
    },
    chapterList,
    detectedModeHint: null,
    source: adapter ? `${adapter.id}-heuristic` : "heuristic",
    chapterUrlPattern: inferAdjacentChapterUrls(chapterUrl).chapterUrlPattern,
    pageType: panelCandidates.length >= 2 ? "chapter" : chapterList.length >= 2 ? "series" : "unknown",
    suggestedFirstChapterUrl: null
  };
}

function collectImageElementContext(
  $: cheerio.CheerioAPI,
  chapterUrl: string
): Array<{
  tag: string;
  id: string;
  className: string;
  alt: string;
  src: string | null;
  dataSrc: string | null;
  dataOriginal: string | null;
  dataUrl: string | null;
  width: string;
  height: string;
  parent: string;
}> {
  const items: Array<{
    tag: string;
    id: string;
    className: string;
    alt: string;
    src: string | null;
    dataSrc: string | null;
    dataOriginal: string | null;
    dataUrl: string | null;
    width: string;
    height: string;
    parent: string;
  }> = [];

  $("img, amp-img, source").each((_, node) => {
    if (items.length >= MAX_CANDIDATE_PANELS) {
      return false;
    }

    const el = $(node);
    const parent = el.parent();
    items.push({
      tag: node.tagName ?? "img",
      id: cleanText(el.attr("id")),
      className: cleanText(el.attr("class")),
      alt: cleanText(el.attr("alt")),
      src: toAbsoluteUrl(chapterUrl, el.attr("src")),
      dataSrc: toAbsoluteUrl(chapterUrl, el.attr("data-src") ?? el.attr("data-lazy-src")),
      dataOriginal: toAbsoluteUrl(chapterUrl, el.attr("data-original")),
      dataUrl: toAbsoluteUrl(chapterUrl, el.attr("data-url")),
      width: cleanText(el.attr("width")),
      height: cleanText(el.attr("height")),
      parent: cleanText(
        [parent.prop("tagName"), parent.attr("id") ? `#${parent.attr("id")}` : "", parent.attr("class")]
          .filter(Boolean)
          .join(" ")
      )
    });
  });

  return items;
}

function extractReaderHtmlSnippet($: cheerio.CheerioAPI): string {
  for (const selector of READER_CONTAINER_SELECTORS) {
    const node = $(selector).first();
    if (!node.length) {
      continue;
    }

    const html = $.html(node);
    const snippet = cleanText(html).slice(0, MAX_HTML_SNIPPET);
    if (snippet) {
      return snippet;
    }
  }

  return cleanText($.html()).slice(0, MAX_HTML_SNIPPET);
}

function buildAIContext(html: string, chapterUrl: string, heuristic: ExtractedChapterData): {
  chapterUrl: string;
  pageTitle: string;
  metaDescription: string;
  ogTitle: string;
  ogImage: string;
  candidatePanelUrls: string[];
  candidateImageElements: ReturnType<typeof collectImageElementContext>;
  candidateChapterLinks: Array<{ number: string; title: string | null; url: string }>;
  candidatePageLinks: Array<{ text: string; url: string }>;
  inferredUrlStructure: AdjacentChapterUrls;
  readerHtmlSnippet: string;
} {
  const sanitized = sanitizeHtml(html);
  const $ = cheerio.load(sanitized);

  const pageTitle = cleanText($("title").first().text());
  const metaDescription = cleanText($("meta[name='description']").first().attr("content"));
  const ogTitle = cleanText($("meta[property='og:title']").first().attr("content"));
  const ogImage = cleanText($("meta[property='og:image']").first().attr("content"));
  const inferredUrlStructure = inferAdjacentChapterUrls(chapterUrl);

  return {
    chapterUrl,
    pageTitle,
    metaDescription,
    ogTitle,
    ogImage,
    candidatePanelUrls: heuristic.chapter.panelCandidates
      .map((panel) => panel.url)
      .slice(0, MAX_CANDIDATE_PANELS),
    candidateImageElements: collectImageElementContext($, chapterUrl),
    candidateChapterLinks: heuristic.chapterList.slice(0, MAX_CANDIDATE_LINKS).map((chapter) => ({
      number: chapter.number,
      title: chapter.title,
      url: chapter.url
    })),
    candidatePageLinks: extractCandidatePageLinks($, chapterUrl),
    inferredUrlStructure,
    readerHtmlSnippet: extractReaderHtmlSnippet($)
  };
}

function extractJsonObject(raw: string): unknown | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced?.[1]?.trim() ?? trimmed;

  try {
    return JSON.parse(candidate);
  } catch {
    const firstBrace = candidate.indexOf("{");
    const lastBrace = candidate.lastIndexOf("}");

    if (firstBrace < 0 || lastBrace <= firstBrace) {
      return null;
    }

    const narrowed = candidate.slice(firstBrace, lastBrace + 1);
    try {
      return JSON.parse(narrowed);
    } catch {
      return null;
    }
  }
}

async function requestAIExtraction(
  context: ReturnType<typeof buildAIContext>
): Promise<AIExtractionAttempt> {
  const provider = resolveAIProvider();
  if (!provider) {
    return {
      ok: false,
      source: null,
      failure: {
        reason: "no-provider"
      }
    };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AI_TIMEOUT_MS);

  try {
    const requestBody: {
      model: string;
      temperature: number;
      messages: Array<{ role: "system" | "user"; content: string }>;
      response_format?: { type: "json_object" };
    } = {
      model: provider.model,
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
            "You extract manga/manhwa chapter metadata and panel image URLs from webpage context. Respond with strict JSON only. Exclude logos, covers, ads, avatars, and thumbnails from panelUrls. Keep panelUrls in reading order."
        },
        {
          role: "user",
          content: JSON.stringify({
            task: "Extract chapter data from this webpage context.",
            outputSchema: {
              pageType: "chapter|series|unknown",
              firstChapterUrl: "string|null",
              seriesTitle: "string",
              chapterNumber: "string",
              chapterTitle: "string|null",
              coverUrl: "string|null",
              genres: ["string"],
              status: "ongoing|completed|unknown",
              panelUrls: ["string"],
              chapterList: [{ number: "string", title: "string|null", url: "string" }],
              detectedMode: "paginated|scroll|unknown",
              navigation: {
                previousChapterUrl: "string|null",
                nextChapterUrl: "string|null",
                chapterUrlPattern: "string|null"
              }
            },
            constraints: [
              "Use absolute URLs.",
              "Classify pageType as series when the page is a manga/manhwa info/list page with chapter links but no readable panel sequence.",
              "If pageType is series, set firstChapterUrl to the earliest readable chapter URL, usually Chapter 1, Prologue, or the lowest numbered chapter.",
              "If pageType is chapter, inspect candidate image elements, attributes, scripts, and reader HTML to identify the ordered panel image URLs.",
              "Keep only chapter page images in panelUrls.",
              "Infer chapterList entries from HTML links and URL structure.",
              "Infer navigation.previousChapterUrl and navigation.nextChapterUrl from chapter URL pattern when chapterList is incomplete.",
              "If uncertain, prefer returning fewer panelUrls rather than wrong non-panel images."
            ],
            context
          })
        }
      ]
    };

    if (provider.supportsResponseFormat) {
      requestBody.response_format = { type: "json_object" };
    }

    const response = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${provider.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      return {
        ok: false,
        source: `ai-${provider.name}`,
        failure: {
          reason: "http-error",
          status: response.status,
          detail: (await response.text()).slice(0, 500)
        }
      };
    }

    const payload = (await response.json()) as {
      choices?: Array<{
        message?: {
          content?: string | null;
        };
      }>;
    };

    const content = payload.choices?.[0]?.message?.content;
    if (typeof content !== "string" || !content.trim()) {
      return {
        ok: false,
        source: `ai-${provider.name}`,
        failure: {
          reason: "empty-response"
        }
      };
    }

    const parsed = extractJsonObject(content);
    if (!parsed) {
      return {
        ok: false,
        source: `ai-${provider.name}`,
        failure: {
          reason: "json-parse-failed",
          detail: content.slice(0, 500)
        }
      };
    }

    const validated = aiOutputSchema.safeParse(parsed);
    if (!validated.success) {
      return {
        ok: false,
        source: `ai-${provider.name}`,
        failure: {
          reason: "schema-validation-failed",
          detail: validated.error.message.slice(0, 500)
        }
      };
    }

    return {
      ok: true,
      output: validated.data,
      source: `ai-${provider.name}`
    };
  } catch (error) {
    return {
      ok: false,
      source: `ai-${provider.name}`,
      failure: {
        reason: error instanceof Error && error.name === "AbortError" ? "timeout" : "request-failed",
        detail: error instanceof Error ? error.message.slice(0, 500) : undefined
      }
    };
  } finally {
    clearTimeout(timeout);
  }
}

function normalizeAIPanelCandidates(panelUrls: string[], chapterUrl: string): PanelCandidate[] {
  const normalized = panelUrls
    .map((value) => toAbsoluteUrl(chapterUrl, value))
    .filter((value): value is string => Boolean(value))
    .map((url) => ({ url }));

  return dedupePanelCandidates(normalized);
}

function normalizeAIChapterList(
  chapterList: Array<{ number: string; title: string | null; url: string }>,
  chapterUrl: string
): ChapterListItem[] {
  const map = new Map<string, ChapterListItem>();

  for (const chapter of chapterList) {
    const absoluteUrl = toAbsoluteUrl(chapterUrl, chapter.url);
    if (!absoluteUrl) {
      continue;
    }

    map.set(absoluteUrl, {
      number: cleanText(chapter.number) || inferChapterNumber(cleanText(chapter.title), absoluteUrl),
      title: cleanText(chapter.title),
      url: absoluteUrl
    });
  }

  return Array.from(map.values());
}

function mergeAIAndHeuristic(
  ai: AIOutput,
  heuristic: ExtractedChapterData,
  chapterUrl: string,
  aiSource: string
): ExtractedChapterData {
  const aiPanelCandidates = normalizeAIPanelCandidates(ai.panelUrls, chapterUrl);
  const aiChapterList = normalizeAIChapterList(ai.chapterList, chapterUrl);
  const usedAIPanels = aiPanelCandidates.length >= 2;

  const chapterTitle = cleanText(ai.chapterTitle) || heuristic.chapter.title;
  const seriesTitle = normalizeSeriesTitle(cleanText(ai.seriesTitle) || heuristic.series.title, chapterTitle);

  const chapterNumber =
    cleanText(ai.chapterNumber) || inferChapterNumber(chapterTitle ?? seriesTitle, chapterUrl) || heuristic.chapter.number;

  const mergedGenres =
    ai.genres.length > 0
      ? dedupeStrings(ai.genres)
      : heuristic.series.genres;

  const mergedStatus = (ai.status === "unknown" ? heuristic.series.status : ai.status) as SeriesStatus;
  const mergedChapterList = mergeChapterLists(aiChapterList, heuristic.chapterList);
  const aiNavigation: AdjacentChapterUrls = {
    previousChapterUrl: toAbsoluteUrl(chapterUrl, ai.navigation.previousChapterUrl),
    nextChapterUrl: toAbsoluteUrl(chapterUrl, ai.navigation.nextChapterUrl),
    chapterUrlPattern: cleanText(ai.navigation.chapterUrlPattern) || null
  };
  const fallbackNavigation = inferAdjacentChapterUrls(chapterUrl);

  return {
    series: {
      title: seriesTitle,
      coverUrl: toAbsoluteUrl(chapterUrl, ai.coverUrl) || heuristic.series.coverUrl,
      genres: mergedGenres,
      status: mergedStatus
    },
    chapter: {
      number: chapterNumber,
      title: chapterTitle,
      panelCandidates: usedAIPanels ? aiPanelCandidates : heuristic.chapter.panelCandidates
    },
    chapterList: augmentChapterListWithInferredNavigation({
      chapterList: mergedChapterList,
      chapterUrl,
      chapterNumber,
      chapterTitle,
      navigation: {
        previousChapterUrl: aiNavigation.previousChapterUrl ?? fallbackNavigation.previousChapterUrl,
        nextChapterUrl: aiNavigation.nextChapterUrl ?? fallbackNavigation.nextChapterUrl,
        chapterUrlPattern: aiNavigation.chapterUrlPattern ?? fallbackNavigation.chapterUrlPattern
      }
    }),
    detectedModeHint: ai.detectedMode === "unknown" ? null : (ai.detectedMode as ReaderMode),
    source: usedAIPanels ? aiSource : `${aiSource}-heuristic`,
    chapterUrlPattern: aiNavigation.chapterUrlPattern ?? fallbackNavigation.chapterUrlPattern,
    pageType: ai.pageType,
    suggestedFirstChapterUrl: toAbsoluteUrl(chapterUrl, ai.firstChapterUrl)
  };
}

export async function extractChapterData(options: {
  html: string;
  chapterUrl: string;
  adapter: Adapter | null;
}): Promise<ExtractedChapterData> {
  const heuristic = buildHeuristicExtraction(options.html, options.chapterUrl, options.adapter);
  const heuristicWithNavigation = {
    ...heuristic,
    chapterList: augmentChapterListWithInferredNavigation({
      chapterList: heuristic.chapterList,
      chapterUrl: options.chapterUrl,
      chapterNumber: heuristic.chapter.number,
      chapterTitle: heuristic.chapter.title,
      navigation: inferAdjacentChapterUrls(options.chapterUrl)
    })
  };
  return heuristicWithNavigation;
}
