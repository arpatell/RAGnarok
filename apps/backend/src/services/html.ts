import * as cheerio from "cheerio";
import type { Cheerio, CheerioAPI } from "cheerio";
import type { ChapterListItem, PanelCandidate, SeriesStatus } from "../types.js";

const IMAGE_ATTRIBUTES = [
  "data-src",
  "data-original",
  "data-lazy-src",
  "data-url",
  "data-srcset",
  "srcset",
  "src"
];
const URL_BLACKLIST = /(?:^|[\/._-])(logo|avatar|icon|banner|sprite|thumb|cover|ads?)(?:[\/._-]|$)/i;
const CHAPTER_PREFIX = /^(chapter|ch\.?|episode|ep\.?)\s*/i;
const NUMBER_PATTERN = /-?\d+(?:\.\d+)?/;
const SCRIPT_IMAGE_URL_PATTERN = /https?:\/\/[^"'\s<>]+\.(?:jpg|jpeg|png|webp|gif|avif)(?:\?[^"'\s<>]*)?/gi;
const SCRIPT_ESCAPED_IMAGE_URL_PATTERN = /https?:\\\/\\\/[^"'\s<>]+\.(?:jpg|jpeg|png|webp|gif|avif)(?:\?[^"'\s<>]*)?/gi;
const SERIES_ROOT_SEGMENTS = new Set([
  "manga",
  "manhwa",
  "manhua",
  "comic",
  "series",
  "title",
  "titles"
]);
const CHAPTER_ROOT_SEGMENTS = new Set(["chapter", "chapters", "viewer", "read", "episode", "episodes"]);
const SERIES_QUERY_KEYS = ["title_no", "series", "manga", "comic", "id"] as const;
const CHAPTER_IN_SEGMENT_PATTERN =
  /(?:-|_| )?(?:chapter|chap|ch|episode|ep|c)[-_ ]*\d+(?:\.\d+)?(?:[-_ ]?(?:v|p)\d+)?$/i;

function splitLowerPathSegments(pathname: string): string[] {
  return pathname
    .split("/")
    .map((segment) => segment.trim().toLowerCase())
    .filter(Boolean);
}

function normalizeHostname(hostname: string): string {
  return hostname
    .toLowerCase()
    .replace(/^(www\d*|www|m|mobile|read)\./i, "")
    .trim();
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
    const host = normalizeHostname(parsed.hostname);
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
    if (segments.length >= 2) {
      const first = segments[0] ?? "";
      const second = segments[1] ?? "";
      const normalizedSecond = normalizeSeriesSegment(second) || second;

      if (SERIES_ROOT_SEGMENTS.has(first) && second) {
        return `${host}/${first}/${normalizedSecond}`;
      }

      if (segments.length >= 3 && first.length === 2) {
        const third = segments[2] ?? "";
        const normalizedThird = normalizeSeriesSegment(third) || third;
        if (third) {
          return `${host}/${first}/${normalizedSecond}/${normalizedThird}`;
        }
      }

      if (!CHAPTER_ROOT_SEGMENTS.has(first)) {
        return `${host}/${first}/${normalizedSecond}`;
      }
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

function toAbsoluteUrl(baseUrl: string, value: string | undefined | null): string | null {
  if (!value) {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed || trimmed === "#" || trimmed.startsWith("#")) {
    return null;
  }

  if (/^(javascript|about):/i.test(trimmed)) {
    return null;
  }

  try {
    return new URL(trimmed, baseUrl).toString();
  } catch {
    return null;
  }
}

function isLikelyPanelUrl(url: string): boolean {
  if (URL_BLACKLIST.test(url)) {
    return false;
  }

  if (/\.(jpg|jpeg|png|webp|gif|avif)(\?|$)/i.test(url)) {
    return true;
  }

  return /\/(image|images|uploads?)\b/i.test(url);
}

function parseDimension(value: string | undefined): number | undefined {
  if (!value) {
    return undefined;
  }

  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function extractUrlsFromSrcset(rawValue: string): string[] {
  return rawValue
    .split(",")
    .map((part) => part.trim())
    .map((part) => part.split(/\s+/)[0] ?? "")
    .filter(Boolean);
}

function readImageCandidate(element: Cheerio<any>, baseUrl: string): PanelCandidate | null {
  for (const attr of IMAGE_ATTRIBUTES) {
    const raw = element.attr(attr);
    if (!raw) {
      continue;
    }

    const candidates = attr.includes("srcset") ? extractUrlsFromSrcset(raw) : [raw];
    for (const candidate of candidates) {
      const url = toAbsoluteUrl(baseUrl, candidate);
      if (!url || url.startsWith("data:")) {
        continue;
      }

      if (!isLikelyPanelUrl(url)) {
        continue;
      }

      return {
        url,
        width: parseDimension(element.attr("width")),
        height: parseDimension(element.attr("height"))
      };
    }
  }

  return null;
}

function collectImageElements($: CheerioAPI, selector: string): Cheerio<any>[] {
  const selected = $(selector);
  if (selected.length === 0) {
    return [];
  }

  const imageElements: Cheerio<any>[] = [];
  selected.each((_, node) => {
    const wrapped = $(node);
    if (wrapped.is("img, amp-img, source")) {
      imageElements.push(wrapped);
      return;
    }

    wrapped.find("img, amp-img, source").each((__, nestedNode) => {
      imageElements.push($(nestedNode));
    });
  });

  return imageElements;
}

export function sanitizeHtml(html: string): string {
  const $ = cheerio.load(html);
  $("noscript, iframe, object, embed").remove();
  return $.html();
}

export function getFirstText($: CheerioAPI, selectors: string[]): string | null {
  for (const selector of selectors) {
    const text = $(selector).first().text().replace(/\s+/g, " ").trim();
    if (text) {
      return text;
    }
  }

  return null;
}

export function getFirstAttr($: CheerioAPI, selectors: string[], attr: string, baseUrl: string): string | null {
  for (const selector of selectors) {
    const value = $(selector).first().attr(attr) ?? $(selector).first().attr("content");
    const absolute = toAbsoluteUrl(baseUrl, value);
    if (absolute) {
      return absolute;
    }
  }

  return null;
}

export function extractTexts($: CheerioAPI, selectors: string[], limit = 20): string[] {
  const values: string[] = [];

  for (const selector of selectors) {
    $(selector).each((_, node) => {
      if (values.length >= limit) {
        return;
      }

      const text = $(node).text().replace(/\s+/g, " ").trim();
      if (text) {
        values.push(text);
      }
    });

    if (values.length >= limit) {
      break;
    }
  }

  return Array.from(new Set(values));
}

export function inferChapterNumber(sourceText: string, fallbackUrl: string): string {
  const cleaned = sourceText.replace(CHAPTER_PREFIX, "").trim();
  const source = cleaned || fallbackUrl;
  const match = source.match(NUMBER_PATTERN);
  if (match) {
    return match[0];
  }

  const slug = fallbackUrl.split("/").filter(Boolean).at(-1) ?? "Unknown";
  return slug.slice(0, 64);
}

export function normalizeStatus(rawStatus: string | null | undefined): SeriesStatus {
  if (!rawStatus) {
    return "unknown";
  }

  const lowered = rawStatus.toLowerCase();
  if (lowered.includes("ongoing")) {
    return "ongoing";
  }

  if (lowered.includes("complete") || lowered.includes("finished")) {
    return "completed";
  }

  return "unknown";
}

export function extractPanelCandidates($: CheerioAPI, selectors: string[], baseUrl: string): PanelCandidate[] {
  const unique = new Map<string, PanelCandidate>();

  const addCandidate = (candidate: PanelCandidate | null) => {
    if (!candidate) {
      return;
    }
    unique.set(candidate.url, candidate);
  };

  const candidateSelectors = selectors.length > 0 ? selectors : ["img"];
  for (const selector of candidateSelectors) {
    for (const imageElement of collectImageElements($, selector)) {
      addCandidate(readImageCandidate(imageElement, baseUrl));
    }
  }

  if (unique.size < 2) {
    $("img, amp-img, source").each((_, node) => {
      addCandidate(readImageCandidate($(node), baseUrl));
    });
  }

  if (unique.size < 2) {
    $("meta[property='og:image'], meta[name='twitter:image'], meta[itemprop='image']").each((_, node) => {
      const raw = $(node).attr("content") ?? $(node).attr("src");
      const url = toAbsoluteUrl(baseUrl, raw);
      if (!url || !isLikelyPanelUrl(url)) {
        return;
      }

      addCandidate({ url });
    });
  }

  if (unique.size < 2) {
    $("script").each((_, node) => {
      const content = $(node).html() ?? "";
      const matches = [
        ...(content.match(SCRIPT_IMAGE_URL_PATTERN) ?? []),
        ...(content
          .match(SCRIPT_ESCAPED_IMAGE_URL_PATTERN)
          ?.map((value) => value.replace(/\\\//g, "/")) ?? [])
      ];

      for (const raw of matches) {
        const url = toAbsoluteUrl(baseUrl, raw);
        if (!url || !isLikelyPanelUrl(url)) {
          continue;
        }

        addCandidate({ url });
      }
    });
  }

  return Array.from(unique.values());
}

function cleanChapterLabel(value: string | null | undefined): string {
  const cleaned = (value ?? "").replace(/\s+/g, " ").trim();
  const chapterMatch = cleaned.match(/\b(?:chapter|ch\.?|episode|ep\.?)\s*-?\d+(?:\.\d+)?\b/i);
  return chapterMatch?.[0]?.trim() || cleaned;
}

function extractWeebcentralChapterLinks($: CheerioAPI, baseUrl: string): ChapterListItem[] {
  const unique = new Map<string, ChapterListItem>();
  const containers = $("div.grid.grid-cols-2.gap-3.justify-items-center").filter((_, node) =>
    ($(node).attr("class") ?? "").split(/\s+/).includes("lg:grid-cols-3")
  );

  const addAnchor = (anchor: Cheerio<any>) => {
    const href = toAbsoluteUrl(baseUrl, anchor.attr("href"));
    if (!href || !/\/chapters\//i.test(href)) {
      return;
    }

    const title = cleanChapterLabel(anchor.text());
    unique.set(href, {
      number: inferChapterNumber(title, href),
      title: title || null,
      url: href
    });
  };

  containers.find("a[href]").each((_, node) => {
    addAnchor($(node));
  });

  containers.find("button#selected_chapter").each((_, node) => {
    const title = cleanChapterLabel($(node).text());
    if (!title) {
      return;
    }

    unique.set(baseUrl, {
      number: inferChapterNumber(title, baseUrl),
      title,
      url: baseUrl
    });
  });

  $("#chapter-list a[href*='/chapters/']").each((_, node) => {
    addAnchor($(node));
  });

  if (/\/full-chapter-list(?:[?#]|$)/i.test(baseUrl)) {
    $("a[href*='/chapters/']").each((_, node) => {
      addAnchor($(node));
    });
  }

  return Array.from(unique.values());
}

function extractManhwaZoneChapterLinks($: CheerioAPI, baseUrl: string): ChapterListItem[] {
  const unique = new Map<string, ChapterListItem>();

  $("ol[role='list'][aria-label='List of chapters'] li[role='listitem']").each((_, node) => {
    const listItem = $(node);
    const anchor = listItem.find("a[href]").first();
    const href = toAbsoluteUrl(baseUrl, anchor.attr("href"));
    if (!href) {
      return;
    }

    const title =
      cleanChapterLabel(anchor.find("span.truncate.flex-1").first().text()) ||
      cleanChapterLabel(anchor.text());

    unique.set(href, {
      number: inferChapterNumber(title, href),
      title: title || null,
      url: href
    });
  });

  return Array.from(unique.values());
}

export function extractChapterLinks($: CheerioAPI, selectors: string[], baseUrl: string): ChapterListItem[] {
  const unique = new Map<string, ChapterListItem>();
  let baseHostname = "";

  try {
    baseHostname = new URL(baseUrl).hostname.toLowerCase();
  } catch {
    baseHostname = "";
  }

  const normalizedBaseHostname = baseHostname ? normalizeHostname(baseHostname) : "";
  const baseSeriesKey = buildSeriesKey(baseUrl);

  if (normalizedBaseHostname.endsWith("weebcentral.com")) {
    const weebcentralChapters = extractWeebcentralChapterLinks($, baseUrl);
    if (weebcentralChapters.length > 0) {
      return weebcentralChapters;
    }
  }

  if (normalizedBaseHostname.endsWith("manhwazone.com")) {
    const manhwaZoneChapters = extractManhwaZoneChapterLinks($, baseUrl);
    if (manhwaZoneChapters.length > 0) {
      return manhwaZoneChapters;
    }
  }

  const candidateSelectors = selectors.length > 0 ? selectors : ["a[href]"];
  for (const selector of candidateSelectors) {
    const elements = $(selector);
    if (elements.length === 0) {
      continue;
    }

    elements.each((_, node) => {
      const anchor = $(node);
      if (!anchor.is("a")) {
        return;
      }

      const href = toAbsoluteUrl(baseUrl, anchor.attr("href"));
      if (!href) {
        return;
      }

      try {
        const parsedHref = new URL(href);
        if (
          normalizedBaseHostname &&
          normalizeHostname(parsedHref.hostname) !== normalizedBaseHostname
        ) {
          return;
        }

        const isManhwaZonePreviewLink =
          normalizedBaseHostname.endsWith("manhwazone.com") && parsedHref.pathname.startsWith("/preview/");
        if (baseSeriesKey && !isManhwaZonePreviewLink && !isSameSeriesUrl(baseUrl, parsedHref.toString())) {
          return;
        }
      } catch {
        return;
      }

      if (/\/(tag|genre|category|author|artists?)\//i.test(href)) {
        return;
      }

      const titleText = anchor.text().replace(/\s+/g, " ").trim();
      const relText = (anchor.attr("rel") ?? "").toLowerCase();

      const chapterKeyword = /chapter|ch\.?|episode|ep\.?/i.test(titleText);
      const chapterNumberHint =
        /\d/.test(titleText) || /chapter[-_/ ]?\d|episode[-_/ ]?\d|\/c(?:h(?:apter)?)?[-_/ ]?\d+/i.test(href);
      const navHint = /\b(prev(?:ious)?|next)\b/i.test(titleText) || /\b(prev|next)\b/i.test(relText);

      if (!chapterKeyword && !chapterNumberHint && !navHint) {
        return;
      }

      unique.set(href, {
        number: inferChapterNumber(titleText, href),
        title: titleText || null,
        url: href
      });
    });
  }

  return Array.from(unique.values());
}
