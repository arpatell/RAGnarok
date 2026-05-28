import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type TouchEvent as ReactTouchEvent } from "react";
import { ApiRequestError, fetchJikanMangaChapterCount, fetchSuggestions, ingestChapter, relayImageUrl } from "../lib/api";
import { isImageLoaded, isImageLoading, preloadImageOnce } from "../lib/imagePreload";
import {
  addBookmark,
  clearProgress,
  deleteBookmark,
  exportBookmarks,
  getBookmarks,
  getHistory,
  getProgress,
  getReaderSettings,
  getSavedMode,
  hasReadChapter,
  isSeriesFavorited,
  markChapterRead,
  pushHistory,
  removeFavoriteBySeries,
  saveMode,
  saveProgress,
  saveReaderSettings,
  upsertFavorite,
  updateBookmarkLabel
} from "../lib/storage";
import type {
  Bookmark,
  ChapterListItem,
  IngestResponse,
  ReaderMode,
  ReaderSettings,
  SuggestionsPayload
} from "../types";
import { ChapterSidebar } from "./ChapterSidebar";
import { PaginatedReader } from "./PaginatedReader";
import { ReaderToolbar } from "./ReaderToolbar";
import { ScrollReader } from "./ScrollReader";
import { SettingsPanel } from "./SettingsPanel";

interface ReaderPageProps {
  chapterUrl: string;
  onNavigate: (url: string) => void;
  onBackHome: () => void;
  onNavigateNotFound: (payload: {
    failedUrl: string;
    failedReason: string;
    errorCode?: string;
    fallbackChapterUrl?: string;
  }) => boolean;
  onHistoryChanged: () => void;
}

function normalizeUrl(input: string): string {
  try {
    const parsed = new URL(input);
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return input;
  }
}

function findPreviousSuccessfulChapterUrl(failedUrl: string): string {
  const normalizedFailed = normalizeUrl(failedUrl);
  const history = getHistory();

  for (const entry of history) {
    const candidateUrl = entry.chapterUrl.trim();
    if (!candidateUrl) {
      continue;
    }

    if (normalizeUrl(candidateUrl) === normalizedFailed) {
      continue;
    }

    return candidateUrl;
  }

  return "";
}

function makeBookmarkId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function toRgb(hexColor: string): [number, number, number] {
  const cleaned = hexColor.replace("#", "").trim();
  if (!/^[0-9a-fA-F]{3,8}$/.test(cleaned)) {
    return [0, 0, 0];
  }

  const normalized =
    cleaned.length === 3
      ? cleaned
          .split("")
          .map((chunk) => `${chunk}${chunk}`)
          .join("")
      : cleaned;

  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);
  return [r, g, b];
}

function withOpacity(hexColor: string, opacity: number): string {
  const [r, g, b] = toRgb(hexColor);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

function lighten(hexColor: string, amount: number): string {
  const [r, g, b] = toRgb(hexColor);
  const clamp = (value: number) => Math.min(255, Math.max(0, value));
  return `rgb(${clamp(r + amount)}, ${clamp(g + amount)}, ${clamp(b + amount)})`;
}

type ChapterEntryPoint = "first" | "last";
const MOBILE_BREAKPOINT = 1080;

const SERIES_ROOT_SEGMENTS = new Set(["manga", "manhwa", "manhua", "comic", "series", "title", "titles"]);
const CHAPTER_ROOT_SEGMENTS = new Set(["chapter", "chapters", "viewer", "read", "episode", "episodes"]);
const SERIES_QUERY_KEYS = ["title_no", "series", "manga", "comic", "id"] as const;
const CHAPTER_IN_SEGMENT_PATTERN =
  /(?:-|_| )?(?:chapter|chap|ch|episode|ep|c)[-_ ]*\d+(?:\.\d+)?(?:[-_ ]?(?:v|p)\d+)?$/i;
const CHAPTER_URL_QUERY_KEY = "chapter";
const PAGE_QUERY_KEY = "page";

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

function filterChapterListBySeries(baseUrl: string, chapters: ChapterListItem[]): ChapterListItem[] {
  const normalizedBase = normalizeUrl(baseUrl);
  const filtered = chapters.filter((chapter) => isSameSeriesUrl(baseUrl, chapter.url));
  const hasCurrent = filtered.some((chapter) => normalizeUrl(chapter.url) === normalizedBase);

  if (hasCurrent) {
    return filtered;
  }

  const currentFromOriginal = chapters.find((chapter) => normalizeUrl(chapter.url) === normalizedBase);
  if (currentFromOriginal) {
    return [...filtered, currentFromOriginal];
  }

  return filtered;
}

function parseChapterNumber(value: string | null | undefined): number | null {
  const match = (value ?? "").match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatChapterNumberLike(rawTemplate: string, value: number): string {
  const decimalPlaces = rawTemplate.includes(".") ? (rawTemplate.split(".")[1] ?? "").length : 0;
  const absolute = Math.abs(value);
  let formatted = decimalPlaces > 0 ? absolute.toFixed(decimalPlaces) : String(Math.round(absolute));

  const integerTemplate = rawTemplate.replace(/^-/, "").split(".")[0] ?? rawTemplate;
  const leadingZeros = integerTemplate.match(/^0+/)?.[0].length ?? 0;
  if (leadingZeros > 0) {
    const [intPart, decimalPart] = formatted.split(".");
    const padded = (intPart ?? "0").padStart(Math.max((intPart ?? "").length, leadingZeros + 1), "0");
    formatted = decimalPart ? `${padded}.${decimalPart}` : padded;
  }

  return value < 0 ? `-${formatted}` : formatted;
}

function findChapterUrlToken(url: string): { start: number; end: number; raw: string } | null {
  try {
    const parsed = new URL(url);
    const keywordPattern = /(chapter|chap|ch|episode|ep|c)([-_ /]*)(\d+(?:\.\d+)?)/gi;
    let selected: { start: number; end: number; raw: string } | null = null;

    for (const match of parsed.pathname.matchAll(keywordPattern)) {
      const fullMatch = match[0] ?? "";
      const raw = match[3] ?? "";
      const index = match.index ?? -1;
      if (!fullMatch || !raw || index < 0) {
        continue;
      }

      const start = index + fullMatch.lastIndexOf(raw);
      selected = {
        start,
        end: start + raw.length,
        raw
      };
    }

    if (selected) {
      return selected;
    }

    const segments = parsed.pathname.split("/");
    let offset = 0;
    for (const segment of segments) {
      const match = segment.match(/^\d+(?:\.\d+)?$/);
      if (match) {
        return {
          start: offset,
          end: offset + segment.length,
          raw: segment
        };
      }
      offset += segment.length + 1;
    }
  } catch {
    return null;
  }

  return null;
}

function renderChapterUrl(chapterUrl: string, chapterNumber: number): string | null {
  try {
    const parsed = new URL(chapterUrl);
    const token = findChapterUrlToken(chapterUrl);
    if (token) {
      const replacement = formatChapterNumberLike(token.raw, chapterNumber);
      parsed.pathname = `${parsed.pathname.slice(0, token.start)}${replacement}${parsed.pathname.slice(token.end)}`;
      return parsed.toString();
    }

    for (const [key, value] of parsed.searchParams.entries()) {
      if (!/(chapter|chap|ch|episode|ep|c)/i.test(key)) {
        continue;
      }

      const numericMatch = value.match(/-?\d+(?:\.\d+)?/);
      parsed.searchParams.set(
        key,
        numericMatch ? value.replace(numericMatch[0], String(chapterNumber)) : String(chapterNumber)
      );
      return parsed.toString();
    }
  } catch {
    return null;
  }

  return null;
}

function sortChapterEntries(chapters: ChapterListItem[]): ChapterListItem[] {
  return [...chapters].sort((a, b) => {
    const aNumber = parseChapterNumber(a.number || a.title || a.url);
    const bNumber = parseChapterNumber(b.number || b.title || b.url);
    if (aNumber !== null && bNumber !== null && aNumber !== bNumber) {
      return aNumber - bNumber;
    }

    if (aNumber !== null) {
      return -1;
    }

    if (bNumber !== null) {
      return 1;
    }

    return (a.title ?? a.number).localeCompare(b.title ?? b.number, undefined, { sensitivity: "base" });
  });
}

function expandChapterListWithJikanCount(
  chapterUrl: string,
  chapters: ChapterListItem[],
  chapterCount: number | null
): ChapterListItem[] {
  if (!chapterCount || chapterCount <= 0 || !renderChapterUrl(chapterUrl, 1)) {
    return chapters;
  }

  const byUrl = new Map<string, ChapterListItem>();
  for (let chapterNumber = 1; chapterNumber <= chapterCount; chapterNumber += 1) {
    const url = renderChapterUrl(chapterUrl, chapterNumber);
    if (!url) {
      continue;
    }

    byUrl.set(normalizeUrl(url), {
      number: `Chapter ${chapterNumber}`,
      title: null,
      url
    });
  }

  for (const chapter of chapters) {
    byUrl.set(normalizeUrl(chapter.url), chapter);
  }

  return sortChapterEntries(Array.from(byUrl.values()));
}

function readPageIndexFromLocation(totalPages: number, expectedChapterUrl: string): number | null {
  if (typeof window === "undefined") {
    return null;
  }

  const params = new URLSearchParams(window.location.search);
  const queryChapter = params.get(CHAPTER_URL_QUERY_KEY)?.trim();
  if (!queryChapter) {
    return null;
  }

  if (normalizeUrl(queryChapter) !== normalizeUrl(expectedChapterUrl)) {
    return null;
  }

  const pageParam = params.get(PAGE_QUERY_KEY);
  if (!pageParam) {
    return null;
  }

  const parsed = Number.parseInt(pageParam, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }

  const lastPageIndex = Math.max(totalPages - 1, 0);
  return Math.min(Math.max(parsed - 1, 0), lastPageIndex);
}

function buildAdjacentPreloadOrder(
  panelUrls: string[],
  centerIndex: number,
  maxItems = Number.POSITIVE_INFINITY
): Array<{ index: number; url: string }> {
  const totalPages = panelUrls.length;
  const lastPageIndex = Math.max(totalPages - 1, 0);
  const safeCenter = Math.min(Math.max(centerIndex, 0), lastPageIndex);
  const safeMaxItems = Math.max(1, maxItems);
  const order: Array<{ index: number; url: string }> = [];

  const addIfWorthPreloading = (index: number) => {
    const url = panelUrls[index];
    if (!url || isImageLoaded(relayImageUrl(url)) || isImageLoading(relayImageUrl(url))) {
      return;
    }

    order.push({ index, url });
  };

  addIfWorthPreloading(safeCenter);
  if (order.length >= safeMaxItems || totalPages === 0) {
    return order;
  }

  let nextOutOfBounds = false;
  let previousOutOfBounds = false;

  for (let offset = 1; order.length < safeMaxItems && !(nextOutOfBounds && previousOutOfBounds); offset += 1) {
    const next = safeCenter + offset;
    const previous = safeCenter - offset;

    if (next > lastPageIndex) {
      nextOutOfBounds = true;
    } else {
      addIfWorthPreloading(next);
      if (order.length >= safeMaxItems) {
        return order;
      }
    }

    if (previous < 0) {
      previousOutOfBounds = true;
    } else {
      addIfWorthPreloading(previous);
      if (order.length >= safeMaxItems) {
        return order;
      }
    }
  }

  return order;
}

function logPreloadTiming(label: string, panelUrl: string, startedAt: number) {
  if (!import.meta.env.DEV) {
    return;
  }

  // eslint-disable-next-line no-console
  console.info("[RAGnarok] image preloaded", {
    label,
    ms: Math.round(performance.now() - startedAt),
    url: panelUrl
  });
}

export function ReaderPage({ chapterUrl, onNavigate, onBackHome, onNavigateNotFound, onHistoryChanged }: ReaderPageProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<IngestResponse | null>(null);
  const [mode, setMode] = useState<ReaderMode>("paginated");
  const [settings, setSettings] = useState<ReaderSettings>(() => getReaderSettings());
  const [sidebarOpen, setSidebarOpen] = useState(() => (typeof window !== "undefined" ? window.innerWidth > 1080 : true));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isMobileViewport, setIsMobileViewport] = useState(
    () => (typeof window !== "undefined" ? window.innerWidth <= MOBILE_BREAKPOINT : false)
  );
  const [chromeHidden, setChromeHidden] = useState(false);
  const [chapterFilter, setChapterFilter] = useState("");
  const [currentPage, setCurrentPage] = useState(0);
  const [scrollPage, setScrollPage] = useState(0);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [suggestions, setSuggestions] = useState<SuggestionsPayload | null>(null);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(() => getBookmarks());
  const [draftUrl, setDraftUrl] = useState(chapterUrl);
  const [initialScrollPage, setInitialScrollPage] = useState(0);
  const [chapterNavigationStatus, setChapterNavigationStatus] = useState<string | null>(null);
  const [jikanChapterCount, setJikanChapterCount] = useState<number | null>(null);
  const [isFavorited, setIsFavorited] = useState(false);
  const [mobileSelectorVisible, setMobileSelectorVisible] = useState(false);
  const [loadedChapterUrl, setLoadedChapterUrl] = useState<string | null>(null);
  const [topBarPeekVisible, setTopBarPeekVisible] = useState(false);
  const fallbackChapterUrl = useMemo(() => findPreviousSuccessfulChapterUrl(chapterUrl), [chapterUrl]);

  const longPanelPromptedRef = useRef(false);
  const pendingEntryPointRef = useRef<ChapterEntryPoint>("first");
  const settingsPanelRef = useRef<HTMLElement>(null);
  const previousSettingsOpenRef = useRef(false);
  const topBarPeekHideTimerRef = useRef<number | null>(null);
  const mobileTopSwipeStartRef = useRef<{ x: number; y: number; tracking: boolean }>({
    x: 0,
    y: 0,
    tracking: false
  });

  useEffect(() => {
    setDraftUrl(chapterUrl);
  }, [chapterUrl]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const onResize = () => {
      setIsMobileViewport(window.innerWidth <= MOBILE_BREAKPOINT);
    };

    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    if (!isMobileViewport || mode !== "paginated") {
      setMobileSelectorVisible(false);
    }
  }, [isMobileViewport, mode]);

  useEffect(() => {
    if (mode !== "scroll") {
      return;
    }

    setSettings((current) => {
      if (current.panelGap !== 12) {
        return current;
      }

      return {
        ...current,
        panelGap: 0
      };
    });
  }, [mode]);

  useEffect(() => {
    const wasOpen = previousSettingsOpenRef.current;
    previousSettingsOpenRef.current = settingsOpen;

    if (!settingsOpen || wasOpen || chromeHidden || !isMobileViewport) {
      return;
    }

    const panel = settingsPanelRef.current;
    if (!panel) {
      return;
    }

    panel.scrollTo({ top: 0, behavior: "auto" });

    if (mode === "scroll") {
      return;
    }

    const rafId = window.requestAnimationFrame(() => {
      panel.scrollIntoView({ behavior: "smooth", block: "end" });
    });

    return () => window.cancelAnimationFrame(rafId);
  }, [chromeHidden, isMobileViewport, mode, settingsOpen]);

  useEffect(() => {
    setChapterNavigationStatus(null);
  }, [chapterUrl]);

  useEffect(() => {
    if (!chromeHidden) {
      setTopBarPeekVisible(false);
      if (topBarPeekHideTimerRef.current !== null) {
        window.clearTimeout(topBarPeekHideTimerRef.current);
        topBarPeekHideTimerRef.current = null;
      }
    }
  }, [chromeHidden]);

  useEffect(() => {
    return () => {
      if (topBarPeekHideTimerRef.current !== null) {
        window.clearTimeout(topBarPeekHideTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadChapter() {
      setLoading(true);
      setError(null);
      setSuggestions(null);
      setLoadedChapterUrl(null);

      try {
        const result = await ingestChapter(chapterUrl);
        if (cancelled) {
          return;
        }

        if (import.meta.env.DEV) {
          // eslint-disable-next-line no-console
          console.info("[RAGnarok] extraction", {
            source: result.extractionSource ?? "unknown",
            adapter: result.sourceAdapter,
            pages: result.chapter.totalPages,
            requestUrl: chapterUrl
          });
        }

        setData(result);
        setLoadedChapterUrl(chapterUrl);
        longPanelPromptedRef.current = false;
        markChapterRead(chapterUrl);
        setIsFavorited(isSeriesFavorited(result.series.title));

        const preferredMode = getSavedMode(result.sourceAdapter) ?? result.detectedMode;
        setMode(preferredMode);

        const entryPoint = pendingEntryPointRef.current;
        pendingEntryPointRef.current = "first";
        const fallbackInitialPageIndex =
          entryPoint === "last" ? Math.max(result.chapter.totalPages - 1, 0) : 0;
        const sharedPageIndex = readPageIndexFromLocation(result.chapter.totalPages, chapterUrl);
        const mostRecentChapterUrl = getHistory()[0]?.chapterUrl ?? "";
        const mostRecentMatchesCurrent = normalizeUrl(mostRecentChapterUrl) === normalizeUrl(chapterUrl);
        const savedProgressIndex = mostRecentMatchesCurrent ? getProgress(chapterUrl)?.pageIndex ?? null : null;
        const normalizedSavedProgressIndex =
          savedProgressIndex !== null
            ? Math.min(Math.max(savedProgressIndex, 0), Math.max(result.chapter.totalPages - 1, 0))
            : null;
        const initialPageIndex =
          entryPoint === "last"
            ? fallbackInitialPageIndex
            : sharedPageIndex ?? normalizedSavedProgressIndex ?? fallbackInitialPageIndex;

        setCurrentPage(initialPageIndex);
        setScrollPage(initialPageIndex);
        setInitialScrollPage(initialPageIndex);
        setScrollProgress(initialPageIndex > 0 ? 1 : 0);

        saveMode(result.sourceAdapter, preferredMode);

        pushHistory({
          seriesTitle: result.series.title,
          chapterTitle: result.chapter.title ?? `Chapter ${result.chapter.number}`,
          chapterUrl,
          coverUrl: result.series.coverUrl,
          mode: preferredMode,
          visitedAt: new Date().toISOString()
        });
        onHistoryChanged();

        setBookmarks(getBookmarks());

        fetchSuggestions(result.series.title, result.series.genres, result.sourceAdapter)
          .then((payload) => {
            if (!cancelled) {
              setSuggestions(payload);
            }
          })
          .catch(() => {
            if (!cancelled) {
              setSuggestions(null);
            }
          });
      } catch (requestError) {
        if (!cancelled) {
          const message = requestError instanceof Error ? requestError.message : "Failed to ingest chapter.";
          const errorCode =
            requestError instanceof ApiRequestError && typeof requestError.code === "string"
              ? requestError.code
              : undefined;
          if (
            onNavigateNotFound({
              failedUrl: chapterUrl,
              failedReason: message,
              errorCode,
              fallbackChapterUrl
            })
          ) {
            return;
          }

          if (
            typeof window !== "undefined" &&
            /does not expose panel image links in html/i.test(message)
          ) {
            const shouldRedirect = window.confirm(
              "This site hides panel image links from HTML. Open this chapter on the source site instead?"
            );
            if (shouldRedirect) {
              window.location.assign(chapterUrl);
              return;
            }
          }

          setError(message);
          setData(null);
          setLoadedChapterUrl(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadChapter();

    return () => {
      cancelled = true;
    };
  }, [chapterUrl, fallbackChapterUrl, onHistoryChanged, onNavigateNotFound]);

  useEffect(() => {
    if (!data) {
      setJikanChapterCount(null);
      return;
    }

    const currentList = filterChapterListBySeries(chapterUrl, data.chapterList);
    const hasSparseChapterList = currentList.length > 0 && currentList.length < 8;
    const canEnumerateFromUrl = Boolean(renderChapterUrl(chapterUrl, 1));
    if (!hasSparseChapterList || !canEnumerateFromUrl) {
      setJikanChapterCount(null);
      return;
    }

    const controller = new AbortController();
    setJikanChapterCount(null);

    void fetchJikanMangaChapterCount(
      {
        seriesTitle: data.series.title,
        chapterTitle: data.chapter.title,
        chapterUrl
      },
      controller.signal
    )
      .then((count) => {
        if (!controller.signal.aborted) {
          setJikanChapterCount(count);
        }
      })
      .catch((error) => {
        if (error instanceof Error && error.name === "AbortError") {
          return;
        }

        if (!controller.signal.aborted) {
          setJikanChapterCount(null);
        }
      });

    return () => {
      controller.abort();
    };
  }, [chapterUrl, data]);

  useEffect(() => {
    saveReaderSettings(settings);
  }, [settings]);

  useEffect(() => {
    if (!data) {
      return;
    }

    saveMode(data.sourceAdapter, mode);
  }, [data, mode]);

  useEffect(() => {
    if (!data || !settings.saveProgress) {
      return;
    }

    const timeout = window.setTimeout(() => {
      saveProgress({
        chapterUrl,
        pageIndex: currentPage,
        scrollPercent: scrollProgress,
        updatedAt: new Date().toISOString()
      });
    }, 300);

    return () => window.clearTimeout(timeout);
  }, [chapterUrl, currentPage, data, scrollProgress, settings.saveProgress]);

  useEffect(() => {
    if (!data) {
      return;
    }

    const panelUrls = Array.from(new Set(data.chapter.panelUrls));
    if (panelUrls.length === 0) {
      return;
    }

    const preferredStartIndex = mode === "paginated" ? currentPage : scrollPage;
    const preloadLimit = Math.max(5, settings.preloadDepth * 2 + 1);
    const preloadOrder = buildAdjacentPreloadOrder(panelUrls, preferredStartIndex, preloadLimit);

    if (preloadOrder.length === 0) {
      return;
    }

    let cancelled = false;
    const timerIds = new Set<number>();
    const workerCount = typeof window !== "undefined" && window.innerWidth < 900 ? 1 : 2;
    let cursor = 0;

    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        const id = window.setTimeout(() => {
          timerIds.delete(id);
          resolve();
        }, ms);
        timerIds.add(id);
      });

    const preloadWithRetry = async (panelUrl: string, index: number) => {
      let attempt = 0;

      while (!cancelled) {
        const loaded = await preloadImageOnce(relayImageUrl(panelUrl), (startedAt) =>
          logPreloadTiming(`reader-preload:${index + 1}`, panelUrl, startedAt)
        );

        if (loaded || cancelled) {
          return;
        }

        attempt += 1;
        const delayMs = Math.min(8_000, 250 * 2 ** Math.min(attempt, 6) + Math.random() * 200);
        await sleep(delayMs);
      }
    };

    const workers = Array.from({ length: workerCount }, async () => {
      while (!cancelled) {
        const entry = preloadOrder[cursor];
        cursor += 1;

        if (!entry) {
          return;
        }

        await preloadWithRetry(entry.url, entry.index);
      }
    });

    void Promise.all(workers);

    return () => {
      cancelled = true;
      for (const id of timerIds) {
        window.clearTimeout(id);
      }
      timerIds.clear();
    };
  }, [chapterUrl, currentPage, data, mode, scrollPage, settings.preloadDepth]);

  const visibleChapterList = useMemo(() => {
    if (!data) {
      return [];
    }

    const filtered = filterChapterListBySeries(chapterUrl, data.chapterList);
    return expandChapterListWithJikanCount(chapterUrl, filtered, jikanChapterCount);
  }, [chapterUrl, data, jikanChapterCount]);

  const currentChapterIndex = useMemo(() => {
    if (!data) {
      return -1;
    }

    const target = normalizeUrl(chapterUrl);
    return visibleChapterList.findIndex((chapter) => normalizeUrl(chapter.url) === target);
  }, [chapterUrl, data, visibleChapterList]);

  const previousChapter =
    data && currentChapterIndex > 0 ? visibleChapterList[currentChapterIndex - 1] ?? null : null;
  const nextChapter =
    data && currentChapterIndex >= 0 ? visibleChapterList[currentChapterIndex + 1] ?? null : null;

  function navigateTo(
    url: string,
    options: {
      entryPoint?: ChapterEntryPoint;
      clearSavedProgress?: boolean;
      restrictToCurrentSeries?: boolean;
    } = {}
  ): boolean {
    if (!url || normalizeUrl(url) === normalizeUrl(chapterUrl)) {
      return true;
    }

    if (options.restrictToCurrentSeries && !isSameSeriesUrl(chapterUrl, url)) {
      setChapterNavigationStatus("No chapter data found for that chapter on this manga.");
      return false;
    }

    setChapterNavigationStatus(null);
    pendingEntryPointRef.current = options.entryPoint ?? "first";
    if (options.clearSavedProgress) {
      clearProgress(url);
    }

    setChapterFilter("");
    setSettingsOpen(false);
    onNavigate(url);
    return true;
  }

  function toggleReaderChrome() {
    setChromeHidden((hidden) => {
      const nextHidden = !hidden;
      setSidebarOpen(!nextHidden);
      if (!nextHidden) {
        setTopBarPeekVisible(false);
      }
      if (nextHidden) {
        setSettingsOpen(false);
      }

      return nextHidden;
    });
  }

  function toggleSettingsPanel() {
    setSettingsOpen((open) => {
      const nextOpen = !open;

      if (nextOpen) {
        setChromeHidden(false);
        setSidebarOpen(true);
      }

      return nextOpen;
    });
  }

  function closeSettingsPanel() {
    setSettingsOpen(false);
    if (isMobileViewport && !chromeHidden) {
      setSidebarOpen(true);
    }
  }

  function revealTopBarPeekFromMobileGesture() {
    if (!chromeHidden || !isMobileViewport || sidebarOpen) {
      return;
    }

    setTopBarPeekVisible(true);
    if (topBarPeekHideTimerRef.current !== null) {
      window.clearTimeout(topBarPeekHideTimerRef.current);
    }
    topBarPeekHideTimerRef.current = window.setTimeout(() => {
      setTopBarPeekVisible(false);
      topBarPeekHideTimerRef.current = null;
    }, 3000);
  }

  function handleShellMouseMove(event: ReactMouseEvent<HTMLDivElement>) {
    if (isMobileViewport || !chromeHidden || sidebarOpen) {
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const relativeY = event.clientY - bounds.top;
    const shouldPeek = relativeY <= 74;
    setTopBarPeekVisible((visible) => (visible === shouldPeek ? visible : shouldPeek));
  }

  function handleShellMouseLeave() {
    if (isMobileViewport || !chromeHidden || sidebarOpen) {
      return;
    }

    setTopBarPeekVisible(false);
  }

  function handleShellTouchStart(event: ReactTouchEvent<HTMLDivElement>) {
    if (!isMobileViewport || !chromeHidden || sidebarOpen) {
      mobileTopSwipeStartRef.current = { x: 0, y: 0, tracking: false };
      return;
    }

    const touch = event.touches[0];
    if (!touch) {
      mobileTopSwipeStartRef.current = { x: 0, y: 0, tracking: false };
      return;
    }

    mobileTopSwipeStartRef.current = {
      x: touch.clientX,
      y: touch.clientY,
      tracking: touch.clientY <= 78
    };
  }

  function handleShellTouchMove(event: ReactTouchEvent<HTMLDivElement>) {
    const swipe = mobileTopSwipeStartRef.current;
    if (!swipe.tracking) {
      return;
    }

    const touch = event.touches[0];
    if (!touch) {
      return;
    }

    const deltaY = touch.clientY - swipe.y;
    const deltaX = Math.abs(touch.clientX - swipe.x);
    if (deltaY >= 56 && deltaY > deltaX + 10) {
      swipe.tracking = false;
      revealTopBarPeekFromMobileGesture();
    }
  }

  function handleShellTouchEnd() {
    mobileTopSwipeStartRef.current = { x: 0, y: 0, tracking: false };
  }

  function goToPreviousChapter(entryPoint: ChapterEntryPoint = "first", clearSavedProgress = false) {
    if (previousChapter) {
      navigateTo(previousChapter.url, {
        entryPoint,
        clearSavedProgress,
        restrictToCurrentSeries: true
      });
    }
  }

  function goToNextChapter(entryPoint: ChapterEntryPoint = "first", clearSavedProgress = false) {
    if (nextChapter) {
      navigateTo(nextChapter.url, {
        entryPoint,
        clearSavedProgress,
        restrictToCurrentSeries: true
      });
    }
  }

  function handleSidebarChapterSelect(url: string): string | null {
    const didNavigate = navigateTo(url, {
      entryPoint: "first",
      clearSavedProgress: true,
      restrictToCurrentSeries: true
    });

    return didNavigate ? null : "No chapter data found for that chapter on this manga.";
  }

  function handleLongPanelDetected() {
    if (longPanelPromptedRef.current || mode !== "paginated" || settings.fitMode !== "height-fit") {
      return;
    }

    longPanelPromptedRef.current = true;
    const shouldSwitch = window.confirm(
      "Long vertical images detected. Switch to Manhwa mode for smoother reading?"
    );

    if (shouldSwitch) {
      setMode("scroll");
    }
  }

  function createBookmark() {
    if (!data) {
      return;
    }

    const labelSeed =
      mode === "paginated"
        ? `Page ${Math.min(currentPage + 1, data.chapter.totalPages)}`
        : `${Math.round(scrollProgress * 100)}%`;

    const label = window.prompt("Bookmark label", `${data.series.title} • ${labelSeed}`)?.trim();
    if (!label) {
      return;
    }

    addBookmark({
      id: makeBookmarkId(),
      label,
      chapterUrl,
      seriesTitle: data.series.title,
      pageIndex: currentPage,
      scrollPercent: scrollProgress,
      createdAt: new Date().toISOString()
    });

    setBookmarks(getBookmarks());
  }

  function toggleFavoriteSeries() {
    if (!data) {
      return;
    }

    if (isSeriesFavorited(data.series.title)) {
      removeFavoriteBySeries(data.series.title);
      setIsFavorited(false);
      onHistoryChanged();
      return;
    }

    upsertFavorite({
      seriesTitle: data.series.title,
      chapterTitle: data.chapter.title ?? `Chapter ${data.chapter.number}`,
      chapterUrl,
      coverUrl: data.series.coverUrl,
      updatedAt: new Date().toISOString()
    });

    setIsFavorited(true);
    onHistoryChanged();
  }

  function goToBookmark(bookmark: Bookmark) {
    navigateTo(bookmark.chapterUrl);
  }

  useEffect(() => {
    if (typeof window === "undefined" || !data) {
      return;
    }

    if (!loadedChapterUrl || normalizeUrl(loadedChapterUrl) !== normalizeUrl(chapterUrl)) {
      return;
    }

    const totalPages = Math.max(1, data.chapter.totalPages);
    const activePage = mode === "paginated" ? currentPage : scrollPage;
    const page = Math.min(Math.max(activePage + 1, 1), totalPages);
    const params = new URLSearchParams();
    params.set(CHAPTER_URL_QUERY_KEY, chapterUrl);
    params.set(PAGE_QUERY_KEY, String(page));

    const query = params.toString();
    const target = `${window.location.pathname}${query ? `?${query}` : ""}`;
    const previousState = (window.history.state as Record<string, unknown> | null) ?? {};
    window.history.replaceState(previousState, "", target);
  }, [chapterUrl, currentPage, data, loadedChapterUrl, mode, scrollPage]);

  function editBookmark(bookmark: Bookmark) {
    const label = window.prompt("Edit bookmark label", bookmark.label)?.trim();
    if (!label) {
      return;
    }

    updateBookmarkLabel(bookmark.id, label);
    setBookmarks(getBookmarks());
  }

  function removeBookmark(bookmark: Bookmark) {
    deleteBookmark(bookmark.id);
    setBookmarks(getBookmarks());
  }

  function downloadBookmarks() {
    const blob = new Blob([exportBookmarks()], {
      type: "application/json"
    });

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `panelflow-bookmarks-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    function onKeydown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable)) {
        return;
      }

      if (event.key === "o" || event.key === "O") {
        event.preventDefault();
        toggleSettingsPanel();
        return;
      }

      if (event.key === "s" || event.key === "S") {
        event.preventDefault();
        toggleReaderChrome();
        return;
      }

      if (event.key === "[") {
        event.preventDefault();
        goToPreviousChapter();
        return;
      }

      if (event.key === "]") {
        event.preventDefault();
        goToNextChapter();
        return;
      }

      if (mode !== "paginated" || !data) {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        const lastPageIndex = Math.max(0, data.chapter.totalPages - 1);

        if (settings.readingDirection === "rtl") {
          if (currentPage >= lastPageIndex) {
            goToNextChapter();
            return;
          }

          setCurrentPage((page) => Math.min(lastPageIndex, page + 1));
        } else {
          if (currentPage <= 0) {
            goToPreviousChapter("last");
            return;
          }

          setCurrentPage((page) => Math.max(0, page - 1));
        }
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();
        const lastPageIndex = Math.max(0, data.chapter.totalPages - 1);

        if (settings.readingDirection === "rtl") {
          if (currentPage <= 0) {
            goToPreviousChapter("last");
            return;
          }

          setCurrentPage((page) => Math.max(0, page - 1));
        } else {
          if (currentPage >= lastPageIndex) {
            goToNextChapter();
            return;
          }

          setCurrentPage((page) => Math.min(lastPageIndex, page + 1));
        }
      }
    }

    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [currentPage, data, mode, settings.readingDirection, chapterUrl]);

  if (loading) {
    return (
      <section className="reader-loading" aria-live="polite">
        <div className="spinner" />
        <p>Extracting chapter panels and metadata...</p>
      </section>
    );
  }

  if (error || !data) {
    return (
      <section className="reader-error">
        <h2>Unable to load this URL</h2>
        <p>{error ?? "Unknown ingestion error."}</p>

        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (draftUrl.trim()) {
              onNavigate(draftUrl.trim());
            }
          }}
          className="error-retry-form"
        >
          <input
            type="url"
            value={draftUrl}
            onChange={(event) => setDraftUrl(event.target.value)}
            placeholder="Paste a chapter URL"
          />
          <button type="submit">Try again</button>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              if (fallbackChapterUrl) {
                onNavigate(fallbackChapterUrl);
                return;
              }

              if (typeof window !== "undefined" && window.history.length > 1) {
                window.history.back();
                return;
              }

              onBackHome();
            }}
          >
            Back
          </button>
          <button type="button" className="ghost" onClick={onBackHome}>
            Back home
          </button>
        </form>
      </section>
    );
  }

  const chapterLabel = data.chapter.title ?? `Chapter ${data.chapter.number}`;
  const pageIndicator =
    mode === "paginated"
      ? `${Math.min(currentPage + 1, data.chapter.totalPages)} / ${data.chapter.totalPages}`
      : `${Math.min(scrollPage + 1, data.chapter.totalPages)} / ${data.chapter.totalPages}`;
  const sidebarVisible = sidebarOpen && !chromeHidden;
  const showMobileChromeToggle =
    !isMobileViewport || (sidebarOpen && !chromeHidden) || mobileSelectorVisible;
  const toolbarHidden = chromeHidden && !topBarPeekVisible;

  const shellStyle = {
    "--reader-bg": settings.backgroundColor,
    "--bg-surface": settings.interfaceColor,
    "--bg-surface-2": withOpacity(settings.interfaceColor, 0.86),
    "--text-primary": settings.textColor,
    "--text-muted": withOpacity(settings.textColor, 0.74),
    "--accent": settings.accentColor,
    "--accent-2": lighten(settings.accentColor, 26)
  } as React.CSSProperties;

  return (
    <div
      className={`reader-shell ${settings.highContrast ? "high-contrast" : ""} ${chromeHidden ? "chrome-hidden" : ""}`}
      style={shellStyle}
      onMouseMove={handleShellMouseMove}
      onMouseLeave={handleShellMouseLeave}
      onTouchStart={handleShellTouchStart}
      onTouchMove={handleShellTouchMove}
      onTouchEnd={handleShellTouchEnd}
      onTouchCancel={handleShellTouchEnd}
    >
      <ReaderToolbar
        hidden={toolbarHidden}
        seriesTitle={data.series.title}
        chapterLabel={chapterLabel}
        pageIndicator={pageIndicator}
        hasPreviousChapter={Boolean(previousChapter)}
        hasNextChapter={Boolean(nextChapter)}
        isFavorited={isFavorited}
        mode={mode}
        onPreviousChapter={() => goToPreviousChapter("first", true)}
        onNextChapter={() => goToNextChapter("first", true)}
        onModeChange={setMode}
        onToggleSettings={toggleSettingsPanel}
        onBackHome={onBackHome}
        onToggleFavorite={toggleFavoriteSeries}
        onAddBookmark={createBookmark}
      />

      <div className={`reader-layout ${sidebarVisible ? "sidebar-open" : "sidebar-closed"}`}>
        <button
          type="button"
          className={`chrome-edge-toggle ${chromeHidden ? "hidden-state" : "visible-state"} ${
            showMobileChromeToggle ? "" : "mobile-hidden"
          }`}
          onClick={toggleReaderChrome}
          aria-label={chromeHidden ? "Show reader UI" : "Hide reader UI"}
        >
          <span aria-hidden="true">{chromeHidden ? "›" : "‹"}</span>
        </button>

        <ChapterSidebar
          isOpen={sidebarOpen && !chromeHidden}
          data={{
            ...data,
            chapterList: visibleChapterList
          }}
          currentChapterUrl={chapterUrl}
          filterQuery={chapterFilter}
          navigationStatus={chapterNavigationStatus}
          suggestions={suggestions}
          bookmarks={bookmarks}
          hasReadChapter={hasReadChapter}
          onFilterQueryChange={setChapterFilter}
          onSelectChapter={handleSidebarChapterSelect}
          onGoBookmark={goToBookmark}
          onEditBookmark={editBookmark}
          onDeleteBookmark={removeBookmark}
          onExportBookmarks={downloadBookmarks}
        />

        <main className="reader-main" aria-label="Reader canvas">
          {mode === "paginated" ? (
            <PaginatedReader
              panelUrls={data.chapter.panelUrls}
              currentPage={currentPage}
              settings={settings}
              onPageChange={setCurrentPage}
              onReadPreviousChapter={() => goToPreviousChapter("last")}
              onReadNextChapter={() => goToNextChapter("first")}
              onLongPanelDetected={handleLongPanelDetected}
              onMobileSelectorVisibilityChange={setMobileSelectorVisible}
            />
          ) : (
            <ScrollReader
              panelUrls={data.chapter.panelUrls}
              initialPageIndex={initialScrollPage}
              settings={settings}
              onProgressChange={setScrollProgress}
              onVisiblePageChange={setScrollPage}
              onReadPreviousChapter={() => goToPreviousChapter("last")}
              onReadNextChapter={() => goToNextChapter("first")}
              onMobileSelectorVisibilityChange={setMobileSelectorVisible}
            />
          )}
        </main>

        <SettingsPanel
          isOpen={settingsOpen && !chromeHidden}
          mode={mode}
          settings={settings}
          panelRef={settingsPanelRef}
          onClose={closeSettingsPanel}
          onChangeSettings={setSettings}
        />
      </div>
    </div>
  );
}
