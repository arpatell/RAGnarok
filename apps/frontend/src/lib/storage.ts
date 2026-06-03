import {
  createDefaultSettings,
  type Bookmark,
  type FavoriteEntry,
  type HistoryEntry,
  type ReaderMode,
  type ReaderProgress,
  type ReaderSettings
} from "../types";

const memoryFallback = new Map<string, string>();
let storageEnabled = true;
let warnedUnavailable = false;
let warningHandler: ((message: string) => void) | null = null;

const STORAGE_KEYS = {
  settings: "panelflow:settings",
  history: "panelflow:history",
  favorites: "panelflow:favorites",
  bookmarks: "panelflow:bookmarks",
  progressMap: "panelflow:progress",
  readSet: "panelflow:read-set"
};

function emitWarningOnce() {
  if (warnedUnavailable) {
    return;
  }

  warnedUnavailable = true;
  if (warningHandler) {
    warningHandler("Storage is unavailable or full. Reading works, but persistence is disabled.");
  }
}

function detectStorageAvailability(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  try {
    const testKey = "panelflow:test";
    window.localStorage.setItem(testKey, "1");
    window.localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

storageEnabled = detectStorageAvailability();
if (!storageEnabled) {
  emitWarningOnce();
}

function readRaw(key: string): string | null {
  if (storageEnabled) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      storageEnabled = false;
      emitWarningOnce();
    }
  }

  return memoryFallback.get(key) ?? null;
}

function writeRaw(key: string, value: string): boolean {
  if (storageEnabled) {
    try {
      window.localStorage.setItem(key, value);
      return true;
    } catch {
      storageEnabled = false;
      emitWarningOnce();
    }
  }

  memoryFallback.set(key, value);
  return false;
}

function readJson<T>(key: string, fallback: T): T {
  const raw = readRaw(key);
  if (!raw) {
    return fallback;
  }

  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJson<T>(key: string, value: T): boolean {
  return writeRaw(key, JSON.stringify(value));
}

function normalizeUrl(url: string): string {
  try {
    const parsed = new URL(url);
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return url;
  }
}

function normalizeSeriesKey(seriesTitle: string): string {
  return seriesTitle.trim().toLowerCase();
}

function looksLikeChapterLabel(value: string): boolean {
  return /^(?:chapter|chap|ch\.?|episode|ep\.?)\s*[-#: ]?\d+(?:\.\d+)?\b/i.test(value.trim());
}

function stripChapterPrefix(value: string): string {
  const cleaned = value.trim();
  const withoutPrefix = cleaned.replace(
    /^(?:chapter|chap|ch\.?|episode|ep\.?)\s*[-#: ]?\d+(?:\.\d+)?\s*[-–—:|]\s*/i,
    ""
  ).trim();

  return withoutPrefix || cleaned;
}

function normalizeHistoryEntry(entry: HistoryEntry): HistoryEntry {
  const seriesTitle = entry.seriesTitle.trim();
  const chapterTitle = entry.chapterTitle.trim();
  const normalizedSeriesTitle =
    looksLikeChapterLabel(seriesTitle) && seriesTitle === chapterTitle
      ? stripChapterPrefix(seriesTitle)
      : seriesTitle;

  return {
    ...entry,
    seriesTitle: normalizedSeriesTitle || seriesTitle || chapterTitle,
    chapterTitle: chapterTitle || seriesTitle
  };
}

function normalizeHistorySeriesUrlKey(chapterUrl: string): string | null {
  try {
    const parsed = new URL(chapterUrl);
    const host = parsed.hostname.toLowerCase().replace(/^(www\d*|www|m|mobile|read)\./i, "");
    const segments = parsed.pathname.split("/").filter(Boolean);
    if (segments.length === 0) {
      return host;
    }

    const chapterIndex = segments.findIndex((segment) =>
      /^(?:c\d+(?:\.\d+)?|chapter[-_ ]?\d+(?:\.\d+)?|chap[-_ ]?\d+(?:\.\d+)?|ch[-_ ]?\d+(?:\.\d+)?|episode[-_ ]?\d+(?:\.\d+)?|ep[-_ ]?\d+(?:\.\d+)?)$/i.test(segment)
    );

    const firstChapterKeywordIndex = segments.findIndex((segment) =>
      /^(?:chapter|chapters|episode|episodes|read|viewer)$/i.test(segment)
    );

    let seriesSegments = segments;
    if (chapterIndex > 0) {
      seriesSegments = segments.slice(0, chapterIndex);
    } else if (firstChapterKeywordIndex > 0) {
      seriesSegments = segments.slice(0, firstChapterKeywordIndex);
    } else {
      const last = segments[segments.length - 1] ?? "";
      const strippedLast = last
        .replace(/(?:[-_])?(?:chapter|chap|ch|episode|ep|c)[-_ ]?\d+(?:\.\d+)?(?:[-_].*)?$/i, "")
        .replace(/[-_]+$/g, "");

      if (strippedLast && strippedLast !== last) {
        seriesSegments = [...segments.slice(0, -1), strippedLast];
      } else if (segments.length > 1 && /\d/.test(last) && /(?:chapter|chapters|episode|episodes|read|viewer)/i.test(parsed.pathname)) {
        seriesSegments = segments.slice(0, -1);
      }
    }

    const normalizedPath = seriesSegments.join("/").toLowerCase();
    return normalizedPath ? `${host}/${normalizedPath}` : host;
  } catch {
    return null;
  }
}

function historySeriesKey(entry: HistoryEntry): string {
  return normalizeHistorySeriesUrlKey(entry.chapterUrl) ?? normalizeSeriesKey(normalizeHistoryEntry(entry).seriesTitle);
}

function dedupeHistoryBySeries(history: HistoryEntry[]): HistoryEntry[] {
  const deduped: HistoryEntry[] = [];
  const seen = new Set<string>();

  for (const rawEntry of history) {
    const entry = normalizeHistoryEntry(rawEntry);
    const key = historySeriesKey(entry);
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    deduped.push(entry);
  }

  return deduped;
}

export function setStorageWarningHandler(handler: (message: string) => void) {
  warningHandler = handler;
  if (!storageEnabled) {
    emitWarningOnce();
  }
}

export function getReaderSettings(): ReaderSettings {
  const defaults = createDefaultSettings();
  const saved = readJson<Partial<ReaderSettings>>(STORAGE_KEYS.settings, {});

  const normalizedPreloadDepth =
    typeof saved.preloadDepth === "number"
      ? Math.min(9, Math.max(1, Math.round(saved.preloadDepth)))
      : defaults.preloadDepth;

  const normalizedVerticalArrowStep =
    typeof saved.verticalArrowStepPx === "number"
      ? Math.min(50, Math.max(5, Math.round(saved.verticalArrowStepPx / 5) * 5))
      : defaults.verticalArrowStepPx;

  const normalizedPanelGap =
    typeof saved.panelGap === "number"
      ? Math.min(32, Math.max(0, Math.round(saved.panelGap)))
      : defaults.panelGap;

  return {
    ...defaults,
    ...saved,
    fitMode: saved.fitMode ?? defaults.fitMode,
    readingDirection: saved.readingDirection ?? defaults.readingDirection,
    spreadMode: saved.spreadMode === "double" ? "double" : "single",
    preloadDepth: normalizedPreloadDepth,
    verticalArrowStepPx: normalizedVerticalArrowStep,
    panelGap: normalizedPanelGap,
    turnPagesByClicking:
      typeof saved.turnPagesByClicking === "boolean"
        ? saved.turnPagesByClicking
        : defaults.turnPagesByClicking,
    turnPagesWithArrowsInVerticalView:
      typeof saved.turnPagesWithArrowsInVerticalView === "boolean"
        ? saved.turnPagesWithArrowsInVerticalView
        : defaults.turnPagesWithArrowsInVerticalView,
    enableSwipeGestures:
      typeof saved.enableSwipeGestures === "boolean"
        ? saved.enableSwipeGestures
        : defaults.enableSwipeGestures,
    pinVerticalPageSelector:
      typeof saved.pinVerticalPageSelector === "boolean"
        ? saved.pinVerticalPageSelector
        : defaults.pinVerticalPageSelector,
    resetPageScrollAfterFlip:
      typeof saved.resetPageScrollAfterFlip === "boolean"
        ? saved.resetPageScrollAfterFlip
        : defaults.resetPageScrollAfterFlip,
    transitionStyle:
      saved.transitionStyle === "fade" || saved.transitionStyle === "slide" || saved.transitionStyle === "none"
        ? saved.transitionStyle
        : defaults.transitionStyle,
    browserHistoryBehavior: saved.browserHistoryBehavior ?? defaults.browserHistoryBehavior
  };
}

export function saveReaderSettings(settings: ReaderSettings): boolean {
  return writeJson(STORAGE_KEYS.settings, settings);
}

export function getSavedMode(adapterId: string): ReaderMode | null {
  const mode = readRaw(`panelflow:mode:${adapterId}`);
  return mode === "paginated" || mode === "scroll" ? mode : null;
}

export function saveMode(adapterId: string, mode: ReaderMode): boolean {
  return writeRaw(`panelflow:mode:${adapterId}`, mode);
}

export function getHistory(): HistoryEntry[] {
  return dedupeHistoryBySeries(readJson<HistoryEntry[]>(STORAGE_KEYS.history, []));
}

export function pushHistory(entry: HistoryEntry): boolean {
  const normalizedEntry = normalizeHistoryEntry(entry);
  const targetSeriesKey = historySeriesKey(normalizedEntry);
  const history = readJson<HistoryEntry[]>(STORAGE_KEYS.history, []);
  const deduped = dedupeHistoryBySeries(history).filter((item) => historySeriesKey(item) !== targetSeriesKey);
  deduped.unshift(normalizedEntry);
  return writeJson(STORAGE_KEYS.history, deduped.slice(0, 50));
}

export function removeHistorySeries(seriesTitle: string): boolean {
  const target = normalizeSeriesKey(seriesTitle);
  const filtered = getHistory().filter((entry) => {
    const normalizedEntry = normalizeHistoryEntry(entry);
    return (
      normalizeSeriesKey(normalizedEntry.seriesTitle) !== target &&
      normalizeSeriesKey(entry.seriesTitle) !== target
    );
  });
  return writeJson(STORAGE_KEYS.history, filtered);
}

export function getFavorites(): FavoriteEntry[] {
  return readJson<FavoriteEntry[]>(STORAGE_KEYS.favorites, []);
}

export function saveFavorites(favorites: FavoriteEntry[]): boolean {
  return writeJson(STORAGE_KEYS.favorites, favorites);
}

export function upsertFavorite(entry: FavoriteEntry): boolean {
  const favorites = getFavorites();
  const target = normalizeSeriesKey(entry.seriesTitle);
  const deduped = favorites.filter((item) => normalizeSeriesKey(item.seriesTitle) !== target);
  deduped.unshift(entry);
  return saveFavorites(deduped.slice(0, 100));
}

export function removeFavoriteBySeries(seriesTitle: string): boolean {
  const target = normalizeSeriesKey(seriesTitle);
  const filtered = getFavorites().filter((entry) => normalizeSeriesKey(entry.seriesTitle) !== target);
  return saveFavorites(filtered);
}

export function isSeriesFavorited(seriesTitle: string): boolean {
  const target = normalizeSeriesKey(seriesTitle);
  return getFavorites().some((entry) => normalizeSeriesKey(entry.seriesTitle) === target);
}

export function getProgressMap(): Record<string, ReaderProgress> {
  return readJson<Record<string, ReaderProgress>>(STORAGE_KEYS.progressMap, {});
}

export function getProgress(chapterUrl: string): ReaderProgress | null {
  const map = getProgressMap();
  return map[normalizeUrl(chapterUrl)] ?? null;
}

export function saveProgress(progress: ReaderProgress): boolean {
  const normalized = normalizeUrl(progress.chapterUrl);
  const nextMap: Record<string, ReaderProgress> = {
    [normalized]: {
      ...progress,
      chapterUrl: normalized
    }
  };

  return writeJson(STORAGE_KEYS.progressMap, nextMap);
}

export function clearProgress(chapterUrl: string): boolean {
  const map = getProgressMap();
  const normalized = normalizeUrl(chapterUrl);

  if (!(normalized in map)) {
    return true;
  }

  delete map[normalized];
  return writeJson(STORAGE_KEYS.progressMap, map);
}

export function getBookmarks(): Bookmark[] {
  return readJson<Bookmark[]>(STORAGE_KEYS.bookmarks, []);
}

export function saveBookmarks(bookmarks: Bookmark[]): boolean {
  return writeJson(STORAGE_KEYS.bookmarks, bookmarks);
}

export function addBookmark(bookmark: Bookmark): boolean {
  const bookmarks = getBookmarks();
  bookmarks.unshift(bookmark);
  return saveBookmarks(bookmarks);
}

export function updateBookmarkLabel(bookmarkId: string, label: string): boolean {
  const bookmarks = getBookmarks().map((bookmark) =>
    bookmark.id === bookmarkId ? { ...bookmark, label } : bookmark
  );
  return saveBookmarks(bookmarks);
}

export function deleteBookmark(bookmarkId: string): boolean {
  const bookmarks = getBookmarks().filter((bookmark) => bookmark.id !== bookmarkId);
  return saveBookmarks(bookmarks);
}

export function exportBookmarks(): string {
  return JSON.stringify(getBookmarks(), null, 2);
}

export function getReadSet(): string[] {
  return readJson<string[]>(STORAGE_KEYS.readSet, []);
}

export function markChapterRead(chapterUrl: string): boolean {
  const normalized = normalizeUrl(chapterUrl);
  const readSet = new Set(getReadSet());
  readSet.add(normalized);
  return writeJson(STORAGE_KEYS.readSet, Array.from(readSet));
}

export function hasReadChapter(chapterUrl: string): boolean {
  return new Set(getReadSet()).has(normalizeUrl(chapterUrl));
}

export function isStorageEnabled(): boolean {
  return storageEnabled;
}
