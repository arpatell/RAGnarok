import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HomeScreen } from "./components/HomeScreen";
import { ReaderPage } from "./components/ReaderPage";
import {
  getFavorites,
  getHistory,
  getReaderSettings,
  removeFavoriteBySeries,
  removeHistorySeries,
  setStorageWarningHandler
} from "./lib/storage";
import type { FavoriteEntry, HistoryEntry } from "./types";

const ROOT_ROUTE_PATH = "/";
const NOT_FOUND_ROUTE_PATH = "/not-found";
const CHAPTER_QUERY_KEY = "chapter";
const PAGE_QUERY_KEY = "page";
const SERIES_QUERY_KEY = "series";
const CHAPTER_TITLE_QUERY_KEY = "chapterTitle";
const FAILED_URL_QUERY_KEY = "url";
const FAILED_REASON_QUERY_KEY = "reason";
const PREVIOUS_CHAPTER_QUERY_KEY = "previous";
const LEGACY_QUERY_KEYS = ["url", "u"] as const;
const NOT_FOUND_ERROR_CODES = new Set(["NOT_CHAPTER", "UNSUPPORTED_DOMAIN", "NOT_PUBLIC"]);

interface ReaderHistoryState {
  chapterUrl?: string;
  failedUrl?: string;
  failedReason?: string;
  previousChapterUrl?: string;
  route?: "home" | "reader" | "not-found";
}

interface AppRouteState {
  kind: "home" | "reader" | "not-found";
  chapterUrl: string;
  failedUrl: string;
  failedReason: string;
  previousChapterUrl: string;
}

interface NotFoundPayload {
  failedUrl: string;
  failedReason: string;
  errorCode?: string;
  fallbackChapterUrl?: string;
}

function findLegacyChapterParam(params: URLSearchParams): string {
  for (const key of LEGACY_QUERY_KEYS) {
    const value = params.get(key)?.trim();
    if (value) {
      return value;
    }
  }

  return "";
}

function normalizePath(pathname: string): string {
  if (!pathname) {
    return ROOT_ROUTE_PATH;
  }

  const trimmed = pathname.replace(/\/+$/, "");
  if (!trimmed) {
    return ROOT_ROUTE_PATH;
  }

  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

function normalizeComparableUrl(url: string): string {
  try {
    const parsed = new URL(url.trim());
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return url.trim();
  }
}

function readRouteStateFromLocation(): AppRouteState {
  if (typeof window === "undefined") {
    return {
      kind: "home",
      chapterUrl: "",
      failedUrl: "",
      failedReason: "",
      previousChapterUrl: ""
    };
  }

  const path = normalizePath(window.location.pathname);
  const params = new URLSearchParams(window.location.search);

  if (path === NOT_FOUND_ROUTE_PATH) {
    return {
      kind: "not-found",
      chapterUrl: "",
      failedUrl: params.get(FAILED_URL_QUERY_KEY)?.trim() ?? "",
      failedReason: params.get(FAILED_REASON_QUERY_KEY)?.trim() ?? "",
      previousChapterUrl: params.get(PREVIOUS_CHAPTER_QUERY_KEY)?.trim() ?? ""
    };
  }

  const queryChapter = params.get(CHAPTER_QUERY_KEY)?.trim() ?? findLegacyChapterParam(params);
  if (queryChapter) {
    return {
      kind: "reader",
      chapterUrl: queryChapter,
      failedUrl: "",
      failedReason: "",
      previousChapterUrl: ""
    };
  }

  const state = window.history.state as ReaderHistoryState | null;
  const stateChapter = typeof state?.chapterUrl === "string" ? state.chapterUrl.trim() : "";
  if (stateChapter) {
    return {
      kind: "reader",
      chapterUrl: stateChapter,
      failedUrl: "",
      failedReason: "",
      previousChapterUrl: ""
    };
  }

  return {
    kind: "home",
    chapterUrl: "",
    failedUrl: "",
    failedReason: "",
    previousChapterUrl: ""
  };
}

function normalizeReaderQueryInRootLocation() {
  if (typeof window === "undefined") {
    return;
  }

  const path = normalizePath(window.location.pathname);
  if (path !== ROOT_ROUTE_PATH) {
    return;
  }

  const params = new URLSearchParams(window.location.search);
  const chapter = params.get(CHAPTER_QUERY_KEY)?.trim();
  const legacy = findLegacyChapterParam(params);
  const normalizedChapter = chapter || legacy;
  const page = params.get(PAGE_QUERY_KEY)?.trim() ?? "";

  params.delete(CHAPTER_QUERY_KEY);
  for (const key of LEGACY_QUERY_KEYS) {
    params.delete(key);
  }
  params.delete(PAGE_QUERY_KEY);
  params.delete(SERIES_QUERY_KEY);
  params.delete(CHAPTER_TITLE_QUERY_KEY);

  if (normalizedChapter) {
    params.set(CHAPTER_QUERY_KEY, normalizedChapter);
    if (/^\d+$/.test(page) && Number.parseInt(page, 10) > 0) {
      params.set(PAGE_QUERY_KEY, page);
    }
  }

  const previousState = (window.history.state as ReaderHistoryState | null) ?? {};
  const nextState: ReaderHistoryState = {
    ...previousState,
    route: normalizedChapter ? "reader" : "home",
    chapterUrl: normalizedChapter || ""
  };

  const query = params.toString();
  const target = `${ROOT_ROUTE_PATH}${query ? `?${query}` : ""}`;
  window.history.replaceState(nextState, "", target);
}

function updateReaderLocationState(url: string, replace = false) {
  const params = new URLSearchParams();

  const trimmed = url.trim();

  if (trimmed) {
    params.set(CHAPTER_QUERY_KEY, trimmed);
    params.set(PAGE_QUERY_KEY, "1");
  }

  const query = params.toString();
  const target = `${ROOT_ROUTE_PATH}${query ? `?${query}` : ""}`;
  const previousState = (window.history.state as ReaderHistoryState | null) ?? {};
  const nextState: ReaderHistoryState = {
    ...previousState,
    route: trimmed ? "reader" : "home",
    chapterUrl: trimmed,
    failedUrl: "",
    failedReason: "",
    previousChapterUrl: ""
  };

  if (replace) {
    window.history.replaceState(nextState, "", target);
  } else {
    window.history.pushState(nextState, "", target);
  }
}

function updateNotFoundLocationState(payload: NotFoundPayload) {
  const params = new URLSearchParams();
  const failedUrl = payload.failedUrl.trim();
  const failedReason = payload.failedReason.trim();
  const fallbackChapterUrl = payload.fallbackChapterUrl?.trim() ?? "";

  if (failedUrl) {
    params.set(FAILED_URL_QUERY_KEY, failedUrl);
  }
  if (failedReason) {
    params.set(FAILED_REASON_QUERY_KEY, failedReason);
  }
  if (fallbackChapterUrl) {
    params.set(PREVIOUS_CHAPTER_QUERY_KEY, fallbackChapterUrl);
  }

  const query = params.toString();
  const target = `${NOT_FOUND_ROUTE_PATH}${query ? `?${query}` : ""}`;
  const previousState = (window.history.state as ReaderHistoryState | null) ?? {};

  if (fallbackChapterUrl) {
    const recoveryParams = new URLSearchParams();
    recoveryParams.set(CHAPTER_QUERY_KEY, fallbackChapterUrl);
    recoveryParams.set(PAGE_QUERY_KEY, "1");
    const recoveryQuery = recoveryParams.toString();
    const recoveryTarget = `${ROOT_ROUTE_PATH}${recoveryQuery ? `?${recoveryQuery}` : ""}`;
    const recoveryState: ReaderHistoryState = {
      ...previousState,
      route: "reader",
      chapterUrl: fallbackChapterUrl,
      failedUrl: "",
      failedReason: "",
      previousChapterUrl: ""
    };
    window.history.replaceState(recoveryState, "", recoveryTarget);
  }

  const stateAfterRecovery = (window.history.state as ReaderHistoryState | null) ?? previousState;
  const nextState: ReaderHistoryState = {
    ...stateAfterRecovery,
    route: "not-found",
    chapterUrl: "",
    failedUrl,
    failedReason,
    previousChapterUrl: fallbackChapterUrl
  };

  window.history.pushState(nextState, "", target);
}

function goToBrowserPreviousOrHome() {
  if (typeof window === "undefined") {
    return;
  }

  if (window.history.length > 1) {
    window.history.back();
    return;
  }

  const nextState: ReaderHistoryState = {
    route: "home",
    chapterUrl: "",
    failedUrl: "",
    failedReason: "",
    previousChapterUrl: ""
  };
  window.history.replaceState(nextState, "", ROOT_ROUTE_PATH);
}

function resolveNotFoundReason(route: AppRouteState): string {
  const message = route.failedReason.trim();
  if (message) {
    return message;
  }
  return "We couldn't locate a readable chapter for that URL.";
}

function renderFailedUrl(route: AppRouteState): string {
  return route.failedUrl.trim();
}

export default function App() {
  const [route, setRoute] = useState<AppRouteState>(readRouteStateFromLocation());
  const [historyEntries, setHistoryEntries] = useState<HistoryEntry[]>(() => getHistory());
  const [favoriteEntries, setFavoriteEntries] = useState<FavoriteEntry[]>(() => getFavorites());
  const [toast, setToast] = useState<string | null>(null);
  const [notFoundUrlDraft, setNotFoundUrlDraft] = useState("");
  const pendingPreviousChapterRef = useRef("");

  useEffect(() => {
    normalizeReaderQueryInRootLocation();
    setRoute(readRouteStateFromLocation());
  }, []);

  useEffect(() => {
    const popStateHandler = () => setRoute(readRouteStateFromLocation());
    window.addEventListener("popstate", popStateHandler);
    return () => window.removeEventListener("popstate", popStateHandler);
  }, []);

  useEffect(() => {
    setStorageWarningHandler((message) => setToast(message));
  }, []);

  const navigateTo = useCallback((url: string) => {
    const trimmed = url.trim();
    if (!trimmed) {
      return;
    }
    const currentReaderChapter = route.kind === "reader" ? route.chapterUrl.trim() : "";
    pendingPreviousChapterRef.current =
      currentReaderChapter && normalizeComparableUrl(currentReaderChapter) !== normalizeComparableUrl(trimmed)
        ? currentReaderChapter
        : "";

    const behavior = getReaderSettings().browserHistoryBehavior;
    const mustNormalizePath =
      typeof window !== "undefined" && normalizePath(window.location.pathname) !== ROOT_ROUTE_PATH;
    const isCurrentlyReader = route.kind === "reader" && route.chapterUrl.trim().length > 0;
    if (behavior !== "dont-touch" || mustNormalizePath) {
      const shouldReplace = behavior === "title-only" && !mustNormalizePath && isCurrentlyReader;
      updateReaderLocationState(trimmed, shouldReplace);
    }

    setRoute({
      kind: "reader",
      chapterUrl: trimmed,
      failedUrl: "",
      failedReason: "",
      previousChapterUrl: ""
    });
  }, [route.chapterUrl, route.kind]);

  const backHome = useCallback(() => {
    updateReaderLocationState("", true);
    setRoute({
      kind: "home",
      chapterUrl: "",
      failedUrl: "",
      failedReason: "",
      previousChapterUrl: ""
    });
  }, []);

  const showNotFoundRoute = useCallback((payload: NotFoundPayload) => {
    const errorCode = payload.errorCode?.trim().toUpperCase() ?? "";
    const message = payload.failedReason.trim();
    const messageLooksNotFound = /not\s+found|unable to find|could not find|no readable chapter/i.test(message);
    const shouldRouteToNotFound = messageLooksNotFound || NOT_FOUND_ERROR_CODES.has(errorCode);

    if (!shouldRouteToNotFound) {
      return false;
    }

    const navigationPrevious = pendingPreviousChapterRef.current.trim();
    pendingPreviousChapterRef.current = "";
    const payloadFallback = payload.fallbackChapterUrl?.trim() ?? "";
    const trustedFallback =
      navigationPrevious &&
      payloadFallback &&
      normalizeComparableUrl(navigationPrevious) === normalizeComparableUrl(payloadFallback)
        ? payloadFallback
        : "";

    const normalizedPayload: NotFoundPayload = {
      ...payload,
      fallbackChapterUrl: trustedFallback
    };

    updateNotFoundLocationState(normalizedPayload);
    setRoute({
      kind: "not-found",
      chapterUrl: "",
      failedUrl: payload.failedUrl.trim(),
      failedReason: message,
      previousChapterUrl: trustedFallback
    });

    return true;
  }, []);

  const refreshLibrary = useCallback(() => {
    setHistoryEntries(getHistory());
    setFavoriteEntries(getFavorites());
  }, []);

  const removeRecent = useCallback((seriesTitle: string) => {
    removeHistorySeries(seriesTitle);
    setHistoryEntries(getHistory());
  }, []);

  const removeFavorite = useCallback((seriesTitle: string) => {
    removeFavoriteBySeries(seriesTitle);
    setFavoriteEntries(getFavorites());
  }, []);

  const hasActiveChapter = useMemo(() => route.kind === "reader" && route.chapterUrl.length > 0, [route]);
  const showNotFound = route.kind === "not-found";
  const failedUrl = renderFailedUrl(route);
  const notFoundReason = resolveNotFoundReason(route);
  const previousChapterUrl = route.previousChapterUrl.trim();
  const isExternalImageBlockedReason = /does not expose images externally/i.test(notFoundReason);
  const showBackToPreviousChapter =
    previousChapterUrl.length > 0 &&
    !isExternalImageBlockedReason &&
    normalizeComparableUrl(previousChapterUrl) !== normalizeComparableUrl(failedUrl);

  useEffect(() => {
    if (showNotFound) {
      setNotFoundUrlDraft(failedUrl);
    }
  }, [failedUrl, showNotFound]);

  return (
    <>
      {hasActiveChapter ? (
        <ReaderPage
          chapterUrl={route.chapterUrl}
          onNavigate={navigateTo}
          onBackHome={backHome}
          onNavigateNotFound={showNotFoundRoute}
          onHistoryChanged={refreshLibrary}
        />
      ) : showNotFound ? (
        <section className="reader-error">
          <h2>Chapter not found</h2>
          <p>{notFoundReason}</p>
          <form
            className="error-retry-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (notFoundUrlDraft.trim()) {
                navigateTo(notFoundUrlDraft.trim());
              }
            }}
          >
            <input
              type="url"
              value={notFoundUrlDraft}
              onChange={(event) => setNotFoundUrlDraft(event.target.value)}
              placeholder="Paste a chapter URL"
            />
            <button
              type="submit"
            >
              Try again
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                if (showBackToPreviousChapter) {
                  navigateTo(previousChapterUrl);
                  return;
                }
                goToBrowserPreviousOrHome();
              }}
            >
              {showBackToPreviousChapter ? "Back to previous chapter" : "Back"}
            </button>
            <button type="button" className="ghost" onClick={backHome}>
              Back home
            </button>
          </form>
        </section>
      ) : (
        <HomeScreen
          initialUrl=""
          history={historyEntries}
          favorites={favoriteEntries}
          onSubmitUrl={navigateTo}
          onRemoveRecent={removeRecent}
          onRemoveFavorite={removeFavorite}
        />
      )}

      {toast ? (
        <div className="toast" role="status" aria-live="polite">
          <span>{toast}</span>
          <button type="button" className="ghost" onClick={() => setToast(null)}>
            Dismiss
          </button>
        </div>
      ) : null}
    </>
  );
}
