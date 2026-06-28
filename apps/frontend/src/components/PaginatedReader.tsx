import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { relayImageUrl } from "../lib/api";
import { isImageLoaded, isImageLoading, markImageLoaded, preloadImageOnce } from "../lib/imagePreload";
import type { ReaderSettings } from "../types";

interface PaginatedReaderProps {
  panelUrls: string[];
  currentPage: number;
  settings: ReaderSettings;
  onPageChange: (nextPage: number) => void;
  onReadNextChapter: () => void;
  onReadPreviousChapter: () => void;
  onMobileSelectorVisibilityChange?: (visible: boolean) => void;
  onReaderAtTopChange?: (atTop: boolean) => void;
  onTopScrollAttempt?: () => void;
}

function normalizePanelStyle(settings: ReaderSettings): CSSProperties {
  const zoomFactor = settings.zoomPercent / 100;
  const zoomPercent = Math.min(200, Math.max(50, settings.zoomPercent));
  const heightOffsetPx = Math.round(120 * zoomFactor);

  switch (settings.fitMode) {
    case "height-fit":
      return {
        height: `calc(${zoomPercent}vh - ${heightOffsetPx}px)`,
        width: "auto",
        maxWidth: "none",
        maxHeight: "none",
        flexShrink: 0
      };
    case "original":
      return {
        width: "auto",
        height: "auto",
        maxWidth: "none",
        maxHeight: "none",
        flexShrink: 0,
        transform: `scale(${zoomFactor})`,
        transformOrigin: "top center"
      };
    case "custom":
      return {
        width: `${zoomPercent}%`,
        height: "auto",
        maxWidth: "none",
        maxHeight: "none",
        flexShrink: 0
      };
    case "width-fit":
    default:
      return {
        width: `${Math.max(60, zoomPercent)}%`,
        height: "auto",
        maxWidth: "none",
        maxHeight: "none",
        flexShrink: 0
      };
  }
}

function buildBidirectionalPreloadIndices(
  panelUrls: string[],
  centerIndex: number,
  targetCount: number,
  excludeIndices: Set<number>
): number[] {
  const totalPages = panelUrls.length;
  const lastPageIndex = Math.max(totalPages - 1, 0);
  const safeCenter = Math.min(Math.max(centerIndex, 0), lastPageIndex);
  const safeTargetCount = Math.max(1, targetCount);
  const indices: number[] = [];
  let nextOutOfBounds = false;
  let previousOutOfBounds = false;

  for (let offset = 1; indices.length < safeTargetCount && !(nextOutOfBounds && previousOutOfBounds); offset += 1) {
    const next = safeCenter + offset;
    const previous = safeCenter - offset;

    if (next > lastPageIndex) {
      nextOutOfBounds = true;
    } else {
      const nextUrl = panelUrls[next];
      if (nextUrl && !excludeIndices.has(next) && !isImageLoaded(nextUrl) && !isImageLoading(nextUrl)) {
        indices.push(next);
      }
    }

    if (indices.length >= safeTargetCount) {
      break;
    }

    if (previous < 0) {
      previousOutOfBounds = true;
    } else {
      const previousUrl = panelUrls[previous];
      if (previousUrl && !excludeIndices.has(previous) && !isImageLoaded(previousUrl) && !isImageLoading(previousUrl)) {
        indices.push(previous);
      }
    }
  }

  return indices;
}

function logImageLoadTiming(label: string, panelUrl: string, startedAt: number) {
  if (!import.meta.env.DEV) {
    return;
  }

  // eslint-disable-next-line no-console
  console.info("[RAGnarok] image loaded", {
    label,
    ms: Math.round(performance.now() - startedAt),
    url: panelUrl
  });
}

export function PaginatedReader({
  panelUrls,
  currentPage,
  settings,
  onPageChange,
  onReadNextChapter,
  onReadPreviousChapter,
  onMobileSelectorVisibilityChange,
  onReaderAtTopChange,
  onTopScrollAttempt
}: PaginatedReaderProps) {
  const [failed, setFailed] = useState<Set<number>>(new Set());
  const [retryVersion, setRetryVersion] = useState<Record<number, number>>({});
  const [retryAttempts, setRetryAttempts] = useState<Record<number, number>>({});
  const [hoverZone, setHoverZone] = useState<"left" | "right" | null>(null);
  const [tooltipZone, setTooltipZone] = useState<"left" | "right" | null>(null);
  const [selectorVisible, setSelectorVisible] = useState(false);
  const touchStartX = useRef<number | null>(null);
  const touchStartY = useRef<number | null>(null);
  const touchStartBottomRef = useRef(false);
  const touchSelectorOpenRef = useRef(false);
  const suppressNextClickRef = useRef(false);
  const retryTimersRef = useRef<Map<number, number>>(new Map());
  const retryAttemptsRef = useRef<Map<number, number>>(new Map());
  const tooltipTimerRef = useRef<number | null>(null);
  const selectorIdleTimerRef = useRef<number | null>(null);
  const hasShownEdgeHintRef = useRef(false);

  const totalPages = panelUrls.length;
  const lastPageIndex = Math.max(totalPages - 1, 0);
  const safeCurrentPage = Math.min(Math.max(currentPage, 0), lastPageIndex);
  const forwardZone = settings.readingDirection === "rtl" ? "left" : "right";
  const selectorPinned = settings.pinVerticalPageSelector && !(typeof window !== "undefined" && window.innerWidth <= 1080);
  const showSelector = selectorPinned || selectorVisible;

  const visibleIndices = useMemo(() => {
    if (totalPages === 0) {
      return [];
    }

    const first = safeCurrentPage;
    const second = safeCurrentPage + 1;
    const canUseSecond = second < totalPages;

    const single = settings.spreadMode !== "double" || !canUseSecond;

    return single ? [first] : [first, second];
  }, [safeCurrentPage, settings.spreadMode, totalPages]);

  const orderedIndices = useMemo(() => {
    if (visibleIndices.length < 2) {
      return visibleIndices;
    }

    if (settings.readingDirection === "rtl") {
      return [...visibleIndices].reverse();
    }

    return visibleIndices;
  }, [settings.readingDirection, visibleIndices]);

  const selectorIndices = useMemo(() => {
    if (totalPages <= 24) {
      return Array.from({ length: totalPages }, (_, index) => index);
    }

    const selected = new Set<number>([0, totalPages - 1, safeCurrentPage]);
    const localStart = Math.max(0, safeCurrentPage - 5);
    const localEnd = Math.min(totalPages - 1, safeCurrentPage + 5);

    for (let index = localStart; index <= localEnd; index += 1) {
      selected.add(index);
    }

    const step = Math.max(1, Math.floor(totalPages / 12));
    for (let index = 0; index < totalPages; index += step) {
      selected.add(index);
    }

    return Array.from(selected).sort((a, b) => a - b);
  }, [safeCurrentPage, totalPages]);

  const panelStyle = useMemo(() => normalizePanelStyle(settings), [settings]);

  const pageStep = visibleIndices.length > 1 ? 2 : 1;

  function isMobileViewport(): boolean {
    return typeof window !== "undefined" && window.innerWidth <= 1080;
  }

  function clearSelectorIdleTimer() {
    if (selectorIdleTimerRef.current !== null) {
      window.clearTimeout(selectorIdleTimerRef.current);
      selectorIdleTimerRef.current = null;
    }
  }

  function bumpSelectorInactivityTimeout() {
    if (selectorPinned || !selectorVisible || !isMobileViewport()) {
      return;
    }

    clearSelectorIdleTimer();
    selectorIdleTimerRef.current = window.setTimeout(() => {
      setSelectorVisible(false);
      selectorIdleTimerRef.current = null;
    }, 4000);
  }

  function handlePageFlip(nextPage: number) {
    const scrollX = typeof window !== "undefined" ? window.scrollX : 0;
    const scrollY = typeof window !== "undefined" ? window.scrollY : 0;
    onPageChange(nextPage);
    if (typeof window !== "undefined") {
      window.requestAnimationFrame(() => {
        window.scrollTo({ left: scrollX, top: scrollY, behavior: "auto" });
      });
    }
  }

  function goNext() {
    if (safeCurrentPage + pageStep >= totalPages) {
      onReadNextChapter();
      return;
    }

    handlePageFlip(safeCurrentPage + pageStep);
  }

  function goPrevious() {
    if (safeCurrentPage <= 0) {
      onReadPreviousChapter();
      return;
    }

    handlePageFlip(Math.max(0, safeCurrentPage - pageStep));
  }

  function getZoneForX(width: number, x: number): "left" | "right" | null {
    if (x < width / 3) {
      return "left";
    }

    if (x > (width * 2) / 3) {
      return "right";
    }

    return null;
  }

  function handleZoneNavigation(zone: "left" | "right" | null) {
    if (!zone) {
      return;
    }

    if (zone === forwardZone) {
      goNext();
    } else {
      goPrevious();
    }
  }

  function handleClick(event: React.MouseEvent<HTMLDivElement>) {
    if (suppressNextClickRef.current) {
      suppressNextClickRef.current = false;
      event.preventDefault();
      return;
    }

    if (!settings.turnPagesByClicking) {
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const width = bounds.width;
    const x = event.clientX - bounds.left;
    handleZoneNavigation(getZoneForX(width, x));
  }

  function handleMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    if (typeof window !== "undefined" && window.innerWidth <= 1080) {
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const width = bounds.width;
    const x = event.clientX - bounds.left;
    const zone = getZoneForX(width, x);

    if (!selectorPinned) {
      setSelectorVisible(x <= 112);
    }

    setHoverZone(zone);

    if (!zone) {
      return;
    }

    if (hasShownEdgeHintRef.current) {
      return;
    }

    hasShownEdgeHintRef.current = true;
    setTooltipZone(zone);

    if (tooltipTimerRef.current !== null) {
      window.clearTimeout(tooltipTimerRef.current);
    }

    tooltipTimerRef.current = window.setTimeout(() => {
      setTooltipZone(null);
      tooltipTimerRef.current = null;
    }, 3000);
  }

  function handleMouseLeave() {
    setHoverZone(null);
    if (!selectorPinned) {
      setSelectorVisible(false);
    }
  }

  function handleTouchStart(event: React.TouchEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement | null;
    if (target?.closest(".vertical-page-selector")) {
      touchStartX.current = null;
      touchStartY.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      return;
    }

    const touch = event.changedTouches[0];
    const bounds = event.currentTarget.getBoundingClientRect();
    const touchX = touch ? touch.clientX - bounds.left : null;
    touchStartX.current = touchX;
    touchStartY.current = touch?.clientY ?? null;
    touchStartBottomRef.current = Boolean(touch && bounds.bottom - touch.clientY <= 34);
    touchSelectorOpenRef.current = false;
  }

  function handleTouchMove(event: React.TouchEvent<HTMLDivElement>) {
    if (touchStartX.current === null || touchStartY.current === null) {
      return;
    }

    const touch = event.changedTouches[0];
    if (!touch) {
      return;
    }

    if (typeof window !== "undefined" && window.innerWidth <= 1080 && window.scrollY <= 0) {
      const pullDown = touch.clientY - touchStartY.current;
      const horizontalMovement = Math.abs(touch.clientX - touchStartX.current);
      if (pullDown > 42 && horizontalMovement < 56) {
        onTopScrollAttempt?.();
      }
    }

    if (!settings.enableSwipeGestures) {
      return;
    }

    if (typeof window === "undefined" || window.innerWidth > 1080 || selectorPinned || !touchStartBottomRef.current) {
      return;
    }

    const deltaX = Math.abs(touch.clientX - touchStartX.current);
    const deltaY = touch.clientY - touchStartY.current;

    if (deltaY < -42 && deltaX < 52) {
      touchSelectorOpenRef.current = true;
      suppressNextClickRef.current = true;
      setSelectorVisible(true);
      bumpSelectorInactivityTimeout();
    }
  }

  function handleTouchEnd(event: React.TouchEvent<HTMLDivElement>) {
    if (touchStartX.current === null) {
      touchStartX.current = null;
      touchStartY.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      return;
    }

    if (touchSelectorOpenRef.current) {
      touchStartX.current = null;
      touchStartY.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const endX = event.changedTouches[0]
      ? event.changedTouches[0].clientX - bounds.left
      : touchStartX.current;
    const delta = endX - touchStartX.current;
    const threshold = 50;

    touchStartX.current = null;
    touchStartY.current = null;
    touchStartBottomRef.current = false;

    if (Math.abs(delta) < threshold) {
      return;
    }

    if (delta < 0) {
      handleZoneNavigation(forwardZone);
    } else {
      handleZoneNavigation(forwardZone === "left" ? "right" : "left");
    }
  }

  function clearRetryState(index: number) {
    const timer = retryTimersRef.current.get(index);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      retryTimersRef.current.delete(index);
    }

    retryAttemptsRef.current.delete(index);
    setRetryAttempts((prev) => {
      if (!(index in prev)) {
        return prev;
      }

      const next = { ...prev };
      delete next[index];
      return next;
    });

    setFailed((prev) => {
      if (!prev.has(index)) {
        return prev;
      }

      const next = new Set(prev);
      next.delete(index);
      return next;
    });
  }

  function scheduleRetry(index: number) {
    if (retryTimersRef.current.has(index)) {
      return;
    }

    const nextAttempt = (retryAttemptsRef.current.get(index) ?? 0) + 1;
    retryAttemptsRef.current.set(index, nextAttempt);
    setRetryAttempts((prev) => ({ ...prev, [index]: nextAttempt }));

    const retryDelayMs = Math.min(8_000, 300 * 2 ** Math.min(nextAttempt, 6) + Math.floor(Math.random() * 200));
    const timer = window.setTimeout(() => {
      retryTimersRef.current.delete(index);
      setFailed((prev) => {
        const next = new Set(prev);
        next.delete(index);
        return next;
      });
      setRetryVersion((prev) => ({
        ...prev,
        [index]: (prev[index] ?? 0) + 1
      }));
    }, retryDelayMs);

    retryTimersRef.current.set(index, timer);
  }

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      const preloadIndices = buildBidirectionalPreloadIndices(
        panelUrls,
        safeCurrentPage,
        settings.preloadDepth,
        new Set(visibleIndices)
      );

      for (const targetIndex of preloadIndices) {
        const panel = panelUrls[targetIndex];
        if (!panel) {
          continue;
        }

        void preloadImageOnce(relayImageUrl(panel), (startedAt) =>
          logImageLoadTiming(`paginated-preload:${targetIndex + 1}`, panel, startedAt)
        );
      }
    }, 120);

    return () => window.clearTimeout(timeout);
  }, [panelUrls, safeCurrentPage, settings.preloadDepth, visibleIndices]);

  useEffect(() => {
    setFailed(new Set());
    setRetryVersion({});
    setRetryAttempts({});
    setSelectorVisible(selectorPinned);
    setTooltipZone(null);
    hasShownEdgeHintRef.current = false;

    for (const timer of retryTimersRef.current.values()) {
      window.clearTimeout(timer);
    }
    retryTimersRef.current.clear();
    retryAttemptsRef.current.clear();
  }, [panelUrls, selectorPinned]);

  useEffect(() => {
    setSelectorVisible(selectorPinned);
  }, [selectorPinned]);

  useEffect(() => {
    if (!onMobileSelectorVisibilityChange) {
      return;
    }

    onMobileSelectorVisibilityChange(showSelector && isMobileViewport());
  }, [onMobileSelectorVisibilityChange, showSelector]);

  useEffect(() => {
    if (!onMobileSelectorVisibilityChange) {
      return;
    }

    if (!settings.enableSwipeGestures) {
      touchStartX.current = null;
      touchStartY.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      return;
    }

    return () => onMobileSelectorVisibilityChange(false);
  }, [onMobileSelectorVisibilityChange]);

  useEffect(() => {
    if (!onReaderAtTopChange || typeof window === "undefined") {
      return;
    }

    const report = () => onReaderAtTopChange(window.scrollY <= 0);
    report();
    window.addEventListener("scroll", report, { passive: true });
    return () => window.removeEventListener("scroll", report);
  }, [onReaderAtTopChange]);

  function handleWheel(event: React.WheelEvent<HTMLDivElement>) {
    if (typeof window !== "undefined" && window.innerWidth <= 1080 && window.scrollY <= 0 && event.deltaY < 0) {
      onTopScrollAttempt?.();
    }
  }

  useEffect(() => {
    if (!showSelector || selectorPinned || !isMobileViewport()) {
      clearSelectorIdleTimer();
      return;
    }

    bumpSelectorInactivityTimeout();
    return () => clearSelectorIdleTimer();
  }, [selectorPinned, showSelector]);

  useEffect(() => {
    return () => {
      for (const timer of retryTimersRef.current.values()) {
        window.clearTimeout(timer);
      }
      if (tooltipTimerRef.current !== null) {
        window.clearTimeout(tooltipTimerRef.current);
      }
      clearSelectorIdleTimer();
      retryTimersRef.current.clear();
      retryAttemptsRef.current.clear();
    };
  }, []);

  return (
    <section
      className={`paginated-reader transition-${settings.transitionStyle} ${showSelector ? "selector-visible" : ""} ${
        selectorPinned ? "selector-pinned" : ""
      }`}
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      onWheel={handleWheel}
      aria-label="Paginated manga reader"
    >
      <aside
        className={`vertical-page-selector ${showSelector ? "visible" : "hidden"} ${selectorPinned ? "pinned" : ""}`}
        aria-label="Page selector"
        onMouseEnter={() => {
          if (!selectorPinned) {
            setSelectorVisible(true);
          }
        }}
        onMouseLeave={() => {
          if (!selectorPinned) {
            setSelectorVisible(false);
          }
        }}
        onPointerDown={bumpSelectorInactivityTimeout}
        onTouchMove={bumpSelectorInactivityTimeout}
        onScroll={bumpSelectorInactivityTimeout}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="vertical-page-selector-header">
          <strong>{safeCurrentPage + 1}</strong>
          <small>/{totalPages}</small>
        </div>
        <div className="vertical-page-selector-list">
          {selectorIndices.map((index, indexPosition) => {
            const previous = selectorIndices[indexPosition - 1] ?? index;
            const showGap = indexPosition > 0 && index - previous > 1;

            return (
              <div key={`selector-${index}`} className="vertical-page-selector-row">
                {showGap ? <span className="vertical-page-selector-gap">...</span> : null}
                <button
                  type="button"
                  className={`vertical-page-selector-item ${index === safeCurrentPage ? "active" : ""}`}
                  onClick={() => handlePageFlip(index)}
                  aria-label={`Go to page ${index + 1}`}
                >
                  {index + 1}
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      <div className="page-zone-hints" aria-hidden="true">
        <div className={`page-zone-hint left ${hoverZone === "left" ? "active" : ""}`}>
          <span className={`page-corner-tooltip ${tooltipZone === "left" ? "visible" : ""}`}>
            {forwardZone === "left" ? "Next" : "Previous"}
          </span>
        </div>
        <div className={`page-zone-hint right ${hoverZone === "right" ? "active" : ""}`}>
          <span className={`page-corner-tooltip ${tooltipZone === "right" ? "visible" : ""}`}>
            {forwardZone === "right" ? "Next" : "Previous"}
          </span>
        </div>
      </div>

      {totalPages > 0 ? (
        <div className={`page-spread ${visibleIndices.length > 1 ? "double" : "single"}`}>
          {orderedIndices.map((index) => {
            const panelUrl = panelUrls[index];
            if (!panelUrl) {
              return null;
            }

            if (failed.has(index)) {
              return (
                <div className="panel-error" key={`error:${index}`}>
                  <p>Reconnecting panel...</p>
                  <small>Retry attempt {retryAttempts[index] ?? 1}</small>
                </div>
              );
            }

            return (
              <img
                key={`${panelUrl}:${retryVersion[index] ?? 0}`}
                src={relayImageUrl(panelUrl)}
                alt={`Panel ${index + 1}`}
                style={panelStyle}
                loading="eager"
                onLoad={(event) => {
                  clearRetryState(index);
                  markImageLoaded(event.currentTarget.currentSrc || relayImageUrl(panelUrl));
                  if (import.meta.env.DEV) {
                    const entries = performance.getEntriesByName(event.currentTarget.currentSrc);
                    const latest = entries[entries.length - 1] as PerformanceResourceTiming | undefined;
                    // eslint-disable-next-line no-console
                    console.info("[RAGnarok] image displayed", {
                      label: `paginated-visible:${index + 1}`,
                      durationMs: latest ? Math.round(latest.duration) : null,
                      url: panelUrl
                    });
                  }

                }}
                onError={() => {
                  setFailed((prev) => {
                    const next = new Set(prev);
                    next.add(index);
                    return next;
                  });
                  scheduleRetry(index);
                }}
              />
            );
          })}
        </div>
      ) : (
        <div className="panel-placeholder">No panels available.</div>
      )}
    </section>
  );
}
