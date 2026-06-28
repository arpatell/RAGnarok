import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type TouchEvent as ReactTouchEvent
} from "react";
import { relayImageUrl } from "../lib/api";
import { markImageLoaded } from "../lib/imagePreload";
import type { ReaderSettings } from "../types";

interface ScrollReaderProps {
  panelUrls: string[];
  initialPageIndex: number;
  settings: ReaderSettings;
  onProgressChange: (progress: number) => void;
  onVisiblePageChange: (pageIndex: number) => void;
  onReadPreviousChapter: () => void;
  onReadNextChapter: () => void;
  onMobileSelectorVisibilityChange?: (visible: boolean) => void;
  onReaderAtTopChange?: (atTop: boolean) => void;
  onTopScrollAttempt?: () => void;
}

const LONG_STRIP_HEIGHT_WIDTH_RATIO = 1.77;

function resolveAutoFitMode(settings: ReaderSettings, heightWidthRatio: number | undefined): "height-fit" | "width-fit" {
  if (heightWidthRatio === undefined) {
    return settings.fitMode === "height-fit" ? "height-fit" : "width-fit";
  }

  return heightWidthRatio <= LONG_STRIP_HEIGHT_WIDTH_RATIO ? "height-fit" : "width-fit";
}

function panelStyle(settings: ReaderSettings, heightWidthRatio?: number): CSSProperties {
  const zoom = Math.min(200, Math.max(50, settings.zoomPercent));
  const zoomFactor = zoom / 100;
  const heightOffsetPx = Math.round(120 * zoomFactor);
  const fitMode =
    settings.fitMode === "original" || settings.fitMode === "custom"
      ? settings.fitMode
      : resolveAutoFitMode(settings, heightWidthRatio);

  if (fitMode === "height-fit") {
    return {
      height: `calc(${zoom}dvh - ${heightOffsetPx}px)`,
      width: "auto",
      maxWidth: "none",
      maxHeight: "none"
    };
  }

  if (fitMode === "original") {
    return {
      width: "auto",
      maxWidth: "none",
      maxHeight: "none"
    };
  }

  if (fitMode === "custom") {
    return {
      width: `${zoom}%`,
      height: "auto",
      maxWidth: "none",
      maxHeight: "none"
    };
  }

  return {
    width: `${Math.max(60, zoom)}%`,
    height: "auto",
    maxWidth: "none",
    maxHeight: "none"
  };
}

function buildNearbyPanelSet(totalPanels: number, centerIndex: number, radius: number): Set<number> {
  const lastPanelIndex = Math.max(totalPanels - 1, 0);
  const safeCenter = Math.min(Math.max(centerIndex, 0), lastPanelIndex);
  const safeRadius = Math.max(2, radius);
  const indices = new Set<number>();

  for (let offset = 0; offset <= safeRadius; offset += 1) {
    const next = safeCenter + offset;
    const previous = safeCenter - offset;

    if (next <= lastPanelIndex) {
      indices.add(next);
    }

    if (previous >= 0) {
      indices.add(previous);
    }
  }

  return indices;
}

function logImageLoadTiming(label: string, panelUrl: string, image: HTMLImageElement) {
  if (!import.meta.env.DEV) {
    return;
  }

  const entries = performance.getEntriesByName(image.currentSrc);
  const latest = entries[entries.length - 1] as PerformanceResourceTiming | undefined;
  // eslint-disable-next-line no-console
  console.info("[RAGnarok] image displayed", {
    label,
    durationMs: latest ? Math.round(latest.duration) : null,
    url: panelUrl
  });
}

export function ScrollReader({
  panelUrls,
  initialPageIndex,
  settings,
  onProgressChange,
  onVisiblePageChange,
  onReadPreviousChapter,
  onReadNextChapter,
  onMobileSelectorVisibilityChange,
  onReaderAtTopChange,
  onTopScrollAttempt
}: ScrollReaderProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const anchorIndexRef = useRef(0);
  const ratioMapRef = useRef<Map<number, number>>(new Map());

  const [progress, setProgress] = useState(0);
  const [currentPanelIndex, setCurrentPanelIndex] = useState(0);
  const [activePanels, setActivePanels] = useState<Set<number>>(new Set([0, 1, 2, 3, 4]));
  const [failedPanels, setFailedPanels] = useState<Set<number>>(new Set());
  const [panelAspectRatios, setPanelAspectRatios] = useState<Record<number, number>>({});
  const [retryVersion, setRetryVersion] = useState<Record<number, number>>({});
  const [retryAttempts, setRetryAttempts] = useState<Record<number, number>>({});
  const [selectorVisible, setSelectorVisible] = useState(false);
  const retryTimersRef = useRef<Map<number, number>>(new Map());
  const retryAttemptsRef = useRef<Map<number, number>>(new Map());
  const selectorIdleTimerRef = useRef<number | null>(null);
  const touchStartXRef = useRef<number | null>(null);
  const touchStartYRef = useRef<number | null>(null);
  const touchStartBottomRef = useRef(false);
  const touchSelectorOpenRef = useRef(false);
  const suppressNextClickRef = useRef(false);
  const endNextHideTimerRef = useRef<number | null>(null);
  const [endNextVisible, setEndNextVisible] = useState(false);

  const selectorPinned = settings.pinVerticalPageSelector;
  const showSelector = selectorPinned || selectorVisible;
  const isOnLastPanel = currentPanelIndex >= Math.max(panelUrls.length - 1, 0);

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

  function clearEndNextHideTimer() {
    if (endNextHideTimerRef.current !== null) {
      window.clearTimeout(endNextHideTimerRef.current);
      endNextHideTimerRef.current = null;
    }
  }

  function revealEndNextButton(force = false) {
    if (!force && !isOnLastPanel) {
      return;
    }

    setEndNextVisible(true);
    clearEndNextHideTimer();
    endNextHideTimerRef.current = window.setTimeout(() => {
      setEndNextVisible(false);
      endNextHideTimerRef.current = null;
    }, 3000);
  }

  const selectorIndices = useMemo(() => {
    const total = panelUrls.length;
    if (total <= 24) {
      return Array.from({ length: total }, (_, index) => index);
    }

    const selected = new Set<number>([0, total - 1, currentPanelIndex]);
    const localStart = Math.max(0, currentPanelIndex - 5);
    const localEnd = Math.min(total - 1, currentPanelIndex + 5);

    for (let index = localStart; index <= localEnd; index += 1) {
      selected.add(index);
    }

    const step = Math.max(1, Math.floor(total / 12));
    for (let index = 0; index < total; index += step) {
      selected.add(index);
    }

    return Array.from(selected).sort((a, b) => a - b);
  }, [currentPanelIndex, panelUrls.length]);

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

    setFailedPanels((prev) => {
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

    const retryDelayMs = Math.min(8_000, 350 * 2 ** Math.min(nextAttempt, 6) + Math.floor(Math.random() * 200));
    const timer = window.setTimeout(() => {
      retryTimersRef.current.delete(index);
      setFailedPanels((prev) => {
        const next = new Set(prev);
        next.delete(index);
        return next;
      });
      setRetryVersion((prev) => ({
        ...prev,
        [index]: (prev[index] ?? 0) + 1
      }));
      setActivePanels((prev) => new Set(prev).add(index));
    }, retryDelayMs);

    retryTimersRef.current.set(index, timer);
  }

  function scrollToPanel(index: number, behavior: ScrollBehavior = "smooth") {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const clamped = Math.min(Math.max(index, 0), Math.max(panelUrls.length - 1, 0));
    const targetNode = container.querySelector<HTMLElement>(`[data-index="${clamped}"]`);
    if (!targetNode) {
      return;
    }

    setCurrentPanelIndex(clamped);
    setActivePanels((prev) => new Set(prev).add(clamped));
    targetNode.scrollIntoView({
      behavior,
      block: "start"
    });
  }

  function goNextPanel() {
    if (currentPanelIndex >= Math.max(panelUrls.length - 1, 0)) {
      onReadNextChapter();
      return;
    }

    scrollToPanel(currentPanelIndex + 1);
  }

  function goPreviousPanel() {
    if (currentPanelIndex <= 0) {
      onReadPreviousChapter();
      return;
    }

    scrollToPanel(currentPanelIndex - 1);
  }

  useEffect(() => {
    const startIndex = Math.min(Math.max(initialPageIndex, 0), Math.max(panelUrls.length - 1, 0));

    setProgress(0);
    setCurrentPanelIndex(startIndex);
    setFailedPanels(new Set());
    setPanelAspectRatios({});
    setRetryVersion({});
    setRetryAttempts({});
    setSelectorVisible(selectorPinned);
    clearSelectorIdleTimer();
    touchStartXRef.current = null;
    touchStartYRef.current = null;
    touchStartBottomRef.current = false;
    touchSelectorOpenRef.current = false;
    suppressNextClickRef.current = false;
    setEndNextVisible(false);
    clearEndNextHideTimer();
    anchorIndexRef.current = startIndex;
    ratioMapRef.current.clear();
    setActivePanels(buildNearbyPanelSet(panelUrls.length, startIndex, Math.max(4, settings.preloadDepth)));

    for (const timer of retryTimersRef.current.values()) {
      window.clearTimeout(timer);
    }
    retryTimersRef.current.clear();
    retryAttemptsRef.current.clear();

    const container = containerRef.current;
    if (container) {
      container.scrollTop = 0;
      onReaderAtTopChange?.(true);

      if (startIndex > 0) {
        requestAnimationFrame(() => {
          scrollToPanel(startIndex, "auto");
        });
      }
    }
  }, [initialPageIndex, onReaderAtTopChange, panelUrls, selectorPinned, settings.preloadDepth]);

  useEffect(() => {
    if (!isOnLastPanel) {
      setEndNextVisible(false);
      clearEndNextHideTimer();
      return;
    }

    revealEndNextButton();
  }, [isOnLastPanel]);

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

    return () => onMobileSelectorVisibilityChange(false);
  }, [onMobileSelectorVisibilityChange]);

  useEffect(() => {
    if (!showSelector || selectorPinned || !isMobileViewport()) {
      clearSelectorIdleTimer();
      return;
    }

    bumpSelectorInactivityTimeout();
    return () => clearSelectorIdleTimer();
  }, [selectorPinned, showSelector]);

  useEffect(() => {
    onVisiblePageChange(currentPanelIndex);
  }, [currentPanelIndex, onVisiblePageChange]);

  useEffect(() => {
    return () => {
      for (const timer of retryTimersRef.current.values()) {
        window.clearTimeout(timer);
      }
      retryTimersRef.current.clear();
      retryAttemptsRef.current.clear();
      clearSelectorIdleTimer();
      clearEndNextHideTimer();
    };
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        let anchor = anchorIndexRef.current;

        setActivePanels((prev) => {
          const next = new Set(prev);

          for (const entry of entries) {
            const raw = (entry.target as HTMLElement).dataset.index;
            const index = Number(raw);
            if (Number.isNaN(index)) {
              continue;
            }

            if (entry.isIntersecting) {
              anchor = index;
              for (const nearbyIndex of buildNearbyPanelSet(panelUrls.length, index, Math.max(2, settings.preloadDepth))) {
                next.add(nearbyIndex);
              }
              ratioMapRef.current.set(index, entry.intersectionRatio);
            } else {
              ratioMapRef.current.delete(index);
            }
          }

          let bestIndex = anchor;
          let bestRatio = ratioMapRef.current.get(anchor) ?? 0;
          for (const [index, ratio] of ratioMapRef.current.entries()) {
            if (ratio > bestRatio) {
              bestRatio = ratio;
              bestIndex = index;
            }
          }

          anchorIndexRef.current = bestIndex;

          if (next.size <= 50) {
            return next;
          }

          const prioritized = Array.from(next)
            .sort((a, b) => Math.abs(a - bestIndex) - Math.abs(b - bestIndex))
            .slice(0, 50);

          return new Set(prioritized);
        });

        setCurrentPanelIndex(anchorIndexRef.current);
      },
      {
        root: containerRef.current,
        rootMargin: "120% 0px",
        threshold: [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1]
      }
    );

    const nodes = containerRef.current?.querySelectorAll<HTMLElement>("[data-index]") ?? [];
    nodes.forEach((node) => observer.observe(node));

    return () => observer.disconnect();
  }, [panelUrls, settings.preloadDepth]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const onScroll = () => {
      const maxScroll = Math.max(container.scrollHeight - container.clientHeight, 1);
      const nextProgress = Math.min(1, Math.max(0, container.scrollTop / maxScroll));
      setProgress(nextProgress);
      onProgressChange(nextProgress);
      onReaderAtTopChange?.(container.scrollTop <= 0);

      const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 36;
      if (isOnLastPanel || nearBottom) {
        revealEndNextButton(true);
      }
    };

    onScroll();
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, [isOnLastPanel, onProgressChange, onReaderAtTopChange, panelUrls]);

  useEffect(() => {
    function onKeydown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      const container = containerRef.current;
      if (!container) {
        return;
      }

      if (settings.turnPagesWithArrowsInVerticalView) {
        if (event.key === "ArrowRight" || event.key === "ArrowDown" || event.key === "PageDown") {
          event.preventDefault();
          goNextPanel();
          return;
        }

        if (event.key === "ArrowLeft" || event.key === "ArrowUp" || event.key === "PageUp") {
          event.preventDefault();
          goPreviousPanel();
        }
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        container.scrollBy({ top: settings.verticalArrowStepPx, behavior: "smooth" });
        return;
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();
        container.scrollBy({ top: -settings.verticalArrowStepPx, behavior: "smooth" });
        return;
      }

      if (event.key === "PageDown") {
        event.preventDefault();
        container.scrollBy({ top: container.clientHeight * 0.9, behavior: "smooth" });
        return;
      }

      if (event.key === "PageUp") {
        event.preventDefault();
        container.scrollBy({ top: -(container.clientHeight * 0.9), behavior: "smooth" });
      }
    }

    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [
    currentPanelIndex,
    panelUrls.length,
    settings.turnPagesWithArrowsInVerticalView,
    settings.verticalArrowStepPx
  ]);

  function handleReaderMouseMove(event: ReactMouseEvent<HTMLElement>) {
    if (isMobileViewport()) {
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const relativeX = event.clientX - bounds.left;

    if (!selectorPinned) {
      setSelectorVisible(relativeX <= 112);
    }

    revealEndNextButton();
  }

  function handleReaderMouseLeave() {
    if (!selectorPinned) {
      setSelectorVisible(false);
    }
  }

  function handleReaderTouchStart(event: ReactTouchEvent<HTMLElement>) {
    const target = event.target as HTMLElement | null;
    if (target?.closest(".vertical-page-selector")) {
      touchStartXRef.current = null;
      touchStartYRef.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      revealEndNextButton();
      return;
    }

    const touch = event.changedTouches[0];
    if (!touch) {
      touchStartXRef.current = null;
      touchStartYRef.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      revealEndNextButton();
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const touchX = touch.clientX - bounds.left;
    touchStartXRef.current = touchX;
    touchStartYRef.current = touch.clientY;
    touchStartBottomRef.current = Boolean(isMobileViewport() && bounds.bottom - touch.clientY <= 34);
    touchSelectorOpenRef.current = false;
    revealEndNextButton();
  }

  function handleReaderTouchMove(event: ReactTouchEvent<HTMLElement>) {
    if (touchStartXRef.current === null || touchStartYRef.current === null) {
      return;
    }

    const touch = event.changedTouches[0];
    if (!touch) {
      return;
    }

    const container = containerRef.current;
    if (isMobileViewport() && container && container.scrollTop <= 0) {
      const pullDown = touch.clientY - touchStartYRef.current;
      const horizontalMovement = Math.abs(touch.clientX - touchStartXRef.current);
      if (pullDown > 42 && horizontalMovement < 56) {
        onTopScrollAttempt?.();
      }
    }

    if (!settings.enableSwipeGestures || !isMobileViewport() || selectorPinned || !touchStartBottomRef.current) {
      return;
    }

    const deltaX = Math.abs(touch.clientX - touchStartXRef.current);
    const deltaY = touch.clientY - touchStartYRef.current;

    if (deltaY < -42 && deltaX < 52) {
      touchSelectorOpenRef.current = true;
      suppressNextClickRef.current = true;
      setSelectorVisible(true);
      bumpSelectorInactivityTimeout();
    }

    revealEndNextButton();
  }

  function handleReaderTouchEnd() {
    if (touchSelectorOpenRef.current) {
      touchStartXRef.current = null;
      touchStartYRef.current = null;
      touchStartBottomRef.current = false;
      touchSelectorOpenRef.current = false;
      return;
    }

    touchStartXRef.current = null;
    touchStartYRef.current = null;
    touchStartBottomRef.current = false;
  }

  function handleReaderClick(event: ReactMouseEvent<HTMLElement>) {
    if (suppressNextClickRef.current) {
      suppressNextClickRef.current = false;
      event.preventDefault();
    }
  }

  function handleReaderWheel(event: React.WheelEvent<HTMLElement>) {
    const container = containerRef.current;
    if (isMobileViewport() && container && container.scrollTop <= 0 && event.deltaY < 0) {
      onTopScrollAttempt?.();
    }
  }

  return (
    <section
      className={`scroll-reader ${showSelector ? "selector-visible" : ""} ${selectorPinned ? "selector-pinned" : ""}`}
      aria-label="Manhwa mode reader"
      onClick={handleReaderClick}
      onMouseMove={handleReaderMouseMove}
      onMouseLeave={handleReaderMouseLeave}
      onTouchStart={handleReaderTouchStart}
      onTouchMove={handleReaderTouchMove}
      onTouchEnd={handleReaderTouchEnd}
      onWheel={handleReaderWheel}
    >
      <aside
        className={`vertical-page-selector ${showSelector ? "visible" : "hidden"} ${selectorPinned ? "pinned" : ""}`}
        aria-label="Vertical page selector"
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
          <strong>{Math.min(currentPanelIndex + 1, panelUrls.length)}</strong>
          <small>/{panelUrls.length}</small>
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
                  className={`vertical-page-selector-item ${index === currentPanelIndex ? "active" : ""}`}
                  onClick={() => scrollToPanel(index)}
                  aria-label={`Go to panel ${index + 1}`}
                >
                  {index + 1}
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      <div
        className="scroll-progress-bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
      >
        <span style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>

      <div
        className={`scroll-panel-stack ${settings.panelGap <= 0 ? "zero-gap" : ""}`}
        ref={containerRef}
        style={{
          gap: `${settings.panelGap}px`
        }}
      >
        {panelUrls.map((panelUrl, index) => {
          const isActive = activePanels.has(index);
          const isLastPanel = index === panelUrls.length - 1;
          const showEndNextOnPanel = isLastPanel && endNextVisible;
          const computedPanelStyle = panelStyle(settings, panelAspectRatios[index]);

          return (
            <article className="scroll-panel" key={`${panelUrl}:${index}`} data-index={index}>
              {isActive ? (
                failedPanels.has(index) ? (
                  <div className="panel-error">
                    <p>Reconnecting panel...</p>
                    <small>Retry attempt {retryAttempts[index] ?? 1}</small>
                  </div>
                ) : (
                  <div className="scroll-panel-media">
                    <img
                      key={`${panelUrl}:${retryVersion[index] ?? 0}`}
                      src={relayImageUrl(panelUrl)}
                      alt={`Panel ${index + 1}`}
                      style={computedPanelStyle}
                      loading={Math.abs(index - currentPanelIndex) <= 1 ? "eager" : "lazy"}
                      onLoad={(event) => {
                        clearRetryState(index);
                        const image = event.currentTarget;
                        const ratio =
                          image.naturalWidth > 0 && image.naturalHeight > 0
                            ? image.naturalHeight / image.naturalWidth
                            : undefined;
                        if (ratio !== undefined) {
                          setPanelAspectRatios((prev) => {
                            if (prev[index] === ratio) {
                              return prev;
                            }
                            return { ...prev, [index]: ratio };
                          });
                        }
                        markImageLoaded(event.currentTarget.currentSrc || relayImageUrl(panelUrl));
                        logImageLoadTiming(`scroll-visible:${index + 1}`, panelUrl, event.currentTarget);
                      }}
                      onError={() => {
                        setFailedPanels((prev) => {
                          const next = new Set(prev);
                          next.add(index);
                          return next;
                        });
                        scheduleRetry(index);
                      }}
                    />

                    {showEndNextOnPanel ? (
                      <button
                        type="button"
                        className="end-next-button"
                        onClick={onReadNextChapter}
                        aria-label="Read next chapter"
                      >
                        Next
                      </button>
                    ) : null}
                  </div>
                )
              ) : (
                <div className="panel-placeholder">Panel {index + 1}</div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
