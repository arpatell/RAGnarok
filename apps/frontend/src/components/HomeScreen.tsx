import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { fetchRagResultLiveMeta, relayImageUrl, resolveReadNowByTitle, searchMangaRagStream } from "../lib/api";
import type { FavoriteEntry, HistoryEntry, RagSearchResult } from "../types";

interface HomeScreenProps {
  initialUrl: string;
  history: HistoryEntry[];
  favorites: FavoriteEntry[];
  onSubmitUrl: (url: string) => void;
  onRemoveRecent: (seriesTitle: string) => void;
  onRemoveFavorite: (seriesTitle: string) => void;
}

type InputMode = "smart" | "url";
type LibraryTab = "recents" | "favorites";

const KOFI_URL = "https://ko-fi.com/phos174";
const LIVE_META_RETRY_WINDOW_MS = 65_000;
const LIVE_META_RETRY_PASS_DELAY_MS = 2_500;
const LIVE_META_BETWEEN_CARD_DELAY_MS = 250;

const CRYPTO_WALLETS = [
  {
    id: "sol",
    label: "SOL",
    icon: "SOL",
    address: "8Vs3NX3pN7M7CjmgM6tENgef1LfJLHkWLn6SK3oPEX9r"
  },
  {
    id: "eth",
    label: "ETH",
    icon: "ETH",
    address: "0x0b41B491E96cc626E009BAdDA83d535A94Ab4a85"
  },
  {
    id: "btc",
    label: "BTC",
    icon: "BTC",
    address: "bc1p2l627u3gwyj55pnpn4pkrehvvrlwm6y74cnx006ztdnkut6vhtns0fy2c8"
  },
  {
    id: "base",
    label: "BASE",
    icon: "BASE",
    address: "0x0b41B491E96cc626E009BAdDA83d535A94Ab4a85"
  },
  {
    id: "monad",
    label: "MONAD",
    icon: "MON",
    address: "0x0b41B491E96cc626E009BAdDA83d535A94Ab4a85"
  },
  {
    id: "sui",
    label: "SUI",
    icon: "SUI",
    address: "0xbe1bb13a1bbe47a5948c9c190d32dc70984fffd43fc128428fe67ac3dc11c4ba"
  },
  {
    id: "polygon",
    label: "POLYGON",
    icon: "POLY",
    address: "0x0b41B491E96cc626E009BAdDA83d535A94Ab4a85"
  }
];

function toSeriesKey(value: string): string {
  return value.trim().toLowerCase();
}

function formatHighlightJustification(justification: string): string {
  return justification.replace(/\s+/g, " ").trim();
}

function includesCitation(text: string, citation: string): boolean {
  const normalizedText = text.trim().toLowerCase();
  const normalizedCitation = citation.trim().toLowerCase();
  if (!normalizedText || !normalizedCitation) {
    return false;
  }
  return normalizedText.includes(normalizedCitation);
}

function getPrimaryWatchOptionUrl(result: RagSearchResult): string | null {
  const preferredKeys = ["anikaitv", "aniwatch", "crunchyroll"];
  for (const key of preferredKeys) {
    const candidate = result.watch_options[key]?.trim();
    if (candidate) {
      return candidate;
    }
  }

  for (const value of Object.values(result.watch_options)) {
    const candidate = value.trim();
    if (candidate) {
      return candidate;
    }
  }

  return null;
}

function toDisplayTypeLabel(result: RagSearchResult): string {
  const mediaType = (result.media_type || "").trim().toLowerCase();
  const displayType = (result.display_type || "").trim();

  if (mediaType === "anime") {
    const subtype = !displayType || displayType.toLowerCase() === "anime" ? "TV" : displayType;
    return `Anime (${subtype})`;
  }

  if (displayType) {
    return displayType;
  }

  if (mediaType === "manga") {
    return "Manga";
  }

  return "Manga";
}

function isAnimeResult(result: RagSearchResult): boolean {
  return (result.media_type || "").trim().toLowerCase() === "anime";
}

function handleResultImageError(event: React.SyntheticEvent<HTMLImageElement>): void {
  const image = event.currentTarget;
  const directSrc = image.dataset.directSrc?.trim();
  if (directSrc && image.src !== directSrc) {
    image.src = directSrc;
    return;
  }

  image.onerror = null;
}

async function waitFor(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }

  await new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);

    const onAbort = () => {
      window.clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      reject(new DOMException("Aborted", "AbortError"));
    };

    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function HomeScreen({
  initialUrl,
  history,
  favorites,
  onSubmitUrl,
  onRemoveRecent,
  onRemoveFavorite
}: HomeScreenProps) {
  const [url, setUrl] = useState(initialUrl);
  const [activeTab, setActiveTab] = useState<LibraryTab>("recents");
  const [inputMode, setInputMode] = useState<InputMode>("smart");

  const [smartQuery, setSmartQuery] = useState("");
  const [smartResults, setSmartResults] = useState<RagSearchResult[]>([]);
  const [smartAnswer, setSmartAnswer] = useState("");
  const [lastSearchQuery, setLastSearchQuery] = useState("");
  const [smartHighlight, setSmartHighlight] = useState<{
    title: string;
    mal_id: number | null;
    justification: string;
    citation: string;
  } | null>(null);
  const [isSmartSearching, setIsSmartSearching] = useState(false);
  const [smartError, setSmartError] = useState<string | null>(null);
  const [hasSmartSearched, setHasSmartSearched] = useState(false);
  const [supportOpen, setSupportOpen] = useState(false);
  const [supportShowCrypto, setSupportShowCrypto] = useState(false);
  const [supportCopyStatus, setSupportCopyStatus] = useState<string | null>(null);

  const [readNowLoadingKey, setReadNowLoadingKey] = useState<string | null>(null);
  const [readNowError, setReadNowError] = useState<string | null>(null);
  const readNowAbortRef = useRef<AbortController | null>(null);
  const liveMetaAbortRef = useRef<AbortController | null>(null);

  const smartAbortRef = useRef<AbortController | null>(null);
  const smartSubmitAtRef = useRef(0);

  const recentSeries = history.reduce<HistoryEntry[]>((acc, entry) => {
    const key = toSeriesKey(entry.seriesTitle);
    if (acc.some((item) => toSeriesKey(item.seriesTitle) === key)) {
      return acc;
    }
    acc.push(entry);
    return acc;
  }, []);

  useEffect(() => {
    return () => {
      smartAbortRef.current?.abort();
      readNowAbortRef.current?.abort();
      liveMetaAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    setUrl(initialUrl);
  }, [initialUrl]);

  const runSmartSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSmartResults([]);
      setSmartAnswer("");
      setLastSearchQuery("");
      setSmartHighlight(null);
      setSmartError(null);
      setReadNowError(null);
      setHasSmartSearched(false);
      return;
    }

    smartAbortRef.current?.abort();
    const controller = new AbortController();
    smartAbortRef.current = controller;

    setIsSmartSearching(true);
    setSmartResults([]);
    setSmartAnswer("");
    setLastSearchQuery(query.trim());
    setSmartHighlight(null);
    setHasSmartSearched(true);
    setSmartError(null);
    setReadNowError(null);

    try {
      const payload = await searchMangaRagStream(
        query.trim(),
        (streamedPayload) => {
          if (controller.signal.aborted) {
            return;
          }

          const streamedResults = streamedPayload.results.slice(0, 10);
          setSmartResults(streamedResults);
          setSmartAnswer(streamedPayload.answer);
          setLastSearchQuery(streamedPayload.query.trim() || query.trim());
          setSmartHighlight(streamedPayload.highlight);
          setHasSmartSearched(true);
        },
        controller.signal
      );
      const topResults = payload.results.slice(0, 10);
      setSmartResults(topResults);
      setSmartAnswer(payload.answer);
      setLastSearchQuery(payload.query.trim() || query.trim());
      setSmartHighlight(payload.highlight);
      setHasSmartSearched(true);

      liveMetaAbortRef.current?.abort();
      const metaController = new AbortController();
      liveMetaAbortRef.current = metaController;

      void (async () => {
        const pending = new Map(
          topResults
            .filter((result) => result.mal_id !== null && !result.image_url?.trim())
            .map((result) => [`${result.mal_id}:${result.title}`, result])
        );
        const deadlineAt = Date.now() + LIVE_META_RETRY_WINDOW_MS;

        while (pending.size > 0 && Date.now() < deadlineAt) {
          for (const [pendingKey, result] of Array.from(pending.entries())) {
            if (metaController.signal.aborted) {
              return;
            }

            if (result.mal_id === null) {
              pending.delete(pendingKey);
              continue;
            }

            try {
              const liveMeta = await fetchRagResultLiveMeta(
                result.mal_id,
                result.title,
                result.media_type,
                metaController.signal
              );

              if (metaController.signal.aborted) {
                return;
              }

              setSmartResults((current) =>
                current.map((row) => {
                  if (row.mal_id !== liveMeta.mal_id) {
                    return row;
                  }

                  return {
                    ...row,
                    media_type: liveMeta.media_type || row.media_type,
                    display_type: liveMeta.display_type || row.display_type,
                    title_candidates: liveMeta.title_candidates.length > 0 ? liveMeta.title_candidates : row.title_candidates,
                    image_url: liveMeta.image_url || row.image_url,
                    status: liveMeta.status ?? row.status,
                    chapters: liveMeta.chapters ?? row.chapters,
                    volumes: liveMeta.volumes ?? row.volumes,
                    episodes: liveMeta.episodes ?? row.episodes,
                    jikan_score: liveMeta.jikan_score ?? row.jikan_score
                  };
                })
              );

              if (liveMeta.image_url?.trim()) {
                pending.delete(pendingKey);
              }
            } catch (error) {
              if (error instanceof Error && error.name === "AbortError") {
                return;
              }
              // Keep this card in the queue; transient Jikan failures and 429s are retried below.
            }

            if (pending.size > 0) {
              await waitFor(LIVE_META_BETWEEN_CARD_DELAY_MS, metaController.signal);
            }
          }

          if (pending.size > 0 && Date.now() < deadlineAt) {
            await waitFor(LIVE_META_RETRY_PASS_DELAY_MS, metaController.signal);
          }
        }
      })();
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setSmartError(err instanceof Error ? err.message : "Smart Search failed. Try again.");
    } finally {
      setIsSmartSearching(false);
    }
  }, []);

  const handleReadNow = useCallback(
    async (result: RagSearchResult) => {
      const key = `${result.title}-${result.mal_id ?? "na"}`;
      const title = result.title.trim();
      if (!title) {
        setReadNowError("Could not start reading because this result has no title.");
        return;
      }

      readNowAbortRef.current?.abort();
      const controller = new AbortController();
      readNowAbortRef.current = controller;
      setReadNowError(null);
      setReadNowLoadingKey(key);

      try {
        let titleCandidates = [title, ...(result.title_candidates ?? [])];
        if (result.mal_id !== null) {
          try {
            const liveMeta = await fetchRagResultLiveMeta(
              result.mal_id,
              result.title,
              result.media_type,
              controller.signal
            );
            titleCandidates = [title, liveMeta.title, ...liveMeta.title_candidates, ...(result.title_candidates ?? [])];
          } catch (error) {
            if (error instanceof Error && error.name === "AbortError") {
              throw error;
            }
            // Read Now can still proceed with the stored RAG title when Jikan aliases are unavailable.
          }
        }

        const resolved = await resolveReadNowByTitle(
          title,
          titleCandidates,
          controller.signal,
          result.media_type || result.display_type || ""
        );
        const chapterUrl = resolved.chapterUrl.trim();
        if (!chapterUrl) {
          throw new Error(
            `We couldn't find "${title}" on WeebCentral or ManhwaZone. Try searching for a dedicated website hosting this manga/manhwa.`
          );
        }
        onSubmitUrl(chapterUrl);
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          return;
        }
        const message =
          error instanceof Error && error.message.trim().length > 0
            ? error.message
            : `We couldn't find "${title}" on WeebCentral or ManhwaZone. Try searching for a dedicated website hosting this manga/manhwa.`;
        setReadNowError(message);
      } finally {
        setReadNowLoadingKey((current) => (current === key ? null : current));
      }
    },
    [onSubmitUrl]
  );

  function handleSmartSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSmartSearching) {
      return;
    }
    const now = Date.now();
    if (now - smartSubmitAtRef.current < 300) {
      return;
    }
    smartSubmitAtRef.current = now;
    runSmartSearch(smartQuery);
  }

  function handleUrlSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!url.trim()) return;
    onSubmitUrl(url.trim());
  }

  async function copyCryptoAddress(label: string, address: string) {
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(address);
        setSupportCopyStatus(`${label} address copied.`);
        return;
      }
    } catch {
      // Fall back to execCommand below.
    }

    try {
      const textarea = document.createElement("textarea");
      textarea.value = address;
      textarea.setAttribute("readonly", "true");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      document.body.removeChild(textarea);
      setSupportCopyStatus(copied ? `${label} address copied.` : "Copy failed. Please copy manually.");
    } catch {
      setSupportCopyStatus("Copy failed. Please copy manually.");
    }
  }

  const showSmartResults =
    inputMode === "smart" &&
    (isSmartSearching || smartResults.length > 0 || (hasSmartSearched && !isSmartSearching) || Boolean(smartError));
  const supportModal = supportOpen ? (
    <div
      className="coffee-modal-backdrop"
      onClick={() => setSupportOpen(false)}
      role="presentation"
    >
      <div
        className="coffee-modal home-support-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Support developer"
        onClick={(event) => event.stopPropagation()}
      >
        <h4>Buy Developer a Coffee!</h4>
        <p className="muted">
          Thanks for using my site. If you wanted to support development, feel free to use any of these methods, and
          don&rsquo;t forget to leave any feedback or ideas.
        </p>
        <div className="home-support-modal-actions">
          <a className="primary" href={KOFI_URL} target="_blank" rel="noreferrer">
            Ko-fi
          </a>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setSupportCopyStatus(null);
              setSupportShowCrypto((current) => !current);
            }}
            aria-expanded={supportShowCrypto}
          >
            Crypto
          </button>
        </div>
        {supportShowCrypto ? (
          <ul className="coffee-wallet-list" aria-label="Crypto wallet addresses">
            {CRYPTO_WALLETS.map((wallet) => (
              <li key={wallet.id} className="coffee-wallet-item">
                <div className="wallet-network">
                  <span className="wallet-icon" aria-hidden="true">{wallet.icon}</span>
                  <strong>{wallet.label}</strong>
                </div>
                <div className="coffee-address-row wallet-address-row">
                  <input type="text" readOnly value={wallet.address} aria-label={`${wallet.label} address`} />
                  <button
                    type="button"
                    onClick={() => {
                      void copyCryptoAddress(wallet.label, wallet.address);
                    }}
                  >
                    Copy
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ) : null}
        {supportCopyStatus ? <p className="muted">{supportCopyStatus}</p> : null}
        <button type="button" className="ghost" onClick={() => setSupportOpen(false)}>
          Close
        </button>
      </div>
    </div>
  ) : null;

  return (
    <div className="home-shell">
      <div className="home-gradient" />
      <button
        type="button"
        className="home-support-button"
        aria-label="Support development"
        title="Support development"
        onClick={() => {
          setSupportCopyStatus(null);
          setSupportShowCrypto(false);
          setSupportOpen(true);
        }}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M4 7h13v8a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4V7zm13 2h1a2 2 0 0 1 0 4h-1V9zm-9 12h8M7 4h2M11 4h2" />
        </svg>
      </button>
      <main className="home-content">
        <p className="home-kicker">AI-Powered Discovery &amp; Universal Manga Reader</p>
        <h1>RAGnarok🌀</h1>
        <p className="home-subtitle">
          Discover your next favorite series. Search by describing a plot, typing a character&rsquo;s name, or entering a title. You can also paste a
          supported chapter URL directly to start reading.
        </p>
        <p className="home-doc-links">
          Learn more:
          {" "}
          <a href="/docs/product-guide.html">Product Guide</a>
        </p>

        <div className="home-input-tabs">
          <button
            type="button"
            className={`home-input-tab ${inputMode === "smart" ? "active" : ""}`}
            onClick={() => setInputMode("smart")}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M10 1.5l1.8 4.6L16.5 8l-4.7 1.9L10 14.5 8.2 9.9 3.5 8l4.7-1.9L10 1.5zm6.2 10.8.9 2.3 2.4.9-2.4.9-.9 2.4-.9-2.4-2.3-.9 2.3-.9.9-2.3zM4.2 12l1.1 2.8L8 15.9l-2.7 1.1-1.1 2.7L3 17l-2.8-1.1L3 14.8 4.2 12z" />
            </svg>
            Smart Search
          </button>
          <button
            type="button"
            className={`home-input-tab ${inputMode === "url" ? "active" : ""}`}
            onClick={() => setInputMode("url")}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M12.586 4.586a2 2 0 1 1 2.828 2.828l-3 3a2 2 0 0 1-2.828 0 1 1 0 0 0-1.414 1.414 4 4 0 0 0 5.656 0l3-3a4 4 0 0 0-5.656-5.656l-1.5 1.5a1 1 0 1 0 1.414 1.414l1.5-1.5zm-5 5a2 2 0 0 1 2.828 0 1 1 0 1 0 1.414-1.414 4 4 0 0 0-5.656 0l-3 3a4 4 0 1 0 5.656 5.656l1.5-1.5a1 1 0 1 0-1.414-1.414l-1.5 1.5a2 2 0 1 1-2.828-2.828l3-3z" />
            </svg>
            Paste URL
          </button>
        </div>

        {inputMode === "smart" && (
          <form className="url-form" onSubmit={handleSmartSubmit}>
            <label htmlFor="smart-search" className="sr-only">
              Smart Search query
            </label>
            <input
              id="smart-search"
              type="text"
              value={smartQuery}
              onChange={(e) => setSmartQuery(e.target.value)}
              placeholder="e.g. guy finds a notebook that kills people"
              autoFocus
            />
            <button type="submit" disabled={isSmartSearching}>
              {isSmartSearching ? "Searching..." : "Search"}
            </button>
          </form>
        )}

        {inputMode === "url" && (
          <form className="url-form" onSubmit={handleUrlSubmit}>
            <label htmlFor="chapter-url" className="sr-only">
              Chapter URL
            </label>
            <input
              id="chapter-url"
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste a manga chapter URL to start reading"
              autoFocus
            />
            <button type="submit">Read</button>
          </form>
        )}

        {showSmartResults && (
          <section className="search-results-section" aria-label="Smart Search results">
            {isSmartSearching ? (
              <div className="search-loading-panel" role="status" aria-live="polite">
                <div className="search-loading-mark" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
                <div>
                  <p>Searching...</p>
                  <small>Matching titles, plots, and aliases.</small>
                </div>
              </div>
            ) : null}
            {smartError && !isSmartSearching ? (
              <p className="muted search-empty">{smartError}</p>
            ) : smartResults.length === 0 && !isSmartSearching ? (
              <p className="muted search-empty">
                No Smart Search matches found for &ldquo;{lastSearchQuery || smartQuery}&rdquo;. Try adding character names,
                abilities, or setting clues.
              </p>
            ) : smartResults.length > 0 ? (
              <>
                {smartHighlight ? (
                  <article className="rag-answer-card">
                    <h2>Why this matches</h2>
                    <p>{formatHighlightJustification(smartHighlight.justification)}</p>
                    {smartHighlight.citation &&
                    !includesCitation(smartHighlight.justification, smartHighlight.citation) ? (
                      <p className="rag-answer-citation">{smartHighlight.citation}</p>
                    ) : null}
                  </article>
                ) : smartAnswer ? (
                  <article className="rag-answer-card">
                    <h2>Search Summary</h2>
                    <p>{smartAnswer}</p>
                  </article>
                ) : null}

                <div className="rag-results-grid">
                  {smartResults.map((result) => {
                    const resultKey = `${result.title}-${result.mal_id ?? "na"}`;
                    const isReadNowLoading = readNowLoadingKey === resultKey;
                    const watchUrl = getPrimaryWatchOptionUrl(result);
                    const mediaLabel = toDisplayTypeLabel(result);
                    const canReadNow = !isAnimeResult(result);
                    return (
                    <article className="rag-result-card" key={resultKey}>
                      <div className="rag-result-body">
                        <div className="rag-result-cover-wrap">
                          {result.image_url ? (
                            <img
                              className="rag-result-cover"
                              src={relayImageUrl(result.image_url)}
                              data-direct-src={result.image_url}
                              alt={result.title}
                              loading="lazy"
                              onError={handleResultImageError}
                            />
                          ) : (
                            <div className="search-result-cover search-cover-fallback">
                              <span>{result.title.charAt(0).toUpperCase()}</span>
                            </div>
                          )}
                        </div>

                        <div className="rag-result-meta">
                          <p className="rag-result-title">{result.title}</p>
                          <p className="rag-result-subtitle">{mediaLabel}</p>
                          <p className="rag-result-synopsis">{result.synopsis}</p>
                          {result.characters.length > 0 ? (
                            <p className="rag-result-characters">
                              <strong>Characters:</strong> {result.characters.join(", ")}
                            </p>
                          ) : null}

                          {result.genres.length > 0 ? (
                            <div className="search-genre-list">
                              {result.genres.slice(0, 3).map((genre) => (
                                <span key={genre} className="genre-pill">
                                  {genre}
                                </span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      </div>

                      <div className="search-result-actions rag-action-row">
                        {canReadNow ? (
                          <button
                            type="button"
                            className="search-result-read-btn"
                            disabled={isReadNowLoading}
                            onClick={() => handleReadNow(result)}
                          >
                            {isReadNowLoading ? "Loading..." : "Read Now"}
                          </button>
                        ) : null}
                        {watchUrl ? (
                          <a
                            className="ghost rag-watch-btn"
                            href={watchUrl}
                            target="_blank"
                            rel="noreferrer noopener"
                          >
                            Watch Now
                          </a>
                        ) : null}
                      </div>
                    </article>
                    );
                  })}
                </div>
                {readNowError ? (
                  <p className="search-read-now-error" role="alert">
                    {readNowError}
                  </p>
                ) : null}
              </>
            ) : null}
          </section>
        )}

        <section className="home-card" aria-labelledby="about-ragnarok-title">
          <div className="home-card-title-row">
            <h2 id="about-ragnarok-title">About RAGnarok Reader</h2>
          </div>
          <p className="muted">
            RAGnarok Reader is an AI-powered manga and anime discovery app designed for natural-language search. You
            can find a series by plot, character name, ability, setting, or exact title, then jump directly into a
            chapter reader experience.
          </p>
          <p className="muted">
            Under the hood, it combines hybrid semantic retrieval and keyword retrieval with model-based ranking so
            vague story queries and specific name queries both work reliably. The reader supports both paginated manga
            flow and continuous manhwa-style scrolling.
          </p>
        </section>

        <section className="home-card" aria-labelledby="supported-title">
          <div className="home-card-title-row">
            <h2 id="supported-title">Supported Sources</h2>
          </div>
          <p className="muted">
            <strong>Search Results:</strong> Automated reading is currently optimized to search and fetch directly
            from WeebCentral and ManhwaZone.
          </p>
          <p className="muted">
            <strong>Manual Import:</strong> Our paste feature works with almost any site that publicly exposes image
            panels. While manga-specific domains (like{" "}
            <a href="https://jjkaisen.com/chapter/1/" target="_blank" rel="noreferrer">
              https://jjkaisen.com/chapter/1/
            </a>
            ) have the highest success rates, you can paste a chapter URL from any domain to try loading it directly
            into the reader.
          </p>
        </section>

        <section className="home-card" aria-labelledby="continue-title">
          <div className="home-card-title-row">
            <h2 id="continue-title">Library</h2>
            <span>{activeTab === "recents" ? recentSeries.length : favorites.length} shown</span>
          </div>

          <div className="library-tabs" role="tablist" aria-label="Library views">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "recents"}
              className={`library-tab ${activeTab === "recents" ? "active" : ""}`}
              onClick={() => setActiveTab("recents")}
            >
              Recents ({recentSeries.length})
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "favorites"}
              className={`library-tab ${activeTab === "favorites" ? "active" : ""}`}
              onClick={() => setActiveTab("favorites")}
            >
              Favorites ({favorites.length})
            </button>
          </div>

          {activeTab === "recents" && recentSeries.length === 0 ? (
            <p className="muted">No successful reads yet.</p>
          ) : activeTab === "favorites" && favorites.length === 0 ? (
            <p className="muted">No favorites saved yet.</p>
          ) : (
            <div className="continue-grid">
              {(activeTab === "recents" ? recentSeries : favorites).map((entry) => {
                const updatedAt =
                  activeTab === "recents"
                    ? new Date((entry as HistoryEntry).visitedAt).toLocaleString()
                    : new Date((entry as FavoriteEntry).updatedAt).toLocaleString();

                return (
                  <article className="continue-card" key={`${activeTab}-${toSeriesKey(entry.seriesTitle)}`}>
                    <div className="continue-open" role="group" aria-label={`${entry.seriesTitle} recent entry`}>
                      <span className="continue-title">{entry.seriesTitle}</span>
                      <span className="continue-subtitle">{entry.chapterTitle}</span>
                      <span className="continue-meta">{updatedAt}</span>
                    </div>

                    <div className="continue-actions">
                      <button
                        className="continue-action-trash"
                        type="button"
                        aria-label={`Remove ${entry.seriesTitle}`}
                        onClick={() =>
                          activeTab === "recents"
                            ? onRemoveRecent(entry.seriesTitle)
                            : onRemoveFavorite(entry.seriesTitle)
                        }
                      >
                        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                          <path d="M9 3.5h6a1 1 0 0 1 1 1V6h3a1 1 0 1 1 0 2h-1l-1 11.2A2.2 2.2 0 0 1 14.8 21H9.2A2.2 2.2 0 0 1 7 19.2L6 8H5a1 1 0 1 1 0-2h3V4.5a1 1 0 0 1 1-1Zm1 2V6h4v-.5h-4Zm-1.1 13a1 1 0 0 0 1 .9h5.2a1 1 0 0 0 1-.9L16.9 8H7.1l.8 10.5Zm2.1-8.8a1 1 0 1 1 2 0v6.2a1 1 0 1 1-2 0V9.7Zm4 0a1 1 0 1 1 2 0v6.2a1 1 0 1 1-2 0V9.7Z" />
                        </svg>
                      </button>

                      <button
                        className="ghost continue-action-btn continue-action-read"
                        type="button"
                        onClick={() => onSubmitUrl(entry.chapterUrl)}
                      >
                        Read
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </main>
      {supportModal && typeof document !== "undefined" ? createPortal(supportModal, document.body) : null}
    </div>
  );
}
