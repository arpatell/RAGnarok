import { useState, type FormEvent } from "react";
import { createPortal } from "react-dom";
import type { Bookmark, ChapterListItem, IngestResponse, SuggestionsPayload } from "../types";

const KOFI_URL = "https://ko-fi.com/phos174";

interface CryptoWallet {
  id: string;
  label: string;
  icon: string;
  address: string;
}

const CRYPTO_WALLETS: CryptoWallet[] = [
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

interface ChapterSidebarProps {
  isOpen: boolean;
  data: IngestResponse;
  currentChapterUrl: string;
  filterQuery: string;
  navigationStatus?: string | null;
  suggestions: SuggestionsPayload | null;
  bookmarks: Bookmark[];
  hasReadChapter: (url: string) => boolean;
  onFilterQueryChange: (query: string) => void;
  onSelectChapter: (url: string) => string | null;
  onGoBookmark: (bookmark: Bookmark) => void;
  onEditBookmark: (bookmark: Bookmark) => void;
  onDeleteBookmark: (bookmark: Bookmark) => void;
  onExportBookmarks: () => void;
}

function normalizeUrl(url: string): string {
  try {
    const normalized = new URL(url);
    normalized.hash = "";
    return normalized.toString();
  } catch {
    return url;
  }
}

function chapterLabel(chapter: ChapterListItem): string {
  if (chapter.title && chapter.title !== chapter.number) {
    return `${chapter.number} - ${chapter.title}`;
  }
  return chapter.title || chapter.number || "Chapter";
}

function parseQueryNumber(query: string): number | null {
  const match = query.toLowerCase().match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseChapterNumber(value: string): number | null {
  const match = value.toLowerCase().match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

export function ChapterSidebar({
  isOpen,
  data,
  currentChapterUrl,
  filterQuery,
  navigationStatus,
  suggestions,
  bookmarks,
  hasReadChapter,
  onFilterQueryChange,
  onSelectChapter,
  onGoBookmark,
  onEditBookmark,
  onDeleteBookmark,
  onExportBookmarks
}: ChapterSidebarProps) {
  const [coffeeOpen, setCoffeeOpen] = useState(false);
  const [cryptoOpen, setCryptoOpen] = useState(false);
  const [cryptoCopyStatus, setCryptoCopyStatus] = useState<string | null>(null);
  const [findStatus, setFindStatus] = useState<string | null>(null);

  const normalizedCurrent = normalizeUrl(currentChapterUrl);
  const filtered = data.chapterList.filter((chapter) => {
    const query = filterQuery.trim();
    if (!query) {
      return true;
    }

    const parsedQueryNumber = parseQueryNumber(query);
    if (parsedQueryNumber !== null) {
      const parsedChapterNumber = parseChapterNumber(chapter.number);
      if (parsedChapterNumber !== null && parsedChapterNumber === parsedQueryNumber) {
        return true;
      }
    }

    const target = `${chapter.number} ${chapter.title ?? ""}`.toLowerCase();
    return target.includes(query.toLowerCase());
  });

  function handleFindChapterSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const query = filterQuery.trim();
    if (!query) {
      setFindStatus(null);
      return;
    }

    const lowered = query.toLowerCase();
    const exactMatch = filtered.find((chapter) => {
      const number = chapter.number.toLowerCase();
      const title = (chapter.title ?? "").toLowerCase();
      const label = chapterLabel(chapter).toLowerCase();
      return number === lowered || title === lowered || label === lowered;
    });

    if (exactMatch) {
      const selectionStatus = onSelectChapter(exactMatch.url);
      setFindStatus(selectionStatus);
      return;
    }

    if (filtered.length === 1) {
      const onlyMatch = filtered[0];
      if (!onlyMatch) {
        setFindStatus("No chapter matched.");
        return;
      }

      const selectionStatus = onSelectChapter(onlyMatch.url);
      setFindStatus(selectionStatus);
      return;
    }

    if (filtered.length > 1) {
      setFindStatus("Multiple matches found. Keep typing to narrow it down.");
      return;
    }

    const numericQuery = parseQueryNumber(query);
    if (numericQuery !== null && numericQuery <= 0) {
      setFindStatus("No chapter data found for that chapter on this manga.");
      return;
    }

    setFindStatus(`No chapter data found for "${query}" on this manga.`);
  }

  async function copyCryptoAddress(wallet: CryptoWallet) {
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(wallet.address);
        setCryptoCopyStatus(`${wallet.label} address copied.`);
        return;
      }
    } catch {
      // Fall back to execCommand if Clipboard API is unavailable.
    }

    try {
      const textarea = document.createElement("textarea");
      textarea.value = wallet.address;
      textarea.setAttribute("readonly", "true");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      document.body.removeChild(textarea);

      setCryptoCopyStatus(copied ? `${wallet.label} address copied.` : "Copy failed. Please copy manually.");
    } catch {
      setCryptoCopyStatus("Copy failed. Please copy manually.");
    }
  }

  const cryptoModal = cryptoOpen ? (
    <div
      className="coffee-modal-backdrop"
      onClick={() => setCryptoOpen(false)}
      role="presentation"
    >
      <div
        className="coffee-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Crypto donation"
        onClick={(event) => event.stopPropagation()}
      >
        <h4>Crypto Support</h4>
        <p className="muted">
          Send with crypto and include a message with your payment if you want to share feedback.
        </p>

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
                    void copyCryptoAddress(wallet);
                  }}
                >
                  Copy
                </button>
              </div>
            </li>
          ))}
        </ul>

        {cryptoCopyStatus ? <p className="muted">{cryptoCopyStatus}</p> : null}
        <button type="button" className="ghost" onClick={() => setCryptoOpen(false)}>
          Close
        </button>
      </div>
    </div>
  ) : null;

  return (
    <>
      <aside className={`chapter-sidebar ${isOpen ? "open" : ""}`} aria-hidden={!isOpen}>
      <div className="series-head">
        {data.series.coverUrl ? <img src={data.series.coverUrl} alt={`${data.series.title} cover`} /> : null}
        <div>
          <h2>{data.series.title}</h2>
          <p>{data.chapter.totalPages} panels</p>
        </div>
      </div>

      <form className="chapter-search" onSubmit={handleFindChapterSubmit}>
        <label>
          <span>Find chapter</span>
          <input
            type="search"
            value={filterQuery}
            onChange={(event) => {
              onFilterQueryChange(event.target.value);
              setFindStatus(null);
            }}
            placeholder="Search chapter or enter number"
          />
        </label>
        {findStatus || navigationStatus ? <p className="muted">{findStatus ?? navigationStatus}</p> : null}
      </form>

      <ul className="chapter-list" role="listbox" aria-label="Chapter list">
        {filtered.map((chapter) => {
          const isCurrent = normalizeUrl(chapter.url) === normalizedCurrent;
          const isRead = hasReadChapter(chapter.url);

          return (
            <li key={chapter.url}>
              <button
                type="button"
                className={`chapter-item ${isCurrent ? "active" : ""}`}
                onClick={() => {
                  const selectionStatus = onSelectChapter(chapter.url);
                  setFindStatus(selectionStatus);
                }}
              >
                <span>{chapterLabel(chapter)}</span>
                <div className="chapter-meta">
                  <small>{isRead ? "Read" : "Unread"}</small>
                </div>
              </button>
            </li>
          );
        })}
      </ul>

      <section className="sidebar-section" aria-label="Bookmarks">
        <div className="sidebar-section-row">
          <h3>Bookmarks</h3>
          <button className="ghost" type="button" onClick={onExportBookmarks}>
            Export JSON
          </button>
        </div>

        {bookmarks.length === 0 ? (
          <p className="muted">No bookmarks yet.</p>
        ) : (
          <ul className="bookmark-list">
            {bookmarks.map((bookmark) => (
              <li key={bookmark.id}>
                <button type="button" className="bookmark-main" onClick={() => onGoBookmark(bookmark)}>
                  <strong>{bookmark.label}</strong>
                  <small>
                    {new Date(bookmark.createdAt).toLocaleString()} - {bookmark.seriesTitle}
                  </small>
                </button>
                <div className="bookmark-actions">
                  <button type="button" className="ghost" onClick={() => onEditBookmark(bookmark)}>
                    Edit
                  </button>
                  <button type="button" className="ghost" onClick={() => onDeleteBookmark(bookmark)}>
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      {suggestions ? (
        <section className="sidebar-section" aria-label="Suggestions">
          <h3>Suggestions</h3>

          <div className="suggestion-group">
            <h4>Jikan Recommendations</h4>
            {suggestions.similarSeries.length ? (
              <ul>
                {suggestions.similarSeries.map((series) => (
                  <li key={series.title}>
                    {series.url ? (
                      <a href={series.url} target="_blank" rel="noreferrer">
                        {series.title}
                      </a>
                    ) : (
                      <span>{series.title}</span>
                    )}
                    {typeof series.votes === "number" ? <small> {series.votes} votes</small> : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No Jikan recommendations found.</p>
            )}
          </div>
        </section>
      ) : null}

      <section className="sidebar-section" aria-label="Support">
        <div className="suggestion-group support-coffee-group">
          <button
            type="button"
            className={`coffee-toggle ${coffeeOpen ? "open" : ""}`}
            onClick={() => setCoffeeOpen((open) => !open)}
            aria-expanded={coffeeOpen}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M4 7h13v8a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4V7zm13 2h1a2 2 0 0 1 0 4h-1V9zm-9 12h8M7 4h2M11 4h2" />
            </svg>
            <span>Buy Developer a Coffee</span>
          </button>

          {coffeeOpen ? (
            <div className="coffee-panel">
              <div className="coffee-actions">
                <a className="ghost" href={KOFI_URL} target="_blank" rel="noreferrer">
                  Ko-fi
                </a>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setCryptoOpen(true);
                    setCryptoCopyStatus(null);
                  }}
                >
                  Crypto
                </button>
              </div>
              <p className="muted">
                Leave feedback directly on Ko-fi, or include a note in your crypto payment message.
              </p>
            </div>
          ) : null}
        </div>
      </section>

      </aside>
      {cryptoModal && typeof document !== "undefined" ? createPortal(cryptoModal, document.body) : null}
    </>
  );
}
