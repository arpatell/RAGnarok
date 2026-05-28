import type { ReaderMode } from "../types";

interface ReaderToolbarProps {
  hidden: boolean;
  seriesTitle: string;
  chapterLabel: string;
  pageIndicator: string;
  hasPreviousChapter: boolean;
  hasNextChapter: boolean;
  isFavorited: boolean;
  mode: ReaderMode;
  onPreviousChapter: () => void;
  onNextChapter: () => void;
  onModeChange: (mode: ReaderMode) => void;
  onToggleSettings: () => void;
  onBackHome: () => void;
  onToggleFavorite: () => void;
  onAddBookmark: () => void;
}

export function ReaderToolbar({
  hidden,
  seriesTitle,
  chapterLabel,
  pageIndicator,
  hasPreviousChapter,
  hasNextChapter,
  isFavorited,
  mode,
  onPreviousChapter,
  onNextChapter,
  onModeChange,
  onToggleSettings,
  onBackHome,
  onToggleFavorite,
  onAddBookmark
}: ReaderToolbarProps) {
  return (
    <header className={`reader-toolbar ${hidden ? "is-hidden" : ""}`}>
      <div className="toolbar-left">
        <button className="ghost" onClick={onBackHome} type="button">
          Home
        </button>
        <button className="ghost" onClick={onToggleSettings} type="button" aria-label="Open settings (O)">
          Settings
        </button>
      </div>

      <div className="toolbar-center">
        <strong>{seriesTitle}</strong>
        <span>{chapterLabel}</span>
      </div>

      <div className="toolbar-right">
        <div className="chapter-nav" role="group" aria-label="Chapter navigation">
          <button type="button" className="ghost" onClick={onPreviousChapter} disabled={!hasPreviousChapter}>
            Prev
          </button>
          <button type="button" className="ghost" onClick={onNextChapter} disabled={!hasNextChapter}>
            Next
          </button>
        </div>

        <div className="mode-toggle" role="group" aria-label="Reading mode">
          <button
            type="button"
            className={mode === "paginated" ? "active" : ""}
            onClick={() => onModeChange("paginated")}
          >
            Manga
          </button>
          <button
            type="button"
            className={mode === "scroll" ? "active" : ""}
            onClick={() => onModeChange("scroll")}
          >
            Manhwa
          </button>
        </div>

        <button className={`ghost ${isFavorited ? "active-favorite" : ""}`} onClick={onToggleFavorite} type="button">
          {isFavorited ? "Favorited" : "Favorite"}
        </button>

        <button className="ghost" onClick={onAddBookmark} type="button">
          Bookmark
        </button>

        <span className="page-indicator" aria-live="polite">
          {pageIndicator}
        </span>
      </div>
    </header>
  );
}
