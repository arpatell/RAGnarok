export type ReaderMode = "paginated" | "scroll";
export type FitMode = "width-fit" | "height-fit" | "original" | "custom";
export type ReadingDirection = "rtl" | "ltr";
export type SpreadMode = "single" | "double" | "smart";
export type TransitionStyle = "none" | "fade" | "slide";
export type ThemePresetId = "classic" | "graphite" | "paper" | "neon";
export type BrowserHistoryBehavior =
  | "dont-touch"
  | "title-only"
  | "chapter-history"
  | "chapter-and-pages"
  | "all-moves";

export interface ThemePreset {
  id: ThemePresetId;
  label: string;
  interfaceColor: string;
  textColor: string;
  accentColor: string;
  readerBackground: string;
}

const CLASSIC_THEME_PRESET: ThemePreset = {
  id: "classic",
  label: "Classic",
  interfaceColor: "#323840",
  textColor: "#e6ebf0",
  accentColor: "#607387",
  readerBackground: "#2a3037"
};

export const THEME_PRESETS: ThemePreset[] = [
  CLASSIC_THEME_PRESET,
  {
    id: "graphite",
    label: "Graphite",
    interfaceColor: "#22262d",
    textColor: "#f1f4f9",
    accentColor: "#8aa3ba",
    readerBackground: "#14171b"
  },
  {
    id: "paper",
    label: "Paper",
    interfaceColor: "#e7dfce",
    textColor: "#241d14",
    accentColor: "#6e5432",
    readerBackground: "#f5f1e8"
  },
  {
    id: "neon",
    label: "Neon",
    interfaceColor: "#1f2330",
    textColor: "#dff6ff",
    accentColor: "#35c5c9",
    readerBackground: "#0f1220"
  }
];

export interface IngestResponse {
  series: {
    title: string;
    coverUrl: string | null;
    genres: string[];
    status: "ongoing" | "completed" | "unknown";
  };
  chapter: {
    number: string;
    title: string | null;
    panelUrls: string[];
    totalPages: number;
  };
  chapterList: ChapterListItem[];
  detectedMode: ReaderMode;
  sourceAdapter: string;
  extractionSource?: string;
}

export interface ChapterListItem {
  number: string;
  title: string | null;
  url: string;
  special?: boolean;
}

export interface SuggestionsPayload {
  alternatives: Array<{
    adapter: string;
    title: string;
    chapterUrl: string;
    confidence: number;
  }>;
  similarSeries: Array<{
    title: string;
    coverUrl: string | null;
    chapterCount: number;
  }>;
  trending: Array<{
    title: string;
    adapter: string;
    chapterUrl: string;
  }>;
}

export interface SupportedAdapter {
  id: string;
  displayName: string;
  domains: string[];
  defaultMode: ReaderMode;
}

export interface ReaderSettings {
  fitMode: FitMode;
  readingDirection: ReadingDirection;
  spreadMode: SpreadMode;
  readerThemePreset: ThemePresetId;
  interfaceColor: string;
  textColor: string;
  accentColor: string;
  backgroundColor: string;
  panelGap: number;
  preloadDepth: number;
  verticalArrowStepPx: number;
  resetPageScrollAfterFlip: boolean;
  turnPagesByClicking: boolean;
  turnPagesWithArrowsInVerticalView: boolean;
  enableSwipeGestures: boolean;
  pinVerticalPageSelector: boolean;
  browserHistoryBehavior: BrowserHistoryBehavior;
  transitionStyle: TransitionStyle;
  saveProgress: boolean;
  zoomPercent: number;
  highContrast: boolean;
}

export interface ReaderProgress {
  chapterUrl: string;
  pageIndex: number;
  scrollPercent: number;
  updatedAt: string;
}

export interface HistoryEntry {
  seriesTitle: string;
  chapterTitle: string;
  chapterUrl: string;
  coverUrl: string | null;
  mode: ReaderMode;
  visitedAt: string;
}

export interface FavoriteEntry {
  seriesTitle: string;
  chapterTitle: string;
  chapterUrl: string;
  coverUrl: string | null;
  updatedAt: string;
}

export interface Bookmark {
  id: string;
  label: string;
  chapterUrl: string;
  seriesTitle: string;
  pageIndex: number;
  scrollPercent: number;
  createdAt: string;
}

export const DEFAULT_BACKGROUND = CLASSIC_THEME_PRESET.readerBackground;

export function inferDefaultPreloadDepth(): number {
  return 3;
}

export function createDefaultSettings(): ReaderSettings {
  const preset = CLASSIC_THEME_PRESET;

  return {
    fitMode: "height-fit",
    readingDirection: "ltr",
    spreadMode: "single",
    readerThemePreset: preset.id,
    interfaceColor: preset.interfaceColor,
    textColor: preset.textColor,
    accentColor: preset.accentColor,
    backgroundColor: preset.readerBackground,
    panelGap: 0,
    preloadDepth: inferDefaultPreloadDepth(),
    verticalArrowStepPx: 25,
    resetPageScrollAfterFlip: false,
    turnPagesByClicking: true,
    turnPagesWithArrowsInVerticalView: false,
    enableSwipeGestures: true,
    pinVerticalPageSelector: false,
    browserHistoryBehavior: "title-only",
    transitionStyle: "none",
    saveProgress: true,
    zoomPercent: 100,
    highContrast: false
  };
}

export interface SearchResult {
  seriesTitle: string;
  coverUrl: string | null;
  firstChapterUrl: string;
  chapterCount: number;
  genres: string[];
  adapter: string;
  sourceDomain: string;
}

export interface RagSearchResult {
  title: string;
  title_candidates?: string[];
  media_type: string;
  display_type: string | null;
  mal_id: number | null;
  synopsis: string;
  characters: string[];
  genres: string[];
  citations: string[];
  score: number | null;
  source: "pinecone";
  source_label: string;
  image_url: string | null;
  status: string | null;
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  jikan_score: number | null;
  read_options: Record<string, string>;
  watch_options: Record<string, string>;
  anime_companion: {
    mal_id: number | null;
    title: string;
    status: string | null;
    episodes: number | null;
    score: number | null;
    image_url: string | null;
    watch_url: string;
    citation: string;
  } | null;
}

export interface RagResultLiveMetaPayload {
  mal_id: number;
  media_type: string;
  display_type: string | null;
  title: string;
  title_candidates: string[];
  image_url: string | null;
  status: string | null;
  chapters: number | null;
  volumes: number | null;
  episodes: number | null;
  jikan_score: number | null;
}

export interface RagSearchPayload {
  query: string;
  answer: string;
  retrieval_mode: string;
  results: RagSearchResult[];
  highlight: {
    title: string;
    mal_id: number | null;
    justification: string;
    citation: string;
  } | null;
}

export interface ReadNowPayload {
  requestedTitle: string;
  matchedTitle: string;
  seriesUrl: string;
  chapterUrl: string;
  sourceDomain: string;
}
