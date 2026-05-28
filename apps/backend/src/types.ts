export type ReaderMode = "paginated" | "scroll";
export type SeriesStatus = "ongoing" | "completed" | "unknown";

export interface PanelCandidate {
  url: string;
  width?: number;
  height?: number;
}

export interface ChapterListItem {
  number: string;
  title: string | null;
  url: string;
  special?: boolean;
}

export interface SeriesMeta {
  title: string;
  coverUrl: string | null;
  genres: string[];
  status: SeriesStatus;
}

export interface ChapterMeta {
  number: string;
  title: string | null;
  panelUrls: string[];
  totalPages: number;
}

export interface IngestResponse {
  series: SeriesMeta;
  chapter: ChapterMeta;
  chapterList: ChapterListItem[];
  detectedMode: ReaderMode;
  sourceAdapter: string;
  extractionSource?: string;
}

export interface ExtractedChapterData {
  series: SeriesMeta;
  chapter: {
    number: string;
    title: string | null;
    panelCandidates: PanelCandidate[];
  };
  chapterList: ChapterListItem[];
  detectedModeHint: ReaderMode | null;
  source: string;
  chapterUrlPattern: string | null;
  pageType?: "chapter" | "series" | "unknown";
  suggestedFirstChapterUrl?: string | null;
}

export interface AdapterParseResult {
  series: Partial<SeriesMeta> & Pick<SeriesMeta, "title">;
  chapter: {
    number: string;
    title: string | null;
    panelCandidates: PanelCandidate[];
  };
  chapterList: ChapterListItem[];
}

export interface Adapter {
  id: string;
  displayName: string;
  domains: string[];
  defaultMode: ReaderMode;
  parse: (html: string, chapterUrl: string) => AdapterParseResult;
}

export interface SupportedAdapterSummary {
  id: string;
  displayName: string;
  domains: string[];
  defaultMode: ReaderMode;
}
