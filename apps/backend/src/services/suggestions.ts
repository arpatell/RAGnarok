interface BuildSuggestionsInput {
  seriesTitle: string;
  genres: string[];
  currentAdapter: string;
}

export interface SuggestionsResponse {
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

interface CatalogSeries {
  title: string;
  adapter: string;
  chapterUrl: string;
  genres: string[];
  chapterCount: number;
  coverUrl: string | null;
}

const CATALOG: CatalogSeries[] = [
  {
    title: "One Piece",
    adapter: "manga-plus",
    chapterUrl: "https://mangaplus.shueisha.co.jp/titles/100020",
    genres: ["action", "adventure", "shonen"],
    chapterCount: 1111,
    coverUrl: null
  },
  {
    title: "Solo Leveling",
    adapter: "manganato",
    chapterUrl: "https://manganato.com/manga-dr980200",
    genres: ["action", "fantasy", "manhwa"],
    chapterCount: 201,
    coverUrl: null
  },
  {
    title: "Tower of God",
    adapter: "webtoons",
    chapterUrl: "https://www.webtoons.com/en/fantasy/tower-of-god/list",
    genres: ["fantasy", "action", "webtoon"],
    chapterCount: 640,
    coverUrl: null
  },
  {
    title: "Blue Lock",
    adapter: "mangadex",
    chapterUrl: "https://mangadex.org/title/4141c5dc-c525-4df5-8a9b-997ff941d0cc/blue-lock",
    genres: ["sports", "drama", "shonen"],
    chapterCount: 302,
    coverUrl: null
  },
  {
    title: "Omniscient Reader",
    adapter: "bato-to",
    chapterUrl: "https://bato.to/title/84989-omniscient-reader-s-viewpoint",
    genres: ["action", "fantasy", "manhwa"],
    chapterCount: 260,
    coverUrl: null
  },
  {
    title: "Lookism",
    adapter: "webtoons",
    chapterUrl: "https://www.webtoons.com/en/drama/lookism/list",
    genres: ["drama", "action", "webtoon"],
    chapterCount: 520,
    coverUrl: null
  },
  {
    title: "Kingdom",
    adapter: "mangasee",
    chapterUrl: "https://mangasee123.com/manga/Kingdom",
    genres: ["historical", "action", "seinen"],
    chapterCount: 810,
    coverUrl: null
  },
  {
    title: "The Beginning After The End",
    adapter: "toonily",
    chapterUrl: "https://toonily.com/webtoon/the-beginning-after-the-end",
    genres: ["fantasy", "action", "isekai"],
    chapterCount: 212,
    coverUrl: null
  }
];

const SUPPORTED_ADAPTER_IDS = new Set<string>([
  "mangakatana",
  "mangadex",
  "webtoons",
  "mangasee",
  "manga-plus",
  "bato-to",
  "toonily",
  "manganato"
]);

const STOP_WORDS = new Set([
  "the",
  "a",
  "an",
  "part",
  "season",
  "chapter",
  "episode",
  "manga",
  "manhwa",
  "comic",
  "webtoon"
]);

const trendingCache: {
  value: SuggestionsResponse["trending"];
  expiresAt: number;
} = {
  value: [],
  expiresAt: 0
};

function normalizeSeriesName(title: string): string[] {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0 && !STOP_WORDS.has(token));
}

function jaccardConfidence(aTokens: string[], bTokens: string[]): number {
  const a = new Set(aTokens);
  const b = new Set(bTokens);
  const intersection = Array.from(a).filter((token) => b.has(token)).length;
  const union = new Set([...a, ...b]).size;

  if (union === 0) {
    return 0;
  }

  return Math.round((intersection / union) * 100);
}

function getTrending(): SuggestionsResponse["trending"] {
  const now = Date.now();
  if (trendingCache.expiresAt > now && trendingCache.value.length > 0) {
    return trendingCache.value;
  }

  const value = CATALOG.filter((series) => SUPPORTED_ADAPTER_IDS.has(series.adapter))
    .slice(0, 6)
    .map((series) => ({
      title: series.title,
      adapter: series.adapter,
      chapterUrl: series.chapterUrl
    }));

  trendingCache.value = value;
  trendingCache.expiresAt = now + 6 * 60 * 60 * 1000;
  return value;
}

export function buildSuggestions(input: BuildSuggestionsInput): SuggestionsResponse {
  const baseTokens = normalizeSeriesName(input.seriesTitle);

  const alternatives = CATALOG.map((series) => ({
    ...series,
    confidence: jaccardConfidence(baseTokens, normalizeSeriesName(series.title))
  }))
    .filter((series) => series.adapter !== input.currentAdapter)
    .filter((series) => series.confidence >= 80)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5)
    .map((series) => ({
      adapter: series.adapter,
      title: series.title,
      chapterUrl: series.chapterUrl,
      confidence: series.confidence
    }));

  const normalizedGenres = input.genres.map((genre) => genre.toLowerCase());
  const similarSeries = CATALOG.filter((series) => series.title !== input.seriesTitle)
    .map((series) => {
      const overlap = series.genres.filter((genre) => normalizedGenres.includes(genre.toLowerCase())).length;
      return {
        series,
        overlap
      };
    })
    .filter((entry) => entry.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap)
    .slice(0, 6)
    .map((entry) => ({
      title: entry.series.title,
      coverUrl: entry.series.coverUrl,
      chapterCount: entry.series.chapterCount
    }));

  return {
    alternatives,
    similarSeries,
    trending: getTrending()
  };
}
