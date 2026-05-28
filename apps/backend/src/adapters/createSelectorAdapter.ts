import * as cheerio from "cheerio";
import type { Adapter, ReaderMode } from "../types.js";
import {
  extractChapterLinks,
  extractPanelCandidates,
  extractTexts,
  getFirstAttr,
  getFirstText,
  inferChapterNumber,
  normalizeStatus
} from "../services/html.js";

interface SelectorAdapterConfig {
  id: string;
  displayName: string;
  domains: string[];
  defaultMode: ReaderMode;
  selectors: {
    title: string[];
    cover: string[];
    chapterTitle: string[];
    chapterLinks: string[];
    panel: string[];
    genres?: string[];
    status?: string[];
  };
}

export function createSelectorAdapter(config: SelectorAdapterConfig): Adapter {
  return {
    id: config.id,
    displayName: config.displayName,
    domains: config.domains,
    defaultMode: config.defaultMode,
    parse(html: string, chapterUrl: string) {
      const $ = cheerio.load(html);

      const title =
        getFirstText($, config.selectors.title) ||
        getFirstText($, ["meta[property='og:title']", "title"]) ||
        "Unknown Series";

      const chapterTitle = getFirstText($, config.selectors.chapterTitle);
      const chapterNumber = inferChapterNumber(chapterTitle ?? "", chapterUrl);
      const panelCandidates = extractPanelCandidates($, config.selectors.panel, chapterUrl);
      const chapterList = extractChapterLinks($, config.selectors.chapterLinks, chapterUrl);
      const genres = config.selectors.genres ? extractTexts($, config.selectors.genres, 12) : [];
      const statusText = config.selectors.status ? getFirstText($, config.selectors.status) : null;

      return {
        series: {
          title,
          coverUrl: getFirstAttr($, config.selectors.cover, "src", chapterUrl),
          genres,
          status: normalizeStatus(statusText)
        },
        chapter: {
          number: chapterNumber,
          title: chapterTitle,
          panelCandidates
        },
        chapterList
      };
    }
  };
}
