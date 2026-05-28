import { createSelectorAdapter } from "./createSelectorAdapter.js";
import type { Adapter, SupportedAdapterSummary } from "../types.js";

const ADAPTERS: Adapter[] = [
  createSelectorAdapter({
    id: "mangakatana",
    displayName: "MangaKatana",
    domains: ["mangakatana.com"],
    defaultMode: "paginated",
    selectors: {
      title: ["h1.heading", "h1.title", "meta[property='og:title']"],
      cover: [".cover img", "meta[property='og:image']"],
      chapterTitle: ["h1.heading", ".chapter-heading", "title"],
      chapterLinks: [".chapters a", ".chapter-list a", "a[href*='/c']"],
      panel: ["#imgs img", ".chapter-content img", ".vung-doc img"],
      genres: ["a[href*='/genre/']", ".genres a"],
      status: [".d-cell-small-value", ".status"]
    }
  }),
  createSelectorAdapter({
    id: "webtoons",
    displayName: "Webtoons",
    domains: ["webtoons.com"],
    defaultMode: "scroll",
    selectors: {
      title: ["h1.subj", "h1", "meta[property='og:title']"],
      cover: [".detail_header img", "meta[property='og:image']"],
      chapterTitle: ["h1.subj_episode", "title"],
      chapterLinks: ["#_listUl a", ".episode_lst a", "a[href*='episode_no']"],
      panel: ["#_imageList img", "._images img", "img[data-url]"],
      genres: [".genre", ".info a"],
      status: [".day_info", ".info"]
    }
  }),
  createSelectorAdapter({
    id: "weebcentral",
    displayName: "Weebcentral",
    domains: ["weebcentral.com"],
    defaultMode: "scroll",
    selectors: {
      title: ["h1", "meta[property='og:title']", "title"],
      cover: ["meta[property='og:image']", ".series-cover img", ".poster img"],
      chapterTitle: ["h1", ".chapter-title", "title"],
      chapterLinks: [".chapter-list a", "a[href*='/chapters/']", "a[href*='/chapter']"],
      panel: [".chapter-content img", ".reader img", "main img"],
      genres: ["a[href*='/genre']", ".genres a"],
      status: [".status", ".series-status", ".meta-item"]
    }
  }),
  createSelectorAdapter({
    id: "mangakakalot",
    displayName: "Mangakakalot",
    domains: ["mangakakalot.com"],
    defaultMode: "scroll",
    selectors: {
      title: [".story-info-right h1", "h1", "meta[property='og:title']"],
      cover: [".manga-info-pic img", "meta[property='og:image']"],
      chapterTitle: [".panel-chapter-info-top h1", ".chapter-heading", "title"],
      chapterLinks: [".row-content-chapter a", ".chapter-list a", "a[href*='/chapter-']"],
      panel: [".container-chapter-reader img", ".vung-doc img", ".chapter-content img"],
      genres: [".story-info-right-extent a[href*='genre']", ".genres a"],
      status: [".story-info-right-extent", ".status"]
    }
  }),
  createSelectorAdapter({
    id: "assortedscans",
    displayName: "AssortedScans",
    domains: ["assortedscans.com"],
    defaultMode: "scroll",
    selectors: {
      title: ["h1", ".entry-title", "meta[property='og:title']"],
      cover: [".summary_image img", ".thumb img", "meta[property='og:image']"],
      chapterTitle: ["h1", ".entry-title", "title"],
      chapterLinks: [".wp-manga-chapter a", ".chapter-list a", "a[href*='/chapter-']"],
      panel: [".reading-content img", ".entry-content img", ".chapter-content img"],
      genres: [".genres-content a", "a[href*='/genre/']"],
      status: [".post-status .summary-content", ".summary-content", ".status"]
    }
  }),
  createSelectorAdapter({
    id: "mangafire",
    displayName: "MangaFire (Experimental)",
    domains: ["mangafire.to"],
    defaultMode: "scroll",
    selectors: {
      title: ["h1", "meta[property='og:title']", "title"],
      cover: ["meta[property='og:image']", ".manga-poster img", ".poster img"],
      chapterTitle: ["h1", ".chapter-name", ".chapter-title", "title"],
      chapterLinks: ["a[href*='/chapter-']", "a[href*='/read/']", ".chapter-list a"],
      panel: ["#reader img", ".reader img", ".chapter-content img", "main img"],
      genres: ["a[href*='/genre/']", ".genres a"],
      status: [".status", ".meta-item"]
    }
  })
];

export function resolveAdapterForUrl(inputUrl: string): Adapter | null {
  let host: string;
  try {
    host = new URL(inputUrl).hostname.toLowerCase();
  } catch {
    return null;
  }

  const normalizedHost = host.replace(/^www\./, "");

  return (
    ADAPTERS.find((adapter) =>
      adapter.domains.some(
        (domain) =>
          normalizedHost === domain ||
          normalizedHost === `www.${domain}` ||
          normalizedHost.endsWith(`.${domain}`)
      )
    ) ?? null
  );
}

export function listSupportedAdapters(): SupportedAdapterSummary[] {
  return ADAPTERS.map((adapter) => ({
    id: adapter.id,
    displayName: adapter.displayName,
    domains: adapter.domains,
    defaultMode: adapter.defaultMode
  }));
}
