export const PRODUCTION_ORIGIN = "https://ragnarok-reader.vercel.app";
export const SITE_NAME = "RAGnarok Reader";
export const SITE_TITLE = "RAGnarok | Manga & Anime Discovery + Manga Reader";
export const SITE_DESCRIPTION = "Discover manga, manhwa, and anime by plot, character, or title with RAGnarok's AI search. Paste a supported chapter URL to read manga in your browser.";
export const GUIDE_PATH = "/docs/product-guide.html";

export function resolveSiteOrigin(env = process.env) {
  const url = new URL(env.SITE_URL?.trim() || env.VITE_SITE_URL?.trim() || PRODUCTION_ORIGIN);
  if (url.protocol !== "https:" || url.username || url.password || url.search || url.hash || url.pathname !== "/") {
    throw new Error("SITE_URL must be an HTTPS origin without credentials, a path, a query, or a fragment.");
  }
  return url.origin;
}

export function structuredData(origin) {
  const url = `${origin}/`;
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite", "@id": `${url}#website`, url, name: SITE_NAME,
        alternateName: ["RAGnarok", "Ragnarok Reader"], inLanguage: "en",
        description: SITE_DESCRIPTION
      },
      {
        "@type": "WebPage", "@id": `${url}#webpage`, url, name: SITE_TITLE,
        description: SITE_DESCRIPTION, inLanguage: "en",
        isPartOf: { "@id": `${url}#website` }, mainEntity: { "@id": `${url}#app` }
      },
      {
        "@type": "WebApplication", "@id": `${url}#app`, url, name: SITE_NAME,
        applicationCategory: "MultimediaApplication", operatingSystem: "Web Browser",
        browserRequirements: "Requires JavaScript", description: SITE_DESCRIPTION,
        featureList: [
          "Manga, manhwa, and anime discovery by natural-language descriptions",
          "Hybrid semantic and keyword search with retrieval-augmented generation (RAG)",
          "Search by title, alternate title, plot, character, and metadata",
          "Read manga from a supported chapter URL",
          "Paginated manga and continuous manhwa reading",
          "Reading history, favorites, and bookmarks"
        ],
        isAccessibleForFree: true
      }
    ]
  };
}

export function seoTags(origin, preview = false) {
  const meta = (key, value, content) => ({ tag: "meta", attrs: { [key]: value, content }, injectTo: "head" });
  return [
    { tag: "title", children: SITE_TITLE, injectTo: "head" },
    meta("name", "description", SITE_DESCRIPTION),
    meta("name", "robots", preview ? "noindex,follow" : "index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1"),
    meta("name", "application-name", SITE_NAME),
    meta("property", "og:type", "website"),
    meta("property", "og:site_name", SITE_NAME),
    meta("property", "og:locale", "en_US"),
    meta("property", "og:title", SITE_TITLE),
    meta("property", "og:description", SITE_DESCRIPTION),
    meta("property", "og:url", `${origin}/`),
    meta("property", "og:image", `${origin}/social-preview.png`),
    meta("property", "og:image:alt", "RAGnarok Reader - Manga and anime discovery, plus reading from a chapter URL"),
    meta("property", "og:image:width", "1200"),
    meta("property", "og:image:height", "630"),
    meta("name", "twitter:card", "summary_large_image"),
    meta("name", "twitter:title", SITE_TITLE),
    meta("name", "twitter:description", SITE_DESCRIPTION),
    meta("name", "twitter:image", `${origin}/social-preview.png`),
    meta("name", "twitter:image:alt", "RAGnarok Reader - Manga and anime discovery, plus reading from a chapter URL"),
    { tag: "link", attrs: { rel: "canonical", href: `${origin}/` }, injectTo: "head" },
    { tag: "script", attrs: { type: "application/ld+json" }, children: JSON.stringify(structuredData(origin)).replace(/</g, "\\u003c"), injectTo: "head" }
  ];
}

export function sitemapXml(origin) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${["/", GUIDE_PATH].map((path) => `  <url><loc>${origin}${path}</loc></url>`).join("\n")}\n</urlset>\n`;
}

export function robotsTxt(origin) {
  return `User-agent: *\nAllow: /\n\nSitemap: ${origin}/sitemap.xml\n`;
}
