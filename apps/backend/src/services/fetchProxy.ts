import { LRUCache } from "lru-cache";

const HTML_CACHE_TTL_MS = 15 * 60 * 1000;
const FETCH_TIMEOUT_MS = 15_000;

const USER_AGENTS = [
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
  "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
];

const htmlCache = new LRUCache<string, FetchResult>({
  max: 500,
  ttl: HTML_CACHE_TTL_MS
});

const inFlightFetches = new Map<string, Promise<FetchResult>>();

export interface FetchResult {
  requestedUrl: string;
  finalUrl: string;
  status: number;
  html: string;
}

function randomUserAgent(): string {
  const candidate = USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
  return candidate || USER_AGENTS[0] || "Mozilla/5.0";
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchChapterHtml(url: string): Promise<FetchResult> {
  const cached = htmlCache.get(url);
  if (cached) {
    return cached;
  }

  const existingInFlight = inFlightFetches.get(url);
  if (existingInFlight) {
    return existingInFlight;
  }

  const requestPromise = (async () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      let lastResult: FetchResult | null = null;

      for (let attempt = 0; attempt < 2; attempt += 1) {
        const response = await fetch(url, {
          redirect: "follow",
          signal: controller.signal,
          headers: {
            "User-Agent": randomUserAgent(),
            Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache"
          }
        });

        const html = await response.text();
        const result: FetchResult = {
          requestedUrl: url,
          finalUrl: response.url || url,
          status: response.status,
          html
        };
        lastResult = result;

        if (response.ok) {
          htmlCache.set(url, result);
          return result;
        }

        if (![403, 429].includes(response.status)) {
          return result;
        }

        if (attempt === 0) {
          await delay(200);
        }
      }

      if (lastResult) {
        return lastResult;
      }

      return {
        requestedUrl: url,
        finalUrl: url,
        status: 520,
        html: ""
      };
    } finally {
      clearTimeout(timeout);
    }
  })();

  inFlightFetches.set(url, requestPromise);

  try {
    return await requestPromise;
  } finally {
    inFlightFetches.delete(url);
  }
}
