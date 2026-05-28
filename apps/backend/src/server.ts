import fs from "node:fs";
import path from "node:path";
import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import rateLimit from "express-rate-limit";
import { LRUCache } from "lru-cache";
import { z } from "zod";
import { listSupportedAdapters } from "./adapters/index.js";
import { isIngestError, IngestError, toErrorPayload } from "./errors.js";
import { ingestUrl } from "./services/ingest.js";
import { searchMangaRag } from "./services/ragProxy.js";
import { buildSuggestions } from "./services/suggestions.js";
import { resolveMangaKatanaReadNow, searchManga } from "./services/webSearch.js";

function loadEnvironment(): void {
  const cwd = process.cwd();
  const candidates = [
    path.resolve(cwd, ".env"),
    path.resolve(cwd, "..", ".env"),
    path.resolve(cwd, "..", "..", ".env")
  ];

  for (const envPath of candidates) {
    if (!fs.existsSync(envPath)) {
      continue;
    }

    dotenv.config({ path: envPath, override: false });
  }
}

loadEnvironment();

const PORT = Number.parseInt(process.env.PORT ?? "8787", 10);
const corsAllowedOrigins = (process.env.CORS_ALLOWED_ORIGINS ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter((origin) => origin.length > 0);
const MAX_IMAGE_FETCH_CONCURRENCY = 6;
const IMAGE_RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504, 520, 522, 524]);

interface CachedImage {
  body: Buffer;
  contentType: string;
}

const imageResponseCache = new LRUCache<string, CachedImage>({
  maxSize: 64 * 1024 * 1024,
  sizeCalculation: (value) => value.body.byteLength,
  ttl: 10 * 60 * 1000
});

const inFlightImageFetches = new Map<string, Promise<CachedImage>>();
let activeImageFetches = 0;
const pendingImageFetchWaiters: Array<() => void> = [];

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeRetryDelayMs(attempt: number, retryAfterHeader: string | null): number {
  if (retryAfterHeader) {
    const seconds = Number.parseInt(retryAfterHeader, 10);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.min(seconds * 1000, 8_000);
    }
  }

  const exponential = Math.min(8_000, 250 * 2 ** attempt);
  const jitter = Math.floor(Math.random() * 200);
  return exponential + jitter;
}

async function withImageConcurrencySlot<T>(operation: () => Promise<T>): Promise<T> {
  if (activeImageFetches >= MAX_IMAGE_FETCH_CONCURRENCY) {
    await new Promise<void>((resolve) => {
      pendingImageFetchWaiters.push(resolve);
    });
  }

  activeImageFetches += 1;

  try {
    return await operation();
  } finally {
    activeImageFetches = Math.max(0, activeImageFetches - 1);
    const next = pendingImageFetchWaiters.shift();
    if (next) {
      next();
    }
  }
}

async function fetchImageWithRetry(imageUrl: URL): Promise<CachedImage> {
  const key = imageUrl.toString();
  const cached = imageResponseCache.get(key);
  if (cached) {
    return cached;
  }

  const inFlight = inFlightImageFetches.get(key);
  if (inFlight) {
    return inFlight;
  }

  const fetchPromise = withImageConcurrencySlot(async () => {
    let lastStatus = 502;

    for (let attempt = 0; attempt < 6; attempt += 1) {
      const response = await fetch(imageUrl.toString(), {
        headers: {
          Referer: imageUrl.origin,
          Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
          "Cache-Control": "no-cache",
          "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
      });

      if (response.ok) {
        const body = Buffer.from(await response.arrayBuffer());
        const image: CachedImage = {
          body,
          contentType: response.headers.get("content-type") ?? "image/jpeg"
        };

        imageResponseCache.set(key, image);
        return image;
      }

      lastStatus = response.status;

      if (!IMAGE_RETRYABLE_STATUS.has(response.status)) {
        throw new IngestError("INGEST_FAILED", `Source image returned ${response.status}.`, response.status);
      }

      const retryDelayMs = computeRetryDelayMs(attempt, response.headers.get("retry-after"));
      await delay(retryDelayMs);
    }

    throw new IngestError("INGEST_FAILED", `Source image returned ${lastStatus}.`, lastStatus);
  });

  inFlightImageFetches.set(key, fetchPromise);

  try {
    return await fetchPromise;
  } finally {
    inFlightImageFetches.delete(key);
  }
}

const app = express();
app.use(
  cors(
    corsAllowedOrigins.length === 0
      ? {}
      : {
          origin(origin, callback) {
            if (!origin) {
              callback(null, true);
              return;
            }
            callback(null, corsAllowedOrigins.includes(origin));
          }
        }
  )
);
app.use(express.json({ limit: "1mb" }));
app.set("etag", false);

app.use((req, res, next) => {
  if (!req.path.startsWith("/api")) {
    next();
    return;
  }

  const startedAt = Date.now();
  res.on("finish", () => {
    const elapsedMs = Date.now() - startedAt;
    const logLine = `[api] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${elapsedMs}ms)`;

    if (res.statusCode >= 400) {
      // eslint-disable-next-line no-console
      console.error(logLine);
      return;
    }

    // eslint-disable-next-line no-console
    console.log(logLine);
  });

  next();
});

const ingestLimiter = rateLimit({
  windowMs: 60_000,
  limit: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Rate limit exceeded. Please wait before requesting another chapter.",
    code: "RATE_LIMIT"
  }
});

const imageLimiter = rateLimit({
  windowMs: 60_000,
  limit: 1200,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many image relay requests. Please wait briefly.",
    code: "RATE_LIMIT"
  }
});

const searchLimiter = rateLimit({
  windowMs: 60_000,
  limit: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many search requests. Please wait before searching again.",
    code: "RATE_LIMIT"
  }
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    now: new Date().toISOString()
  });
});

app.get("/adapters", (_req, res) => {
  res.json({
    adapters: listSupportedAdapters()
  });
});

app.post("/api/ingest", ingestLimiter, async (req, res, next) => {
  try {
    const payload = z
      .object({
        url: z.string().min(1)
      })
      .parse(req.body);

    const targetUrl = payload.url.trim();
    const result = await ingestUrl(targetUrl);

    try {
      const host = new URL(targetUrl).hostname;
      // eslint-disable-next-line no-console
      console.log(
        `[ingest] ok host=${host} adapter=${result.sourceAdapter} source=${result.extractionSource ?? "unknown"} panels=${result.chapter.totalPages}`
      );
    } catch {
      // Ignore malformed URL logging failures because URL is validated upstream.
    }

    res.json(result);
  } catch (error) {
    next(error);
  }
});

app.get("/api/image", imageLimiter, async (req, res, next) => {
  try {
    const rawUrl = z.string().url().parse(req.query.url);
    const imageUrl = new URL(rawUrl);

    if (!["http:", "https:"].includes(imageUrl.protocol)) {
      throw new IngestError("INVALID_URL", "Only http(s) images can be proxied.", 400);
    }

    const image = await fetchImageWithRetry(imageUrl);

    res.setHeader("Content-Type", image.contentType);
    res.setHeader("Cache-Control", "public, max-age=900");
    res.send(image.body);
  } catch (error) {
    next(error);
  }
});

app.get("/api/search", searchLimiter, async (req, res, next) => {
  try {
    const q = z.string().min(1).max(200).parse(req.query.q);
    // Hard outer cap — prevents the connection from hanging if search runs over budget
    const results = await Promise.race([
      searchManga(q),
      new Promise<[]>((resolve) => setTimeout(() => resolve([]), 45_000))
    ]);
    res.json({ results });
  } catch (error) {
    next(error);
  }
});

app.get("/api/read-now", searchLimiter, async (req, res, next) => {
  try {
    const title = z.string().min(1).max(220).parse(req.query.title).trim();
    const resolved = await resolveMangaKatanaReadNow(title);
    if (!resolved) {
      throw new IngestError(
        "INGEST_FAILED",
        `Could not find a readable MangaKatana Chapter 1 match for "${title}".`,
        404
      );
    }
    res.json(resolved);
  } catch (error) {
    next(error);
  }
});

app.get("/api/rag/search", searchLimiter, async (req, res, next) => {
  try {
    const q = z.string().min(1).max(400).parse(req.query.q);
    res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    res.setHeader("Surrogate-Control", "no-store");
    const ragResult = await searchMangaRag(q.trim());
    res.json(ragResult);
  } catch (error) {
    next(error);
  }
});

app.post("/api/suggestions", ingestLimiter, (req, res, next) => {
  try {
    const payload = z
      .object({
        seriesTitle: z.string().min(1),
        genres: z.array(z.string()).default([]),
        currentAdapter: z.string().default("unknown")
      })
      .parse(req.body);

    const suggestions = buildSuggestions(payload);
    res.json(suggestions);
  } catch (error) {
    next(error);
  }
});

app.use((error: unknown, req: express.Request, res: express.Response, _next: express.NextFunction) => {
  const requestUrl = typeof req.body?.url === "string" ? req.body.url : "";

  if (isIngestError(error)) {
    // eslint-disable-next-line no-console
    console.error(
      `[ingest] error code=${error.code} status=${error.status} path=${req.path} url=${requestUrl} message=${error.message}`
    );
    res.status(error.status).json(toErrorPayload(error));
    return;
  }

  if (error instanceof z.ZodError) {
    // eslint-disable-next-line no-console
    console.error(
      `[api] bad-request path=${req.path} url=${requestUrl} issues=${error.issues.length}`
    );
    res.status(400).json({
      error: "Invalid request payload.",
      code: "BAD_REQUEST",
      details: error.flatten()
    });
    return;
  }

  const message = error instanceof Error ? error.message : "Unexpected server error.";
  // eslint-disable-next-line no-console
  console.error(`[api] unexpected-error path=${req.path} url=${requestUrl} message=${message}`);
  res.status(500).json({
    error: message,
    code: "INGEST_FAILED"
  });
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`RAGnarok backend running on http://localhost:${PORT}`);
});
