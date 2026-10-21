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
import {
  fetchMediaMeta,
  resolveChapterCount,
  resolveRecommendations,
} from "./services/metadata.js";
import { ingestUrl } from "./services/ingest.js";
import { fetchChapterHtml } from "./services/fetchProxy.js";
import { fetchImageWithRetry } from "./services/imageProxy.js";
import { searchMangaRag } from "./services/ragProxy.js";
import { buildSuggestions } from "./services/suggestions.js";
import { ReadNowUnavailableError, resolveDirectReadNow, searchManga } from "./services/webSearch.js";

function loadEnvironment(): void {
  const cwd = process.cwd();
  const candidates = [
    path.resolve(cwd, ".env"),
    path.resolve(cwd, "..", ".env"),
    path.resolve(cwd, "..", "..", ".env"),
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
          },
        },
  ),
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
    error:
      "Rate limit exceeded. Please wait before requesting another chapter.",
    code: "RATE_LIMIT",
  },
});

const imageLimiter = rateLimit({
  windowMs: 60_000,
  limit: 1200,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many image relay requests. Please wait briefly.",
    code: "RATE_LIMIT",
  },
});

const searchLimiter = rateLimit({
  windowMs: 60_000,
  limit: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many search requests. Please wait before searching again.",
    code: "RATE_LIMIT",
  },
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    now: new Date().toISOString(),
  });
});

app.get("/adapters", (_req, res) => {
  res.json({
    adapters: listSupportedAdapters(),
  });
});

app.post("/api/ingest", ingestLimiter, async (req, res, next) => {
  try {
    const payload = z
      .object({
        url: z.string().min(1),
      })
      .parse(req.body);

    const targetUrl = payload.url.trim();
    const result = await ingestUrl(targetUrl);

    try {
      const host = new URL(targetUrl).hostname;
      // eslint-disable-next-line no-console
      console.log(
        `[ingest] ok host=${host} adapter=${result.sourceAdapter} source=${result.extractionSource ?? "unknown"} panels=${result.chapter.totalPages}`,
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
      throw new IngestError(
        "INVALID_URL",
        "Only http(s) images can be proxied.",
        400,
      );
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
      new Promise<[]>((resolve) => setTimeout(() => resolve([]), 45_000)),
    ]);
    res.json({ results });
  } catch (error) {
    next(error);
  }
});

app.get("/api/read-now", searchLimiter, async (req, res, next) => {
  try {
    const title = z.string().min(1).max(220).parse(req.query.title).trim();
    const rawTitles =
      typeof req.query.titles === "string" ? req.query.titles : "";
    const titleCandidates = (() => {
      if (!rawTitles) {
        return [title];
      }

      try {
        const parsed = JSON.parse(rawTitles) as unknown;
        if (Array.isArray(parsed)) {
          return [
            title,
            ...parsed.filter(
              (value): value is string => typeof value === "string",
            ),
          ];
        }
      } catch {
        // Fall back to the primary title when optional candidates are malformed.
      }

      return [title];
    })();

    const mediaHint =
      typeof req.query.media_type === "string" ? req.query.media_type : "";
    const preferManhwa = /manhwa/i.test(mediaHint);
    const malId = z.coerce.number().int().positive().optional().parse(req.query.mal_id);
    const resolved = await resolveDirectReadNow(titleCandidates, {
      preferManhwa,
      malId,
    });
    if (!resolved) {
      throw new IngestError(
        "INGEST_FAILED",
        `We couldn't find "${title}" on WeebCentral or ManhwaZone. Try searching online for a dedicated website hosting this manga/manhwa and paste a chapter link into the Paste URL option above.`,
        404,
      );
    }
    // Start only the chapter the user selected. The subsequent ingest joins
    // this same cached/in-flight fetch; no second vendor request is needed.
    void fetchChapterHtml(resolved.chapterUrl).catch(() => {});
    res.json(resolved);
  } catch (error) {
    next(error instanceof ReadNowUnavailableError ? new IngestError("INGEST_FAILED", error.message, 503) : error);
  }
});

app.get("/api/rag/search", searchLimiter, async (req, res, next) => {
  try {
    const q = z.string().min(1).max(400).parse(req.query.q);
    res.setHeader(
      "Cache-Control",
      "no-store, no-cache, must-revalidate, proxy-revalidate",
    );
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    res.setHeader("Surrogate-Control", "no-store");
    const ragResult = await searchMangaRag(q.trim());
    res.json(ragResult);
  } catch (error) {
    next(error);
  }
});

app.get("/api/rag/search/stream", searchLimiter, async (req, res) => {
  let lastWriteAt = 0;
  let waitingTimer: ReturnType<typeof setTimeout> | undefined;

  const writeLine = (payload: unknown) => {
    if (res.destroyed || res.writableEnded) return;
    res.write(`${JSON.stringify(payload)}\n`);
  };

  try {
    const q = z.string().min(1).max(400).parse(req.query.q).trim();
    res.status(200);
    res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
    res.setHeader(
      "Cache-Control",
      "no-store, no-cache, must-revalidate, proxy-revalidate",
    );
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    res.setHeader("Surrogate-Control", "no-store");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();
    waitingTimer = setTimeout(() => {
      writeLine({ type: "status", message: "Search is taking a little longer. Thanks for waiting." });
    }, 2000);
    waitingTimer.unref();
    res.once("close", () => clearTimeout(waitingTimer));

    const result = await searchMangaRag(q, {
      onProgress: async (progress) => {
        const now = Date.now();
        if (progress.stage !== "final" && now - lastWriteAt < 650) {
          return;
        }
        lastWriteAt = now;
        writeLine({ type: "result", ...progress });
      },
    });

    writeLine({ type: "done", payload: result });
    res.end();
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Smart Search failed.";
    if (!res.headersSent) {
      res.status(500);
      res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
    }
    writeLine({
      type: "error",
      error: message,
      code: isIngestError(error) ? error.code : "INGEST_FAILED",
    });
    res.end();
  } finally {
    clearTimeout(waitingTimer);
  }
});

app.post("/api/suggestions", ingestLimiter, (req, res, next) => {
  try {
    const payload = z
      .object({
        seriesTitle: z.string().min(1),
        genres: z.array(z.string()).default([]),
        currentAdapter: z.string().default("unknown"),
      })
      .parse(req.body);

    const suggestions = buildSuggestions(payload);
    res.json(suggestions);
  } catch (error) {
    next(error);
  }
});

const metaResponseCache = new LRUCache<string, { body: unknown }>({
  max: 2048,
  maxSize: 16 * 1024 * 1024,
  sizeCalculation: (value) => Math.max(1, Buffer.byteLength(JSON.stringify(value), "utf8")),
  ttl: 30 * 60 * 1000,
});
const inFlightMetaResponses = new Map<string, Promise<unknown>>();

function sendCachedMeta(
  res: express.Response,
  cacheKey: string,
  produce: () => Promise<unknown>,
): Promise<void> {
  return new Promise<void>((resolve) => {
    setImmediate(() => {
      void (async () => {
        try {
          res.setHeader("Cache-Control", "public, max-age=3600");

          const cached = metaResponseCache.get(cacheKey);
          if (cached) {
            res.json(cached.body);
            return;
          }

          let pending = inFlightMetaResponses.get(cacheKey);
          if (!pending) {
            pending = produce().finally(() => inFlightMetaResponses.delete(cacheKey));
            inFlightMetaResponses.set(cacheKey, pending);
          }
          const body = await pending;
          if (body !== null && body !== undefined) metaResponseCache.set(cacheKey, { body });
          else res.setHeader("Cache-Control", "no-store");
          res.json(body);
        } catch (error) {
          if (!res.headersSent) {
            res.setHeader("Cache-Control", "no-store");
            res.status(502).json({
              error:
                error instanceof Error
                  ? error.message
                  : "Metadata lookup failed.",
              code: "META_UNAVAILABLE",
            });
          } else {
            res.end();
          }
        } finally {
          resolve();
        }
      })();
    });
  });
}

const metaLimiter = rateLimit({
  windowMs: 60_000,
  limit: 240,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many metadata requests. Please wait briefly.",
    code: "RATE_LIMIT",
  },
});

app.get("/api/meta", metaLimiter, async (req, res, next) => {
  try {
    const payload = z
      .object({
        type: z.enum(["manga", "anime"]),
        id: z.coerce.number().int().positive(),
        title: z.string().max(300).optional(),
      })
      .parse(req.query);

    await sendCachedMeta(
      res,
      `meta:${payload.type}:${payload.id}:${payload.title ?? ""}`,
      () =>
        fetchMediaMeta({
          malId: payload.id,
          fallbackTitle: payload.title,
          preferredType: payload.type,
          strictMediaType: false,
        }),
    );
  } catch (error) {
    next(error);
  }
});

app.get("/api/meta/chapter-count", metaLimiter, async (req, res, next) => {
  try {
    const payload = z
      .object({
        title: z.string().min(1).max(300),
        chapterTitle: z.string().max(300).optional(),
        url: z.string().url().max(2000).optional(),
      })
      .parse(req.query);

    await sendCachedMeta(
      res,
      `chapters:${payload.title}:${payload.chapterTitle ?? ""}:${payload.url ?? ""}`,
      () =>
        resolveChapterCount({
          seriesTitle: payload.title,
          chapterTitle: payload.chapterTitle,
          chapterUrl: payload.url,
        }).then((count) => ({ chapters: count })),
    );
  } catch (error) {
    next(error);
  }
});

app.get("/api/meta/recommendations", metaLimiter, async (req, res, next) => {
  try {
    const payload = z
      .object({
        title: z.string().min(1).max(300),
      })
      .parse(req.query);

    await sendCachedMeta(res, `recs:${payload.title}`, () =>
      resolveRecommendations(payload.title),
    );
  } catch (error) {
    next(error);
  }
});

app.use(
  (
    error: unknown,
    req: express.Request,
    res: express.Response,
    _next: express.NextFunction,
  ) => {
    const requestUrl = typeof req.body?.url === "string" ? req.body.url : "";

    if (isIngestError(error)) {
      // eslint-disable-next-line no-console
      console.error(
        `[ingest] error code=${error.code} status=${error.status} path=${req.path} url=${requestUrl} message=${error.message}`,
      );
      res.status(error.status).json(toErrorPayload(error));
      return;
    }

    if (error instanceof z.ZodError) {
      // eslint-disable-next-line no-console
      console.error(
        `[api] bad-request path=${req.path} url=${requestUrl} issues=${error.issues.length}`,
      );
      res.status(400).json({
        error: "Invalid request payload.",
        code: "BAD_REQUEST",
        details: error.flatten(),
      });
      return;
    }

    const message =
      error instanceof Error ? error.message : "Unexpected server error.";
    // eslint-disable-next-line no-console
    console.error(
      `[api] unexpected-error path=${req.path} url=${requestUrl} message=${message}`,
    );
    res.status(500).json({
      error: message,
      code: "INGEST_FAILED",
    });
  },
);

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`RAGnarok backend running on http://localhost:${PORT}`);
});
