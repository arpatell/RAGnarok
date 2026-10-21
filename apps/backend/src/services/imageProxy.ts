import { setTimeout as wait } from "node:timers/promises";
import { LRUCache } from "lru-cache";
import { IngestError } from "../errors.js";

interface CachedImage { body: Buffer; contentType: string }
interface Limits { concurrency: number; maxQueued: number; queueWaitMs: number; timeoutMs: number; maxBytes: number; cacheBytes: number }
const DEFAULTS: Limits = { concurrency: 4, maxQueued: 32, queueWaitMs: 5000, timeoutMs: 20000, maxBytes: 8 * 1024 * 1024, cacheBytes: 32 * 1024 * 1024 };
const RETRYABLE = new Set([408, 425, 429, 500, 502, 503, 504, 520, 522, 524]);

export class ImageProxy {
  private readonly limits: Limits;
  private readonly cache: LRUCache<string, CachedImage>;
  private readonly pending = new Map<string, Promise<CachedImage>>();
  private active = 0;
  private readonly queue: Array<{ start: () => void }> = [];

  constructor(limits: Partial<Limits> = {}, private readonly fetcher: typeof fetch = fetch) {
    this.limits = { ...DEFAULTS, ...limits };
    this.cache = new LRUCache({ maxSize: this.limits.cacheBytes, sizeCalculation: (value: CachedImage) => value.body.byteLength, ttl: 600000 });
  }

  private async acquire(): Promise<void> {
    if (this.active < this.limits.concurrency) { this.active++; return; }
    if (this.queue.length >= this.limits.maxQueued) throw new IngestError("INGEST_FAILED", "Images are busy loading. Please try again shortly.", 503);
    await new Promise<void>((resolve, reject) => {
      const entry = { start: () => { clearTimeout(timer); resolve(); } };
      const timer = setTimeout(() => {
        const position = this.queue.indexOf(entry);
        if (position >= 0) this.queue.splice(position, 1);
        reject(new IngestError("INGEST_FAILED", "Images are busy loading. Please try again shortly.", 503));
      }, this.limits.queueWaitMs);
      this.queue.push(entry);
    });
  }

  private release(): void {
    // Hand the reserved slot directly to the next waiter, without a race that
    // lets a new request and the waiter both claim the same slot.
    const next = this.queue.shift();
    if (next) next.start(); else this.active--;
  }

  private async read(response: Response, signal: AbortSignal): Promise<CachedImage> {
    if (Number(response.headers.get("content-length")) > this.limits.maxBytes) {
      await response.body?.cancel();
      throw new IngestError("INGEST_FAILED", "Source image exceeds the size limit.", 502);
    }
    const contentType = response.headers.get("content-type") ?? "image/jpeg";
    if (!contentType.startsWith("image/") && !contentType.startsWith("application/octet-stream")) {
      await response.body?.cancel();
      throw new IngestError("INGEST_FAILED", "The source did not return an image.", 502);
    }
    const reader = response.body?.getReader();
    if (!reader) throw new IngestError("INGEST_FAILED", "Source image was empty.", 502);
    const abort = () => { void reader.cancel().catch(() => {}); };
    signal.addEventListener("abort", abort, { once: true });
    const chunks: Buffer[] = [];
    let size = 0;
    try {
      signal.throwIfAborted();
      while (true) {
        const result = await reader.read();
        signal.throwIfAborted();
        if (result.done) break;
        size += result.value.byteLength;
        if (size > this.limits.maxBytes) {
          await reader.cancel();
          throw new IngestError("INGEST_FAILED", "Source image exceeds the size limit.", 502);
        }
        chunks.push(Buffer.from(result.value.buffer, result.value.byteOffset, result.value.byteLength));
      }
    } finally { signal.removeEventListener("abort", abort); reader.releaseLock(); }
    if (!size) throw new IngestError("INGEST_FAILED", "Source image was empty.", 502);
    return { body: Buffer.concat(chunks, size), contentType };
  }

  async get(url: URL): Promise<CachedImage> {
    const key = url.toString();
    const cached = this.cache.get(key);
    if (cached) return cached;
    const existing = this.pending.get(key);
    if (existing) return existing;
    const promise = (async () => {
      await this.acquire();
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.limits.timeoutMs);
      const deadline = Date.now() + this.limits.timeoutMs;
      try {
        for (let attempt = 0; attempt < 2; attempt++) {
          let response: Response;
          try {
            response = await this.fetcher(key, { signal: controller.signal, headers: {
              Referer: url.origin, Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
              "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            } });
          } catch (error) {
            if (attempt || controller.signal.aborted) throw error;
            await wait(250, undefined, { signal: controller.signal });
            continue;
          }
          if (response.ok) {
            const image = await this.read(response, controller.signal);
            this.cache.set(key, image);
            return image;
          }
          await response.body?.cancel();
          const retry = response.headers.get("retry-after");
          const seconds = retry ? Number(retry) : NaN;
          const retryMs = retry ? Math.max(0, Number.isFinite(seconds) ? seconds * 1000 : Date.parse(retry) - Date.now()) : 250;
          if (attempt || !RETRYABLE.has(response.status) || retryMs >= deadline - Date.now()) {
            throw new IngestError("INGEST_FAILED", `Source image returned ${response.status}.`, response.status);
          }
          await wait(Number.isFinite(retryMs) ? retryMs : 250, undefined, { signal: controller.signal });
        }
        throw new IngestError("INGEST_FAILED", "Image source is unavailable. Please try again shortly.", 502);
      } catch (error) {
        if (controller.signal.aborted) throw new IngestError("INGEST_FAILED", "The image source took too long to respond. Please try again.", 504);
        throw error;
      } finally { clearTimeout(timer); this.release(); }
    })();
    this.pending.set(key, promise);
    try { return await promise; } finally { this.pending.delete(key); }
  }
}

const images = new ImageProxy();
export const fetchImageWithRetry = (url: URL) => images.get(url);
