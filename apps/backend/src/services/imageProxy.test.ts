import assert from "node:assert/strict";
import test from "node:test";
import { setTimeout as wait } from "node:timers/promises";
import { ImageProxy } from "./imageProxy.js";

const url = (id: number) => new URL(`https://images.example.test/${id}.png`);
const image = () => new Response(new Uint8Array([1, 2, 3]), { headers: { "content-type": "image/png" } });

test("image cache coalesces readers and evicts by bytes", async () => {
  let calls = 0;
  const proxy = new ImageProxy({ cacheBytes: 4 }, async () => { calls++; await wait(5); return image(); });
  const first = await Promise.all([proxy.get(url(1)), proxy.get(url(1))]);
  assert.deepEqual(first[0]?.body, Buffer.from([1, 2, 3]));
  assert.equal(calls, 1);
  await proxy.get(url(1));
  assert.equal(calls, 1);
  await proxy.get(url(2));
  await proxy.get(url(1));
  assert.equal(calls, 3);
});

test("a stalled image is aborted and its slot is released", async () => {
  let calls = 0;
  const proxy = new ImageProxy({ timeoutMs: 15, concurrency: 1 }, async (_url, options) => {
    if (++calls > 1) return image();
    return new Promise((_resolve, reject) => options?.signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true }));
  });
  await assert.rejects(proxy.get(url(1)), (error: any) => error.status === 504);
  assert.equal((await proxy.get(url(2))).body.length, 3);
});

test("queued images have bounded wait and do not leak capacity", async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let active = 0, peak = 0;
  const proxy = new ImageProxy({ concurrency: 1, maxQueued: 1, queueWaitMs: 15 }, async () => {
    active++; peak = Math.max(active, peak); await gate; active--; return image();
  });
  const first = proxy.get(url(1));
  const second = proxy.get(url(2));
  await assert.rejects(proxy.get(url(3)), (error: any) => error.status === 503);
  await assert.rejects(second, (error: any) => error.status === 503);
  release(); await first;
  await proxy.get(url(4));
  assert.equal(peak, 1);
});

test("chunked oversized images are stopped and never cached", async () => {
  let calls = 0;
  const proxy = new ImageProxy({ maxBytes: 2 }, async () => { calls++; return image(); });
  for (let i = 0; i < 2; i++) await assert.rejects(proxy.get(url(1)), /size limit/);
  assert.equal(calls, 2);
});

test("a slow response body shares the network deadline", async () => {
  const proxy = new ImageProxy({ timeoutMs: 15 }, async () => new Response(new ReadableStream({ start() {} }), { headers: { "content-type": "image/png" } }));
  await assert.rejects(proxy.get(url(1)), (error: any) => error.status === 504);
});

test("HTML error pages are not cached as images", async () => {
  const proxy = new ImageProxy({}, async () => new Response("Challenge", { headers: { "content-type": "text/html" } }));
  await assert.rejects(proxy.get(url(1)), /did not return an image/);
});

test("Retry-After beyond the request budget does not cause a retry storm", async () => {
  let calls = 0;
  const proxy = new ImageProxy({ timeoutMs: 100 }, async () => { calls++; return new Response("Busy", { status: 429, headers: { "retry-after": "60" } }); });
  await assert.rejects(proxy.get(url(1)), (error: any) => error.status === 429);
  assert.equal(calls, 1);
});
