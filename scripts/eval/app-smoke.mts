import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";

const base = process.env.BASE_URL ?? "http://127.0.0.1:8788";
const results: Array<Record<string, unknown>> = [];
async function check(name: string, action: () => Promise<unknown>) {
  const started = performance.now();
  try {
    const detail = await action();
    results.push({ name, pass: true, ms: Math.round(performance.now() - started), detail });
  } catch (error) {
    results.push({ name, pass: false, ms: Math.round(performance.now() - started), error: String(error) });
  }
}
async function json(path: string, init?: RequestInit) {
  const response = await fetch(base + path, { ...init, signal: AbortSignal.timeout(45000) });
  assert.ok(response.ok, `${path}: ${response.status} ${await (!response.ok ? response.text() : Promise.resolve(""))}`);
  return response.json();
}

await check("backend health and supported adapters", async () => {
  assert.equal((await json("/health")).ok, true);
  const payload = await json("/adapters");
  assert.ok(payload.adapters.length > 0);
  return payload.adapters.map((a: { id: string }) => a.id);
});
await check("frontend entry and cross-origin API access", async () => {
  const frontend = process.env.FRONTEND_URL ?? "http://127.0.0.1:5174";
  const page = await fetch(frontend);
  assert.ok(page.ok);
  assert.ok((await page.text()).includes("/src/main.tsx"));
  const module = await fetch(frontend + "/src/main.tsx");
  assert.ok(module.ok);
  const preflight = await fetch(base + "/api/ingest", { method: "OPTIONS", headers: { Origin: frontend, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type" } });
  assert.ok(preflight.ok);
  assert.ok([frontend, "*"].includes(preflight.headers.get("access-control-allow-origin") ?? ""));
});
for (const [query, medium, id] of [
  ["death note manga", "manga", 21],
  ["death note anime", "anime", 1535],
  ["attack on titan manga", "manga", 23390],
  ["attack on titan anime", "anime", 16498],
  ["firefly wedding manga", "manga", 154679],
  ["Yuryeongeui norae manhwa", "manga", 175718],
] as const) {
  await check(`typed search: ${query}`, async () => {
    const payload = await json(`/api/rag/search?q=${encodeURIComponent(query)}`);
    assert.equal(payload.results[0]?.mal_id, id);
    assert.ok(payload.results.every((r: { media_type: string }) => r.media_type === medium));
    assert.ok(payload.results[0].image_url?.startsWith("https://"));
    return { title: payload.results[0].title, id };
  });
}
await check("stream delivers results and final completion", async () => {
  const response = await fetch(base + "/api/rag/search/stream?q=firefly%20wedding%20manga", { signal: AbortSignal.timeout(30000) });
  const rows = (await response.text()).trim().split("\n").map((line) => JSON.parse(line));
  assert.ok(rows.some((r) => r.type === "result"));
  assert.equal(rows.at(-1)?.type, "done");
  assert.equal(rows.at(-1)?.payload.results[0]?.mal_id, 154679);
  return { events: rows.length };
});
await check("MAL cover is a real image and matches the manga identity", async () => {
  const meta = await json("/api/meta?type=manga&id=21&title=Death%20Note");
  const cover = meta.image_url ?? meta.coverUrl;
  assert.ok(cover, JSON.stringify(meta));
  const response = await fetch(cover, { signal: AbortSignal.timeout(15000) });
  assert.ok(response.ok);
  assert.ok(response.headers.get("content-type")?.startsWith("image/"));
  return { bytes: (await response.arrayBuffer()).byteLength };
});
await check("not-found Read Now returns a useful error", async () => {
  const response = await fetch(base + "/api/read-now?title=Nonexistent%20Catalogue%20Entry%20987654321", { signal: AbortSignal.timeout(15000) });
  assert.equal(response.status, 404);
  assert.ok((await response.json()).error.length > 20);
});
await check("Read Now resolves the selected MAL identity to readable chapter one", async () => {
  const started = performance.now();
  const match = await json("/api/read-now?title=Death%20Note&mal_id=21");
  const resolveMs = performance.now() - started;
  const chapter = await json("/api/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: match.chapterUrl }) });
  assert.equal(chapter.chapter.number, "1");
  assert.match(chapter.series.title, /death note/i);
  assert.ok(chapter.chapter.totalPages > 1);
  return { resolveMs: Math.round(resolveMs), totalMs: Math.round(performance.now() - started), chapterUrl: match.chapterUrl };
});
for (const [name, chapterUrl] of [
  ["WeebCentral", "https://weebcentral.com/chapters/01J76XZ2FVSZYPS9EE49JP5GKH"],
  ["ManhwaZone", "https://manhwazone.com/preview/2q32e"],
] as const) {
  await check(`${name}: chapter extraction, cached reopen, and first panel image`, async () => {
    const init = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: chapterUrl }) };
    const chapters = await Promise.all([json("/api/ingest", init), json("/api/ingest", init)]);
    assert.ok(chapters[0].chapter.totalPages >= 2);
    assert.equal(chapters[0].chapter.number, "1");
    assert.deepEqual(chapters[0], chapters[1]);
    const cached = await json("/api/ingest", init);
    assert.equal(cached.chapter.panelUrls[0], chapters[0].chapter.panelUrls[0]);
    const image = await fetch(base + `/api/image?url=${encodeURIComponent(cached.chapter.panelUrls[0])}`, { signal: AbortSignal.timeout(30000) });
    assert.ok(image.ok, `panel status ${image.status}`);
    assert.ok(image.headers.get("content-type")?.startsWith("image/"));
    return { series: cached.series.title, chapter: cached.chapter.number, panels: cached.chapter.totalPages, imageBytes: (await image.arrayBuffer()).byteLength };
  });
}
await mkdir(".tmp/eval", { recursive: true });
await writeFile(".tmp/eval/app-smoke.json", JSON.stringify(results, null, 2));
console.log(JSON.stringify(results, null, 2));
if (results.some((r) => !r.pass)) process.exitCode = 1;
