import { createReadStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { createGunzip } from "node:zlib";
import * as cheerio from "cheerio";
import { resolveDirectReadNow } from "../../apps/backend/src/services/webSearch.js";
import { fetchChapterHtml } from "../../apps/backend/src/services/fetchProxy.js";

let seed = 20261020;
function random() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; }
const sample: any[] = [];
let eligible = 0;
const knownRegressions = new Set([14897, 44101, 13666]);
const regressions: any[] = [];
for await (const line of createInterface({ input: createReadStream("rag/data/catalog.jsonl.gz").pipe(createGunzip()), crlfDelay: Infinity })) {
  const row = JSON.parse(line);
  if (row.type !== "manga") continue;
  if (knownRegressions.has(row.mal_id)) regressions.push(row);
  if (!(row.score > 8)) continue;
  eligible += 1;
  if (sample.length < 8) sample.push(row);
  else { const at = Math.floor(random() * eligible); if (at < sample.length) sample[at] = row; }
}
const originalFetch = globalThis.fetch;
let requests: { url: string; status?: number; ms?: number }[] = [];
globalThis.fetch = async (input, options) => {
  const record = { url: String(input) } as (typeof requests)[number];
  requests.push(record);
  const started = performance.now();
  try { const response = await originalFetch(input, options); record.status = response.status; return response; }
  finally { record.ms = Math.round(performance.now() - started); }
};
const results = [];
for (const row of [...sample, ...regressions]) {
  requests = [];
  const titles = [row.title, row.alternative_titles?.en, ...(row.alternative_titles?.synonyms ?? [])].filter(Boolean);
  const started = performance.now();
  try {
    const match = await resolveDirectReadNow(titles, { malId: row.mal_id, preferManhwa: row.display_type === "manhwa" });
    const resolveMs = Math.round(performance.now() - started);
    const searches = requests.filter((r) => r.url.includes("/search"));
    let chapterOne: boolean | undefined;
    if (match) {
      const response = await fetchChapterHtml(match.chapterUrl);
      const $ = cheerio.load(response.html);
      chapterOne = response.status < 400 && /(?:\b(?:chapter|episode|volume|entry\s*(?:no\.?)?)|#)\s*0*1(?![\d.])/i.test($("title").text());
    }
    results.push({ title: row.title, malId: row.mal_id, resolveMs, status: match ? chapterOne ? "verified-chapter-one" : "invalid-chapter" : "not-found-after-searches", searches, requests, match, chapterOne });
  } catch (error) {
    results.push({ title: row.title, malId: row.mal_id, status: "source-unavailable", error: String(error), requests });
  }
}
await mkdir(".tmp/eval", { recursive: true });
await writeFile(".tmp/eval/read-now-sample.json", JSON.stringify({ seed: 20261020, eligible, results }, null, 2));
console.log(JSON.stringify(results.map(({ title, status, ...rest }) => ({ title, status, resolveMs: rest.resolveMs, requests: rest.requests.length })), null, 2));
if (results.some((r) => r.status === "invalid-chapter" || r.requests.length === 0)) process.exitCode = 1;
