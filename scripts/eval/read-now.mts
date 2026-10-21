import { mkdir, writeFile } from "node:fs/promises";
import * as cheerio from "cheerio";
import { resolveDirectReadNow } from "../../apps/backend/src/services/webSearch.js";

const cases = [
  { titles: ["Death Note"], slug: "Death-Note" },
  { titles: ["Attack on Titan", "Shingeki no Kyojin"], slug: "Shingeki-No-Kyojin" },
  { titles: ["Hotaru no Yomeiri", "Firefly Wedding"], slug: "Hotaru-no-Yomeiri" },
  { titles: ["Solo Leveling"], slug: "Solo-Leveling", reject: /ragnarok/i, requiredManhwaZone: true },
  { titles: ["One Piece"], slug: "One-Piece", reject: /gakuen|party|novel/i },
  { titles: ["Naruto"], slug: "Naruto", reject: /boruto|novel|gaiden/i },
  { titles: ["The Beginning After the End"], slug: "The-Beginning-After-the-End", requiredManhwaZone: true },
  { titles: ["Nonexistent Catalogue Entry 987654321"], missing: true },
];
const results = [];
for (const source of ["weebcentral", "manhwazone"] as const) {
  for (const item of cases) {
    const start = performance.now();
    try {
      const match = await resolveDirectReadNow(item.titles, { source });
      const resolveMs = performance.now() - start;
      if (!match) {
        results.push({ source, titles: item.titles, status: item.missing ? "pass" : source === "weebcentral" || item.requiredManhwaZone ? "fail" : "unavailable", resolveMs, ms: performance.now() - start });
        continue;
      }
      let identity = !item.missing && !item.reject?.test(match.seriesUrl);
      if (source === "weebcentral") identity &&= new URL(match.seriesUrl).pathname.split("/").at(-1)?.toLowerCase() === item.slug?.toLowerCase();
      const response = await fetch(match.chapterUrl, { signal: AbortSignal.timeout(15000) });
      const $ = cheerio.load(await response.text());
      const selectedSeriesPath = new URL(match.seriesUrl).pathname.split("/").slice(0, 3).join("/");
      const backlink = $("a[href]").toArray().some((el) => {
        try { return new URL($(el).attr("href")!, match.chapterUrl).pathname.startsWith(selectedSeriesPath); } catch { return false; }
      });
      const chapterTitle = $("title").text();
      const chapterOne = /\b(?:chapter|episode)\s*0*1(?![\d.])/i.test(chapterTitle);
      const cachedStart = performance.now();
      const cached = await resolveDirectReadNow(item.titles, { source });
      const cachedResolveMs = performance.now() - cachedStart;
      identity &&= cached?.chapterUrl === match.chapterUrl;
      results.push({ source, titles: item.titles, status: identity && response.ok && backlink && chapterOne ? "pass" : "fail", identity, backlink, chapterOne, chapterTitle, match, resolveMs, cachedResolveMs, ms: performance.now() - start });
    } catch (error) {
      results.push({ source, titles: item.titles, status: "error", error: String(error), ms: performance.now() - start });
    }
  }
}
await mkdir(".tmp/eval", { recursive: true });
await writeFile(".tmp/eval/read-now.json", JSON.stringify(results, null, 2));
console.log(JSON.stringify(results, null, 2));
if (results.some((row) => row.status === "fail" || row.status === "error")) process.exitCode = 1;
