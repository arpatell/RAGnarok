#!/usr/bin/env node
/**
 * Load-test harness for RAGnarok under simulated e2-micro conditions.
 *
 * IMPORTANT: run this from a SEPARATE machine (your laptop) against the VM's
 * public endpoint — generating load on the VM itself steals CPU from the
 * system under test.
 *
 * Usage:
 *   BASE_URL=https://YOUR_VM.sslip.io node scripts/bench/run-load.mjs --scenario search-realistic
 *   node scripts/bench/run-load.mjs --scenario all
 *   node scripts/bench/run-load.mjs --scenario soak --record PERF.md
 *
 * Scenarios:
 *   search-realistic  Mixed title/plot/character queries, 3 concurrent, 60s
 *   search-burst      Same mix at 5 concurrent (small traffic spike)
 *   meta-burst        Result-card metadata enrichment bursts (10 ids, 5 conn)
 *   reader-cachehit   Repeated ingest of one chapter URL (parse cache hits)
 *   reader-cachemiss  Ingest of distinct chapter URLs (upstream-bound)
 *   soak              Mixed search+meta at 2 concurrent for SOAK_SECONDS (default 900)
 *   all               Everything except soak
 */

import { writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

const BASE_URL = (process.env.BASE_URL ?? "http://127.0.0.1:8787").replace(
  /\/+$/,
  "",
);
const RESULTS_DIR = process.env.RESULTS_DIR ?? ".tmp/bench";
const RECORD_FILE = process.env.RECORD_FILE ?? "";
const SOAK_SECONDS = Number.parseInt(process.env.SOAK_SECONDS ?? "900", 10);

const args = process.argv.slice(2);
const scenarioIdx = args.indexOf("--scenario");
const scenario = scenarioIdx >= 0 ? args[scenarioIdx + 1] : "search-realistic";

// Weighted query mix mirroring real usage: mostly exact titles, some plot
// descriptions, occasional character-name lookups.
const SEARCH_MIX = [
  { q: "death note", kind: "title", weight: 12 },
  { q: "one piece", kind: "title", weight: 8 },
  { q: "attack on titan", kind: "title", weight: 8 },
  { q: "guy finds notebook that can kill people", kind: "plot", weight: 6 },
  { q: "blind swordsman wandering feudal japan", kind: "plot", weight: 3 },
  { q: "gods vs humans battle royale", kind: "plot", weight: 3 },
  { q: "satoru gojo", kind: "character", weight: 4 },
  { q: "satoko kirigaya", kind: "character", weight: 2 },
];

function pickWeighted() {
  const total = SEARCH_MIX.reduce((sum, item) => sum + item.weight, 0);
  let roll = Math.random() * total;
  for (const item of SEARCH_MIX) {
    roll -= item.weight;
    if (roll <= 0) {
      return item;
    }
  }
  return SEARCH_MIX[0];
}

const META_IDS = [
  [1, "manga"], // Monster
  [6471, "manga"], // Death Note (wait: 6471 is Berserk one-shot; id variety matters more than identity here)
  [2, "manga"],
  [21, "manga"], // One Piece
  [23390, "manga"], // Attack on Titan
  [16498, "manga"], // Tokyo Ghoul (re)
  [116778, "manga"], // Solo Leveling
  [11061, "manga"], // Hunter x Hunter
  [44347, "manga"], // Vinland Saga (wait: verify not needed; 404s count as errors but exercise the negative-cache path too)
  [13, "anime"], // One Piece anime
];

async function loadAutocannon() {
  try {
    return await import("autocannon");
  } catch {
    console.error(
      "[bench] autocannon is missing. Run `npm install` at repo root first.",
    );
    process.exit(1);
  }
}

function percentileOf(sortedValues, pct) {
  if (sortedValues.length === 0) return NaN;
  const idx = Math.min(
    sortedValues.length - 1,
    Math.ceil((pct / 100) * sortedValues.length) - 1,
  );
  return sortedValues[Math.max(0, idx)];
}

/** Minimal latency sampler used for scenarios autocannon cannot express well
 *  (e.g., cache-miss ingests that need unique URLs per request). */
async function sampledLatencies(requestFactories, { durationSeconds }) {
  const samples = [];
  const deadline = Date.now() + durationSeconds * 1000;
  let index = 0;
  while (Date.now() < deadline) {
    const factory = requestFactories[index % requestFactories.length];
    index += 1;
    const startedAt = performance.now();
    try {
      const response = await fetch(factory());
      await response.arrayBuffer();
      if (!response.ok && response.status !== 404) {
        // Count as error marker: huge latency so percentiles expose it
        samples.push(Number.POSITIVE_INFINITY);
        continue;
      }
    } catch {
      samples.push(Number.POSITIVE_INFINITY);
      continue;
    }
    samples.push(performance.now() - startedAt);
  }
  return samples.sort((a, b) => a - b);
}

function summarize(name, result, extra = {}) {
  const summary = {
    scenario: name,
    baseUrl: BASE_URL,
    ...extra,
    requests: result.requests?.total ?? extra.samples,
    avgMs: result.latency?.average ?? null,
    p50Ms: result.latency?.p50 ?? extra.p50Ms,
    p90Ms: result.latency?.p90 ?? extra.p90Ms,
    p99Ms: result.latency?.p99 ?? extra.p99Ms,
    maxMs: result.latency?.max ?? extra.maxMs,
    errors: result.errors ?? result.non2xx ?? extra.errors,
    durationSeconds: result.duration ?? extra.durationSeconds,
    bytesPerSec: result.throughput?.average ?? null,
    finishedAt: new Date().toISOString(),
  };
  return summary;
}

async function runAutocannonScenario(name, autocannon, opts, extra = {}) {
  process.stdout.write(
    `[bench] ${name}: ${opts.connections} conn x ${opts.duration}s -> `,
  );
  const result = await autocannon({
    url: BASE_URL,
    ...opts,
  });
  const summary = summarize(name, result, extra);
  console.log(
    `req=${summary.requests} avg=${Math.round(summary.avgMs)}ms p50=${summary.p50Ms}ms p99=${summary.p99Ms}ms err=${summary.errors}`,
  );
  return summary;
}

async function main() {
  const autocannonMod = await loadAutocannon();
  const autocannon = autocannonMod.default ?? autocannonMod;
  const results = [];
  await mkdir(RESULTS_DIR, { recursive: true });

  const scenariosToRun =
    scenario === "all"
      ? ["search-realistic", "search-burst", "meta-burst", "reader-cachehit"]
      : [scenario];

  for (const name of scenariosToRun) {
    switch (name) {
      case "search-realistic": {
        results.push(
          await runAutocannonScenario(name, autocannon, {
            connections: 3,
            duration: 60,
            requests: Array.from({ length: 24 }, () => {
              const picked = pickWeighted();
              return {
                path: `/api/rag/search?q=${encodeURIComponent(picked.q)}&_mix=${picked.kind}`,
              };
            }),
          }),
        );
        break;
      }
      case "search-burst": {
        results.push(
          await runAutocannonScenario(name, autocannon, {
            connections: 5,
            duration: 30,
            requests: Array.from({ length: 24 }, () => {
              const picked = pickWeighted();
              return {
                path: `/api/rag/search?q=${encodeURIComponent(picked.q)}&_burst=1`,
              };
            }),
          }),
        );
        break;
      }
      case "meta-burst": {
        results.push(
          await runAutocannonScenario(name, autocannon, {
            connections: 5,
            duration: 30,
            requests: META_IDS.map(([id, type]) => ({
              path: `/api/meta?type=${type}&id=${id}&title=x&_burst=1`,
            })),
          }),
        );
        break;
      }
      case "reader-cachehit": {
        const url = process.env.BENCH_INGEST_URL;
        if (!url) {
          console.error(
            "[bench] reader-cachehit needs BENCH_INGEST_URL (a public chapter URL)",
          );
          break;
        }
        results.push(
          await runAutocannonScenario(name, autocannon, {
            connections: 2,
            duration: 30,
            requests: [
              {
                method: "POST",
                path: "/api/ingest",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ url }),
              },
            ],
          }),
        );
        break;
      }
      case "reader-cachemiss": {
        const templates = (process.env.BENCH_INGEST_TEMPLATES ?? "")
          .split(",")
          .filter(Boolean);
        if (templates.length === 0) {
          console.error(
            "[bench] reader-cachemiss needs BENCH_INGEST_TEMPLATES (comma-separated URLs containing {N})",
          );
          break;
        }
        let n = Number.parseInt(process.env.BENCH_START_CHAPTER ?? "100", 10);
        const factories = templates.map((tpl) => () => {
          n += 1;
          return `/api/ingest-direct?url=${encodeURIComponent(tpl.replace("{N}", String(n)))}`;
        });
        // Note: distinct-URL ingests are upstream-latency bound; use the sampler
        // at low rate instead of autocannon to avoid hammering source sites.
        const samples = await sampledLatencies(
          factories.map((f) => () => BASE_URL + f()),
          { durationSeconds: 120 },
        );
        const finite = samples.filter((v) => Number.isFinite(v));
        const summary = {
          scenario: name,
          samples: samples.length,
          p50Ms: Math.round(percentileOf(finite, 50)),
          p90Ms: Math.round(percentileOf(finite, 90)),
          p99Ms: Math.round(percentileOf(finite, 99)),
          errors: samples.length - finite.length,
          durationSeconds: 120,
          finishedAt: new Date().toISOString(),
        };
        console.log(
          `[bench] ${name}: samples=${summary.samples} p50=${summary.p50Ms}ms p99=${summary.p99Ms}ms err=${summary.errors}`,
        );
        results.push(summary);
        break;
      }
      case "soak": {
        results.push(
          await runAutocannonScenario(`soak-${SOAK_SECONDS}s`, autocannon, {
            connections: 2,
            duration: SOAK_SECONDS,
            requests: [
              ...Array.from({ length: 18 }, () => {
                const picked = pickWeighted();
                return {
                  path: `/api/rag/search?q=${encodeURIComponent(picked.q)}&_soak=1`,
                };
              }),
              ...META_IDS.map(([id, type]) => ({
                path: `/api/meta?type=${type}&id=${id}&title=x&_soak=1`,
              })),
            ],
          }),
        );
        break;
      }
      default:
        console.error(`[bench] unknown scenario "${scenario}"`);
        process.exit(1);
    }
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const jsonPath = path.join(RESULTS_DIR, `results-${stamp}.json`);
  await writeFile(
    jsonPath,
    JSON.stringify({ baseUrl: BASE_URL, results }, null, 2),
  );
  console.log(`\n[bench] wrote ${jsonPath}`);

  if (RECORD_FILE) {
    const lines = [
      "",
      `### Load test ${stamp} (${BASE_URL})`,
      "",
      "| Scenario | Requests | avg | p50 | p90 | p99 | Errors |",
      "|---|---|---|---|---|---|---|",
    ];
    for (const r of results) {
      lines.push(
        `| ${r.scenario} | ${r.requests ?? r.samples} | ${fmt(r.avgMs)} | ${r.p50Ms} | ${r.p90Ms} | ${r.p99Ms} | ${r.errors} |`,
      );
    }
    const { appendFile } = await import("node:fs/promises");
    await appendFile(RECORD_FILE, lines.join("\n") + "\n");
    console.log(`[bench] appended table to ${RECORD_FILE}`);
  }
}

function fmt(value) {
  return value === null || value === undefined ? "-" : `${Math.round(value)}ms`;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
