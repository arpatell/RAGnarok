import assert from "node:assert/strict";
import { test } from "node:test";
import { isChapterOneLabel, ReadNowUnavailableError, resolveDirectReadNow, titleSimilarityScore } from "./webSearch.js";

test("chapter 1 labels do not accept chapter 1.5, 10 or opaque IDs", () => {
  for (const label of ["Chapter 1.5", "Chapter 10", "Chapter 1.01", "01J76XZ2FVSZYPS9EE49JP5GKH"]) {
    assert.equal(isChapterOneLabel(label), false, label);
  }
  for (const label of ["Chapter 1", "Episode 01", "Ch. 1.0", "Chapter 1 - Beginning"]) {
    assert.equal(isChapterOneLabel(label), true, label);
  }
});

test("Read Now rejects related but distinct titles", () => {
  for (const [query, unrelated] of [
    ["Solo Leveling", "Solo Leveling: Ragnarok"],
    ["Naruto", "Boruto: Naruto Next Generations"],
    ["Death Note", "Death Note: Special One-Shot"],
    ["Attack on Titan", "Attack on Titan: Before the Fall"],
    ["The Beginning After the End", "The End of the World"],
    ["Tokyo Ghoul", "Tokyo Ghoul:re"],
    ["One Piece", "Two Piece"],
    ["The Last Hero", "Hero The Last"],
  ]) assert.ok(titleSimilarityScore(query!, unrelated!) < 0.9, `${query} incorrectly matched ${unrelated}`);
  assert.equal(titleSimilarityScore("Solo Leveling", "Solo Leveling"), 1);
  assert.ok(titleSimilarityScore("Solo Leveling", "Solo Leveling (Full Color)") >= 0.9);
});

test("Read Now keeps subtitles instead of opening the parent work", async (t) => {
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = new URL(String(input));
    assert.equal(url.searchParams.get("text"), "Fixture Saga Return");
    return new Response('<a href="https://weebcentral.com/series/fixture-parent/Fixture-Saga" title="Fixture Saga">Fixture Saga</a>');
  });
  assert.equal(await resolveDirectReadNow(["Fixture Saga: Return"], { source: "weebcentral" }), null);
});

test("translated candidate is verified by authoritative MAL identity, not guessed", async (t) => {
  const series = "https://weebcentral.com/series/fixture-translated/Completely-Different-Translation";
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/search/data?")) return new Response(`<a href="${series}" title="Completely Different Translation">Completely Different Translation</a>`);
    const response = new Response('<h1>Completely Different Translation</h1><a href="https://myanimelist.net/manga/887766/Original">MAL</a><div id="chapter-list"><a href="https://weebcentral.com/chapters/translated-one">Chapter 1</a></div>');
    Object.defineProperty(response, "url", { value: series });
    return response;
  });
  assert.equal((await resolveDirectReadNow("Original Romanized Fixture", { source: "weebcentral", malId: 887766 }))?.seriesUrl, series);
  assert.equal(await resolveDirectReadNow("Original Romanized Fixture", { source: "weebcentral", malId: 887767 }), null);
});

test("provider HTTP failures are unavailable, not an immediate cached not-found", async (t) => {
  let calls = 0;
  t.mock.method(globalThis, "fetch", async () => { calls += 1; return new Response("Unavailable", { status: 503 }); });
  await assert.rejects(resolveDirectReadNow("Unavailable Fixture", { source: "weebcentral" }), ReadNowUnavailableError);
  await assert.rejects(resolveDirectReadNow("Unavailable Fixture", { source: "weebcentral" }), ReadNowUnavailableError);
  assert.equal(calls, 2);
});

test("chapter one is found in a lazy-loaded full chapter list", async (t) => {
  const series = "https://weebcentral.com/series/lazy-fixture/Lazy-Fixture";
  const calls: string[] = [];
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input); calls.push(url);
    const body = url.includes("/search/data?") ? `<a href="${series}" title="Lazy Fixture">Lazy Fixture</a>`
      : url.endsWith("/full-chapter-list") ? '<div><a href="/chapters/lazy-one">Chapter 1</a><a href="/chapters/lazy-ten">Chapter 10</a></div>'
      : '<h1>Lazy Fixture</h1><div hx-get="/series/lazy-fixture/full-chapter-list"></div>';
    const response = new Response(body); Object.defineProperty(response, "url", { value: url }); return response;
  });
  assert.equal((await resolveDirectReadNow("Lazy Fixture", { source: "weebcentral" }))?.chapterUrl, "https://weebcentral.com/chapters/lazy-one");
  assert.equal(calls.length, 3);
});

test("title identity retains comic qualifiers separated from @ by source markup", async (t) => {
  const series = "https://weebcentral.com/series/comic-fixture/Fixture-Comic";
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    const body = url.includes("/search/data?") ? `<a href="${series}" title="Fixture @comic">Fixture @comic</a>`
      : '<h1>Fixture @ comic</h1><div id="chapter-list"><a href="/chapters/comic-fixture-one">Chapter 1</a></div>';
    const response = new Response(body); Object.defineProperty(response, "url", { value: url }); return response;
  });
  assert.equal((await resolveDirectReadNow("Fixture @comic", { source: "weebcentral" }))?.chapterUrl, "https://weebcentral.com/chapters/comic-fixture-one");
});

test("remaining verified aliases are searched after the first pair is exhausted", async (t) => {
  const series = "https://weebcentral.com/series/third-alias-fixture/Third-Verified-Alias";
  const queries: string[] = [];
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/search/data?")) {
      const q = new URL(url).searchParams.get("text")!; queries.push(q);
      return new Response(q === "Third Verified Alias" ? `<a href="${series}" title="Third Verified Alias">Third Verified Alias</a>` : "No results");
    }
    const response = new Response('<h1>Third Verified Alias</h1><div id="chapter-list"><a href="/chapters/third-alias-one">Chapter 1</a></div>');
    Object.defineProperty(response, "url", { value: series }); return response;
  });
  assert.equal((await resolveDirectReadNow(["First Alias Fixture", "Second Alias Fixture", "Third Verified Alias"], { source: "weebcentral" }))?.seriesUrl, series);
  assert.deepEqual(queries, ["First Alias Fixture", "Second Alias Fixture", "Third Verified Alias"]);
});

test("Read Now rejects a search label that points to a differently titled series page", async (t) => {
  const series = "https://weebcentral.com/series/fixture-mismatch/Requested-Fixture";
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/search/data?")) return new Response(`<a href="${series}" title="Requested Fixture">Requested Fixture</a>`);
    const response = new Response('<meta property="og:title" content="Unrelated Fixture"><div id="chapter-list"><a href="https://weebcentral.com/chapters/mismatch-one">Chapter 1</a></div>');
    Object.defineProperty(response, "url", { value: series });
    return response;
  });
  assert.equal(await resolveDirectReadNow("Requested Fixture", { source: "weebcentral" }), null);
});

test("a fast alias can open before a slow primary search finishes", async (t) => {
  const series = "https://weebcentral.com/series/fixture-fast/Fast-Alias";
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/search/data?")) {
      if (new URL(url).searchParams.get("text") === "Slow Primary") await new Promise((resolve) => setTimeout(resolve, 300));
      return new Response(`<a href="${series}" title="Fast Alias">Fast Alias</a>`);
    }
    const response = new Response('<meta property="og:title" content="Fast Alias"><div id="chapter-list"><a href="https://weebcentral.com/chapters/fast-one">Chapter 1</a></div>');
    Object.defineProperty(response, "url", { value: series });
    return response;
  });
  const start = performance.now();
  const result = await resolveDirectReadNow(["Slow Primary", "Fast Alias"], { source: "weebcentral" });
  assert.equal(result?.chapterUrl, "https://weebcentral.com/chapters/fast-one");
  assert.ok(performance.now() - start < 200);
  await new Promise((resolve) => setTimeout(resolve, 310));
});

test("ManhwaZone ignores neighboring preview links and follows the selected series", async (t) => {
  const series = "https://manhwazone.com/series/selected-fixture-a1234";
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/search?")) return new Response(`<div><a href="/preview/unrelated-chapter-1">Chapter 1</a><a href="${series}" title="Selected Fixture">Selected Fixture</a><a href="/series/unrelated-b1234">Unrelated</a></div>`);
    assert.equal(url, series);
    const response = new Response('<meta property="og:title" content="Selected Fixture"><ol role="list" aria-label="List of chapters" class="space-y-3"><li role="listitem"><a href="/preview/selected-one"><span class="truncate flex-1">Chapter 1</span></a></li></ol>');
    Object.defineProperty(response, "url", { value: series });
    return response;
  });
  const result = await resolveDirectReadNow("Selected Fixture", { source: "manhwazone" });
  assert.equal(result?.chapterUrl, "https://manhwazone.com/preview/selected-one");
});

test("provider lookup preserves Attack on Titan and follows chapter 1, not chapter 10", async (t) => {
  const seriesUrl = "https://weebcentral.com/series/fixture-attack/Shingeki-No-Kyojin";
  const calls: string[] = [];
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    calls.push(url);
    if (url.includes("/search/data?")) {
      assert.equal(new URL(url).searchParams.get("text"), "Attack on Titan");
      return new Response(`<article><a href="${seriesUrl}" title="Attack on Titan">Attack on Titan</a></article>`);
    }
    assert.equal(url, seriesUrl);
    const response = new Response(`<meta property="og:title" content="Attack on Titan"><div id="chapter-list"><a href="https://weebcentral.com/chapters/fixture-ten"><span>Chapter 10</span></a><a href="https://weebcentral.com/chapters/fixture-one"><span>Chapter 1</span></a></div>`);
    Object.defineProperty(response, "url", { value: seriesUrl });
    return response;
  });
  const result = await resolveDirectReadNow(["Attack on Titan"], { source: "weebcentral" });
  assert.equal(result?.requestedTitle, "Attack on Titan");
  assert.equal(result?.chapterUrl, "https://weebcentral.com/chapters/fixture-one");
  assert.equal(calls.length, 2);
});

test("MAL identity disambiguates homonyms and partitions the resolution cache", async (t) => {
  const first = "https://weebcentral.com/series/fixture-homonym-first/Shared-Fixture";
  const second = "https://weebcentral.com/series/fixture-homonym-second/Shared-Fixture";
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/search/data?")) return new Response(`<a href="${first}" title="Shared Fixture">Shared Fixture</a><a href="${second}" title="Shared Fixture">Shared Fixture</a>`);
    const correct = url === second;
    const response = new Response(`<h1>Shared Fixture</h1><a href="https://myanimelist.net/manga/${correct ? 42 : 17}/Shared_Fixture">MAL</a><div id="chapter-list"><a href="https://weebcentral.com/chapters/homonym-${correct ? 'correct' : 'wrong'}">Chapter 1</a></div>`);
    Object.defineProperty(response, "url", { value: url });
    return response;
  });
  const result = await resolveDirectReadNow("Shared Fixture", { source: "weebcentral", malId: 42 });
  assert.equal(result?.seriesUrl, second);
  assert.equal(await resolveDirectReadNow("Shared Fixture", { source: "weebcentral", malId: 999 }), null);
});
