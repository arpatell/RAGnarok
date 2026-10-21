import assert from "node:assert/strict";
import { test } from "node:test";
import { searchMangaRag } from "./ragProxy.js";

test("typed search forwards the medium, keeps aliases/covers and performs one lookup", async (t) => {
  let calls = 0;
  t.mock.method(globalThis, "fetch", async (_input: string | URL | Request, init?: RequestInit) => {
    calls += 1;
    const body = JSON.parse(String(init?.body));
    assert.equal(body.query, "Typed Search Fixture");
    assert.equal(body.media_type, "manga");
    return Response.json({ top_results: [
      { title: "Typed Search Fixture", mal_id: 99001, media_type: "manga", snippet: "Type: manga\nTitle: Typed Search Fixture\nDetails: Format: manhwa; Authors: A Creator\nGenres: Mystery\nSynopsis: A mystery.", score: 1, title_candidates: ["Typed Search Fixture", "Native Fixture"], image_url: "https://example.test/correct.jpg" },
      { title: "Other Medium", mal_id: 99001, media_type: "anime", snippet: "Type: anime", score: 1 },
    ] });
  });
  const response = await searchMangaRag("Typed Search Fixture manga");
  assert.equal(calls, 1);
  assert.equal(response.results.length, 1);
  assert.equal(response.results[0]?.display_type, "manhwa");
  assert.equal(response.results[0]?.image_url, "https://example.test/correct.jpg");
  assert.deepEqual(response.results[0]?.title_candidates, ["Typed Search Fixture", "Native Fixture"]);
});

test("a different result's highlight cannot supply the cover", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json({
    top_results: [{ title: "Cover Identity Fixture", mal_id: 99002, media_type: "manga", snippet: "Type: manga", score: 1 }],
    highlight: { mal_id: 99003, image_url: "https://example.test/wrong.jpg" },
  }));
  const response = await searchMangaRag("Cover Identity Fixture manga");
  assert.equal(response.results[0]?.image_url, null);
});

test("natural recollections preserve reranked order and do not make MAL alias requests", async (t) => {
  let calls = 0;
  t.mock.method(globalThis, "fetch", async () => {
    calls += 1;
    return Response.json({ notice: "Search is busy right now.", top_results: [
      { title: "Correct Ranked Fixture", mal_id: 99004, media_type: "manga", snippet: "Type: manga", score: .2 },
      { title: "Incidental Keyword Fixture", mal_id: 99005, media_type: "manga", snippet: "Type: manga", score: 9 },
    ] });
  });
  const response = await searchMangaRag("I remember a traveler searching for a city that moves every night manga");
  assert.equal(calls, 1);
  assert.equal(response.results[0]?.mal_id, 99004);
  assert.equal(response.notice, "Search is busy right now.");
});
