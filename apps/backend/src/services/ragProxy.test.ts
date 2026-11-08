import assert from "node:assert/strict";
import { test } from "node:test";
import { searchMangaRag } from "./ragProxy.js";
import { IngestError } from "../errors.js";

test("timeouts give a retry message instead of claiming the Python service is stopped", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new DOMException("The operation was aborted due to timeout", "TimeoutError"); });
  await assert.rejects(searchMangaRag("timeout fixture manga"), (error: unknown) => {
    assert.ok(error instanceof IngestError);
    assert.equal(error.status, 504);
    assert.equal(error.message, "Search took too long. Please try again in a moment.");
    return true;
  });
});

test("connection failures do not expose internal runtime instructions", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new TypeError("connect ECONNREFUSED 127.0.0.1:8090"); });
  await assert.rejects(searchMangaRag("connection fixture manga"), (error: unknown) => {
    assert.ok(error instanceof IngestError);
    assert.equal(error.status, 503);
    assert.equal(error.message, "Search is temporarily unavailable. Please try again shortly.");
    return true;
  });
});

test("worker overload is reported as busy without leaking upstream details", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json({ detail: "private upstream information" }, { status: 503 }));
  await assert.rejects(searchMangaRag("overload fixture manga"), (error: unknown) => {
    assert.ok(error instanceof IngestError);
    assert.equal(error.status, 503);
    assert.equal(error.message, "Search is busy right now. Please try again shortly.");
    return true;
  });
});

test("a timeout while reading the response body is also friendly", async (t) => {
  t.mock.method(globalThis, "fetch", async () => {
    const response = Response.json({});
    t.mock.method(response, "json", async () => { throw new DOMException("The operation was aborted", "AbortError"); });
    return response;
  });
  await assert.rejects(searchMangaRag("body timeout fixture manga"), (error: unknown) => {
    assert.ok(error instanceof IngestError);
    assert.equal(error.status, 504);
    assert.ok(!error.message.includes("Python"));
    return true;
  });
});

test("typed search forwards the medium, keeps aliases/covers and performs one lookup", async (t) => {
  let calls = 0;
  t.mock.method(globalThis, "fetch", async (_input: string | URL | Request, init?: RequestInit) => {
    calls += 1;
    const body = JSON.parse(String(init?.body));
    assert.equal(body.query, "Typed Search Fixture");
    assert.equal(body.media_type, "manga");
    assert.ok(body.budget_ms > 0 && body.budget_ms <= 24000);
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

test("optional alias failure preserves first-request matches and uses only the remaining budget", async (t) => {
  const previous = process.env.MAL_CLIENT_ID;
  process.env.MAL_CLIENT_ID = "fixture-client";
  t.after(() => { if (previous === undefined) delete process.env.MAL_CLIENT_ID; else process.env.MAL_CLIENT_ID = previous; });
  let now = 1000000;
  t.mock.method(Date, "now", () => now);
  const budgets: number[] = [];
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request, init?: RequestInit) => {
    if (String(input).includes("api.myanimelist.net")) {
      return Response.json({ data: [{ node: { id: 99010, title: "Canonical Budget Work", alternative_titles: { en: "Alias Budget Fixture" } } }] });
    }
    const body = JSON.parse(String(init?.body));
    budgets.push(body.budget_ms);
    if (budgets.length > 1) throw new DOMException("The operation was aborted", "TimeoutError");
    now += 20000;
    return Response.json({ top_results: [{ title: "Available First Match", mal_id: 99011, media_type: "manga", snippet: "Type: manga", score: .5 }] });
  });
  const response = await searchMangaRag("Alias Budget Fixture manga");
  assert.equal(response.results[0]?.mal_id, 99011);
  assert.ok(response.notice?.includes("Showing available matches"));
  assert.deepEqual(budgets, [24000, 8000]);
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
