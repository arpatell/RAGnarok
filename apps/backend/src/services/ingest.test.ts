import assert from "node:assert/strict";
import { test } from "node:test";
import { fetchManhwaZoneLivewireChapterList } from "./ingest.js";

test("ManhwaZone loads ascending chapter list once and coalesces readers", async (t) => {
  const seriesUrl = "https://manhwazone.com/series/fixture-cache-128";
  const initial = JSON.stringify({ memo: { name: "common.chapter-list" }, data: { chapters: null, hasMore: true } });
  const loaded = JSON.stringify({ data: { chapters: [[{ web_url: "/preview/fixture-chapter-one", name: "Chapter 1", chapter_no: "1" }], { s: "arr" }], hasMore: true } });
  const requests: string[] = [];
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    requests.push(url);
    if (url === seriesUrl) {
      return new Response(`<meta name="csrf-token" content="fixture-csrf"><div wire:init="bootLoad" wire:snapshot='${initial}'></div>`, { headers: { "set-cookie": "session=fixture; Path=/" } });
    }
    assert.equal(url, "https://manhwazone.com/livewire/update");
    const body = JSON.parse(String(init?.body));
    assert.equal(body.components[0].updates.sortOrder, "asc");
    assert.equal(body.components[0].calls[0].method, "bootLoad");
    return Response.json({ components: [{ snapshot: loaded }] });
  });
  const results = await Promise.all(Array.from({ length: 5 }, () => fetchManhwaZoneLivewireChapterList(seriesUrl, true)));
  assert.equal(requests.length, 2);
  for (const chapters of results) {
    assert.equal(chapters[0]?.number, "1");
    assert.equal(chapters[0]?.url, "https://manhwazone.com/preview/fixture-chapter-one");
  }
  await fetchManhwaZoneLivewireChapterList(seriesUrl, true);
  assert.equal(requests.length, 2);
});

test("ManhwaZone transient failures do not poison the chapter cache", async (t) => {
  let requests = 0;
  t.mock.method(globalThis, "fetch", async () => {
    requests += 1;
    return new Response("Unavailable", { status: 503 });
  });
  const url = "https://manhwazone.com/series/fixture-unavailable-129";
  assert.deepEqual(await fetchManhwaZoneLivewireChapterList(url, true), []);
  assert.deepEqual(await fetchManhwaZoneLivewireChapterList(url, true), []);
  assert.equal(requests, 2);
});

test("full ManhwaZone lists use bounded larger pages and fall back if rejected", async (t) => {
  const seriesUrl = "https://manhwazone.com/series/fixture-pagination-130";
  const initial = JSON.stringify({ memo: { name: "common.chapter-list" }, data: { chapters: null, hasMore: true } });
  let posts = 0;
  t.mock.method(globalThis, "fetch", async (input: string | URL | Request, init?: RequestInit) => {
    if (String(input) === seriesUrl) return new Response(`<meta name="csrf-token" content="csrf"><div wire:init="bootLoad" wire:snapshot='${initial}'></div>`);
    posts += 1;
    const body = JSON.parse(String(init?.body));
    if (posts === 1) {
      assert.equal(body.components[0].updates.perPage, 250);
      return new Response("Unsupported update", { status: 422 });
    }
    assert.deepEqual(body.components[0].updates, {});
    return Response.json({ components: [{ snapshot: JSON.stringify({ data: { chapters: [{ web_url: "/preview/fallback-one", name: "Chapter 1", chapter_no: "1" }], hasMore: false } }) }] });
  });
  const chapters = await fetchManhwaZoneLivewireChapterList(seriesUrl);
  assert.equal(chapters[0]?.number, "1");
  assert.equal(posts, 2);
});
