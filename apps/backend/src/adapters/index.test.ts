import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveAdapterForUrl } from "./index.js";

test("ManhwaZone chapter label uses page title rather than series heading or opaque preview ID", () => {
  const url = "https://manhwazone.com/preview/2q32e";
  const adapter = resolveAdapterForUrl(url);
  const parsed = adapter!.parse('<html><head><title>Read Example Manhwa - Chapter 1 | ManhwaZone</title></head><body><h1>Example</h1></body></html>', url);
  assert.equal(parsed.chapter.number, "1");
  assert.equal(parsed.series.title, "Example");
});
