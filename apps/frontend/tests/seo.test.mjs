import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import {
  PRODUCTION_ORIGIN, GUIDE_PATH, resolveSiteOrigin, seoTags, sitemapXml, robotsTxt, structuredData
} from "../scripts/seo.mjs";

test("default canonical is the confirmed production URL, never a preview hostname", () => {
  assert.equal(resolveSiteOrigin({ VERCEL_URL: "random-preview.vercel.app" }), PRODUCTION_ORIGIN);
});

test("explicit custom domains are normalized and shared by all SEO assets", () => {
  const origin = resolveSiteOrigin({ SITE_URL: " https://reader.example.org/ " });
  assert.equal(origin, "https://reader.example.org");
  assert.match(sitemapXml(origin), /https:\/\/reader.example.org\//);
  assert.match(robotsTxt(origin), /Sitemap: https:\/\/reader.example.org\/sitemap.xml/);
  assert.equal(seoTags(origin).find((tag) => tag.attrs?.rel === "canonical").attrs.href, `${origin}/`);
});

test("unsafe or non-origin SITE_URL values fail the build", () => {
  for (const SITE_URL of ["http://example.org", "https://user:secret@example.org", "https://example.org/path", "https://example.org/?q=x", "https://example.org/#frag"]) {
    assert.throws(() => resolveSiteOrigin({ SITE_URL }));
  }
});

test("production is indexable and preview metadata is noindex", () => {
  const robots = (preview) => seoTags(PRODUCTION_ORIGIN, preview).find((tag) => tag.attrs?.name === "robots").attrs.content;
  assert.ok(robots(false).startsWith("index,follow"));
  assert.equal(robots(true), "noindex,follow");
});

test("metadata is head-only, unique, and has no keyword stuffing", () => {
  const tags = seoTags(PRODUCTION_ORIGIN);
  assert.ok(tags.every((tag) => tag.injectTo === "head"));
  assert.equal(tags.filter((tag) => tag.tag === "title").length, 1);
  assert.equal(tags.filter((tag) => tag.attrs?.rel === "canonical").length, 1);
  assert.equal(tags.filter((tag) => tag.attrs?.name === "description").length, 1);
  assert.ok(!tags.some((tag) => tag.attrs?.name === "keywords"));
});

test("structured data identifies the actual product without fabricated ratings or search routes", () => {
  const data = structuredData(PRODUCTION_ORIGIN);
  assert.deepEqual(data["@graph"].map((node) => node["@type"]), ["WebSite", "WebPage", "WebApplication"]);
  assert.ok(!JSON.stringify(data).match(/aggregateRating|reviewCount|SearchAction|Jikan/));
  const tag = seoTags(PRODUCTION_ORIGIN).find((entry) => entry.attrs?.type === "application/ld+json");
  assert.deepEqual(JSON.parse(tag.children), data);
});

test("sitemap covers only public canonical pages and does not fabricate modification dates", () => {
  const xml = sitemapXml(PRODUCTION_ORIGIN);
  assert.equal((xml.match(/<loc>/g) ?? []).length, 2);
  assert.ok(xml.includes(`${PRODUCTION_ORIGIN}${GUIDE_PATH}`));
  assert.ok(!xml.match(/lastmod|priority|chapter=|example\.com/));
});

test("Vercel indexing headers only target utility and chapter URLs, not the home page", async () => {
  const config = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
  for (const rule of config.headers) {
    assert.ok(rule.source === "/not-found" || (rule.source === "/" && rule.has?.some((condition) => ["chapter", "url", "u"].includes(condition.key))));
  }
});

test("share card is a real 1200x630 PNG asset", async () => {
  const image = await readFile(new URL("../public/social-preview.png", import.meta.url));
  assert.equal(image.subarray(1, 4).toString(), "PNG");
  assert.equal(image.readUInt32BE(16), 1200);
  assert.equal(image.readUInt32BE(20), 630);
});
