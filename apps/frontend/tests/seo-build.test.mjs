import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import { test } from "node:test";
import { load } from "cheerio";
import { PRODUCTION_ORIGIN, GUIDE_PATH } from "../scripts/seo.mjs";

const dist = new URL("../dist/", import.meta.url);
const html = await readFile(new URL("index.html", dist), "utf8");
const $ = load(html);

test("built home page has one canonical, title and description available without JavaScript", () => {
  assert.equal($("title").length, 1);
  assert.equal($('meta[name="description"]').length, 1);
  assert.equal($('link[rel="canonical"]').length, 1);
  assert.equal($('link[rel="canonical"]').attr("href"), `${PRODUCTION_ORIGIN}/`);
  assert.ok($("title").text().includes("Manga & Anime Discovery"));
  assert.ok(!$('meta[name="robots"]').attr("content").includes("noindex"));
});

test("built body is the existing functional home UI, not an empty root or hidden SEO block", () => {
  assert.equal($("#root h1").length, 1);
  assert.ok($("#root h1").text().includes("RAGnarok"));
  assert.equal($("#smart-search").length, 1);
  assert.equal($("#about-ragnarok-title").length, 1);
  assert.equal($('body script[type="application/ld+json"]').length, 0);
  assert.ok(!html.includes("example.com"));
});

test("built structured data and social previews use the canonical production domain", async () => {
  const json = JSON.parse($('script[type="application/ld+json"]').text());
  assert.equal(json["@graph"].length, 3);
  assert.ok(json["@graph"].every((node) => node.url === `${PRODUCTION_ORIGIN}/`));
  assert.equal($('meta[property="og:url"]').attr("content"), `${PRODUCTION_ORIGIN}/`);
  assert.equal($('meta[property="og:image"]').attr("content"), `${PRODUCTION_ORIGIN}/social-preview.png`);
  await access(new URL("social-preview.png", dist));
});

test("every local static script, stylesheet, icon and documentation link exists in the build", async () => {
  for (const element of $("script[src], link[href], a[href]").toArray()) {
    const href = $(element).attr("src") ?? $(element).attr("href");
    if (!href?.startsWith("/") || href === "/") continue;
    await access(new URL(href.slice(1), dist));
  }
});

test("built sitemap and robots.txt use the same canonical origin", async () => {
  const xml = load(await readFile(new URL("sitemap.xml", dist), "utf8"), { xmlMode: true });
  assert.deepEqual(xml("loc").map((_, node) => xml(node).text()).get(), [`${PRODUCTION_ORIGIN}/`, `${PRODUCTION_ORIGIN}${GUIDE_PATH}`]);
  assert.match(await readFile(new URL("robots.txt", dist), "utf8"), /Sitemap: https:\/\/ragnarok-reader\.vercel\.app\/sitemap.xml/);
});

test("guide canonical is added without changing any existing visible guide content", async () => {
  const original = load(await readFile(new URL("../public/docs/product-guide.html", import.meta.url), "utf8"));
  const guide = load(await readFile(new URL("docs/product-guide.html", dist), "utf8"));
  assert.equal(guide('link[rel="canonical"]').attr("href"), `${PRODUCTION_ORIGIN}${GUIDE_PATH}`);
  assert.equal(guide("body").html(), original("body").html());
});

test("deployment installs and checks FastEmbed using the service interpreter before restart", async () => {
  const workflow = await readFile(new URL("../../../.github/workflows/deploy-vm.yml", import.meta.url), "utf8");
  assert.ok(workflow.includes('PY="$DEPLOY_REPO_REAL_PATH/.venv/bin/python"'));
  const install = workflow.indexOf('"$PY" -m pip install --quiet -r rag/requirements.txt');
  const check = workflow.indexOf("from fastembed import TextEmbedding");
  const restart = workflow.indexOf("sudo systemctl restart ragnarok-rag.service");
  assert.ok(install >= 0 && install < check && check < restart);
});
