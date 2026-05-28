import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const publicDir = path.join(rootDir, "public");

const rawSiteUrl = (process.env.SITE_URL ?? process.env.VITE_SITE_URL ?? "").trim();
const siteUrl = (rawSiteUrl || "https://example.com").replace(/\/+$/, "");

const pages = [
  { path: "/", priority: "1.0", changefreq: "daily" },
  { path: "/docs/product-guide.html", priority: "0.8", changefreq: "weekly" }
];

function toIsoDate(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

const today = toIsoDate();
const sitemapXml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages
  .map(
    (page) => `  <url>
    <loc>${siteUrl}${page.path}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${page.changefreq}</changefreq>
    <priority>${page.priority}</priority>
  </url>`
  )
  .join("\n")}
</urlset>
`;

const robotsTxt = `User-agent: *
Allow: /

Sitemap: ${siteUrl}/sitemap.xml
`;

await mkdir(publicDir, { recursive: true });
await writeFile(path.join(publicDir, "sitemap.xml"), sitemapXml, "utf8");
await writeFile(path.join(publicDir, "robots.txt"), robotsTxt, "utf8");

console.log(`[seo] Wrote sitemap + robots for ${siteUrl}`);
if (!rawSiteUrl) {
  console.log("[seo] SITE_URL not set; using fallback https://example.com. Set SITE_URL before production deploy.");
}
