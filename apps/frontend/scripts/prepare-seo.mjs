import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { loadEnv } from "vite";
import { resolveSiteOrigin, robotsTxt, sitemapXml } from "./seo.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const publicDir = new URL("../public/", import.meta.url);
const origin = resolveSiteOrigin({ ...loadEnv("production", root, ""), ...process.env });
await mkdir(publicDir, { recursive: true });
await writeFile(new URL("sitemap.xml", publicDir), sitemapXml(origin));
await writeFile(new URL("robots.txt", publicDir), robotsTxt(origin));
console.log(`[seo] Prepared sitemap and robots.txt for ${origin}`);
