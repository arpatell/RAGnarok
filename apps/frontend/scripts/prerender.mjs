import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer, loadEnv } from "vite";
import { resolveSiteOrigin, GUIDE_PATH } from "./seo.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const origin = resolveSiteOrigin({ ...loadEnv("production", root, ""), ...process.env });
const server = await createServer({ root, server: { middlewareMode: true, watch: null }, appType: "custom" });
try {
  const { render } = await server.ssrLoadModule("/src/prerender.tsx");
  const indexPath = new URL("../dist/index.html", import.meta.url);
  const html = await readFile(indexPath, "utf8");
  const marker = '<div id="root"></div>';
  if (!html.includes(marker)) throw new Error("Missing app root for static rendering");
  const markup = render();
  if (!markup.includes("<h1>")) throw new Error("Prerender did not produce the existing home screen");
  await writeFile(indexPath, html.replace(marker, () => `<div id="root">${markup}</div>`));

  // Give the existing static guide the same canonical site identity.
  const guidePath = new URL(`../dist${GUIDE_PATH}`, import.meta.url);
  let guide = await readFile(guidePath, "utf8");
  if (process.env.VERCEL_ENV === "preview") {
    guide = guide.replace(/content="index,follow[^"]*"/, 'content="noindex,follow"');
  }
  guide = guide.replace("</head>", `  <link rel="canonical" href="${origin}${GUIDE_PATH}" />\n</head>`);
  await writeFile(guidePath, guide);
  console.log("[seo] Pre-rendered the existing home screen; no new visible content or server required.");
} finally {
  await server.close();
}
