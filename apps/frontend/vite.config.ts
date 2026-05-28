import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "panelflow-legacy-url-query-rewrite",
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          if (!req.url) {
            next();
            return;
          }

          const parsed = new URL(req.url, "http://localhost");
          if (parsed.pathname !== "/" && parsed.pathname !== "/index.html") {
            next();
            return;
          }

          const legacyUrl = parsed.searchParams.get("url")?.trim();
          const chapterParam = parsed.searchParams.get("chapter")?.trim();

          if (!legacyUrl || chapterParam) {
            next();
            return;
          }

          parsed.searchParams.set("chapter", legacyUrl);
          parsed.searchParams.delete("url");
          const query = parsed.searchParams.toString();
          req.url = `${parsed.pathname}${query ? `?${query}` : ""}`;
          next();
        });
      }
    }
  ],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8787",
      "/adapters": "http://localhost:8787",
      "/health": "http://localhost:8787"
    }
  }
});
