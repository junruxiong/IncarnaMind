import { createReadStream, readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, extname, join } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "electron-vite";
import type { Plugin } from "vite";

/** Where the renderer finds pdf.js's data files, relative to its page (see src/renderer/src/viewer/pdfjs.ts). */
const PDFJS_DATA_PATH = "pdfjs";

/**
 * pdf.js's data files, loaded at runtime by the Document viewer: character maps
 * for CJK fonts, the standard fonts, and the WebAssembly image decoders (with
 * their licences). Scripting stays off, so its QuickJS sandbox is left out.
 */
const PDFJS_DATA_FOLDERS = ["cmaps", "standard_fonts", "wasm"];

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  ".wasm": "application/wasm",
  ".js": "text/javascript",
};

/** Serves pdf.js's data files under /pdfjs/ in development, and copies them there in a build. */
function pdfjsData(): Plugin {
  const packageDir = dirname(createRequire(import.meta.url).resolve("pdfjs-dist/package.json"));
  const files = () =>
    PDFJS_DATA_FOLDERS.flatMap((folder) =>
      readdirSync(join(packageDir, folder))
        .filter((name) => !name.startsWith("quickjs"))
        .map((name) => `${folder}/${name}`),
    );

  return {
    name: "incarnamind:pdfjs-data",
    configureServer(server) {
      const available = new Set(files());
      server.middlewares.use(`/${PDFJS_DATA_PATH}/`, (request, response, next) => {
        const name = decodeURIComponent((request.url ?? "").split("?")[0] ?? "").replace(/^\//, "");
        if (!available.has(name)) return next();
        response.setHeader(
          "Content-Type",
          CONTENT_TYPES[extname(name)] ?? "application/octet-stream",
        );
        createReadStream(join(packageDir, name)).pipe(response);
      });
    },
    generateBundle() {
      for (const name of files()) {
        this.emitFile({
          type: "asset",
          fileName: `${PDFJS_DATA_PATH}/${name}`,
          source: readFileSync(join(packageDir, name)),
        });
      }
    },
  };
}

// Entry points follow electron-vite's defaults: src/main, src/preload and src/renderer.
export default defineConfig({
  main: {},
  preload: {},
  renderer: {
    plugins: [react(), tailwindcss(), pdfjsData()],
  },
});
