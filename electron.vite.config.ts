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

/**
 * The licences of the fonts the renderer bundles (src/renderer/src/fonts.css).
 * All are under the SIL Open Font License 1.1, which asks that every copy of a font
 * comes with it.
 */
const FONT_LICENCES: Readonly<Record<string, string>> = {
  "SourceSerif4-OFL.txt": "@fontsource-variable/source-serif-4/LICENSE",
  "SourceSans3-OFL.txt": "@fontsource-variable/source-sans-3/LICENSE",
  "JetBrainsMono-OFL.txt": "@fontsource/jetbrains-mono/LICENSE",
  // Calibri's stand-in in drawn slides (src/renderer/src/viewer/slides/slides.css).
  "Carlito-OFL.txt": "@fontsource/carlito/LICENSE",
};

/** Copies the bundled fonts' licences into a build, under licenses/. */
function fontLicences(): Plugin {
  const require = createRequire(import.meta.url);
  return {
    name: "incarnamind:font-licences",
    generateBundle() {
      for (const [fileName, licence] of Object.entries(FONT_LICENCES)) {
        this.emitFile({
          type: "asset",
          fileName: `licenses/${fileName}`,
          source: readFileSync(require.resolve(licence)),
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
    plugins: [react(), tailwindcss(), pdfjsData(), fontLicences()],
    // electron-vite leaves the renderer unminified; minified, its JavaScript is half the size
    // to load and parse at every launch.
    build: { minify: true },
  },
});
