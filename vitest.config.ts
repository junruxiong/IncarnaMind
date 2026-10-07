import { basename, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build, type Plugin } from "vite";
import { defineConfig } from "vitest/config";

const root = fileURLToPath(new URL(".", import.meta.url));
const NODE_WORKER = "?nodeWorker";

/**
 * electron-vite builds `import start from "./worker?nodeWorker"` into a separate
 * chunk and a function that starts it on a worker thread. Vitest can't run a
 * TypeScript worker, so this plugin does the same for tests: it bundles the
 * worker with Vite into node_modules/.cache (where its npm imports still
 * resolve) and returns a function that starts the bundle. The evaluation's
 * config (eval/vitest.config.ts) uses it too.
 */
export function nodeWorkers(): Plugin {
  const outDir = join(root, "node_modules/.cache/incarnamind-test-workers");
  return {
    name: "incarnamind:node-workers-for-tests",
    enforce: "pre",
    async resolveId(source, importer) {
      if (!source.endsWith(NODE_WORKER)) return null;
      const entry = await this.resolve(source.slice(0, -NODE_WORKER.length), importer, {
        skipSelf: true,
      });
      return entry && `${entry.id}${NODE_WORKER}`;
    },
    async load(id) {
      if (!id.endsWith(NODE_WORKER)) return null;
      const entry = id.slice(0, -NODE_WORKER.length);
      const fileName = `${basename(entry, extname(entry))}.mjs`;
      const output = await build({
        configFile: false,
        root,
        logLevel: "warn",
        build: {
          ssr: entry,
          outDir,
          emptyOutDir: false,
          minify: false,
          sourcemap: "inline",
          rollupOptions: { output: { format: "es", entryFileNames: fileName } },
        },
      });
      for (const result of Array.isArray(output) ? output : [output]) {
        if (!("output" in result)) continue;
        for (const chunk of result.output) {
          if (chunk.type === "chunk")
            for (const module of chunk.moduleIds) this.addWatchFile(module);
        }
      }
      return [
        `import { Worker } from "node:worker_threads";`,
        `export default (options) => new Worker(${JSON.stringify(join(outDir, fileName))}, options);`,
      ].join("\n");
    },
  };
}

export default defineConfig({
  plugins: [nodeWorkers()],
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
