import { readdir, readFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { describe, expect, test } from "vitest";
import { ANTHROPIC_OAUTH } from "../../src/core/providers/claudeAccount/config";
import { COPILOT_PROFILES } from "../../src/core/providers/copilot/config";

const root = fileURLToPath(new URL("../..", import.meta.url));

const required = <T>(value: T | null | undefined): T => {
  if (!value) throw new Error("Tests build with the experimental sign-in routes on.");
  return value;
};

/** Each route's config module, and every text only its experimental profile holds, which nothing else may name. */
function routes(): { config: string; texts: string[] }[] {
  const claude = required(ANTHROPIC_OAUTH);
  const editor = required(COPILOT_PROFILES.editor);
  return [
    {
      config: "src/core/providers/claudeAccount/config.ts",
      texts: [
        claude.clientId,
        claude.authorizeUrl,
        claude.tokenUrl,
        ...(claude.redirect.kind === "page" ? [claude.redirect.uri] : []),
        ...(claude.request?.systemBlocks ?? []),
        ...Object.values(claude.request?.headers ?? {}),
      ],
    },
    {
      config: "src/core/providers/copilot/config.ts",
      texts: [
        editor.clientId,
        ...Object.values(editor.signInHeaders ?? {}),
        ...Object.values(editor.requestHeaders ?? {}),
        ...(editor.tokenExchange ? [editor.tokenExchange.url] : []),
        ...Object.values(editor.tokenExchange?.headers ?? {}),
      ],
    },
  ];
}

async function bundle(config: string, on: boolean): Promise<string> {
  const output = await build({
    configFile: false,
    root,
    logLevel: "silent",
    define: { __EXPERIMENTAL_SIGN_IN_ROUTES__: JSON.stringify(on) },
    build: { ssr: config, write: false, minify: false },
  });
  const results = Array.isArray(output) ? output : "output" in output ? [output] : [];
  return results
    .flatMap((result) => result.output)
    .map((chunk) => (chunk.type === "chunk" ? chunk.code : ""))
    .join("\n");
}

async function sourceFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true, recursive: true });
  return entries
    .filter((entry) => entry.isFile() && /\.tsx?$/.test(entry.name))
    .map((entry) => join(entry.parentPath, entry.name));
}

describe("The experimental sign-in routes", () => {
  test("are in a bundle built with them, and not at all in one built without", async () => {
    for (const { config, texts } of routes()) {
      const withRoutes = await bundle(config, true);
      const without = await bundle(config, false);
      for (const text of texts) {
        expect(withRoutes, text).toContain(text);
        expect(without, text).not.toContain(text);
      }
    }
  });

  test("are named only in their config module", async () => {
    const all = routes();
    for (const file of await sourceFiles(join(root, "src"))) {
      const source = await readFile(file, "utf8");
      for (const { config, texts } of all) {
        if (relative(root, file) === config) continue;
        for (const text of texts) expect(source.includes(text), `${file}: ${text}`).toBe(false);
      }
    }
  });
});
