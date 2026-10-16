import { readdir, readFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { describe, expect, test } from "vitest";
import { ANTHROPIC_OAUTH } from "../../src/core/providers/claudeAccount/config";

const root = fileURLToPath(new URL("../..", import.meta.url));
const CONFIG = "src/core/providers/claudeAccount/config.ts";

/** Every text the profile holds, which nothing but its config module may name. */
function profileTexts(): string[] {
  if (!ANTHROPIC_OAUTH) throw new Error("Tests build with the experimental sign-in routes on.");
  const { clientId, authorizeUrl, tokenUrl, redirect, request } = ANTHROPIC_OAUTH;
  return [
    clientId,
    authorizeUrl,
    tokenUrl,
    ...(redirect.kind === "page" ? [redirect.uri] : []),
    ...(request?.systemBlocks ?? []),
    ...Object.values(request?.headers ?? {}),
  ];
}

async function bundle(on: boolean): Promise<string> {
  const output = await build({
    configFile: false,
    root,
    logLevel: "silent",
    define: { __EXPERIMENTAL_SIGN_IN_ROUTES__: JSON.stringify(on) },
    build: { ssr: CONFIG, write: false, minify: false },
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
    const texts = profileTexts();
    const withRoutes = await bundle(true);
    const without = await bundle(false);

    for (const text of texts) {
      expect(withRoutes, text).toContain(text);
      expect(without, text).not.toContain(text);
    }
  });

  test("are named only in their config module", async () => {
    const texts = profileTexts();
    for (const file of await sourceFiles(join(root, "src"))) {
      if (relative(root, file) === CONFIG) continue;
      const source = await readFile(file, "utf8");
      for (const text of texts) expect(source.includes(text), `${file}: ${text}`).toBe(false);
    }
  });
});
