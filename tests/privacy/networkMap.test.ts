/**
 * docs/privacy.md says everything that leaves the computer (#1615), and this
 * keeps it true as the code grows: every data flow and every other connection
 * has its row, and every file in src/core that can reach the network is
 * covered by a line of the page's code table. A new flow or a new file that
 * fails here needs its words in the page first.
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, test } from "vitest";
import { dataFlowIds, networkTrafficIds } from "../../src/core/api";

const root = join(__dirname, "../..");
const page = readFileSync(join(root, "docs/privacy.md"), "utf8");
const [flows = "", codeTable = ""] = page.split("## Where the code reaches the network");

/** What reaches the network from the core: a call to `fetch`, a socket, the AI SDK's providers, a model's download host. */
const REACHES_NETWORK =
  /\bfetch\s*\(|\bfetch(?:Fn|Impl)\b|node:https?|node:net|new WebSocket|from "@ai-sdk\/|huggingface\.co|\bundici\b/;

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return path.endsWith(".ts") ? [path] : [];
  });
}

const withoutComments = (source: string) =>
  source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'`])\/\/.*$/gm, "$1");

describe("docs/privacy.md", () => {
  test.each([...dataFlowIds, ...networkTrafficIds])("has a row for `%s`", (id) => {
    expect(flows).toContain(`\`${id}\``);
  });

  const listed = [...codeTable.matchAll(/`(src\/[^`]+)`/g)].map((match) => match[1] as string);

  test("names only files and folders that exist", () => {
    expect(listed.length).toBeGreaterThan(0);
    expect(listed.filter((path) => !existsSync(join(root, path)))).toEqual([]);
  });

  test("covers every file in src/core that reaches the network", () => {
    const uncovered = sourceFiles(join(root, "src/core"))
      .filter((file) => REACHES_NETWORK.test(withoutComments(readFileSync(file, "utf8"))))
      .map((file) => relative(root, file).split("\\").join("/"))
      .filter(
        (file) =>
          !listed.some((path) => path === file || (path.endsWith("/") && file.startsWith(path))),
      );
    expect(uncovered).toEqual([]);
  });
});
