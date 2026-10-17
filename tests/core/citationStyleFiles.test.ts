/**
 * What ships for citation styles (ADR-0016): the engine as built from wasm/citation-styles, and
 * every CSL style and locale, unmodified, in one pack, named in its licence notice with their
 * authors, the CSL project and its link, as the CSL repositories and CC BY-SA 3.0 ask.
 */
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { PackFiles, readPackIndex } from "../../src/core/citationStyles/pack";
import { CITATION_STYLES } from "../../src/shared/citationStyles";
import { CITATION_STYLES as FOLDER } from "../helpers/core";

const REPO_ROOT = resolve(__dirname, "../..");
const PACK = join(FOLDER, "styles.pack");
const notice = (name: string) => readFile(join(REPO_ROOT, "resources/notices", name), "utf8");
const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
const decode = (text: string) =>
  text
    .replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name: string) =>
      name.startsWith("#x")
        ? String.fromCodePoint(Number.parseInt(name.slice(2), 16))
        : name.startsWith("#")
          ? String.fromCodePoint(Number(name.slice(1)))
          : (ENTITIES[name] ?? whole),
    )
    .replace(/\s+/g, " ")
    .trim();
const CC_BY_SA = '<rights license="http://creativecommons.org/licenses/by-sa/3.0/">';

/** 2,863 styles and 8,002 journal styles on 2026-10-15, less a margin for an engine update. */
const FLOOR = 10_800;
/** What the download may grow by (#910): the old style files it replaces are left out. */
const BUDGET = 3_000_000;

describe("the files citation styles ship", () => {
  test("the engine is the file its checksum names, which the build script reproduces", async () => {
    const wasm = await readFile(join(FOLDER, "engine.wasm"));
    const recorded = (await readFile(join(FOLDER, "engine.wasm.sha256"), "utf8")).split(/\s+/)[0];
    expect(createHash("sha256").update(wasm).digest("hex")).toBe(recorded);
    // A wasm module: its magic number and version 1.
    expect([...wasm.subarray(0, 8)]).toEqual([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0]);
  });

  test("the pack is the file its checksum names, under the size the download may grow by", async () => {
    const recorded = (await readFile(`${PACK}.sha256`, "utf8")).split(/\s+/)[0];
    expect(
      createHash("sha256")
        .update(await readFile(PACK))
        .digest("hex"),
    ).toBe(recorded);
    expect((await stat(PACK)).size).toBeLessThan(BUDGET);
  });

  test("it holds every style the engine parses, more than the floor, journals beside the style they follow", () => {
    const index = readPackIndex(PACK);
    expect(index.styles.length).toBeGreaterThan(FLOOR);
    const byId = new Map(index.styles.map((style) => [style.id, style]));
    for (const style of index.styles) {
      if (style.parent === undefined) expect(style.at, style.id).toBeDefined();
      else expect(byId.get(style.parent)?.at, style.id).toBeDefined();
    }
    // The common styles, and Nature, open with the first chunk.
    for (const id of [...CITATION_STYLES.flatMap((style) => style.packId ?? []), "nature"]) {
      expect(byId.get(id)?.at?.[0], id).toBe(0);
    }
    expect(byId.get("nature-genetics")?.parent).toBe("nature");
    expect(Object.keys(index.locales)).toEqual(expect.arrayContaining(["en-US", "zh-CN", "de-DE"]));
  });

  test("every style and locale is an unmodified CSL file under CC BY-SA 3.0, its people named in the notice", () => {
    const index = readPackIndex(PACK);
    const files = new PackFiles(PACK, index, 2);
    const full = files.text(index.notice);
    expect(full).toContain("https://citationstyles.org/");
    expect(full).toContain(index.source.styles);
    expect(full).toContain("Attribution-ShareAlike 3.0 Unported");
    const listed = new Set<string>();
    const people = new Set<string>();
    for (const line of full.split("\n")) {
      const style = /^- .*\(([a-z0-9-]+)(?:, [^)]*)?\)$/.exec(line)?.[1];
      if (style) listed.add(style);
      const named = /^ {2}(?:Authors|Contributors|Translators): (.*)$/.exec(line)?.[1];
      for (const name of named?.split("; ") ?? []) people.add(name);
    }
    for (const style of index.styles) {
      expect(listed.has(style.id), style.id).toBe(true);
      if (!style.at) continue;
      const xml = files.text(style.at);
      expect(xml, style.id).toContain(CC_BY_SA);
      expect(xml, style.id).toContain("<citation");
      const info = /<info>([\s\S]*?)<\/info>/.exec(xml)?.[1] ?? "";
      for (const [, name] of info.matchAll(/<name>([^<]+)<\/name>/g)) {
        expect(people.has(decode(name as string)), `${style.id}: ${name}`).toBe(true);
      }
    }
    for (const [code, slot] of Object.entries(index.locales)) {
      const xml = files.text(slot);
      expect(xml, code).toContain(CC_BY_SA);
      for (const [, name] of xml.matchAll(/<translator>\s*<name>([^<]+)<\/name>/g)) {
        expect(people.has(decode(name as string)), `${code}: ${name}`).toBe(true);
      }
    }
  });

  test("the notices folder credits the CSL project and points to the full notice", async () => {
    const short = await notice("citation-styles.txt");
    expect(short).toContain("https://citationstyles.org/");
    expect(short).toContain("Creative Commons Attribution-ShareAlike 3.0 Unported");
    expect(short).toContain("Citation style licences");
    expect(await notice("cc-by-sa-3.0.txt")).toContain("Attribution-ShareAlike 3.0 Unported");
  });

  test("the engine's crates are named in their notice, and both notices and the files are packaged", async () => {
    const engine = await notice("citation-engine.txt");
    for (const crate of ["hayagriva", "citationberg", "icu_collator", "quick-xml", "serde_json"]) {
      expect(engine).toContain(`- ${crate} `);
    }
    expect(engine).toContain("UNICODE LICENSE V3");
    expect(engine).toContain("Boost Software License - Version 1.0");
    const builder = await readFile(join(REPO_ROOT, "electron-builder.yml"), "utf8");
    expect(builder).toMatch(/- from: resources\/citation-styles\n\s+to: citation-styles\n/);
    expect(builder).toMatch(/- from: resources\/notices\n\s+to: notices\n/);
  });
});
