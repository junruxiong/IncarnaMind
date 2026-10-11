/**
 * What ships for citation styles (ADR-0016): the engine as built from wasm/citation-styles,
 * and the CSL styles and locales as unmodified files, named in the licence notices with their
 * authors, the CSL project and its link, as the CSL repositories and CC BY-SA 3.0 ask.
 */
import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { CITATION_STYLES } from "../../src/shared/citationStyles";
import { CITATION_STYLES as FOLDER } from "../helpers/core";

const REPO_ROOT = resolve(__dirname, "../..");
const notice = (name: string) => readFile(join(REPO_ROOT, "resources/notices", name), "utf8");
const decode = (text: string) => text.replace(/&amp;/g, "&").replace(/&apos;/g, "'");

describe("the files citation styles ship", () => {
  test("the engine is the file its checksum names, which the build script reproduces", async () => {
    const wasm = await readFile(join(FOLDER, "engine.wasm"));
    const recorded = (await readFile(join(FOLDER, "engine.wasm.sha256"), "utf8")).split(/\s+/)[0];
    expect(createHash("sha256").update(wasm).digest("hex")).toBe(recorded);
    // A wasm module: its magic number and version 1.
    expect([...wasm.subarray(0, 8)]).toEqual([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0]);
  });

  test("every style is an independent CSL style under CC BY-SA 3.0, its people named in the notice", async () => {
    const styles = await notice("citation-styles.txt");
    expect(styles).toContain("https://citationstyles.org/");
    expect(styles).toContain("Attribution-ShareAlike 3.0 Unported");
    const shipped = await readdir(join(FOLDER, "styles"));
    const files = CITATION_STYLES.flatMap((style) => (style.file ? [style.file] : []));
    expect(shipped.sort()).toEqual([...files].sort());
    for (const file of files) {
      const xml = await readFile(join(FOLDER, "styles", file), "utf8");
      expect(xml, file).toContain(
        '<rights license="http://creativecommons.org/licenses/by-sa/3.0/">',
      );
      // A dependent style would have no citation layout of its own.
      expect(xml, file).toContain("<citation");
      expect(styles, file).toContain(`styles/${file}`);
      const info = /<info>([\s\S]*?)<\/info>/.exec(xml)?.[1] ?? "";
      for (const [, name] of info.matchAll(/<name>([^<]+)<\/name>/g)) {
        expect(styles, `${file}: ${name}`).toContain(decode((name as string).trim()));
      }
    }
  });

  test("the locales the styles use ship, their translators named", async () => {
    const styles = await notice("citation-styles.txt");
    for (const code of ["en-US", "en-GB", "zh-CN"]) {
      const xml = await readFile(join(FOLDER, "locales", `locales-${code}.xml`), "utf8");
      expect(xml).toContain('<rights license="http://creativecommons.org/licenses/by-sa/3.0/">');
      for (const [, name] of xml.matchAll(/<translator>\s*<name>([^<]+)<\/name>/g)) {
        expect(styles, `${code}: ${name}`).toContain(decode((name as string).trim()));
      }
    }
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
