/**
 * The citation-style engine run in the test's own thread, on the style pack the app ships, and
 * the spike's 30 English and Chinese references (#288), for the golden tests (#289).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync } from "node:zlib";
import type { StyledRun } from "../../src/core/api";
import { createCatalogue } from "../../src/core/citationStyles/catalogue";
import type { CslItem } from "../../src/core/citationStyles/csl";
import { type FormatJob, formatJob } from "../../src/core/citationStyles/format";
import {
  PACK_HEADER_BYTES,
  PACK_MAGIC,
  PackFiles,
  type PackIndex,
  type PackStyle,
  readPackIndex,
} from "../../src/core/citationStyles/pack";
import { createPackStyles } from "../../src/core/citationStyles/packStyles";
import {
  CslEngine,
  type EngineCitation,
  type EngineOutput,
} from "../../src/core/citationStyles/wasm";
import type { Language } from "../../src/core/language";
import { type CitationStyleId, packIdOf } from "../../src/shared/citationStyles";
import { CITATION_STYLES } from "./core";

const FIXTURES = fileURLToPath(new URL("../fixtures/citation-styles", import.meta.url));

/** The 30 references (16 English, 14 Chinese) as CSL JSON, and the 34 citations of them. */
export const fixtureReferences = (): CslItem[] =>
  JSON.parse(readFileSync(join(FIXTURES, "references.json"), "utf8")) as CslItem[];

export const fixtureCitations = (): EngineCitation[] =>
  JSON.parse(readFileSync(join(FIXTURES, "citations.json"), "utf8")) as EngineCitation[];

export const goldenFile = (style: CitationStyleId) => join(FIXTURES, "golden", `${style}.txt`);

let engine: CslEngine | undefined;
const handles = new Map<string, number>();
const catalogue = createCatalogue(join(CITATION_STYLES, "styles.pack"));
const pack = createPackStyles(CITATION_STYLES);

/**
 * Formats a job with a shipped style, as the worker does, but on this thread: a common style or
 * any style of the pack, in a Mind whose text is in `language`.
 */
export function formatHere(
  style: CitationStyleId,
  job: Omit<FormatJob, "englishTermsForOthers" | "raisedLocators">,
  language: Language = "en",
) {
  const resolved = catalogue.resolve(style, language);
  const { engine: wanted } = resolved;
  if (!wanted) throw new Error(`${style} isn't a CSL style.`);
  if (!engine) {
    engine = new CslEngine(
      new WebAssembly.Module(readFileSync(join(CITATION_STYLES, "engine.wasm"))),
    );
  }
  const live = engine;
  const key = `${wanted.packId} ${wanted.locale ?? ""}`;
  let handle = handles.get(key);
  if (handle === undefined) {
    const { text, locales } = pack.style(wanted);
    handle = live.parse(text, locales).handle;
    handles.set(key, handle);
  }
  const parsed = handle;
  return formatJob(
    {
      ...job,
      englishTermsForOthers: resolved.language === "zh-CN",
      raisedLocators: resolved.locators === "raised-after",
      ...(wanted.locale ? { locale: wanted.locale } : {}),
    },
    (request) => live.render(parsed, request),
  );
}

const shipped = (() => {
  const path = join(CITATION_STYLES, "styles.pack");
  const index = readPackIndex(path);
  return { index, files: new PackFiles(path, index) };
})();

/** A style's file as the app ships it, by its id in the pack. */
export function shippedStyle(id: string): string {
  const at = shipped.index.styles.find((style) => style.id === id)?.at;
  if (!at) throw new Error(`The pack has no file for ${id}.`);
  return shipped.files.text(at);
}

/** A locale's file as the app ships it: "en-US". */
export function shippedLocale(code: string): string {
  const at = shipped.index.locales[code];
  if (!at) throw new Error(`The pack has no locale ${code}.`);
  return shipped.files.text(at);
}

/** Writes a pack of these files, laid out as scripts/citationStylePack.ts lays out the real one. */
export function writePack(
  path: string,
  styles: Readonly<Record<string, string>>,
  locales: Readonly<Record<string, string>>,
): void {
  const chunks: Buffer[] = [];
  const add = (files: readonly string[]) => {
    chunks.push(brotliCompressSync(JSON.stringify(files)));
    return chunks.length - 1;
  };
  const entries: PackStyle[] = Object.entries(styles).map(([id, text]) => ({
    id,
    title: id,
    format: "numeric",
    at: [add([text]), 0],
  }));
  const localeChunk = add(Object.values(locales));
  const notice = add(["A notice."]);
  let offset = 0;
  const index: PackIndex = {
    version: 1,
    source: { styles: "test", locales: "test" },
    chunks: chunks.map((bytes) => {
      const where = { offset, length: bytes.length };
      offset += bytes.length;
      return where;
    }),
    styles: entries,
    locales: Object.fromEntries(
      Object.keys(locales).map((code, slot) => [code, [localeChunk, slot]]),
    ),
    notice: [notice, 0],
  };
  const indexBytes = brotliCompressSync(JSON.stringify(index));
  const header = Buffer.alloc(PACK_HEADER_BYTES);
  header.write(PACK_MAGIC, 0, "latin1");
  header.writeUInt32LE(indexBytes.length, PACK_MAGIC.length);
  writeFileSync(path, Buffer.concat([header, indexBytes, ...chunks]));
}

const escapeText = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;");

/**
 * Runs as readable text with their formatting as tags: <i>, <b>, <sc> (small capitals), <u>,
 * <sup>, <sub>, and a link as <a href="…"> where its address isn't its text.
 */
export function markup(runs: readonly StyledRun[]): string {
  // Runs that differ only in the work they belong to read as one.
  const merged: StyledRun[] = [];
  for (const { text, italic, bold, smallCaps, underline, script, link } of runs) {
    const run: StyledRun = { text, italic, bold, smallCaps, underline, script, link };
    const last = merged.at(-1);
    const same =
      last !== undefined &&
      (Object.keys(run) as (keyof StyledRun)[]).every(
        (key) => key === "text" || last[key] === run[key],
      );
    if (same) last.text += text;
    else merged.push(run);
  }
  return merged
    .map((run) => {
      let text = escapeText(run.text);
      if (run.script) text = `<${run.script}>${text}</${run.script}>`;
      if (run.underline) text = `<u>${text}</u>`;
      if (run.smallCaps) text = `<sc>${text}</sc>`;
      if (run.bold) text = `<b>${text}</b>`;
      if (run.italic) text = `<i>${text}</i>`;
      if (run.link && run.link !== run.text) text = `<a href="${escapeText(run.link)}">${text}</a>`;
      return text;
    })
    .join("");
}

/** A style's output as the golden file keeps it. */
export function goldenText(style: CitationStyleId, output: EngineOutput): string {
  const lines = [
    `# ${style} (${packIdOf(style)}.csl)`,
    "",
    `## Citations (${output.citations.length})`,
    "",
    ...output.citations.map((runs, index) => `${index + 1}. ${markup(runs)}`),
    "",
    `## Bibliography (${output.bibliography.length}${output.hangingIndent ? ", hanging" : ""})`,
    "",
    ...output.bibliography.map(
      (entry) =>
        `- ${entry.id}: ${entry.firstField ? `${markup(entry.firstField)} ` : ""}${markup(entry.runs)}`,
    ),
  ];
  return `${lines.join("\n")}\n`;
}
