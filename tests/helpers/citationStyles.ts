/**
 * The citation-style engine run in the test's own thread, on the files the app ships, and the
 * spike's 30 English and Chinese references (#288), for the golden tests (#289).
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { StyledRun } from "../../src/core/api";
import type { CslItem } from "../../src/core/citationStyles/csl";
import { type FormatJob, formatJob } from "../../src/core/citationStyles/format";
import {
  CslEngine,
  type EngineCitation,
  type EngineOutput,
} from "../../src/core/citationStyles/wasm";
import { type CitationStyleId, citationStyle } from "../../src/shared/citationStyles";
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

/** Formats a job with a shipped style, as the worker does, but on this thread. */
export function formatHere(
  style: CitationStyleId,
  job: Omit<FormatJob, "englishTermsForOthers" | "raisedLocators">,
) {
  const { file, language, locators } = citationStyle(style);
  if (!file) throw new Error(`${style} isn't a CSL style.`);
  if (!engine) {
    engine = new CslEngine(
      new WebAssembly.Module(readFileSync(join(CITATION_STYLES, "engine.wasm"))),
    );
  }
  const live = engine;
  let handle = handles.get(file);
  if (handle === undefined) {
    const locales = ["en-US", "en-GB", "zh-CN"].map((code) =>
      readFileSync(join(CITATION_STYLES, "locales", `locales-${code}.xml`), "utf8"),
    );
    handle = live.parse(
      readFileSync(join(CITATION_STYLES, "styles", file), "utf8"),
      locales,
    ).handle;
    handles.set(file, handle);
  }
  const parsed = handle;
  return formatJob(
    {
      ...job,
      englishTermsForOthers: language === "zh-CN",
      raisedLocators: locators === "raised-after",
    },
    (request) => live.render(parsed, request),
  );
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
  const { file } = citationStyle(style);
  const lines = [
    `# ${style} (${file})`,
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
