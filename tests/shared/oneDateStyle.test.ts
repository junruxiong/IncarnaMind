import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { describe, expect, test } from "vitest";

/*
 * Every date the interface shows comes from src/shared/dates.ts, so that one
 * language has one style. A date or a time written by hand anywhere else
 * fails here: use the module's kind that fits, or add one to it.
 */

const ROOT = join(__dirname, "../..");
const MODULE = "src/shared/dates.ts";

/** The files that write a date in a way the module's kinds don't, and why. */
const OWN_RULES: Record<string, string> = {
  "src/core/exports/index.ts":
    "an export keeps its own date rules, which come from the citation style",
  "src/shared/sources.ts":
    "the References and every export write the same line, 'accessed 6 October 2026'",
  "src/core/documents/formats/numberDisplay.ts":
    "reads the system's short date as a spreadsheet's format code; it shows no date",
};

const BY_HAND = new RegExp(
  [
    String.raw`\bDateTimeFormat\b`,
    String.raw`\bRelativeTimeFormat\b`,
    String.raw`\.toLocale(?:Date|Time)String\(`,
    String.raw`\.toLocaleString\([^)]*\b(?:year|month|day|weekday|hour|minute|dateStyle|timeStyle)\b`,
    String.raw`\.to(?:Date|Time|UTC|GMT)String\(`,
    String.raw`\.get(?:Hours|Minutes)\(`,
    String.raw`\b(?:dateStyle|timeStyle)\b`,
  ].join("|"),
);

interface Source {
  /** From the repository's root, with "/" on every system. */
  path: string;
  text: string;
}

function sourcesIn(folder: string): Source[] {
  return readdirSync(folder).flatMap((name) => {
    const path = join(folder, name);
    if (statSync(path).isDirectory()) return sourcesIn(path);
    if (!/\.tsx?$/.test(name)) return [];
    return [{ path: relative(ROOT, path).split(sep).join("/"), text: readFileSync(path, "utf8") }];
  });
}

/** The files, other than the module, that write a date by hand. */
const writingDatesByHand = (files: readonly Source[]): string[] =>
  files.filter((file) => file.path !== MODULE && BY_HAND.test(file.text)).map((file) => file.path);

describe("dates", () => {
  const files = sourcesIn(join(ROOT, "src"));

  test("are written in one module: no date or time formatted by hand elsewhere", () => {
    const found = writingDatesByHand(files);
    expect(found.filter((path) => !(path in OWN_RULES))).toEqual([]);
  });

  test("have no file listed as writing its own that no longer does", () => {
    const found = writingDatesByHand(files);
    expect(Object.keys(OWN_RULES).filter((path) => !found.includes(path))).toEqual([]);
  });

  test("fail a format built by hand in a new file, whatever it uses", () => {
    const sample = (text: string): Source[] => [{ path: "src/renderer/src/new.ts", text }];
    for (const text of [
      'new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(at)',
      "at.toLocaleDateString()",
      "at.toLocaleTimeString()",
      'at.toLocaleString("en", { hour: "numeric" })',
      "now.toDateString() === at.toDateString()",
      'at.getHours() + ":" + at.getMinutes()',
      'new Intl.RelativeTimeFormat("en").format(-1, "day")',
    ]) {
      expect(writingDatesByHand(sample(text)), text).toEqual(["src/renderer/src/new.ts"]);
    }
  });

  test("don't fail what isn't one: a number's separators, a day's arithmetic and the module itself", () => {
    const ok = (path: string, text: string): Source[] => [{ path, text }];
    expect(writingDatesByHand(ok("src/a.ts", "rows.toLocaleString()"))).toEqual([]);
    expect(writingDatesByHand(ok("src/a.ts", "new Date(at.getFullYear(), 0, 1)"))).toEqual([]);
    expect(writingDatesByHand(ok(MODULE, "new Intl.DateTimeFormat(locale)"))).toEqual([]);
  });
});
