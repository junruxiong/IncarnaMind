/**
 * The grouping check's fixture set (eval/grouping/set.json, #51): what the
 * ticket asks it to hold, that every file of eval/grouping/fixtures is listed
 * and attributed, and that the core reads the decks and spreadsheets.
 */
import { readdirSync, readFileSync } from "node:fs";
import { basename, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import { HELD_OUT_CASES, loadGroupingSet } from "../../eval/grouping/lib/set";
import { nameFromPath } from "../../src/core/documents/files";
import { extractUnits } from "../../src/core/documents/formats";

const root = fileURLToPath(new URL("../..", import.meta.url));
const set = loadGroupingSet(root);
const fixtures = join(root, "eval/grouping/fixtures");
const byKey = new Map(set.documents.map((document) => [document.key, document]));

describe("the grouping fixture set", () => {
  test("has at least 5 English–Chinese same-subject pairs and 5 decks or spreadsheets in the held-out set", () => {
    expect(set.heldOut.pairs.length).toBeGreaterThanOrEqual(HELD_OUT_CASES);
    expect(set.heldOut.decks.length).toBeGreaterThanOrEqual(HELD_OUT_CASES);
    expect(set.choosing.pairs.length).toBeGreaterThanOrEqual(HELD_OUT_CASES);
    expect(set.choosing.decks.length).toBeGreaterThanOrEqual(HELD_OUT_CASES);
    for (const [en, zh] of [...set.choosing.pairs, ...set.heldOut.pairs]) {
      expect(byKey.get(en)?.language).toBe("en");
      expect(byKey.get(zh)?.language).toBe("zh");
      expect(byKey.get(en)?.subject).toBe(byKey.get(zh)?.subject);
    }
    const forms = [...set.choosing.decks, ...set.heldOut.decks].map((key) => byKey.get(key)?.form);
    expect(forms).toContain("deck");
    expect(forms).toContain("spreadsheet");
  });

  test("keeps the choosing and held-out cases apart, on subjects of their own", () => {
    const subjects = (cases: typeof set.choosing) =>
      new Set([...cases.pairs.flat(), ...cases.decks].map((key) => byKey.get(key)?.subject));
    const choosing = subjects(set.choosing);
    for (const subject of subjects(set.heldOut)) expect(choosing.has(subject)).toBe(false);
  });

  test('has a shared-prefix row of "维基百科-" Documents on different subjects', () => {
    const row = set.sharedNames.find((each) => each.label.includes("维基百科-"));
    const members = set.documents.filter((document) =>
      row?.pattern.test(nameFromPath(document.path)),
    );
    expect(members.length).toBeGreaterThanOrEqual(5);
    expect(new Set(members.map((document) => document.subject)).size).toBeGreaterThanOrEqual(5);
  });

  test("gives every subject at least 3 Documents, so a cluster of one subject isn't dissolved", () => {
    for (const subject of Object.keys(set.subjects)) {
      const count = set.documents.filter((document) => document.subject === subject).length;
      expect(count, subject).toBeGreaterThanOrEqual(3);
    }
  });

  test("has late arrivals on a subject already grouped, and on a new one", () => {
    const late = new Set(set.lateArrivals);
    const firstSubjects = new Set(
      set.documents
        .filter((document) => !late.has(document.key))
        .map((document) => document.subject),
    );
    const subjects = set.lateArrivals.map((key) => byKey.get(key)?.subject as string);
    expect(subjects.some((subject) => firstSubjects.has(subject))).toBe(true);
    expect(subjects.some((subject) => !firstSubjects.has(subject))).toBe(true);
  });

  test("lists and attributes every file in eval/grouping/fixtures", () => {
    const listed = new Set(set.documents.map((document) => document.path));
    const attribution = readFileSync(join(fixtures, "ATTRIBUTION.md"), "utf8");
    for (const name of readdirSync(fixtures)) {
      if (name === "ATTRIBUTION.md") continue;
      expect(listed.has(join(fixtures, name)), `${name} isn't in set.json`).toBe(true);
      expect(attribution.includes(name), `${name} isn't in ATTRIBUTION.md`).toBe(true);
    }
  });

  test("the core reads text from each deck and spreadsheet", async () => {
    for (const document of set.documents.filter((each) => each.form !== "prose")) {
      const extension = extname(document.path).slice(1) as "pptx" | "xlsx" | "csv";
      const units = await extractUnits(extension, new Uint8Array(readFileSync(document.path)));
      const text = units.map((unit) => unit.text).join("\n");
      expect(text.length, basename(document.path)).toBeGreaterThan(200);
    }
  });
});
