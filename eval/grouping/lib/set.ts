/**
 * The grouping check's fixture set (eval/grouping/set.json): every Document
 * with its subject, language and form, the English–Chinese same-subject pairs
 * and the decks and spreadsheets, split into a choosing set and a held-out
 * set (O8a), the shared-name rows (R3), the late arrivals and the corrections
 * of the incremental cases.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { EvalDocument } from "../../lib/evaluationSet";

export const GROUPING_SET = "eval/grouping/set.json";

export type Language = "en" | "zh";
export type Form = "prose" | "deck" | "spreadsheet";

export interface SetDocument extends EvalDocument {
  language: Language;
  subject: string;
  form: Form;
}

/** The cases one part of the set is scored on. */
export interface Cases {
  /** English–Chinese pairs on one subject: both should land in one Topic. */
  pairs: [string, string][];
  /** Decks and spreadsheets: each should land with a Document on its subject. */
  decks: string[];
}

export interface NameRow {
  label: string;
  /** Matched against each Document's name (its file name without the extension). */
  pattern: RegExp;
}

export interface Corrections {
  /** The Topic holding this Document is renamed by the User. */
  rename: string;
  /** Up to this many pairs that were split up are put together: the Chinese Document moved to the English one's Topic. */
  moves: number;
  /** If no pair needs it, this Document is moved to the second one's Topic, so that there is a move. */
  lastResortMove: [string, string];
  /** A Topic the User makes, by moving these Documents into it. */
  create: { name: string; documents: string[] };
}

export interface GroupingSet {
  source: string;
  /** Subject keys and what they are. */
  subjects: Record<string, string>;
  documents: SetDocument[];
  choosing: Cases;
  heldOut: Cases;
  sharedNames: NameRow[];
  /** Left out of the first grouping, then placed by the threshold. */
  lateArrivals: string[];
  corrections: Corrections;
}

/** The held-out set must have this many of each case: the pass bars are "4 of 5" and "3 of 5". */
export const HELD_OUT_CASES = 5;

function fail(message: string): never {
  throw new Error(`${GROUPING_SET}: ${message}`);
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function readDocuments(
  raw: unknown,
  subjects: Record<string, string>,
  root: string,
): SetDocument[] {
  if (!isRecord(raw)) fail("no documents.");
  return Object.entries(raw).map(([key, value]) => {
    if (!isRecord(value)) fail(`${key}: not an object.`);
    const { path, language, subject, form } = value;
    if (typeof path !== "string") fail(`${key}: the path must be text.`);
    const absolute = join(root, path);
    if (!existsSync(absolute)) fail(`${key}: ${path} doesn't exist.`);
    if (language !== "en" && language !== "zh") fail(`${key}: language must be "en" or "zh".`);
    if (typeof subject !== "string" || !(subject in subjects)) fail(`${key}: unknown subject.`);
    if (form !== "prose" && form !== "deck" && form !== "spreadsheet") {
      fail(`${key}: form must be "prose", "deck" or "spreadsheet".`);
    }
    return { key, path: absolute, language, subject, form };
  });
}

function readCases(raw: unknown, name: string, documents: ReadonlyMap<string, SetDocument>): Cases {
  if (!isRecord(raw) || !Array.isArray(raw.pairs) || !Array.isArray(raw.decks)) {
    fail(`${name} needs "pairs" and "decks".`);
  }
  const pairs = raw.pairs.map((pair, index): [string, string] => {
    if (!Array.isArray(pair) || pair.length !== 2) fail(`${name}.pairs[${index}]: two keys.`);
    const [a, b] = pair.map((key) => documents.get(String(key)));
    if (!a || !b) fail(`${name}.pairs[${index}]: unknown Document.`);
    if (a.subject !== b.subject) fail(`${name}.pairs[${index}]: not the same subject.`);
    if (a.language !== "en" || b.language !== "zh") {
      fail(`${name}.pairs[${index}]: the English Document first, then the Chinese one.`);
    }
    return [a.key, b.key];
  });
  const decks = raw.decks.map((key, index) => {
    const document = documents.get(String(key));
    if (!document) fail(`${name}.decks[${index}]: unknown Document.`);
    if (document.form === "prose") fail(`${name}.decks[${index}]: not a deck or spreadsheet.`);
    const others = [...documents.values()].filter(
      (other) => other.subject === document.subject && other.form === "prose",
    );
    if (others.length === 0) fail(`${name}.decks[${index}]: no other Document on its subject.`);
    return document.key;
  });
  return { pairs, decks };
}

function known(documents: ReadonlyMap<string, SetDocument>, key: unknown, where: string): string {
  if (typeof key !== "string" || !documents.has(key)) fail(`${where}: unknown Document.`);
  return key;
}

export function loadGroupingSet(root: string): GroupingSet {
  const raw = JSON.parse(readFileSync(join(root, GROUPING_SET), "utf8")) as Record<string, unknown>;
  if (!isRecord(raw.subjects)) fail("no subjects.");
  const subjects = Object.fromEntries(
    Object.entries(raw.subjects).map(([key, label]) => [key, String(label)]),
  );
  const documents = readDocuments(raw.documents, subjects, root);
  const byKey = new Map(documents.map((document) => [document.key, document]));
  const choosing = readCases(raw.choosing, "choosing", byKey);
  const heldOut = readCases(raw.heldOut, "heldOut", byKey);
  if (heldOut.pairs.length < HELD_OUT_CASES || heldOut.decks.length < HELD_OUT_CASES) {
    fail(
      `the held-out set needs at least ${HELD_OUT_CASES} pairs and ${HELD_OUT_CASES} decks or spreadsheets.`,
    );
  }
  if (choosing.pairs.length === 0 || choosing.decks.length === 0) {
    fail("the choosing set needs pairs and decks or spreadsheets.");
  }
  const inChoosing = new Set([...choosing.pairs.flat(), ...choosing.decks]);
  for (const key of [...heldOut.pairs.flat(), ...heldOut.decks]) {
    if (inChoosing.has(key)) fail(`${key} is a case in both the choosing and the held-out set.`);
  }
  if (!Array.isArray(raw.sharedNames)) fail("no sharedNames.");
  const sharedNames = raw.sharedNames.map((row, index) => {
    if (!isRecord(row) || typeof row.label !== "string" || typeof row.pattern !== "string") {
      fail(`sharedNames[${index}]: a label and a pattern.`);
    }
    return { label: row.label, pattern: new RegExp(row.pattern, "u") };
  });
  if (!Array.isArray(raw.lateArrivals)) fail("no lateArrivals.");
  const lateArrivals = raw.lateArrivals.map((key, index) =>
    known(byKey, key, `lateArrivals[${index}]`),
  );
  const corrections = raw.corrections;
  if (!isRecord(corrections) || !isRecord(corrections.create)) fail("no corrections.");
  const { create } = corrections;
  if (!Array.isArray(create.documents) || typeof create.name !== "string") {
    fail("corrections.create needs a name and documents.");
  }
  if (!Array.isArray(corrections.lastResortMove) || corrections.lastResortMove.length !== 2) {
    fail("corrections.lastResortMove: two keys.");
  }
  return {
    source: GROUPING_SET,
    subjects,
    documents,
    choosing,
    heldOut,
    sharedNames,
    lateArrivals,
    corrections: {
      rename: known(byKey, corrections.rename, "corrections.rename"),
      moves: Math.max(0, Number(corrections.moves) || 0),
      lastResortMove: [
        known(byKey, corrections.lastResortMove[0], "corrections.lastResortMove"),
        known(byKey, corrections.lastResortMove[1], "corrections.lastResortMove"),
      ],
      create: {
        name: create.name,
        documents: create.documents.map((key, index) =>
          known(byKey, key, `corrections.create.documents[${index}]`),
        ),
      },
    },
  };
}
