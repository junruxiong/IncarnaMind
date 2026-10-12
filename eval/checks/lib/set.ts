/**
 * The checks evaluation's labelled sets (eval/checks/labels.json, #86): Find
 * support's sentences and verify's claims, each labelled supported,
 * unsupported or contradicted against one Folder of Documents, with the
 * Location and quote of the evidence. A model drafted the labels; a person
 * confirms each one (`confirmed`, and `review` says who and when).
 */
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { DocumentKind } from "../../../src/core/api";

export const LABELS_FILE = "eval/checks/labels.json";

export type CheckLabel = "supported" | "unsupported" | "contradicted";
export type CheckLanguage = "en" | "zh";
export type CheckSetName = "find-support" | "verify";

export const CHECK_LABELS: readonly CheckLabel[] = ["supported", "unsupported", "contradicted"];
export const CHECK_LANGUAGES: readonly CheckLanguage[] = ["en", "zh"];

/** A quote of a Document at its Location (ADR-0011), as the Citation check finds it. */
export interface LabelledQuote {
  /** The name of a Document in `ChecksSet.folder` or `ChecksSet.received`. */
  document: string;
  /** The Location's label in English, as the app words it ("p. 3", "Staffing, row 22"). */
  location: string;
  /** The first and last Unit the quote is on (src/shared/units.ts), from 1. */
  units: [number, number];
  quote: string;
}

export interface SetDocument {
  /** A file's name with its extension; a web page's title. */
  name: string;
  /** From the repository's root: the file, or a web page's HTML. */
  path: string;
  format: DocumentKind;
  language: CheckLanguage;
  /** "synthetic", or where it was taken from. */
  source: string;
  licence: string;
  /** Of the file at `path`: the labels hold only for these bytes. */
  sha256: string;
  /** A web page's address, which it is saved under (nothing is fetched). */
  address?: string;
}

/** A document someone else sent, to verify claim by claim. */
export interface ReceivedDocument extends SetDocument {
  id: string;
  /** What kind of document it is, e.g. "A contractor's progress report". */
  kind: string;
}

interface LabelledItem {
  id: string;
  language: CheckLanguage;
  label: CheckLabel;
  /** The Folder's Passage that backs it (supported) or says otherwise (contradicted); null when unsupported. */
  evidence: LabelledQuote | null;
  /** Verify only, when contradicted: the conflicting words in the document being verified. */
  contradiction: LabelledQuote | null;
  /** Why the label, for the person confirming it. */
  note: string | null;
  confirmed: boolean;
}

export interface FindSupportItem extends LabelledItem {
  set: "find-support";
  /** The Folder Document the sentence is about, or a near miss of. */
  document: string;
  /** A sentence as the User might write it. */
  sentence: string;
}

export interface VerifyItem extends LabelledItem {
  set: "verify";
  /** The received document the claim is in. */
  document: string;
  /** Where the claim is in it. */
  location: string;
  units: [number, number];
  /** The claim as the document words it: a quote of it. */
  claim: string;
}

export type CheckItem = FindSupportItem | VerifyItem;

export interface ChecksSet {
  version: 1;
  issue: number;
  /** Who confirmed the labels, and when (ISO 8601); null until a person has. */
  review: { by: string | null; at: string | null };
  folder: SetDocument[];
  received: ReceivedDocument[];
  items: CheckItem[];
}

const isUnits = (value: unknown): value is [number, number] =>
  Array.isArray(value) &&
  value.length === 2 &&
  value.every((each) => Number.isInteger(each) && each >= 1) &&
  value[0] <= value[1];

function checkQuote(
  quote: LabelledQuote | null,
  names: ReadonlySet<string>,
  where: string,
  what: string,
): void {
  if (quote === null) return;
  if (!names.has(quote.document)) throw new Error(`${where}: ${what} names an unknown Document.`);
  if (!quote.quote.trim() || !quote.location.trim() || !isUnits(quote.units)) {
    throw new Error(`${where}: ${what} needs a quote, a Location and its Units.`);
  }
}

function checkItem(
  item: CheckItem,
  folder: ReadonlySet<string>,
  received: ReadonlySet<string>,
): void {
  const where = `labels.json, ${item.id}`;
  if (!CHECK_LABELS.includes(item.label)) throw new Error(`${where}: unknown label.`);
  if (!CHECK_LANGUAGES.includes(item.language)) throw new Error(`${where}: unknown language.`);
  if (typeof item.confirmed !== "boolean")
    throw new Error(`${where}: "confirmed" must be true or false.`);
  if ((item.label === "unsupported") !== (item.evidence === null)) {
    throw new Error(`${where}: evidence is given for supported and contradicted items only.`);
  }
  checkQuote(item.evidence, folder, where, "the evidence");
  if (item.set === "find-support") {
    if (!item.sentence.trim()) throw new Error(`${where}: the sentence is empty.`);
    if (!folder.has(item.document)) throw new Error(`${where}: unknown Folder Document.`);
    if (item.contradiction !== null) throw new Error(`${where}: only verify has a contradiction.`);
    return;
  }
  if (item.set !== "verify") throw new Error(`${where}: unknown set.`);
  if (!item.claim.trim()) throw new Error(`${where}: the claim is empty.`);
  if (!received.has(item.document) || !isUnits(item.units)) {
    throw new Error(`${where}: the claim needs its received document and Units.`);
  }
  if ((item.label === "contradicted") !== (item.contradiction !== null)) {
    throw new Error(`${where}: a contradicted claim, and only one, has the conflicting quote.`);
  }
  if (item.contradiction && item.contradiction.document !== item.document) {
    throw new Error(`${where}: the conflicting quote is in the document being verified.`);
  }
  checkQuote(item.contradiction, received, where, "the contradiction");
}

/** Reads the sets and checks them: known names and labels, unique ids, and each file's bytes as labelled. */
export async function readChecksSet(root: string): Promise<ChecksSet> {
  const set = JSON.parse(await readFile(join(root, LABELS_FILE), "utf8")) as ChecksSet;
  if (set.version !== 1) throw new Error(`labels.json: unknown version ${set.version}.`);
  for (const document of [...set.folder, ...set.received]) {
    const sha256 = createHash("sha256")
      .update(await readFile(join(root, document.path)))
      .digest("hex");
    if (sha256 !== document.sha256) {
      throw new Error(
        `labels.json: ${document.path} changed since it was labelled. Check its labels again before updating its sha256.`,
      );
    }
  }
  const folder = new Set(set.folder.map((document) => document.name));
  const received = new Set(set.received.map((document) => document.name));
  const ids = new Set<string>();
  for (const item of set.items) {
    if (ids.has(item.id)) throw new Error(`labels.json: ${item.id} is used twice.`);
    ids.add(item.id);
    checkItem(item, folder, received);
  }
  return set;
}
