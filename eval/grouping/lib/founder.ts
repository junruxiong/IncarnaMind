/**
 * The founder's sample ("Check before building the grouping"): the User's own
 * library is grouped with each vector variant, and a random 30 of its
 * Documents go into a sheet for the User to judge, each with its Topic shown
 * as the Documents nearest the Topic's centre. The bar is at least 70% placed
 * right. Only the User can run it: the check reads their folder, never
 * changes it, and works in a temporary data folder.
 */
import { readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { kindOf } from "../../../src/core/documents/files";
import { dot, seededRandom } from "../../../src/core/topics/grouping";
import type { EvalDocument } from "../../lib/evaluationSet";

export const SAMPLE_SIZE = 30;
/** The founder judges at least this share of the sample placed right. */
export const FOUNDER_BAR = 0.7;
/** How many Documents describe a Topic in the sheet, as for naming (design step 3). */
export const TOPIC_PREVIEW = 8;

/** The items in a seeded random order (Fisher–Yates). */
export function shuffled<T>(items: readonly T[], seed: number): T[] {
  const random = seededRandom(seed);
  const result = [...items];
  for (let index = result.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [result[index], result[other]] = [result[other] as T, result[index] as T];
  }
  return result;
}

/**
 * Every file IncarnaMind can index in the folder, at any depth, leaving out
 * hidden files and folders; keyed by the path inside the folder. With a
 * limit, that many, drawn at random.
 */
export function libraryFiles(folder: string, limit: number | null, seed: number): EvalDocument[] {
  const found: EvalDocument[] = [];
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith(".")) continue;
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.isFile() && kindOf(path)) found.push({ key: relative(folder, path), path });
    }
  };
  walk(folder);
  found.sort((a, b) => a.key.localeCompare(b.key));
  return limit === null || found.length <= limit ? found : shuffled(found, seed).slice(0, limit);
}

export interface SheetDocument {
  key: string;
  name: string;
}

/**
 * A Topic as the sheet shows it: its size and the Documents nearest its
 * centroid, leaving out the one being judged.
 */
export function topicPreview(
  members: readonly (SheetDocument & { vector: Float32Array })[],
  centroid: Float32Array,
  judged: string,
): string {
  const others = members
    .filter((member) => member.key !== judged)
    .map((member) => ({ name: member.name, similarity: dot(member.vector, centroid) }))
    .sort((a, b) => b.similarity - a.similarity);
  if (others.length === 0) return "A Topic of its own";
  const shown = others.slice(0, TOPIC_PREVIEW).map((member) => member.name);
  const more = others.length > TOPIC_PREVIEW ? `; and ${others.length - TOPIC_PREVIEW} more` : "";
  return `With ${others.length} other Document${others.length === 1 ? "" : "s"}: ${shown.join("; ")}${more}`;
}

export interface SheetRow {
  document: string;
  /** Its path inside the folder. */
  file: string;
  /** A letter per grouping, so the judging is blind to the variant (report.json has the key). */
  grouping: string;
  topic: string;
}

const csvCell = (value: string) =>
  /[",\n\r]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;

/** The sheet the User fills in: one row per Document and grouping, with a column for "y" or "n". */
export function founderSheet(rows: readonly SheetRow[]): string {
  const lines = [
    ["document", "file", "grouping", "its Topic", "right Topic? (y/n)", "note"],
    ...rows.map((row) => [row.document, row.file, row.grouping, row.topic, "", ""]),
  ];
  // A byte-order mark, so spreadsheet apps read Chinese names as UTF-8.
  return `﻿${lines.map((line) => line.map(csvCell).join(",")).join("\r\n")}\r\n`;
}
