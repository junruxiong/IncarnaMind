/**
 * Reading a reference export (#287): which of CSL JSON, BibTeX and RIS it is
 * (by what is in it, not by its name), and its references in the record's
 * own fields.
 */
import { isRecord } from "../../errors";
import { fromCsl } from "../csl";
import { readBibtex } from "./bibtex";
import { attachedFiles } from "./common";
import { readRis } from "./ris";
import type { ReadFile, ReadProblem, ReadReference } from "./types";

/** CSL JSON: a list of items (Zotero, Mendeley, Better BibTeX), or an object holding them. */
function readCsl(json: unknown): ReadFile {
  const items = Array.isArray(json)
    ? json
    : isRecord(json) && Array.isArray(json.items)
      ? json.items
      : [json];
  const references: ReadReference[] = [];
  const problems: ReadProblem[] = [];
  items.forEach((item, index) => {
    const label = isRecord(item) && typeof item.id === "string" ? item.id : `item ${index + 1}`;
    if (!isRecord(item)) {
      problems.push({ label, reason: "not-a-reference" });
      return;
    }
    const fields = fromCsl(item);
    if (!fields.title) {
      problems.push({ label, reason: "no-title" });
      return;
    }
    const file = typeof item.file === "string" ? attachedFiles(item.file) : [];
    references.push({ label, fields, files: file });
  });
  return { format: "csl-json", references, problems };
}

/** The file's references, or null if it is none of the three. */
export function readReferenceFile(text: string): ReadFile | null {
  const body = text.replace(/^﻿/, "");
  const head = body.trimStart();
  if (head.startsWith("[") || head.startsWith("{")) {
    try {
      return readCsl(JSON.parse(head));
    } catch {
      return null;
    }
  }
  if (/^\s*TY {1,2}-/m.test(body)) return readRis(body);
  if (/^[ \t]*@[A-Za-z]+\s*[{(]/m.test(body)) return readBibtex(body);
  return null;
}
