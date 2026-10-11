/**
 * RIS files, as EndNote, Zotero, Mendeley and most publishers' "Export
 * citation" write them (#287): lines of `TAG  - value`, each reference
 * starting at `TY` and ending at `ER`.
 */
import { ARXIV_ID, arxivUrl } from "../../../shared/references";
import type { ReferenceFields, ReferenceName, ReferenceType } from "../../api";
import { attachedFiles, dateOf, doiOf, languageOf, nameOf, pagesOf, withValues } from "./common";
import type { ReadFile, ReadProblem, ReadReference } from "./types";

const TYPES: Readonly<Record<string, ReferenceType>> = {
  JOUR: "article-journal",
  JFULL: "article-journal",
  EJOUR: "article-journal",
  MGZN: "article-journal",
  CONF: "paper-conference",
  CPAPER: "paper-conference",
  CHAP: "chapter",
  BOOK: "book",
  EBOOK: "book",
  EDBOOK: "book",
  THES: "thesis",
  RPRT: "report",
  ELEC: "webpage",
  WEB: "webpage",
  BLOG: "webpage",
  UNPB: "article",
};

type Tags = Map<string, string[]>;

/** The tags of each reference, in the order the file gives them. */
function split(text: string): { blocks: { tags: Tags; at: number }[]; problems: ReadProblem[] } {
  const blocks: { tags: Tags; at: number }[] = [];
  const problems: ReadProblem[] = [];
  let tags: Tags | null = null;
  let last: string[] | null = null;
  let start = 0;
  const lines = text.split(/\r\n|\r|\n/);
  const close = () => {
    if (tags && tags.size > 0) blocks.push({ tags, at: start });
    tags = null;
    last = null;
  };
  lines.forEach((raw, index) => {
    const line = raw.replace(/^﻿/, "");
    const match = /^([A-Z][A-Z0-9])\s{1,2}-(?:\s(.*)|\s*)$/.exec(line);
    if (!match) {
      // A line with no tag continues the value before it.
      if (last && line.trim() !== "") last[last.length - 1] += ` ${line.trim()}`;
      return;
    }
    const tag = match[1] as string;
    const value = (match[2] ?? "").trim();
    if (tag === "TY") {
      if (tags) {
        problems.push({ label: `line ${start + 1}`, reason: "broken" });
        close();
      }
      tags = new Map();
      start = index;
    } else if (tag === "ER") {
      close();
      return;
    } else if (!tags) {
      return; // before the first TY: a header or stray line
    }
    const values = tags.get(tag) ?? [];
    values.push(value);
    tags.set(tag, values);
    last = values;
  });
  if (tags) close(); // a last reference with no ER
  return { blocks, problems };
}

function toReference(tags: Tags, at: number): ReadReference {
  const all = (...names: string[]) => names.flatMap((name) => tags.get(name) ?? []);
  const first = (...names: string[]) => all(...names).find((value) => value !== "") ?? "";
  const type = TYPES[first("TY")] ?? "document";
  const authors: ReferenceName[] = all("AU", "A1")
    .map(nameOf)
    .filter((name): name is ReferenceName => name !== null);
  const pages = first("SP");
  const end = first("EP");
  const doi = doiOf(first("DO", "DI"));
  const urls = all("UR", "L3");
  const arxiv = /arxiv/i.test(`${first("JO", "JF", "T2", "J2")} ${urls.join(" ")}`);
  const arxivId = arxiv ? (ARXIV_ID.exec(urls.join(" "))?.[0] ?? "") : "";
  const fields: Partial<ReferenceFields> = {
    type: arxivId && type === "article-journal" ? "article" : type,
    title: first("TI", "T1", "CT"),
    authors,
    issued: dateOf(first("DA")) || dateOf(first("PY", "Y1")),
    containerTitle: type === "book" ? "" : first("JO", "JF", "T2", "BT", "J2"),
    volume: first("VL"),
    issue: first("IS", "CP"),
    pages:
      pages === "" ? "" : pagesOf(end === "" || pages.includes("-") ? pages : `${pages}-${end}`),
    publisher: first("PB"),
    doi,
    url:
      urls.find((url) => !/^https?:\/\/(?:dx\.)?doi\.org\//i.test(url)) ??
      (arxivId ? arxivUrl(arxivId) : ""),
    language: languageOf(first("LA")),
  };
  return {
    label: first("ID") || `line ${at + 1}`,
    fields: withValues(fields),
    files: all("L1", "L2").flatMap((value) => attachedFiles(value.replace(/;/g, "\\;"))),
  };
}

export function readRis(text: string): ReadFile {
  const { blocks, problems } = split(text);
  const references: ReadReference[] = [];
  for (const { tags, at } of blocks) {
    const reference = toReference(tags, at);
    if (!reference.fields.title) problems.push({ label: reference.label, reason: "no-title" });
    else references.push(reference);
  }
  return { format: "ris", references, problems };
}
