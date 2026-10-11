/**
 * BibTeX (and BibLaTeX) files, as Zotero, EndNote, Mendeley, JabRef and
 * Google Scholar write them (#287). A small reader of its own: entries
 * `@type{key, field = {value}, …}`, values in braces, quotes or bare, joined
 * with "#", `@string` macros, and the LaTeX in the text. An entry that is
 * broken is reported and skipped; the next one is read.
 */
import { ARXIV_ID, arxivUrl } from "../../../shared/references";
import type { ReferenceFields, ReferenceName, ReferenceType } from "../../api";
import { attachedFiles, dateOf, doiOf, languageOf, nameOf, pagesOf, withValues } from "./common";
import { unescapeLatex } from "./latex";
import type { ReadFile, ReadProblem, ReadReference } from "./types";

const TYPES: Readonly<Record<string, ReferenceType>> = {
  article: "article-journal",
  inproceedings: "paper-conference",
  conference: "paper-conference",
  incollection: "chapter",
  inbook: "chapter",
  book: "book",
  mvbook: "book",
  phdthesis: "thesis",
  mastersthesis: "thesis",
  thesis: "thesis",
  techreport: "report",
  report: "report",
  online: "webpage",
  electronic: "webpage",
  www: "webpage",
  unpublished: "article",
};

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

interface Entry {
  type: string;
  key: string;
  fields: Map<string, string>;
}

const ENTRY_START = /^[ \t]*@[A-Za-z]+\s*[{(]/m;

/** The value starting at `from`: braces, quotes or a bare word, joined by "#". */
function readValue(
  text: string,
  from: number,
  macros: ReadonlyMap<string, string>,
): { value: string; end: number } | null {
  let at = from;
  let value = "";
  for (;;) {
    while (/\s/.test(text[at] ?? "")) at++;
    const char = text[at];
    if (char === "{") {
      let depth = 0;
      let end = at;
      for (; end < text.length; end++) {
        const each = text[end];
        if (each === "\\") end++;
        else if (each === "{") depth++;
        else if (each === "}" && --depth === 0) break;
      }
      if (end >= text.length) return null;
      value += text.slice(at, end + 1);
      at = end + 1;
    } else if (char === '"') {
      let end = at + 1;
      let depth = 0;
      for (; end < text.length; end++) {
        const each = text[end];
        if (each === "\\") end++;
        else if (each === "{") depth++;
        else if (each === "}") depth--;
        else if (each === '"' && depth === 0) break;
      }
      if (end >= text.length) return null;
      value += `{${text.slice(at + 1, end)}}`;
      at = end + 1;
    } else {
      const word = /^[^\s,{}()"#=]+/.exec(text.slice(at))?.[0];
      if (!word) return null;
      value += /^\d+$/.test(word) ? word : (macros.get(word.toLowerCase()) ?? `{${word}}`);
      at += word.length;
    }
    while (/\s/.test(text[at] ?? "")) at++;
    if (text[at] !== "#") return { value, end: at };
    at++;
  }
}

const line = (text: string, at: number) => text.slice(0, at).split("\n").length;

/** The braces of a value that is one group, taken off: "{Deep Learning}" is "Deep Learning". */
function inner(value: string): string {
  let out = "";
  let at = 0;
  // Joined parts, each "{…}": their contents, one after another.
  while (at < value.length) {
    if (value[at] === "{") {
      let depth = 0;
      let end = at;
      for (; end < value.length; end++) {
        if (value[end] === "\\") end++;
        else if (value[end] === "{") depth++;
        else if (value[end] === "}" && --depth === 0) break;
      }
      out += value.slice(at + 1, end);
      at = end + 1;
    } else {
      out += value[at];
      at++;
    }
  }
  return out;
}

function parseEntries(text: string): { entries: Entry[]; problems: ReadProblem[] } {
  const entries: Entry[] = [];
  const problems: ReadProblem[] = [];
  const macros = new Map<string, string>(MONTHS.map((month, index) => [month, `{${index + 1}}`]));
  let at = 0;
  for (;;) {
    const start = text.indexOf("@", at);
    if (start < 0) break;
    const head = /^@([A-Za-z]+)\s*([{(])/.exec(text.slice(start));
    if (!head) {
      at = start + 1;
      continue;
    }
    const type = (head[1] as string).toLowerCase();
    const close = head[2] === "(" ? ")" : "}";
    let cursor = start + head[0].length;
    /** Reports this entry and goes on at the next "@type{" that starts a line. */
    const skip = (label: string) => {
      problems.push({ label, reason: "broken" });
      const next = ENTRY_START.exec(text.slice(start + 1));
      at = next ? start + 1 + next.index : text.length;
    };
    if (type === "comment" || type === "preamble") {
      const value = type === "comment" ? null : readValue(text, cursor, macros);
      const end = type === "comment" ? text.indexOf(close, cursor) : (value?.end ?? -1);
      at = end < 0 ? start + 1 : end + 1;
      continue;
    }
    if (type === "string") {
      const name = /^\s*([^\s=]+)\s*=/.exec(text.slice(cursor));
      if (name) {
        const value = readValue(text, cursor + name[0].length, macros);
        if (value) {
          macros.set((name[1] as string).toLowerCase(), value.value);
          at = value.end + 1;
          continue;
        }
      }
      at = start + 1;
      continue;
    }
    const keyEnd = text.slice(cursor).search(/[,}]/);
    if (keyEnd < 0) {
      skip(`line ${line(text, start)}`);
      continue;
    }
    const key = text.slice(cursor, cursor + keyEnd).trim();
    const label = key || `line ${line(text, start)}`;
    cursor += keyEnd;
    const fields = new Map<string, string>();
    let broken = false;
    for (;;) {
      while (/[\s,]/.test(text[cursor] ?? "")) cursor++;
      if (cursor >= text.length) {
        broken = true;
        break;
      }
      if (text[cursor] === close) {
        cursor++;
        break;
      }
      const name = /^([A-Za-z][\w:.-]*)\s*=/.exec(text.slice(cursor));
      if (!name) {
        broken = true;
        break;
      }
      const value = readValue(text, cursor + name[0].length, macros);
      if (!value) {
        broken = true;
        break;
      }
      const field = (name[1] as string).toLowerCase();
      if (!fields.has(field)) fields.set(field, value.value);
      cursor = value.end;
    }
    if (broken) {
      skip(label);
      continue;
    }
    entries.push({ type, key, fields });
    at = cursor;
  }
  return { entries, problems };
}

/** Names split at "and" outside braces: "A and B and {World Health Organization}". */
function splitAnd(value: string): string[] {
  const names: string[] = [];
  let depth = 0;
  let current = "";
  const words = value.split(/(\s+)/);
  for (const word of words) {
    for (const char of word) {
      if (char === "{") depth++;
      else if (char === "}") depth--;
    }
    if (depth === 0 && word.toLowerCase() === "and") {
      names.push(current);
      current = "";
    } else {
      current += word;
    }
  }
  names.push(current);
  return names;
}

function namesOf(value: string): ReferenceName[] {
  const found: ReferenceName[] = [];
  for (const raw of splitAnd(value)) {
    const text = raw.trim();
    // A name wholly in braces is one name, as written: an organisation, or a Chinese name.
    const whole = /^\{([^{}]*)\}$/.exec(text);
    if (whole) {
      const literal = unescapeLatex(whole[1] as string);
      if (literal !== "") found.push({ literal });
      continue;
    }
    const name = nameOf(unescapeLatex(text));
    if (name) found.push(name);
  }
  return found;
}

/** An address or DOI: only the backslashes before symbols and the braces around go. */
const plainUrl = (value: string) =>
  inner(value)
    .replace(/\\([_%&#$~^])/g, "$1")
    .replace(/\s+/g, "")
    .trim();

function toReference(entry: Entry): ReadReference {
  const raw = (...names: string[]) => {
    for (const name of names) {
      const value = entry.fields.get(name);
      if (value !== undefined) return value;
    }
    return "";
  };
  const text = (...names: string[]) => unescapeLatex(raw(...names));
  const eprint = text("eprint");
  const prefix = text("archiveprefix", "eprinttype").toLowerCase();
  const arxiv = (prefix === "arxiv" || /^arxiv/i.test(text("journal"))) && eprint !== "";
  const arxivId = arxiv ? (ARXIV_ID.exec(eprint)?.[0] ?? "") : "";
  let type: ReferenceType = TYPES[entry.type] ?? "document";
  const url = plainUrl(raw("url"));
  if (entry.type === "misc" && url !== "" && text("journal", "booktitle") === "") type = "webpage";
  // "arXiv preprint arXiv:1706.03762" is a preprint, not a journal article.
  if (arxivId !== "" && (type === "article-journal" || type === "document")) type = "article";
  if (entry.type === "thesis" && /^\s*\{?\s*master/i.test(text("type"))) type = "thesis";

  const year = text("year");
  const date = text("date");
  let issued = dateOf(date);
  if (issued === "" && year !== "") {
    const monthText = text("month").toLowerCase();
    const index = MONTHS.findIndex((month) => monthText.startsWith(month));
    const month = index >= 0 ? index + 1 : Number(monthText);
    issued = dateOf(
      Number.isInteger(month) && month >= 1 && month <= 12 ? `${year}-${month}` : year,
    );
  }
  const fields: Partial<ReferenceFields> = {
    type,
    title: text("title"),
    authors: namesOf(inner(raw("author"))),
    issued,
    containerTitle:
      type === "webpage"
        ? text("website", "journal", "journaltitle", "booktitle", "organization")
        : text("journal", "journaltitle", "booktitle"),
    volume: text("volume"),
    issue: text("number", "issue"),
    pages: pagesOf(text("pages")),
    publisher:
      type === "thesis" || type === "report"
        ? text("school", "institution", "publisher", "organization")
        : text("publisher", "organization"),
    doi: doiOf(plainUrl(raw("doi"))),
    url: arxivId && url === "" ? arxivUrl(arxivId) : url,
    language: languageOf(text("langid", "language")),
  };
  return {
    label: entry.key || "entry",
    fields: withValues(fields),
    files: attachedFiles(inner(raw("file"))),
  };
}

export function readBibtex(text: string): ReadFile {
  const { entries, problems } = parseEntries(text);
  const references: ReadReference[] = [];
  for (const entry of entries) {
    const reference = toReference(entry);
    if (!reference.fields.title) problems.push({ label: reference.label, reason: "no-title" });
    else references.push(reference);
  }
  return { format: "bibtex", references, problems };
}
