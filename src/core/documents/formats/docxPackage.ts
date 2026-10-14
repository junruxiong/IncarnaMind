/**
 * A Word package (.docx) checked before it is imported as a Mind (#104):
 * refused, with the reason, when it is a legacy or encrypted file, holds
 * macros, is Strict Open XML, isn't a Word document or can't be read, or is
 * over the size limits for Office files (./zip). And who changed and
 * commented on it, read fast from the parts as written, for the import
 * screen to say whose document it looks like.
 */
import type { WordRefusalReason } from "../../api";
import { ExtractionError } from "./errors";
import { decodeEntities, elements, parseXml } from "./xml";
import {
  isCompoundFile,
  openPackage,
  PartTooLargeError,
  resolvePart,
  type ZipArchive,
} from "./zip";

/** Why a Word file isn't imported. */
export class WordRefusedError extends Error {
  override name = "WordRefusedError";
  constructor(
    readonly reason: WordRefusalReason,
    message: string,
  ) {
    super(message);
  }
}

export const NS_TRANSITIONAL = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
export const NS_STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main";
const REL_OFFICE_DOCUMENT = "/officeDocument";
const STRICT_RELATIONSHIPS = "http://purl.oclc.org/ooxml/officeDocument/relationships";

const WORD_CONTENT_TYPES = new Set([
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
]);
const MACRO_CONTENT_TYPES = new Set([
  "application/vnd.ms-word.document.macroEnabled.main+xml",
  "application/vnd.ms-word.template.macroEnabledTemplate.main+xml",
]);

/** An open Word package, checked to be one IncarnaMind imports. */
export interface WordPackage {
  zip: ZipArchive;
  /** The main document part, as the package's relationships name it. */
  part: string;
}

/** Turns what went wrong reading a package into the reason a User is given. */
export function refusalOf(error: unknown, bytes?: Uint8Array): WordRefusedError {
  if (error instanceof WordRefusedError) return error;
  if (error instanceof ExtractionError) {
    if (error.reason === "password-protected") {
      return new WordRefusedError("encrypted", error.message);
    }
    if (error.reason === "too-large" || error instanceof PartTooLargeError) {
      return new WordRefusedError("too-large", error.message);
    }
    if (bytes && isCompoundFile(bytes)) return new WordRefusedError("legacy", error.message);
    return new WordRefusedError("damaged", error.message);
  }
  const message = error instanceof Error ? error.message : String(error);
  return new WordRefusedError("damaged", message);
}

export interface Relationship {
  target: string;
  type: string;
  external: boolean;
}

/** A part's relationships by id: internal targets resolved to part names, external ones as given. */
export async function relationshipsOf(
  zip: ZipArchive,
  part: string,
): Promise<Map<string, Relationship>> {
  const slash = part.lastIndexOf("/");
  const name = `${part.slice(0, slash + 1)}_rels/${part.slice(slash + 1)}.rels`;
  const rels = new Map<string, Relationship>();
  const xml = zip.has(name) ? await zip.readText(name) : undefined;
  for (const rel of xml ? elements(parseXml(xml)) : []) {
    const { Id: id, Target: target, TargetMode: mode, Type: type = "" } = rel.attrs;
    if (!id || target === undefined) continue;
    const external = mode === "External";
    rels.set(id, { target: external ? target : resolvePart(part, target), type, external });
  }
  return rels;
}

/**
 * Opens a .docx to import it, or refuses it with the reason: a legacy (.doc)
 * or encrypted file, a macro-enabled one (.docm), Strict Open XML, a package
 * that isn't a Word document or can't be read, or one over the size limits.
 */
export async function openWordPackage(bytes: Uint8Array): Promise<WordPackage> {
  let zip: ZipArchive;
  try {
    zip = openPackage(bytes);
  } catch (error) {
    throw refusalOf(error, bytes);
  }
  try {
    const packageRels = await relationshipsOf(zip, "");
    const main = [...packageRels.values()].find((rel) => rel.type.endsWith(REL_OFFICE_DOCUMENT));
    if (main?.type.startsWith(STRICT_RELATIONSHIPS)) {
      throw new WordRefusedError("strict", "The package is Strict Open XML.");
    }
    const part = main && !main.external ? main.target : "word/document.xml";
    if (!zip.has(part)) {
      throw new WordRefusedError("damaged", "The package has no Word document part.");
    }
    const contentType = await contentTypeOf(zip, part);
    if (
      (contentType !== null && MACRO_CONTENT_TYPES.has(contentType)) ||
      zip.names().some((name) => /(^|\/)vbaProject\.bin$/i.test(name))
    ) {
      throw new WordRefusedError("macros", "The package holds macros.");
    }
    if (contentType !== null && !WORD_CONTENT_TYPES.has(contentType)) {
      throw new WordRefusedError("damaged", `The package's main part is a ${contentType}.`);
    }
    return { zip, part };
  } catch (error) {
    throw refusalOf(error, bytes);
  }
}

/** A part's content type, from [Content_Types].xml; null when the package doesn't say. */
async function contentTypeOf(zip: ZipArchive, part: string): Promise<string | null> {
  const xml = await zip.readText("[Content_Types].xml");
  if (!xml) return null;
  const types = elements(parseXml(xml));
  const override = types.find(
    (each) => each.name.endsWith("Override") && each.attrs.PartName === `/${part}`,
  );
  if (override?.attrs.ContentType) return override.attrs.ContentType;
  const extension = part.split(".").pop()?.toLowerCase();
  const fallback = types.find(
    (each) => each.name.endsWith("Default") && each.attrs.Extension?.toLowerCase() === extension,
  );
  const type = fallback?.attrs.ContentType ?? null;
  // "application/xml" says nothing about what the part is.
  return type === "application/xml" ? null : type;
}

/** Who changed and commented on a document. */
export interface WordReview {
  /** In the order they first appear. */
  authors: string[];
  /** Tracked changes of every kind: text, paragraph marks, formatting, tables, moves. */
  changes: number;
  comments: number;
}

/**
 * Every tracked change's element, with its author. `w:tblGridChange` is left
 * out: Word records it whenever a table is resized with tracking on, with no
 * author, and it changes no text.
 */
const CHANGE_TAG =
  /<w:(?:ins|del|moveFrom|moveTo|rPrChange|pPrChange|sectPrChange|tblPrChange|trPrChange|tcPrChange|numberingChange|cellIns|cellDel|cellMerge)\b([^>]*)>/g;
const COMMENT_TAG = /<w:comment\b([^>]*)>/g;
const AUTHOR = /\bw:author\s*=\s*"([^"]*)"/;
/** The parts other than the main one that may hold tracked changes. */
const REVIEWED_PART = /^word\/(?:header\d*|footer\d*|footnotes|endnotes)\.xml$/;

/**
 * The tracked changes and comments in a document, read from its parts' text
 * as written (fast, without parsing the whole document), so the import
 * screen can say whose they are before anything is imported. A DOCTYPE in
 * the main part is refused here, as the parser would.
 */
export async function reviewOf(pkg: WordPackage): Promise<WordReview> {
  const authors: string[] = [];
  const seen = new Set<string>();
  const addAuthor = (attributes: string) => {
    const author = decodeEntities(AUTHOR.exec(attributes)?.[1] ?? "").trim();
    if (author && !seen.has(author)) {
      seen.add(author);
      authors.push(author);
    }
  };
  let changes = 0;
  let comments = 0;
  try {
    const parts = [pkg.part, ...pkg.zip.names().filter((name) => REVIEWED_PART.test(name))];
    for (const part of parts) {
      const xml = (await pkg.zip.readText(part)) ?? "";
      if (part === pkg.part) {
        if (/<!(?:DOCTYPE|ENTITY)/i.test(xml)) {
          throw new WordRefusedError("damaged", "Broken XML: a DOCTYPE, which is refused.");
        }
        if (xml.slice(0, 4096).includes(NS_STRICT)) {
          throw new WordRefusedError("strict", "The document is Strict Open XML.");
        }
      }
      for (const match of xml.matchAll(CHANGE_TAG)) {
        changes++;
        addAuthor(match[1] ?? "");
      }
    }
    const rels = await relationshipsOf(pkg.zip, pkg.part);
    const commentsPart = [...rels.values()].find(
      (rel) => !rel.external && rel.type.endsWith("/comments"),
    )?.target;
    const xml = commentsPart ? ((await pkg.zip.readText(commentsPart)) ?? "") : "";
    for (const match of xml.matchAll(COMMENT_TAG)) {
      comments++;
      addAuthor(match[1] ?? "");
    }
  } catch (error) {
    throw refusalOf(error);
  }
  return { authors, changes, comments };
}
