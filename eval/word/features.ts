/** What a .docx holds that a round-trip has to keep, counted from its parts' text. */
export interface Features {
  paragraphs: number;
  /** An estimate from the text (about 3,000 Latin or 700 East Asian characters a page), or the file's own count if higher. */
  estimatedPages: number;
  trackedChanges: number;
  /** The authors of the tracked changes, sorted. */
  revisionAuthors: string[];
  comments: number;
  commentReplies: number;
  footnotes: number;
  endnotes: number;
  fields: number;
  tocFields: number;
  crossReferences: number;
  captions: number;
  bookmarks: number;
  headersFooters: number;
  /** Sections beyond the first. */
  sections: number;
  columns: number;
  tables: number;
  nestedTables: number;
  mergedCells: number;
  listParagraphs: number;
  contentControls: number;
  formFields: number;
  images: number;
  wrappedImages: number;
  equations: number;
  smartArt: number;
  charts: number;
  textBoxes: number;
  shapes: number;
  embeddedObjects: number;
  rightToLeft: number;
  /** 1 when the file is in a compatibility mode older than Word 2013's (or names none). */
  olderCompatibility: number;
  /** 1 when the file is about 100 pages or more. */
  longDocument: number;
}

/** The counts that say "this file has this feature", with the names the README uses. */
export const FEATURES = [
  ["trackedChanges", "Tracked changes"],
  ["comments", "Comments"],
  ["commentReplies", "Comment replies"],
  ["footnotes", "Footnotes"],
  ["endnotes", "Endnotes"],
  ["fields", "Fields"],
  ["tocFields", "TOC"],
  ["crossReferences", "Cross-references"],
  ["captions", "SEQ captions"],
  ["bookmarks", "Bookmarks"],
  ["headersFooters", "Headers and footers"],
  ["sections", "Several sections"],
  ["columns", "Columns"],
  ["tables", "Tables"],
  ["nestedTables", "Nested tables"],
  ["mergedCells", "Merged cells"],
  ["listParagraphs", "Lists"],
  ["contentControls", "Content controls"],
  ["formFields", "Form fields"],
  ["images", "Images"],
  ["wrappedImages", "Images with wrap"],
  ["equations", "Equations"],
  ["smartArt", "SmartArt"],
  ["charts", "Charts"],
  ["textBoxes", "Text boxes"],
  ["shapes", "Shapes"],
  ["embeddedObjects", "Embedded objects"],
  ["rightToLeft", "Right-to-left"],
  ["olderCompatibility", "Older compatibility mode"],
  ["longDocument", "100+ pages"],
] as const satisfies readonly (readonly [keyof Features, string])[];

const count = (text: string, pattern: RegExp) => text.match(pattern)?.length ?? 0;
const REVISION = "(?:ins|del|moveFrom|moveTo|rPrChange|pPrChange|tblPrChange|sectPrChange)";

/** Footnotes and endnotes that carry text; the separators Word writes aren't notes. */
const notes = (text: string, tag: "footnote" | "endnote") =>
  [...text.matchAll(new RegExp(`<w:${tag}\\b[^>]*>`, "g"))].filter(
    (each) => !/w:type="(separator|continuationSeparator|continuationNotice)"/.test(each[0]),
  ).length;

/** Tables inside a table cell. */
function nestedTables(body: string): number {
  let depth = 0;
  let nested = 0;
  for (const each of body.matchAll(/<w:tbl>|<\/w:tbl>/g)) {
    if (each[0] === "</w:tbl>") depth--;
    else if (++depth > 1) nested++;
  }
  return nested;
}

function instructions(all: string): string {
  const complex = [...all.matchAll(/<w:instrText[^>]*>([^<]*)/g)].map((each) => each[1]);
  const simple = [...all.matchAll(/<w:fldSimple\s[^>]*w:instr="([^"]*)"/g)].map((each) => each[1]);
  return [...complex, ...simple].join(" ");
}

/**
 * `parts` maps a part's name to its text for every XML part under word/, and docProps/app.xml;
 * `names` lists every entry of the package.
 */
export function countFeatures(
  parts: ReadonlyMap<string, string>,
  names: readonly string[],
): Features {
  const all = [...parts.entries()]
    .filter(([name]) => name.startsWith("word/"))
    .map(([, text]) => text)
    .join("\n");
  const body = parts.get("word/document.xml") ?? "";
  const authors = new Set<string>();
  for (const each of all.matchAll(new RegExp(`<w:${REVISION}\\s[^>]*?w:author="([^"]*)"`, "g"))) {
    authors.add(each[1] as string);
  }
  const fieldInstructions = instructions(all);
  const compat = /compatibilityMode"[^>]*w:val="(\d+)"/.exec(parts.get("word/settings.xml") ?? "");
  const text = [...body.matchAll(/<w:t(?:\s[^>]*)?>([^<]*)/g)].map((each) => each[1]).join("");
  const eastAsian = count(text, /[぀-ヿ㐀-鿿가-힯]/g);
  const declared = Number(/<Pages>(\d+)/.exec(parts.get("docProps/app.xml") ?? "")?.[1] ?? 0);
  const estimatedPages = Math.max(
    declared,
    Math.round((text.length - eastAsian) / 3000 + eastAsian / 700),
  );
  return {
    paragraphs: count(body, /<w:p[ >]/g),
    estimatedPages,
    trackedChanges: count(all, new RegExp(`<w:${REVISION}\\s`, "g")),
    revisionAuthors: [...authors].sort(),
    comments: count(parts.get("word/comments.xml") ?? "", /<w:comment\s/g),
    commentReplies: count(parts.get("word/commentsExtended.xml") ?? "", /w15:paraIdParent=/g),
    footnotes: notes(parts.get("word/footnotes.xml") ?? "", "footnote"),
    endnotes: notes(parts.get("word/endnotes.xml") ?? "", "endnote"),
    fields: count(all, /<w:fldChar\s[^>]*w:fldCharType="begin"|<w:fldSimple\s/g),
    tocFields: count(fieldInstructions, /\bTOC\b/g),
    crossReferences: count(fieldInstructions, /\b(?:REF|PAGEREF|NOTEREF)\b/g),
    captions: count(fieldInstructions, /\bSEQ\b/g),
    bookmarks: count(body, /<w:bookmarkStart\s/g),
    headersFooters: names.filter((name) => /^word\/(?:header|footer)[^/]*\.xml$/.test(name)).length,
    sections: Math.max(0, count(body, /<w:sectPr[ >]/g) - 1),
    columns: count(body, /<w:cols\s[^>]*w:num="[2-9]/g),
    tables: count(body, /<w:tbl>/g),
    nestedTables: nestedTables(body),
    mergedCells: count(body, /<w:gridSpan\s|<w:vMerge/g),
    listParagraphs: count(body, /<w:numPr>/g),
    contentControls: count(all, /<w:sdt>/g),
    formFields: count(all, /<w:ffData>/g),
    images: names.filter((name) => name.startsWith("word/media/")).length,
    wrappedImages: count(body, /<wp:wrap(?:Square|Tight|Through|TopAndBottom)/g),
    equations: count(all, /<m:oMath[ >]/g),
    smartArt: names.filter((name) => /^word\/diagrams\/data/.test(name)).length,
    charts: names.filter((name) => /^word\/charts\/chart\d*\.xml$/.test(name)).length,
    textBoxes: count(body, /<w:txbxContent/g),
    shapes: count(body, /<wps:wsp>|<v:(?:shape|rect|roundrect|oval)\b/g),
    embeddedObjects: names.filter((name) => name.startsWith("word/embeddings/")).length,
    rightToLeft: count(all, /<w:(?:bidi|rtl)(?:\s*\/>|\s+w:val="(?!0"|false")[^"]*"\s*\/>)/g),
    olderCompatibility: compat && Number(compat[1]) >= 15 ? 0 : 1,
    longDocument: estimatedPages >= 100 ? 1 : 0,
  };
}

/** How many kinds of feature the file has at least one of. */
export function featureKinds(features: Features): number {
  return FEATURES.filter(([key]) => features[key] > 0).length;
}
