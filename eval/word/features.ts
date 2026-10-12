/** What a .docx holds that a round-trip has to keep, counted from its parts' text. */
export interface Features {
  paragraphs: number;
  trackedChanges: number;
  /** The authors of the tracked changes, sorted. */
  revisionAuthors: string[];
  footnotes: number;
  endnotes: number;
  fields: number;
  headersFooters: number;
  comments: number;
  tables: number;
  listParagraphs: number;
  contentControls: number;
  images: number;
}

/** The counts that say "this file has this feature", in the order the README lists them. */
export const FEATURE_COLUMNS = [
  "trackedChanges",
  "footnotes",
  "endnotes",
  "fields",
  "headersFooters",
  "comments",
  "tables",
  "listParagraphs",
  "contentControls",
  "images",
] as const satisfies readonly (keyof Features)[];

const count = (text: string, pattern: RegExp) => text.match(pattern)?.length ?? 0;

/** Footnotes and endnotes that carry text; the separators Word writes aren't notes. */
const notes = (text: string, tag: "footnote" | "endnote") =>
  [...text.matchAll(new RegExp(`<w:${tag}\\b[^>]*>`, "g"))].filter(
    (each) => !/w:type="(separator|continuationSeparator|continuationNotice)"/.test(each[0]),
  ).length;

/**
 * `parts` maps a part's name to its text for every XML part under word/; `names` lists every
 * entry of the package.
 */
export function countFeatures(
  parts: ReadonlyMap<string, string>,
  names: readonly string[],
): Features {
  const all = [...parts.values()].join("\n");
  const body = parts.get("word/document.xml") ?? "";
  const authors = new Set<string>();
  for (const each of all.matchAll(
    /<w:(?:ins|del|moveFrom|moveTo|rPrChange|pPrChange|tblPrChange|sectPrChange)\s[^>]*?w:author="([^"]*)"/g,
  )) {
    authors.add(each[1] as string);
  }
  return {
    paragraphs: count(body, /<w:p[ >]/g),
    trackedChanges: count(
      all,
      /<w:(?:ins|del|moveFrom|moveTo|rPrChange|pPrChange|tblPrChange|sectPrChange)\s/g,
    ),
    revisionAuthors: [...authors].sort(),
    footnotes: notes(parts.get("word/footnotes.xml") ?? "", "footnote"),
    endnotes: notes(parts.get("word/endnotes.xml") ?? "", "endnote"),
    fields: count(all, /<w:fldChar\s[^>]*w:fldCharType="begin"|<w:fldSimple\s/g),
    headersFooters: names.filter((name) => /^word\/(?:header|footer)[^/]*\.xml$/.test(name)).length,
    comments: count(parts.get("word/comments.xml") ?? "", /<w:comment\s/g),
    tables: count(body, /<w:tbl>/g),
    listParagraphs: count(body, /<w:numPr>/g),
    contentControls: count(all, /<w:sdt>/g),
    images: names.filter((name) => name.startsWith("word/media/")).length,
  };
}

/** How many kinds of feature the file has at least one of. */
export function featureKinds(features: Features): number {
  return FEATURE_COLUMNS.filter((column) => features[column] > 0).length;
}
