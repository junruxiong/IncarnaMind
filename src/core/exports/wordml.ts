/**
 * Pieces of WordprocessingML (the XML inside a .docx) that every Word writer
 * shares: text runs, escaping, and footnotes, which Citations become. A Mind
 * exported as a new .docx (./docx) is written with them, so a Citation is the
 * same footnote, and the same text the same run, whichever writer made it.
 *
 * Each piece is a string of XML in the `w:` namespace, which the part it goes
 * into declares.
 */

/** WordprocessingML's namespace (`w:`), and the relationships' (`r:`). */
export const W_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
export const R_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

export const XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n';

// ---------------------------------------------------------------------------
// Text

/** Characters XML can't hold at all, even escaped. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: these are the characters it removes.
const NOT_XML = /[\u0000-\u0008\u000B\u000C\u000E-\u001F￾￿]/g;

/** Text for XML, in an element or a quoted attribute: escaped, and without what XML can't hold. */
export function escapeXml(text: string): string {
  return text
    .replace(NOT_XML, "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export interface RunFormat {
  /** A character style, e.g. "Hyperlink". */
  style?: string;
  bold?: boolean;
  italic?: boolean;
  strike?: boolean;
  /** Word's yellow text highlight, as the editor's. */
  highlight?: boolean;
  underline?: boolean;
  /** Monospaced, for code inside a link (code elsewhere uses its character style). */
  monospace?: boolean;
}

/**
 * A run of text in a format: tabs and line breaks become Word's. Empty text
 * is no run at all.
 */
export function run(text: string, format: RunFormat = {}): string {
  if (!text) return "";
  // In the order the schema requires.
  const rPr = [
    format.style ? `<w:rStyle w:val="${format.style}"/>` : "",
    format.monospace ? '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>' : "",
    format.bold ? "<w:b/>" : "",
    format.italic ? "<w:i/>" : "",
    format.strike ? "<w:strike/>" : "",
    format.highlight ? '<w:highlight w:val="yellow"/>' : "",
    format.underline ? '<w:u w:val="single"/>' : "",
  ].join("");
  let content = "";
  for (const part of text.split(/(\r\n|\r|\n|\t)/)) {
    if (part === "\t") content += "<w:tab/>";
    else if (part === "\n" || part === "\r\n" || part === "\r") content += "<w:br/>";
    else if (part) content += `<w:t xml:space="preserve">${escapeXml(part)}</w:t>`;
  }
  return `<w:r>${rPr ? `<w:rPr>${rPr}</w:rPr>` : ""}${content}</w:r>`;
}

// ---------------------------------------------------------------------------
// Footnotes

/** Where a footnote is referred to in the text: its number, in the FootnoteReference style. */
export function footnoteReference(id: number): string {
  return `<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="${id}"/></w:r>`;
}

/**
 * A footnote, for the footnotes part: one paragraph in the FootnoteText
 * style, its number, then `runs`.
 */
export function footnote(id: number, runs: string): string {
  return `<w:footnote w:id="${id}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/></w:pPr><w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>${runs}</w:p></w:footnote>`;
}

/**
 * The footnotes part (word/footnotes.xml): the separator lines Word draws
 * above the footnotes, as footnotes -1 and 0 (which the settings name), then
 * the footnotes, numbered from 1.
 */
export function footnotesPart(footnotes: readonly string[]): string {
  const separator = (type: string, id: number, mark: string) =>
    `<w:footnote w:type="${type}" w:id="${id}"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><${mark}/></w:r></w:p></w:footnote>`;
  return (
    `${XML_DECLARATION}<w:footnotes xmlns:w="${W_NAMESPACE}" xmlns:r="${R_NAMESPACE}">` +
    separator("separator", -1, "w:separator") +
    separator("continuationSeparator", 0, "w:continuationSeparator") +
    footnotes.join("") +
    "</w:footnotes>"
  );
}
