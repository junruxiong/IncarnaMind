import { crc32, inflateRawSync } from "node:zlib";

/**
 * The files in a ZIP archive (a .docx), by path. Checks each entry's CRC-32
 * and size, so a broken archive fails here rather than in Word.
 */
export function unzip(data: Uint8Array): Map<string, Buffer> {
  const buffer = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  const end = buffer.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  if (end < 0) throw new Error("Not a ZIP archive: no end of central directory.");
  const count = buffer.readUInt16LE(end + 10);
  let offset = buffer.readUInt32LE(end + 16);
  const files = new Map<string, Buffer>();
  for (let index = 0; index < count; index++) {
    if (buffer.readUInt32LE(offset) !== 0x02014b50) throw new Error("Bad central directory.");
    const method = buffer.readUInt16LE(offset + 10);
    const checksum = buffer.readUInt32LE(offset + 16);
    const compressedSize = buffer.readUInt32LE(offset + 20);
    const size = buffer.readUInt32LE(offset + 24);
    const nameLength = buffer.readUInt16LE(offset + 28);
    const extraLength = buffer.readUInt16LE(offset + 30);
    const commentLength = buffer.readUInt16LE(offset + 32);
    const local = buffer.readUInt32LE(offset + 42);
    const name = buffer.toString("utf8", offset + 46, offset + 46 + nameLength);
    if (buffer.readUInt32LE(local) !== 0x04034b50) throw new Error(`Bad local header: ${name}`);
    const start = local + 30 + buffer.readUInt16LE(local + 26) + buffer.readUInt16LE(local + 28);
    const raw = buffer.subarray(start, start + compressedSize);
    const content = method === 8 ? inflateRawSync(raw) : raw;
    if (content.length !== size || crc32(content) !== checksum) {
      throw new Error(`Corrupt entry: ${name}`);
    }
    files.set(name, content);
    offset += 46 + nameLength + extraLength + commentLength;
  }
  return files;
}

/** A text part of a .docx, which must be there. */
export function part(files: Map<string, Buffer>, name: string): string {
  const content = files.get(name);
  if (!content) throw new Error(`The package has no ${name}.`);
  return content.toString("utf8");
}

/**
 * Throws unless `xml` is well-formed: one root element, tags properly nested,
 * attributes quoted and unique, and no stray "<", ">" or "&" in text.
 */
export function checkWellFormed(xml: string): void {
  const markup =
    /<\?[^?]*\?>|<(\/?)([A-Za-z_][\w:.-]*)((?:\s+[A-Za-z_][\w:.-]*="[^"<]*")*)\s*(\/?)>/g;
  const stack: string[] = [];
  let roots = 0;
  let last = 0;
  const checkText = (text: string) => {
    if (/[<>]/.test(text) || /&(?!(amp|lt|gt|quot|apos|#\d+|#x[\da-f]+);)/i.test(text)) {
      throw new Error(`Malformed XML near: ${text.slice(0, 80)}`);
    }
  };
  for (const match of xml.matchAll(markup)) {
    checkText(xml.slice(last, match.index));
    last = match.index + match[0].length;
    if (match[0].startsWith("<?")) continue;
    const [, closing, name, attributes = "", selfClosing] = match;
    const names = [...attributes.matchAll(/([\w:.-]+)="([^"]*)"/g)].map((attribute) => {
      checkText(attribute[2] as string);
      return attribute[1];
    });
    if (new Set(names).size !== names.length) throw new Error(`Repeated attribute on <${name}>.`);
    if (closing) {
      const open = stack.pop();
      if (open !== name) throw new Error(`</${name}> closes <${open}>.`);
    } else {
      if (stack.length === 0) roots++;
      if (!selfClosing) stack.push(name as string);
    }
  }
  checkText(xml.slice(last));
  if (stack.length > 0) throw new Error(`Unclosed <${stack.join(">, <")}>.`);
  if (roots !== 1) throw new Error(`Expected one root element, found ${roots}.`);
}

const decodeEntities = (text: string) =>
  text
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");

/**
 * The text of some WordprocessingML: footnote references as "[n]", tabs and
 * line breaks as "\t" and "\n".
 */
export function textOf(xml: string): string {
  let text = "";
  const pieces =
    /<w:t(?:\s[^>]*)?>([^<]*)<\/w:t>|<w:footnoteReference w:id="(\d+)"\/>|<w:tab\/>|<w:br\/>/g;
  for (const match of xml.matchAll(pieces)) {
    if (match[1] !== undefined) text += decodeEntities(match[1]);
    else if (match[2] !== undefined) text += `[${match[2]}]`;
    else text += match[0] === "<w:tab/>" ? "\t" : "\n";
  }
  return text;
}

export interface DocxParagraph {
  /** Its paragraph style, if it has one. */
  style?: string;
  /** Its list numbering: "numId/level". */
  list?: string;
  text: string;
}

/** The paragraphs of a document body or footnotes part, in order. */
export function paragraphsOf(xml: string): DocxParagraph[] {
  const paragraphs: DocxParagraph[] = [];
  for (const match of xml.matchAll(/<w:p>([\s\S]*?)<\/w:p>|<w:p\/>/g)) {
    const body = match[1] ?? "";
    const paragraph: DocxParagraph = { text: textOf(body) };
    const style = /<w:pStyle w:val="([^"]+)"\/>/.exec(body)?.[1];
    if (style) paragraph.style = style;
    const list = /<w:numPr><w:ilvl w:val="(\d+)"\/><w:numId w:val="(\d+)"\/><\/w:numPr>/.exec(body);
    if (list) paragraph.list = `${list[2]}/${list[1]}`;
    paragraphs.push(paragraph);
  }
  return paragraphs;
}

/** The text of each footnote in footnotes.xml, by id, leaving out Word's separators. */
export function footnotesOf(xml: string): Record<string, string> {
  const footnotes: Record<string, string> = {};
  for (const match of xml.matchAll(/<w:footnote w:id="(\d+)">([\s\S]*?)<\/w:footnote>/g)) {
    footnotes[match[1] as string] = textOf(match[2] as string).trim();
  }
  return footnotes;
}
