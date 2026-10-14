import { parseXml, walk, type XmlElement } from "../../../core/documents/formats/xml";
import { openPackage } from "../../../core/documents/formats/zip";

/**
 * What docx-preview leaves undrawn: the newer branch of `mc:AlternateContent`
 * (it draws the older one, a VML text box's paragraphs included) and a drawing's
 * text box, which it doesn't draw.
 */
const UNDRAWN = new Set(["mc:Choice", "w:drawing"]);

/**
 * Where a paragraph or table at a place in a Word file's main document part
 * ("w:p#12", counted from 0 in document order, as a Word import counts them,
 * #736) is among those the viewer draws, of the same kind: the import counts
 * every one, the viewer draws only one branch of each alternative. One the
 * viewer doesn't draw gives the last drawn before it, the one that holds it.
 * Null for a place the file hasn't.
 */
export async function drawnIndexOf(bytes: Uint8Array, place: string): Promise<number | null> {
  const match = /^(w:p|w:tbl)#(\d+)$/.exec(place);
  if (!match) return null;
  const [, name, ordinal] = match;
  const xml = await openPackage(bytes).readText("word/document.xml");
  if (xml === undefined) return null;
  const root = parseXml(xml);
  const undrawn = new Set<XmlElement>();
  for (const element of walk(root)) {
    if (UNDRAWN.has(element.name)) for (const inner of walk(element)) undrawn.add(inner);
  }
  let seen = -1;
  let drawn = -1;
  for (const element of walk(root)) {
    if (element.name !== name) continue;
    seen++;
    if (!undrawn.has(element)) drawn++;
    if (seen === Number(ordinal)) return Math.max(drawn, 0);
  }
  return null;
}
