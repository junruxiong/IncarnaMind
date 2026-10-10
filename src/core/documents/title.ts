/**
 * A Document's title (#213), for when its file name is machine-made: the
 * title in its properties, else its first heading, else its first line.
 * Pure: the processing worker reads the property and passes the first Unit.
 */
import { looksMachineMade } from "../../shared/documentNames";

/** The longest title kept, in characters. */
const MAX_TITLE = 100;

/** How far into the first Unit a heading is looked for, in lines. */
const HEADING_LINES = 30;

const HEADING = /^#{1,6}\s+(.+?)\s*#*\s*$/;
const DECORATION = /^[\s>*_#=\-|•·]+|[\s*_|=]+$/g;

function tidy(text: string): string | null {
  const flat = text
    .replace(DECORATION, "")
    .replace(/[`*_]{1,3}/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (flat.length < 2 || looksMachineMade(flat)) return null;
  if (flat.length <= MAX_TITLE) return flat;
  const cut = flat.slice(0, MAX_TITLE);
  const space = cut.lastIndexOf(" ");
  return `${cut.slice(0, space > 40 ? space : MAX_TITLE)}…`;
}

/**
 * The title: `property` (a PDF's Title, an Office file's dc:title) if it
 * says something, else the first Markdown heading in `firstUnit`, else its
 * first line of words. Null when there is nothing to take.
 */
export function documentTitle(property: string | null, firstUnit: string | null): string | null {
  const own = property === null ? null : tidy(property);
  if (own !== null) return own;
  if (firstUnit === null) return null;
  const lines = firstUnit.split(/\r?\n/).slice(0, HEADING_LINES);
  for (const line of lines) {
    const heading = HEADING.exec(line.trim());
    const title = heading?.[1] ? tidy(heading[1]) : null;
    if (title !== null) return title;
  }
  for (const line of lines) {
    const title = tidy(line);
    if (title !== null) return title;
  }
  return null;
}
