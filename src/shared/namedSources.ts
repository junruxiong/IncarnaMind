/**
 * Sources a model writes as a name in brackets, the way it reads a Citation in
 * a Mind ("[Tides, p. 3]", "[Deck, slide 2]", "[Model.xlsx, B12]",
 * "[Policy, §4]"), and the Citations they name: the same Document, at a
 * Location that overlaps (#961).
 */
import type { CitationLocation } from "../core/api";
import { citationLocation, parseRequestedLocation, type RequestedLocation } from "./locations";

/** A source written as its Document's name, with the Location after a comma when there is one. */
export interface NamedSource {
  name: string;
  location: RequestedLocation | null;
}

const BRACKETED = /\[([^[\]]+)\](?!\()/g;
const CHINESE_PAGES = /^第\s*(\d+)(?:\s*[-–—]\s*(\d+))?\s*页$/u;

function sourceOf(inner: string): NamedSource | null {
  const text = inner.trim();
  if (/^\^?=?\d+$/.test(text)) return null;
  // A name may hold a comma itself: the first split whose rest reads as a Location.
  for (const comma of text.matchAll(/[,，]/g)) {
    const name = text.slice(0, comma.index).trim();
    const written = text.slice(comma.index + 1).trim();
    const pages = CHINESE_PAGES.exec(written);
    const location = parseRequestedLocation(
      pages ? `p. ${pages[1]}${pages[2] ? `-${pages[2]}` : ""}` : written,
    );
    if (name && location) return { name, location };
  }
  return { name: text, location: null };
}

/**
 * Each source in `text` written as a name in brackets, in place of what
 * `link` gives for it (a Citation's marker), or left as text when it gives
 * null. A bracket with no Location counts only when `link` knows its name:
 * "[sic]" is no source. `unlinked`: the sources left as text.
 */
export function linkSources(
  text: string,
  link: (source: NamedSource) => string | null,
): { text: string; unlinked: number } {
  let unlinked = 0;
  const linked = text.replace(BRACKETED, (whole, inner: string) => {
    const source = sourceOf(inner);
    if (!source) return whole;
    const marker = link(source);
    if (marker !== null) return marker;
    if (source.location !== null) unlinked++;
    return whole;
  });
  return { text: linked, unlinked };
}

const simplified = (text: string) =>
  text.toLowerCase().replace(/[§›>]/g, " ").replace(/\s+/g, " ").trim();

const overlaps = (a: { from: number; to: number }, b: { from: number; to: number }) =>
  a.from <= b.to && b.from <= a.to;

/** Whether a Citation's Location is where a source says; one with none (a whole Markdown file) is anywhere in it. */
function fits(written: RequestedLocation, at: CitationLocation | null): boolean {
  if (at === null) return true;
  switch (written.kind) {
    case "page":
    case "slide":
      // A deck's "page 4" is its slide 4. A Document with no pages has none to tell its Citations apart.
      if (at.kind !== "page" && at.kind !== "slide") return at.kind !== "rows";
      return overlaps(written, at);
    case "lines":
    case "time":
      return at.kind === written.kind && overlaps(written, at);
    case "rows":
      return (
        at.kind === "rows" &&
        overlaps(written, at) &&
        (!written.sheet || !at.sheet || simplified(written.sheet) === simplified(at.sheet))
      );
    case "section": {
      if (at.kind !== "section" || !at.heading) return false;
      const own = simplified(at.heading);
      const wanted = simplified(written.heading);
      // "§4" names "4. Termination" and "§ 4 Termination" by its number.
      return own === wanted || own.startsWith(`${wanted} `) || own.startsWith(`${wanted}.`);
    }
  }
}

const nameKey = (name: string) =>
  name
    .replace(/\.[a-z0-9]{1,5}$/i, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLocaleLowerCase();

/** Whether a Citation is of the source a bracketed name gives: its Document, and where in it, if given. */
export function citesSource(
  citation: { documentName?: unknown; pageFrom?: unknown; pageTo?: unknown; location?: unknown },
  source: NamedSource,
): boolean {
  if (typeof citation.documentName !== "string") return false;
  if (nameKey(citation.documentName) !== nameKey(source.name)) return false;
  if (source.location === null) return true;
  const at = citationLocation({
    location: citation.location,
    pageFrom: typeof citation.pageFrom === "number" ? citation.pageFrom : null,
    pageTo: typeof citation.pageTo === "number" ? citation.pageTo : null,
  });
  return fits(source.location, at);
}
