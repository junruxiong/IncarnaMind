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

const BRACKETED = /([ \t]*)(?:\[([^[\]]+)\](?!\()|【([^【】]+)】)/gu;
const CHINESE_PAGES = /^第\s*(\d+)(?:\s*[-–—]\s*(\d+))?\s*页$/u;
// Written alone, "[B12]" or "[14:26]" may be anything; a page, slide or section reads as a source.
const ALONE: ReadonlySet<RequestedLocation["kind"]> = new Set(["page", "slide", "section"]);

function locationOf(written: string): RequestedLocation | null {
  const pages = CHINESE_PAGES.exec(written);
  return parseRequestedLocation(
    pages ? `p. ${pages[1]}${pages[2] ? `-${pages[2]}` : ""}` : written,
  );
}

function sourceOf(inner: string): NamedSource | null {
  const text = inner.trim();
  if (/^\^?=?\d+$/.test(text)) return null;
  // A name may hold a comma itself: the first split whose rest reads as a Location.
  for (const comma of text.matchAll(/[,，]/g)) {
    const name = text.slice(0, comma.index).trim();
    const location = locationOf(text.slice(comma.index + 1).trim());
    if (name && location) return { name, location };
  }
  const alone = locationOf(text);
  if (alone && ALONE.has(alone.kind)) return { name: "", location: alone };
  return { name: text, location: null };
}

/** The sources one bracket holds: "[Deck, slide 3; Policy, § 4]" holds two. Null: it is a marker. */
function sourcesOf(inner: string): NamedSource[] | null {
  const sources: NamedSource[] = [];
  for (const part of inner.split(/[;；]/)) {
    const source = sourceOf(part);
    if (!source) return null;
    sources.push(source);
  }
  return sources;
}

/**
 * Each source in `text` written as a name in brackets, in place of what
 * `link` gives for it (a Citation's marker); one it gives null for stays as
 * text, or with `unmatched` "drop" is left out. A bracket with no Location
 * counts only when `link` knows its name: "[sic]" is no source. `unlinked`:
 * the sources with a Location that `link` gave null for.
 */
export function linkSources(
  text: string,
  link: (source: NamedSource) => string | null,
  unmatched: "keep" | "drop" = "keep",
): { text: string; unlinked: number } {
  let unlinked = 0;
  const linked = text.replace(
    BRACKETED,
    (whole, space: string, square: string | undefined, full: string | undefined) => {
      const sources = sourcesOf(square ?? full ?? "");
      if (!sources) return whole;
      const markers = sources.map(link);
      const missed = sources.filter(
        (source, index) => markers[index] === null && source.location !== null,
      ).length;
      if (missed === 0 && markers.every((marker) => marker === null)) return whole;
      unlinked += missed;
      if (unmatched === "keep" && markers.some((marker) => marker === null)) return whole;
      const kept = markers.filter((marker) => marker !== null).join("");
      return kept ? `${space}${kept}` : "";
    },
  );
  return { text: linked, unlinked };
}

/**
 * The brackets in `text` that hold a source, as written: one with a Location
 * ("[Deck, slide 3]", "[§ 6.3.1 Forecast]"), or one whose name `known` knows.
 */
export function sourcesAsText(text: string, known: (source: NamedSource) => boolean): string[] {
  return [...text.matchAll(BRACKETED)].flatMap((match) => {
    const sources = sourcesOf(match[2] ?? match[3] ?? "");
    const named = sources?.some((source) => source.location !== null || known(source));
    return named ? [match[0].trim()] : [];
  });
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

/** Whether `name` is one of `names`, as a source written in brackets names a Document: its extension may be left out. */
export const namesOneOf = (name: string, names: readonly string[]): boolean =>
  names.some((each) => nameKey(each) === nameKey(name));

/** Whether a Citation is of the source a bracketed name gives: its Document, and where in it, if given. */
export function citesSource(
  citation: { documentName?: unknown; pageFrom?: unknown; pageTo?: unknown; location?: unknown },
  source: NamedSource,
): boolean {
  if (typeof citation.documentName !== "string") return false;
  if (source.name !== "" && nameKey(citation.documentName) !== nameKey(source.name)) return false;
  if (source.location === null) return true;
  const at = citationLocation({
    location: citation.location,
    pageFrom: typeof citation.pageFrom === "number" ? citation.pageFrom : null,
    pageTo: typeof citation.pageTo === "number" ? citation.pageTo : null,
  });
  return fits(source.location, at);
}
