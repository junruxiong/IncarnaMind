/**
 * Which Document names are machine-made (#213): a UUID, a long run of hex
 * digits or of numbers, a camera or scanner's counter, "Untitled" or
 * "document". The app then shows the Document's own title instead of its
 * name. Pure: the core and the renderer both use it.
 */

/** A UUID, with or without its dashes, as exported files carry it. */
const UUID = /^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$/i;

/** A long run of hex digits, perhaps cut by dashes, underscores or dots, with a digit in it. */
const LONG_CODE = /^(?=.*\d)[0-9a-f]{10,}(?:[-_.][0-9a-f]+)*$/i;

/** A long number, perhaps with separators. */
const NUMBERS = /^\d[\d\s._-]{7,}$/;

/** What software calls a file when it has no title: "Untitled", "document (2)", "IMG_0042", "Scan 12". */
const PLACEHOLDER =
  /^(?:untitled|document|doc|new document|new file|file|download|scan|scanned document|image|img|dsc|dscn|screenshot)(?:[\s._-]*\(?\d+\)?)?$/i;

/** One long word of letters and digits mixed, which nobody typed. */
const MIXED_TOKEN = /^(?=.*\d)(?=.*[a-z])[a-z0-9_-]{24,}$/i;

/** Digits make up a quarter of a token nobody typed: "37-38UpperGrosvenorStreet" has far fewer. */
const mixedToken = (text: string) =>
  MIXED_TOKEN.test(text) && (text.match(/\d/g)?.length ?? 0) / text.length >= 0.25;

/** Whether a Document's name looks machine-made, so that its own title serves the reader better. */
export function looksMachineMade(name: string): boolean {
  const text = name.trim();
  if (text === "") return true;
  return (
    UUID.test(text) ||
    LONG_CODE.test(text) ||
    NUMBERS.test(text) ||
    PLACEHOLDER.test(text) ||
    mixedToken(text)
  );
}

/** The part of a path after its last slash or backslash: the file's name, with its extension. */
export function fileNameOf(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/**
 * The name to show for a Document: its own name, unless that looks
 * machine-made and the Document has a title of its own, which then stands
 * in. A name the User set by renaming is not machine-made, so it stays.
 */
export function displayName(document: { name: string; title?: string | null }): string {
  const title = document.title?.trim();
  return title && looksMachineMade(document.name) ? title : document.name;
}
