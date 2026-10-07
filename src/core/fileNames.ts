/** Longest a file name made from a title gets, before its extension. */
const MAX_NAME_LENGTH = 120;

/** Names Windows keeps for devices, with or without an extension. */
const WINDOWS_DEVICE = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i;

/**
 * A file name (without its extension) from a title, e.g. a Mind's or a
 * Document's: without characters file systems refuse, and not too long.
 * `fallback` if nothing is left.
 */
export function safeFileName(title: string, fallback: string): string {
  const name = title
    // biome-ignore lint/suspicious/noControlCharactersInRegex: control characters can't be in file names.
    .replace(/[\\/:*?"<>|\u0000-\u001f\u007f]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\.+/, "")
    .slice(0, MAX_NAME_LENGTH)
    .replace(/[\s.]+$/, "");
  if (!name) return fallback;
  return WINDOWS_DEVICE.test(name) ? `${name}_` : name;
}
