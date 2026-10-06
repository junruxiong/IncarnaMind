/**
 * Decoding TXT and Markdown Documents. Pure, with no Node imports: processing
 * uses it, and so does the Document viewer in the renderer, so the text the
 * viewer shows (and searches for a quote) is exactly the text Passages came from.
 */

/**
 * Decodes a TXT or Markdown file: UTF-8 or UTF-16 with a byte-order mark,
 * otherwise UTF-8, falling back to GB18030 (common for Chinese text files).
 * Line endings become "\n". Returns null if the bytes are binary data, not text.
 */
export function decodeText(bytes: Uint8Array): string | null {
  const encoding =
    bytes[0] === 0xff && bytes[1] === 0xfe
      ? "utf-16le"
      : bytes[0] === 0xfe && bytes[1] === 0xff
        ? "utf-16be"
        : undefined;
  let text: string | undefined;
  if (encoding) {
    text = new TextDecoder(encoding).decode(bytes);
  } else {
    for (const candidate of ["utf-8", "gb18030"]) {
      try {
        text = new TextDecoder(candidate, { fatal: true }).decode(bytes);
        break;
      } catch {
        // Not this encoding; try the next one.
      }
    }
    text ??= new TextDecoder("utf-8").decode(bytes);
  }
  if (text.includes("\u0000")) return null;
  return text.replace(/\r\n?/g, "\n");
}
