/**
 * A Tool call's input while the model is still writing it (#202): the JSON
 * text so far, read as far as it goes. Only the top-level string fields are
 * read, each with what it holds so far and whether it is complete; other
 * values are passed over. A model writes its fields in order, so a field
 * missing while a later one is being written was left out.
 */

/** A string field of a partial JSON object: its text so far, and whether its closing quote came. */
export interface PartialField {
  value: string;
  done: boolean;
}

const ESCAPES: Readonly<Record<string, string>> = {
  '"': '"',
  "\\": "\\",
  "/": "/",
  b: "\b",
  f: "\f",
  n: "\n",
  r: "\r",
  t: "\t",
};

/**
 * Reads a JSON string from `at` (just after its opening quote): its text, and
 * where it ended (after its closing quote), or null for the end when it isn't
 * closed yet. An escape cut off at the end is left out.
 */
function readString(text: string, at: number): { value: string; end: number | null } {
  let value = "";
  let i = at;
  while (i < text.length) {
    const char = text[i] as string;
    if (char === '"') return { value, end: i + 1 };
    if (char !== "\\") {
      value += char;
      i += 1;
      continue;
    }
    const next = text[i + 1];
    if (next === undefined) break;
    if (next === "u") {
      const hex = text.slice(i + 2, i + 6);
      if (hex.length < 4) break;
      value += String.fromCharCode(Number.parseInt(hex, 16));
      i += 6;
      continue;
    }
    value += ESCAPES[next] ?? next;
    i += 2;
  }
  return { value, end: null };
}

/** Where a value that isn't a string ends: at the comma or brace that closes it; null when it doesn't yet. */
function skipValue(text: string, at: number): number | null {
  let depth = 0;
  let i = at;
  while (i < text.length) {
    const char = text[i] as string;
    if (char === '"') {
      const read = readString(text, i + 1);
      if (read.end === null) return null;
      i = read.end;
      continue;
    }
    if (char === "{" || char === "[") depth += 1;
    else if (char === "}" || char === "]") {
      if (depth === 0) return i;
      depth -= 1;
    } else if (char === "," && depth === 0) return i;
    i += 1;
  }
  return null;
}

const skipSpace = (text: string, at: number) => {
  let i = at;
  while (i < text.length && /\s/.test(text[i] as string)) i += 1;
  return i;
};

/** The top-level string fields of a JSON object as far as `text` goes. */
export function partialFields(text: string): Map<string, PartialField> {
  const fields = new Map<string, PartialField>();
  let i = skipSpace(text, 0);
  if (text[i] !== "{") return fields;
  i += 1;
  while (i < text.length) {
    i = skipSpace(text, i);
    if (text[i] === "}" || text[i] !== '"') return fields;
    const key = readString(text, i + 1);
    if (key.end === null) return fields;
    i = skipSpace(text, key.end);
    if (text[i] !== ":") return fields;
    i = skipSpace(text, i + 1);
    if (i >= text.length) return fields;
    if (text[i] === '"') {
      const read = readString(text, i + 1);
      fields.set(key.value, { value: read.value, done: read.end !== null });
      if (read.end === null) return fields;
      i = read.end;
    } else {
      const end = skipValue(text, i);
      if (end === null) return fields;
      i = end;
    }
    i = skipSpace(text, i);
    if (text[i] === ",") i += 1;
    else return fields;
  }
  return fields;
}
