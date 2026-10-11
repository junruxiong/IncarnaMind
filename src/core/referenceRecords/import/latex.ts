/**
 * BibTeX text as plain text: LaTeX accents and symbols ("{\"o}", "\'e",
 * "\ss", "\&", "--") become the characters they stand for, and the braces
 * BibTeX uses to protect capitals ("{Deep} Learning") are taken out.
 */

/** Accent commands, as the combining mark each puts on the letter after it. */
const ACCENTS: Readonly<Record<string, string>> = {
  "`": "̀",
  "'": "́",
  "^": "̂",
  "~": "̃",
  "=": "̄",
  ".": "̇",
  '"': "̈",
  u: "̆",
  r: "̊",
  H: "̋",
  v: "̌",
  c: "̧",
  k: "̨",
  d: "̣",
  b: "̱",
};

/** Commands that are a letter or a mark on their own. */
const SYMBOLS: Readonly<Record<string, string>> = {
  ss: "ß",
  ae: "æ",
  AE: "Æ",
  oe: "œ",
  OE: "Œ",
  o: "ø",
  O: "Ø",
  aa: "å",
  AA: "Å",
  l: "ł",
  L: "Ł",
  i: "i",
  j: "j",
  textendash: "–",
  textemdash: "—",
  textquoteright: "’",
  textquoteleft: "‘",
  textquotedblleft: "“",
  textquotedblright: "”",
  textellipsis: "…",
  ldots: "…",
  dots: "…",
  textasciitilde: "~",
  textbackslash: "\\",
};

/** The brace group starting at `from` (an opening brace): its content, and where it ends. */
function group(text: string, from: number): { content: string; end: number } | null {
  let depth = 0;
  for (let at = from; at < text.length; at++) {
    const char = text[at];
    if (char === "\\") at++;
    else if (char === "{") depth++;
    else if (char === "}" && --depth === 0) {
      return { content: text.slice(from + 1, at), end: at + 1 };
    }
  }
  return null;
}

/** The letter an accent command sits on, and where it ends: "{e}", "\i", "e". */
function accented(text: string, from: number): { letter: string; end: number } | null {
  const first = text[from];
  if (first === "{") {
    const inner = group(text, from);
    return inner ? { letter: unescapeLatex(inner.content), end: inner.end } : null;
  }
  if (first === "\\") {
    const command = /^\\([A-Za-z]+)/.exec(text.slice(from));
    const symbol = command?.[1] ? SYMBOLS[command[1]] : undefined;
    return command && symbol ? { letter: symbol, end: from + command[0].length } : null;
  }
  return first === undefined ? null : { letter: first, end: from + 1 };
}

export function unescapeLatex(text: string): string {
  let out = "";
  let at = 0;
  while (at < text.length) {
    const char = text[at] as string;
    if (char === "\\") {
      const next = text[at + 1];
      if (next === undefined) break;
      const mark = ACCENTS[next];
      const isLetter = /[A-Za-z]/.test(next);
      // "\'e" and "\"{o}", and the letter-named accents "\c c", "\v{s}" (not "\copyright").
      if (mark && (!isLetter || !/[A-Za-z]/.test(text[at + 2] ?? ""))) {
        let from = at + 2;
        if (isLetter) while (text[from] === " ") from++;
        const base = accented(text, from);
        if (base && base.letter !== "") {
          out += (base.letter.slice(0, 1) + mark + base.letter.slice(1)).normalize("NFC");
          at = base.end;
          continue;
        }
      }
      const command = /^\\([A-Za-z]+)\s*/.exec(text.slice(at));
      if (command?.[1]) {
        // A known symbol, or a command that only styles its argument (\emph{…}), dropped.
        out += SYMBOLS[command[1]] ?? "";
        at += command[0].length;
        continue;
      }
      // An escaped symbol: \& \% \$ \_ \# \{ \}.
      out += next;
      at += 2;
      continue;
    }
    if (char === "{" || char === "}" || char === "$") {
      at++;
    } else if (char === "~") {
      out += " ";
      at++;
    } else if (text.startsWith("---", at)) {
      out += "—";
      at += 3;
    } else if (text.startsWith("--", at)) {
      out += "–";
      at += 2;
    } else if (text.startsWith("``", at)) {
      out += "“";
      at += 2;
    } else if (text.startsWith("''", at)) {
      out += "”";
      at += 2;
    } else {
      out += char;
      at++;
    }
  }
  return out.replace(/\s+/g, " ").trim();
}
