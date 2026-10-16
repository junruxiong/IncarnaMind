/**
 * A help button's Answer during a call (#839), read line by line as the panel
 * lays it out (the shape src/core/answers/meetingHelp.ts asks for): options
 * or readings, the recommended first after "★", each followed by "→" and what
 * it leads to; or up to three short lines. It reads a part still streaming.
 */

/** A run of text, a Citation's marker ("[^2]") or a calculated figure's ("[=1]"). */
export type ReplyPiece =
  | { kind: "text"; text: string }
  | { kind: "citation"; marker: number }
  | { kind: "figure"; marker: number };

export interface ReplyItem {
  recommended: boolean;
  /** The sentence to say, the reading, or the line. */
  lead: ReplyPiece[];
  /** Lines between it and its "→": a reading's moment of the transcript. */
  notes: ReplyPiece[][];
  /** What it leads to; null before its "→" line. */
  effect: ReplyPiece[] | null;
}

const EFFECT = /^(?:→|->)\s*/;
const RECOMMENDED = /^(?:★|⭐)\s*/;
const BULLET = /^(?:[-*•]|\d+[.)])\s+/;
const DEFINITION = /^\[\^\d+\]:/;

export function pieces(text: string): ReplyPiece[] {
  const found: ReplyPiece[] = [];
  let at = 0;
  for (const match of text.matchAll(/\[\^(\d+)\]|\[=(\d+)\]/g)) {
    const before = text.slice(at, match.index);
    if (before) found.push({ kind: "text", text: before });
    found.push(
      match[1] !== undefined
        ? { kind: "citation", marker: Number(match[1]) }
        : { kind: "figure", marker: Number(match[2]) },
    );
    at = match.index + match[0].length;
  }
  const rest = text.slice(at);
  if (rest) found.push({ kind: "text", text: rest });
  return found;
}

/** Bold and italics marks a model may add, which the panel doesn't show. */
const plain = (line: string) => line.replace(/\*\*|__/g, "").trim();

export function readHelpReply(markdown: string): ReplyItem[] {
  const items: ReplyItem[] = [];
  for (const raw of markdown.split("\n")) {
    const line = plain(raw);
    if (line === "" || DEFINITION.test(line) || /^#+\s/.test(line)) continue;
    const current = items.at(-1);
    if (EFFECT.test(line)) {
      const effect = pieces(line.replace(EFFECT, ""));
      if (current && current.effect === null) current.effect = effect;
      else items.push({ recommended: false, lead: [], notes: [], effect });
      continue;
    }
    const recommended = RECOMMENDED.test(line);
    const bullet = BULLET.test(line);
    const text = line.replace(RECOMMENDED, "").replace(BULLET, "");
    if (!recommended && !bullet && current && current.effect === null && current.lead.length > 0) {
      current.notes.push(pieces(text));
      continue;
    }
    items.push({ recommended, lead: pieces(text), notes: [], effect: null });
  }
  return items;
}
