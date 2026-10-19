// A period between digits ("5.5%") ends no sentence, and a marker after the full stop
// ("…a year.[^1]", "…三年。[^1]") is its sentence's, never the next one's.
const SENTENCE =
  /(?:[^.!?。！？\n]|\.(?=\d))+(?:[.!?。！？]+["”’)」』]*(?:\s*\[[\^=]?\d{1,4}\])*|$)\s*/g;
// What a Document or the User said, word for word: never taken apart. A quote left open runs to
// the line's end; a straight single quote counts only when it isn't an apostrophe.
const QUOTED =
  /"[^"\n]*(?:"|$)|“[^”\n]*(?:”|$)|「[^」\n]*(?:」|$)|『[^』\n]*(?:』|$)|‘[^’\n]*’|(?<![\p{L}\p{N}])'[^'\n]+'(?![\p{L}\p{N}])|`[^`\n]*`/gu;
const FENCE = /^\s*(?:```|~~~)/;
const HEADING = /^\s*(#{1,6})\s/;
const NOT_PROSE = /^\s*(?:\||>)/;
const RULE = /^\s*(?:[-*_]\s*){3,}$/;
const LEAD = /^\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?/;
const LEAD_IN = /[:：](?:\*\*|__)?\s*$|^\s*(?:\*\*|__)[^*_\n]+(?:\*\*|__)\s*$/;

interface Line {
  text: string;
  kind: "blank" | "kept" | "rule" | "heading" | "prose";
  level?: number;
  item?: boolean;
  gone?: boolean;
}

function withoutGoing(text: string, goes: (sentence: string) => boolean, wholeItems: boolean) {
  const lead = LEAD.exec(text)?.[0] ?? "";
  const body = text.slice(lead.length);
  const quoted = [...body.matchAll(QUOTED)].map(
    (match) => [match.index, match.index + match[0].length] as const,
  );
  let at = 0;
  const sentences = (body.match(SENTENCE) ?? []).map((sentence) => {
    const from = at;
    at += sentence.length;
    const inQuote = quoted.some(([start, end]) => start < at && end > from);
    return { sentence, going: !inQuote && goes(sentence.trimEnd()) };
  });
  const dropped = sentences.filter((each) => each.going).length;
  const line: Line = { text, kind: "prose", item: lead !== "" };
  if (dropped === 0) return { line, dropped };
  if (dropped === sentences.length) return { line: { ...line, gone: true }, dropped };
  if (wholeItems && lead) return { line, dropped: 0 };
  const rest = sentences
    .filter((each) => !each.going)
    .map((each) => each.sentence)
    .join("")
    .trimEnd();
  return { line: { ...line, text: `${lead}${rest}` }, dropped };
}

const allGone = (lines: readonly Line[]) =>
  lines.some((line) => line.kind === "prose") &&
  lines.every((line) => line.gone || line.kind === "rule");

function blockAfterLeadIn(lines: readonly Line[], from: number): Line[] {
  const start = lines.findIndex((line, index) => index >= from && line.kind !== "blank");
  const first = lines[start];
  if (first?.kind !== "prose") return [];
  const block: Line[] = [];
  for (const [index, line] of lines.entries()) {
    if (index < start) continue;
    if (line.kind === "blank" && first.item && lines[index + 1]?.item) continue;
    if (line.kind !== "prose" || line.item !== first.item) break;
    block.push(line);
  }
  return block;
}

// A "###" under a "##" is part of its section.
function sectionUnder(lines: readonly Line[], at: number): Line[] {
  const level = lines[at]?.level ?? 0;
  const next = lines.findIndex(
    (line, index) => index > at && line.kind === "heading" && (line.level ?? 0) <= level,
  );
  return lines.slice(at + 1, next < 0 ? lines.length : next).filter((l) => l.kind !== "blank");
}

/**
 * Markdown without the sentences of its paragraphs and list items that
 * `goes` picks, and how many went. Headings, tables, block quotes, code and
 * quoted words stay as they are, and with `wholeItems` a list item goes
 * whole or not at all. A line left with nothing goes. With `final` (the
 * whole text, never part of one still coming in), so does a heading or a
 * lead-in ("Key points:", "**Terms**") when all that was under it went;
 * `framing` counts those. Unchanged, the Markdown comes back as it was given.
 */
export function withoutSentences(
  markdown: string,
  goes: (sentence: string) => boolean,
  { final, wholeItems = false }: { final: boolean; wholeItems?: boolean },
): { markdown: string; dropped: number; framing: number } {
  let dropped = 0;
  let fenced = false;
  const lines = markdown.split("\n").map((text): Line => {
    if (FENCE.test(text)) {
      fenced = !fenced;
      return { text, kind: "kept" };
    }
    if (fenced || NOT_PROSE.test(text)) return { text, kind: "kept" };
    if (text.trim() === "") return { text, kind: "blank" };
    if (RULE.test(text)) return { text, kind: "rule" };
    const heading = HEADING.exec(text);
    if (heading) return { text, kind: "heading", level: heading[1]?.length };
    const rewritten = withoutGoing(text, goes, wholeItems);
    dropped += rewritten.dropped;
    return rewritten.line;
  });
  if (dropped === 0) return { markdown, dropped, framing: 0 };
  let framing = 0;
  if (final) {
    for (let at = lines.length - 1; at >= 0; at--) {
      const line = lines[at];
      if (line?.kind === "prose" && !line.gone && LEAD_IN.test(line.text)) {
        line.gone = allGone(blockAfterLeadIn(lines, at + 1));
        if (line.gone) framing++;
      }
    }
    for (let at = lines.length - 1; at >= 0; at--) {
      const line = lines[at];
      if (line?.kind === "heading" && allGone(sectionUnder(lines, at))) {
        line.gone = true;
        framing++;
      }
    }
  }
  const kept: Line[] = [];
  for (const line of lines) {
    if (line.gone || (line.kind === "blank" && (kept.at(-1)?.kind ?? "blank") === "blank")) {
      continue;
    }
    kept.push(line);
  }
  while (kept.at(-1)?.kind === "blank") kept.pop();
  return { markdown: kept.map((line) => line.text).join("\n"), dropped, framing };
}
