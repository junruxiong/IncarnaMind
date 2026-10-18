// A period between digits ("5.5%") ends no sentence, and a marker after the full stop
// ("…a year.[^1]", "…三年。[^1]") is its sentence's, never the next one's.
const SENTENCE =
  /(?:[^.!?。！？\n]|\.(?=\d))+(?:[.!?。！？]+["”’)]*(?:\s*\[[\^=]?\d{1,4}\])*|$)\s*/g;
const FENCE = /^\s*(?:```|~~~)/;
const HEADING = /^\s*(#{1,6})\s/;
const NOT_PROSE = /^\s*(?:\||>)/;
const LEAD = /^\s*(?:[-*+]|\d{1,3}[.)])\s+/;
const LEAD_IN = /[:：]\s*$/;

interface Line {
  text: string;
  kind: "blank" | "kept" | "heading" | "prose";
  level?: number;
  item?: boolean;
  gone?: boolean;
}

function rewriteLine(text: string, rewrite: (sentence: string) => string) {
  const lead = LEAD.exec(text)?.[0] ?? "";
  const sentences = text.slice(lead.length).match(SENTENCE) ?? [];
  let changed = 0;
  const rewritten = sentences.map((sentence) => {
    const space = /\s*$/.exec(sentence)?.[0] ?? "";
    const body = sentence.slice(0, sentence.length - space.length);
    const after = rewrite(body);
    if (after === body) return sentence;
    changed++;
    return after ? `${after}${space}` : "";
  });
  const line: Line = { text, kind: "prose", item: lead !== "" };
  if (changed === 0) return { line, changed };
  const rest = rewritten.join("").trim();
  return { line: rest ? { ...line, text: `${lead}${rest}` } : { ...line, gone: true }, changed };
}

const allGone = (lines: readonly Line[]) =>
  lines.some((line) => line.kind === "prose") && lines.every((line) => line.gone);

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

// Down to the next heading as high as it: a "###" under it belongs to it.
function sectionUnder(lines: readonly Line[], at: number): Line[] {
  const level = lines[at]?.level ?? 0;
  const next = lines.findIndex(
    (line, index) => index > at && line.kind === "heading" && (line.level ?? 0) <= level,
  );
  return lines.slice(at + 1, next < 0 ? lines.length : next).filter((l) => l.kind !== "blank");
}

/**
 * Markdown with each sentence of its paragraphs and list items passed through
 * `rewrite`, which returns it as it is, changed, or "" to take it out, and how
 * many it changed or took out. Headings, tables, quotes and code stay as they
 * are. A line left with nothing goes, and so does a heading or a lead-in
 * ("Key points:") when all that was under it went. Unchanged, the Markdown
 * comes back as it was given.
 */
export function rewriteSentences(
  markdown: string,
  rewrite: (sentence: string) => string,
): { markdown: string; changed: number } {
  let changed = 0;
  let fenced = false;
  const lines = markdown.split("\n").map((text): Line => {
    if (FENCE.test(text)) {
      fenced = !fenced;
      return { text, kind: "kept" };
    }
    if (fenced || NOT_PROSE.test(text)) return { text, kind: "kept" };
    if (text.trim() === "") return { text, kind: "blank" };
    const heading = HEADING.exec(text);
    if (heading) return { text, kind: "heading", level: heading[1]?.length };
    const rewritten = rewriteLine(text, rewrite);
    changed += rewritten.changed;
    return rewritten.line;
  });
  if (changed === 0) return { markdown, changed };
  for (let at = lines.length - 1; at >= 0; at--) {
    const line = lines[at];
    if (line?.kind === "prose" && !line.gone && LEAD_IN.test(line.text)) {
      line.gone = allGone(blockAfterLeadIn(lines, at + 1));
    }
  }
  for (let at = lines.length - 1; at >= 0; at--) {
    const line = lines[at];
    if (line?.kind === "heading") line.gone = allGone(sectionUnder(lines, at));
  }
  const kept: Line[] = [];
  for (const line of lines) {
    if (line.gone || (line.kind === "blank" && (kept.at(-1)?.kind ?? "blank") === "blank")) {
      continue;
    }
    kept.push(line);
  }
  while (kept.at(-1)?.kind === "blank") kept.pop();
  return { markdown: kept.map((line) => line.text).join("\n"), changed };
}
