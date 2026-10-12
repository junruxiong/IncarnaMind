/**
 * A section of a regulation or a statute as text by section, one per
 * paragraph it numbers, so a Citation points at "§ 240.10b-5(b)" and its quote
 * is checked against that paragraph. Paragraphs are numbered by the markers
 * they start with, "(b)", "(1)", "(i)", "(A)", and how deep each is comes from
 * their order: the CFR nests (a), (1), (i), (A), then italic (1) and (i); the
 * US Code (a), (1), (A), (i), (I), (aa). A marker that could be a letter or a
 * Roman numeral ("(i)" after "(h)") is read by the one after it. Headings
 * carry no "§": the Location's label adds it (ADR-0011).
 */
import type { TextSection } from "./types";

export interface Marker {
  label: string;
  /** Set in italics: the CFR's fifth and sixth levels. */
  italic: boolean;
}

/**
 * One block of a section, in order:
 * - "paragraph": it may start with markers ("(b)(1) Except…");
 * - "flush": text after a list, which goes back to the paragraph around it ("in connection with…");
 * - "plain": text read as written, whatever it starts with (a quoted extract, a table's row, a note);
 * - "source": where the provision comes from (its authority, its history), which belongs to the whole of it.
 */
export interface ProvisionBlock {
  kind: "paragraph" | "flush" | "plain" | "source";
  text: string;
  markers: Marker[];
  /** For flush text, how far it is indented: 0 is the provision's own margin. Absent: one level out. */
  depth?: number;
}

type Kind =
  | "lower"
  | "digit"
  | "roman"
  | "upper"
  | "upperRoman"
  | "doubleLower"
  | "doubleUpper"
  | "italicDigit"
  | "italicRoman"
  | "italicLower";

interface Step {
  kind: Kind;
  n: number;
  label: string;
}

const ROMAN: Readonly<Record<string, number>> = { i: 1, v: 5, x: 10, l: 50, c: 100 };

/** A Roman numeral's value, written the usual way; null for anything else. */
function romanValue(label: string): number | null {
  const lower = label.toLowerCase();
  if (!/^[ivxlc]+$/.test(lower)) return null;
  let total = 0;
  for (let at = 0; at < lower.length; at++) {
    const value = ROMAN[lower[at] ?? ""] ?? 0;
    const next = ROMAN[lower[at + 1] ?? ""] ?? 0;
    total += value < next ? -value : value;
  }
  const back = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"];
  const tens = ["", "x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc"];
  const written = `${tens[Math.floor(total / 10)] ?? ""}${back[total % 10] ?? ""}`;
  return total > 0 && total < 100 && written === lower ? total : null;
}

function kindsOf({ label, italic }: Marker): Kind[] {
  const roman = romanValue(label) !== null;
  if (italic) {
    if (/^\d+$/.test(label)) return ["italicDigit"];
    if (/^[a-z]+$/.test(label)) return roman ? ["italicRoman", "italicLower"] : ["italicLower"];
    return [];
  }
  if (/^\d+$/.test(label)) return ["digit"];
  if (/^[a-z]$/.test(label)) return roman ? ["roman", "lower"] : ["lower"];
  if (/^([a-z])\1$/.test(label)) return roman ? ["roman", "doubleLower"] : ["doubleLower"];
  if (/^[a-z]+$/.test(label)) return roman ? ["roman"] : [];
  if (/^[A-Z]$/.test(label)) return roman ? ["upperRoman", "upper"] : ["upper"];
  if (/^([A-Z])\1$/.test(label)) return roman ? ["upperRoman", "doubleUpper"] : ["doubleUpper"];
  if (/^[A-Z]+$/.test(label)) return roman ? ["upperRoman"] : [];
  return [];
}

function ordinal(label: string, kind: Kind): number {
  if (kind === "digit" || kind === "italicDigit") return Number(label);
  if (kind === "roman" || kind === "upperRoman" || kind === "italicRoman") {
    return romanValue(label) ?? 0;
  }
  return label.toLowerCase().charCodeAt(0) - 96;
}

/** Where a marker would go in `path`, read as `kind`: after the step of its kind, or one deeper. */
function placeAs(path: readonly Step[], marker: Marker, kind: Kind): Step[] | null {
  const n = ordinal(marker.label, kind);
  const at = path.findIndex((step) => step.kind === kind);
  if (at >= 0) {
    return path[at]?.n === n - 1 ? [...path.slice(0, at), { kind, n, label: marker.label }] : null;
  }
  return n === 1 ? [...path, { kind, n, label: marker.label }] : null;
}

/** The path after `marker`, read by the marker after it when it could be read two ways. */
function place(path: readonly Step[], marker: Marker, next: Marker | null): Step[] {
  const kinds = kindsOf(marker);
  const options = kinds.flatMap((kind) => {
    const placed = placeAs(path, marker, kind);
    return placed ? [placed] : [];
  });
  if (options.length > 1 && next) {
    const fits = options.filter((option) =>
      kindsOf(next).some((kind) => placeAs(option, next, kind) !== null),
    );
    if (fits.length > 0) options.splice(0, options.length, ...fits);
  }
  // Of two that fit, the one that goes on at a level already open ("(i)" after "(h)").
  const sibling = options.find((option) => option.length <= path.length);
  const chosen = sibling ?? options[0];
  if (chosen) return chosen;
  // Out of order: kept at the level of its kind if one is open, else one deeper.
  const kind = kinds[0];
  if (!kind) return [...path];
  const at = path.findIndex((step) => step.kind === kind);
  const step = { kind, n: ordinal(marker.label, kind), label: marker.label };
  return at >= 0 ? [...path.slice(0, at), step] : [...path, step];
}

/** Where a reader's text is in italics: around it, so an italic marker is known. */
export const ITALIC_START = "";
export const ITALIC_END = "";

/** "(b)(1) Except…" → b, 1, with or without italics around the label or the whole marker. */
const MARKER = /^\s*(?)\((?)([0-9]{1,3}|[a-z]{1,6}|[A-Z]{1,6})?\)(?)/;

/** A paragraph's subject in italics after its marker: "(e) <i>Public disclosure.</i> (1) Except…" is (e)(1). */
const SUBJECT = /^\s*[^]{1,160}[.:]\s*\s*(?=\(|\()/;

/** The markers a paragraph's text starts with, and the text without its italic signs. */
export function leadingMarkers(marked: string): { markers: Marker[]; text: string } {
  const markers: Marker[] = [];
  let rest = marked;
  for (;;) {
    const match = MARKER.exec(rest);
    if (!match?.[3] || markers.length >= 6) break;
    const marker = { label: match[3], italic: match[1] !== "" || match[2] !== "" };
    if (kindsOf(marker).length === 0) break;
    markers.push(marker);
    rest = rest.slice(match[0].length);
    const subject = markers.length === 1 ? SUBJECT.exec(rest) : null;
    if (subject) rest = rest.slice(subject[0].length);
  }
  return { markers, text: marked.replace(/[]/g, "") };
}

/**
 * The sections of one provision: its heading and the text before its first
 * marker at `level`, then a section per numbered paragraph under it, headed
 * by the citation and the path ("240.10b-5(b)", "78j(a)(1)").
 */
export function provisionSections(
  heading: string,
  cite: string,
  level: number,
  blocks: readonly ProvisionBlock[],
): TextSection[] {
  const sections: TextSection[] = [{ heading, level, paragraphs: [] }];
  let path: Step[] = [];
  const open = (next: Step[]) => {
    path = next;
    sections.push({
      heading: `${cite}${next.map((step) => `(${step.label})`).join("")}`,
      level: level + next.length,
      paragraphs: [],
    });
  };
  const markersAfter = (from: number): Marker | null => {
    for (let at = from; at < blocks.length; at++) {
      const block = blocks[at];
      if (block?.kind === "paragraph" && block.markers[0]) return block.markers[0];
    }
    return null;
  };
  blocks.forEach((block, index) => {
    const text = block.text.replace(/\s+/g, " ").trim();
    if (!text) return;
    if (block.kind === "paragraph" && block.markers.length > 0) {
      let next = path;
      block.markers.forEach((marker, at) => {
        next = place(next, marker, block.markers[at + 1] ?? markersAfter(index + 1));
      });
      open(next);
    } else if (block.kind === "flush" && path.length > 0) {
      open(path.slice(0, Math.min(block.depth ?? path.length - 1, path.length - 1)));
    } else if (block.kind === "source" && path.length > 0) {
      open([]);
    }
    sections.at(-1)?.paragraphs.push(text);
  });
  return sections;
}
