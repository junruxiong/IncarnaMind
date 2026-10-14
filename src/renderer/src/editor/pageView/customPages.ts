/**
 * Spike for #755, not for merging: approach (b), a custom layout pass over the live editor.
 *
 * - Every textblock (descending through lists) and every other Block is a unit; its natural
 *   height (without the spacers this plugin adds) is measured once per node identity.
 * - Pages are filled from those heights, Word's way: page and section breaks, page break before,
 *   keep with next, keep lines together, widow and orphan control, and footnotes at the foot of
 *   the page their mark is on (the line with the mark moves on when its note doesn't fit).
 * - A page break between top-level Blocks is an empty block widget; one inside a paragraph (or
 *   before a paragraph in a list) is an empty full-width float at the start of the line, so the
 *   line before it stays justified. Each spacer's height puts the next page's first line at the
 *   next sheet's top margin.
 * - The sheets, headers, footers, footnotes and page numbers are drawn in an overlay outside
 *   the editable DOM, so ProseMirror never sees them.
 * - After an edit, layout starts at the page before the edited one and stops once a page starts
 *   where it started before (convergence); `syncPages` pages are laid out before the frame, the
 *   rest in idle time.
 */
import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet, type EditorView } from "@tiptap/pm/view";
import type * as Y from "yjs";
import type {
  SpikeBlockLook,
  SpikeLook,
  SpikeSection,
} from "../../../../../spikes/755-page-view/wordLayout";
import { WORD_ATOM_NODE } from "../../../../core/api";
import { emptyStats, spikeConfig } from "./spikeConfig";
import { facesOf, looksOf, normalRatio } from "./wordLookSpike";

export const PAGE_GAP = 24;

interface Note {
  pos: number;
  label: string;
  text: string;
}

interface Unit {
  node: ProseMirrorNode;
  pos: number;
  topLevel: boolean;
  textblock: boolean;
  look: SpikeBlockLook | null;
  height: number;
  section: number;
  notes: Note[];
  /** Inside the unit: the position just after a page break, when text follows it. */
  breakInside: number | null;
  /** A page break or a section break ends the page after this unit. */
  breakAfter: boolean;
  /** A table row's unit: where each of its cells' text starts, for the floats that move the row on. */
  rowCells?: number[];
}

interface Lines {
  /** Each line box's top and bottom, natural, from the unit's top. */
  tops: number[];
  bottoms: number[];
  /** Where each line starts, as a document position. */
  starts: number[];
}

interface Page {
  section: number;
  unit: number;
  line: number;
  /** The document position of the break that starts this page (0 for the first). */
  pos: number;
  kind: "first" | "block" | "float" | "row";
  /** A row's other cells, each with a float of the same height. */
  extra?: number[];
  /** Natural y where the page's content ends, from its content top. */
  endY: number;
  notes: Note[];
  notesHeight: number;
}

interface PagesState {
  pages: Page[];
  decorations: DecorationSet;
  /** The smallest position an edit since the last layout touched; null when laid out. */
  dirtyFrom: number | null;
}

const key = new PluginKey<PagesState>("spikePages");

const heights = new WeakMap<ProseMirrorNode, number>();
const lineCache = new WeakMap<ProseMirrorNode, Lines>();
const noteHeights = new Map<string, number>();

function sectionOf(look: SpikeLook, index: number): SpikeSection {
  return (look.sections[Math.min(index, look.sections.length - 1)] ??
    look.sections[0]) as SpikeSection;
}

const contentHeight = (section: SpikeSection) => section.height - section.top - section.bottom;

/** The units of a document, in order, flattening lists and quotes. */
function unitsOf(doc: ProseMirrorNode, looks: Map<ProseMirrorNode, SpikeBlockLook>): Unit[] {
  const units: Unit[] = [];
  let section = 0;
  const visit = (node: ProseMirrorNode, pos: number, topLevel: boolean) => {
    const container =
      !node.isTextblock &&
      !node.isAtom &&
      node.type.name !== "table" &&
      node.type.name !== "comparisonTable" &&
      node.childCount > 0 &&
      /list|listItem|blockquote|taskItem/i.test(node.type.name);
    if (container) {
      node.forEach((child, offset) => visit(child, pos + 1 + offset, false));
      return;
    }
    if (node.type.name === "table") {
      const rows: { row: ProseMirrorNode; pos: number }[] = [];
      node.forEach((row, offset) => rows.push({ row, pos: pos + 1 + offset }));
      // A box (one cell a row) splits like its paragraphs; other tables between rows.
      if (rows.every(({ row }) => row.childCount === 1)) {
        for (const { row, pos: rowPos } of rows) {
          row.forEach((cell, cellOffset) =>
            cell.forEach((child, childOffset) =>
              visit(child, rowPos + 1 + cellOffset + 1 + childOffset, false),
            ),
          );
        }
        return;
      }
      for (const { row, pos: rowPos } of rows) {
        const rowCells: number[] = [];
        const notes: Note[] = [];
        row.forEach((cell, cellOffset) => {
          let first: number | null = null;
          cell.descendants((child, childPos) => {
            const at = rowPos + 1 + cellOffset + 1 + childPos;
            if (first === null && child.isTextblock && child.content.size > 0) first = at + 1;
            if (child.type.name === WORD_ATOM_NODE && child.attrs.kind === "footnote") {
              notes.push({
                pos: at,
                label: String(child.attrs.label ?? ""),
                text: String(child.attrs.text ?? ""),
              });
            }
            return true;
          });
          if (first !== null) rowCells.push(first);
        });
        units.push({
          node: row,
          pos: rowPos,
          topLevel: false,
          textblock: false,
          look: null,
          height: 0,
          section,
          notes,
          breakInside: null,
          breakAfter: false,
          rowCells,
        });
      }
      return;
    }
    const look = node.isTextblock ? (looks.get(node) ?? null) : null;
    if (look) section = look.section;
    const notes: Note[] = [];
    let breakInside: number | null = null;
    let breakAfter = false;
    let sectionBreak = false;
    if (node.isTextblock) {
      let seenText = false;
      node.descendants((child, offset) => {
        const at = pos + 1 + offset;
        if (child.type.name === WORD_ATOM_NODE) {
          const kind = child.attrs.kind;
          if (kind === "footnote") {
            notes.push({
              pos: at,
              label: String(child.attrs.label ?? ""),
              text: String(child.attrs.text ?? ""),
            });
          } else if (kind === "pageBreak") {
            const after = node.textBetween(offset + 1, node.content.size, "", "");
            if (after.trim().length === 0) breakAfter = true;
            else if (!seenText) breakInside = at + 1;
            else breakInside = at + 1;
          } else if (kind === "sectionBreak") {
            sectionBreak = true;
          }
        } else if (child.isText) {
          seenText = true;
        }
        return true;
      });
    }
    units.push({
      node,
      pos,
      topLevel,
      textblock: node.isTextblock,
      look,
      height: 0,
      section,
      notes,
      breakInside,
      breakAfter: breakAfter || sectionBreak,
    });
  };
  doc.forEach((child, offset) => visit(child, offset, true));
  return units;
}

/** The spacers inside an element: what they add to its height, and where each ends. */
function floatsIn(element: HTMLElement): { bottom: number; height: number }[] {
  return [...element.querySelectorAll<HTMLElement>("[data-spike-float]")].map((float) => {
    const rect = float.getBoundingClientRect();
    return { bottom: rect.bottom, height: rect.height };
  });
}

/** The CSS zoom the editor is drawn at (narrow windows): rects come back zoomed, layout is in CSS px. */
const zoomOf = (view: EditorView) =>
  Number(view.dom.closest<HTMLElement>("[data-spike-scale]")?.dataset.spikeScale ?? 0) ||
  ((view.dom as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1);

/** Subtracts the spacers above a point, to get its natural y. */
const natural = (y: number, floats: { bottom: number; height: number }[]) =>
  y - floats.reduce((sum, float) => (y >= float.bottom - 0.5 ? sum + float.height : sum), 0);

function measureHeight(view: EditorView, unit: Unit): number {
  const known = heights.get(unit.node);
  if (known !== undefined) return known;
  const dom = view.nodeDOM(unit.pos);
  if (!(dom instanceof HTMLElement)) return 0;
  const rect = dom.getBoundingClientRect();
  const floats = floatsIn(dom);
  // A row's floats stand side by side, one a cell.
  const spacers = unit.rowCells
    ? Math.max(0, ...floats.map((float) => float.height))
    : floats.reduce((sum, float) => sum + float.height, 0);
  const height = (rect.height - spacers) / zoomOf(view);
  heights.set(unit.node, height);
  return height;
}

/** A textblock's lines, natural, with where each starts. */
function linesOf(view: EditorView, unit: Unit): Lines {
  const known = lineCache.get(unit.node);
  if (known) return known;
  const dom = view.nodeDOM(unit.pos);
  const empty = { tops: [0], bottoms: [unit.height], starts: [unit.pos + 1] };
  if (!(dom instanceof HTMLElement)) return empty;
  const top = dom.getBoundingClientRect().top;
  const floats = floatsIn(dom);
  const zoom = zoomOf(view);
  const style = getComputedStyle(dom);
  const paddingTop = Number.parseFloat(style.paddingTop) || 0;
  const paddingBottom = Number.parseFloat(style.paddingBottom) || 0;
  interface Item {
    top: number;
    bottom: number;
    node: Node;
    offset: number;
  }
  const items: Item[] = [];
  const walker = document.createTreeWalker(dom, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, {
    acceptNode: (node) => {
      if (node instanceof Element && node.closest("[data-spike-float]"))
        return NodeFilter.FILTER_REJECT;
      if (
        node instanceof Element &&
        node.getAttribute("contenteditable") === "false" &&
        node !== dom
      ) {
        return NodeFilter.FILTER_ACCEPT;
      }
      return node instanceof Text ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    },
  });
  const range = document.createRange();
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node instanceof Text) {
      range.selectNodeContents(node);
      for (const rect of range.getClientRects()) {
        if (rect.height === 0) continue;
        items.push({
          top: (natural(rect.top, floats) - top) / zoom,
          bottom: (natural(rect.bottom, floats) - top) / zoom,
          node,
          offset: -1,
        });
      }
    } else if (node instanceof Element) {
      const rect = node.getBoundingClientRect();
      if (rect.height > 0) {
        items.push({
          top: (natural(rect.top, floats) - top) / zoom,
          bottom: (natural(rect.bottom, floats) - top) / zoom,
          node,
          offset: 0,
        });
      }
    }
  }
  if (items.length === 0) {
    lineCache.set(unit.node, empty);
    return empty;
  }
  // Lines from the items' vertical extents: an item whose middle is below the line's bottom starts one.
  const lines: { top: number; bottom: number; first: Item }[] = [];
  for (const item of items) {
    const middle = (item.top + item.bottom) / 2;
    const line = lines.at(-1);
    if (line && middle < line.bottom) {
      line.top = Math.min(line.top, item.top);
      line.bottom = Math.max(line.bottom, item.bottom);
    } else {
      lines.push({ top: item.top, bottom: item.bottom, first: item });
    }
  }
  lines.sort((a, b) => a.top - b.top);
  const tops: number[] = [];
  const bottoms: number[] = [];
  lines.forEach((line, index) => {
    const next = lines[index + 1];
    tops.push(index === 0 ? paddingTop : ((lines[index - 1]?.bottom ?? 0) + line.top) / 2);
    bottoms.push(next ? (line.bottom + next.top) / 2 : unit.height - paddingBottom);
  });
  const starts = lines.map((line, index) => {
    if (index === 0) return unit.pos + 1;
    const first = line.first;
    if (!(first.node instanceof Text)) return view.posAtDOM(first.node, 0);
    // The first character of this text node on this line.
    const text = first.node;
    const boundary = tops[index] ?? 0;
    let low = 0;
    let high = text.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      range.setStart(text, middle);
      range.setEnd(text, middle + 1);
      const rect = range.getClientRects()[0];
      const y = rect ? (natural((rect.top + rect.bottom) / 2, floats) - top) / zoom : 0;
      if (y < boundary) low = middle + 1;
      else high = middle;
    }
    return view.posAtDOM(text, low);
  });
  const result = { tops, bottoms, starts };
  lineCache.set(unit.node, result);
  return result;
}

function noteHeight(note: Note, look: SpikeLook, width: number): number {
  const id = `${width}\u0000${note.label}\u0000${note.text}`;
  const known = noteHeights.get(id);
  if (known !== undefined) return known;
  const config = spikeConfig();
  const faces = facesOf(look.footnote.family, config?.fontMap ?? {});
  const probe = document.createElement("div");
  const line =
    Math.round(normalRatio(faces.latin) * look.footnote.size * look.footnote.lineMultiple * 15) /
    15;
  probe.style.cssText = `position:absolute;visibility:hidden;width:${width}px;font-family:${faces.css};font-size:${look.footnote.size}px;line-height:${line}px`;
  fillNote(probe, note);
  document.body.append(probe);
  const height = probe.getBoundingClientRect().height;
  probe.remove();
  noteHeights.set(id, height);
  return height;
}

/** A note as Word draws it: its number raised and smaller, then its text. */
function fillNote(element: HTMLElement, note: Note): void {
  const mark = document.createElement("sup");
  mark.style.cssText = "font-size:0.65em;line-height:0";
  mark.textContent = note.label;
  element.replaceChildren(mark, document.createTextNode(` ${note.text}`));
}

/** The separator rule's room above the notes. */
const NOTES_RULE = 14;

const notesHeight = (notes: Note[], look: SpikeLook, width: number) =>
  notes.length === 0
    ? 0
    : NOTES_RULE + notes.reduce((sum, note) => sum + noteHeight(note, look, width), 0);

interface LayoutRun {
  pages: Page[];
  converged: boolean;
  laidOut: number;
}

/**
 * Lays pages out from `from` (an index into `previous`), until a page starts where one of
 * `previous` after `stableFrom` did (then the rest is `previous`'s), or `limit` pages are done.
 */
function layOut(
  view: EditorView,
  units: Unit[],
  look: SpikeLook,
  previous: Page[],
  from: number,
  stableFrom: number,
  limit: number,
): LayoutRun {
  const width = sectionOf(look, 0).width - sectionOf(look, 0).left - sectionOf(look, 0).right;
  const pages = previous.slice(0, from);
  const startPage = previous[from];
  let unitIndex = startPage ? Math.max(0, unitAt(units, startPage.pos)) : 0;
  let line = 0;
  if (startPage && startPage.kind === "float") {
    const unit = units[unitIndex];
    if (unit) {
      unit.height = measureHeight(view, unit);
      const lines = linesOf(view, unit);
      line = Math.max(
        0,
        lines.starts.findIndex((start) => start >= startPage.pos),
      );
    }
  }
  let kind: Page["kind"] = startPage?.kind ?? "first";
  let pos = startPage?.pos ?? 0;
  let extra = startPage?.extra;
  let laidOut = 0;
  const oldByPos = new Map(
    previous.slice(stableFrom).map((page, index) => [page.pos, index + stableFrom]),
  );
  while (unitIndex < units.length) {
    if (laidOut >= limit) return { pages, converged: false, laidOut };
    if (pages.length > from) {
      const old = oldByPos.get(pos);
      if (old !== undefined && previous[old]?.kind === kind) {
        return { pages: [...pages, ...previous.slice(old)], converged: true, laidOut };
      }
    }
    const firstUnit = units[unitIndex] as Unit;
    const section = sectionOf(look, firstUnit.section);
    const avail = contentHeight(section);
    const page: Page = {
      section: firstUnit.section,
      unit: unitIndex,
      line,
      pos,
      kind,
      ...(extra && { extra }),
      endY: 0,
      notes: [],
      notesHeight: 0,
    };
    let y = 0;
    let next: {
      unit: number;
      line: number;
      pos: number;
      kind: Page["kind"];
      extra?: number[];
    } | null = null;
    for (let index = unitIndex; index < units.length; index++) {
      const unit = units[index] as Unit;
      unit.height = measureHeight(view, unit);
      const startLine = index === unitIndex ? line : 0;
      const isFirst = index === unitIndex;
      if (!isFirst && unit.look?.pageBreakBefore) {
        next = startOf(unit, index);
        break;
      }
      if (
        !isFirst &&
        unit.section !== page.section &&
        sectionOf(look, unit.section).type !== "continuous"
      ) {
        next = startOf(unit, index);
        break;
      }
      let lines: Lines | null = null;
      const startTop = startLine > 0 ? ((lines = linesOf(view, unit)).tops[startLine] ?? 0) : 0;
      const rest = unit.height - startTop;
      const unitNotes =
        startLine > 0 && lines ? notesFrom(view, unit, lines, startLine, Infinity) : unit.notes;
      const withNotes = notesHeight([...page.notes, ...unitNotes], look, width);
      if (unit.breakInside !== null && unit.textblock) {
        lines ??= linesOf(view, unit);
        const breakLine = lines.starts.findIndex((start) => start >= (unit.breakInside ?? 0));
        if (breakLine > startLine) {
          const bottom = lines.tops[breakLine] ?? unit.height;
          if (y + bottom - startTop + withNotes <= avail || isFirst) {
            page.notes.push(...notesFrom(view, unit, lines, startLine, breakLine));
            y += bottom - startTop;
            next = { unit: index, line: breakLine, pos: unit.breakInside, kind: "float" };
            break;
          }
        }
      }
      // Space after the last paragraph on a page may fall off its foot, as in Word.
      const spaceAfter = unit.textblock ? (unit.look?.after ?? 0) : 0;
      if (y + rest - spaceAfter + withNotes <= avail + 0.5) {
        y += rest;
        page.notes.push(...unitNotes);
        if (unit.breakAfter && units[index + 1]) {
          next = startOf(units[index + 1] as Unit, index + 1);
          break;
        }
        continue;
      }
      // It doesn't fit whole: split it between lines, where the rules let it.
      if (unit.textblock && !unit.look?.keepLines) {
        lines ??= linesOf(view, unit);
        const count = lines.tops.length;
        const least = unit.look?.widowControl ? 2 : 1;
        for (let split = count - least; split >= startLine + (isFirst ? 1 : least); split--) {
          const used = (lines.tops[split] ?? 0) - startTop;
          const notes = [...page.notes, ...notesFrom(view, unit, lines, startLine, split)];
          if (y + used + notesHeight(notes, look, width) <= avail + 0.5) {
            page.notes = notes;
            y += used;
            next = {
              unit: index,
              line: split,
              pos: lines.starts[split] ?? unit.pos + 1,
              kind: "float",
            };
            break;
          }
        }
        if (next) break;
      }
      if (isFirst) {
        // Taller than a page with nowhere to split: it goes here, over the foot.
        y += rest;
        page.notes.push(...unitNotes);
        const following = units[index + 1];
        if (following) next = startOf(following, index + 1);
        break;
      }
      // It moves on whole, with the Blocks before it that keep with the next.
      let moveFrom = index;
      while (moveFrom - 1 > unitIndex && units[moveFrom - 1]?.look?.keepNext) moveFrom--;
      if (moveFrom !== index) {
        for (let back = moveFrom; back < index; back++) {
          const moved = units[back] as Unit;
          y -= moved.height;
          page.notes = page.notes.filter((note) => !moved.notes.includes(note));
        }
      }
      next = startOf(units[moveFrom] as Unit, moveFrom);
      break;
    }
    page.endY = y;
    page.notesHeight = notesHeight(page.notes, look, width);
    pages.push(page);
    laidOut++;
    if (!next) return { pages, converged: true, laidOut };
    unitIndex = next.unit;
    extra = next.extra;
    line = next.line;
    pos = next.pos;
    kind = next.kind;
  }
  return { pages, converged: true, laidOut };
}

function notesFrom(view: EditorView, unit: Unit, lines: Lines, from: number, to: number): Note[] {
  if (unit.notes.length === 0) return [];
  const top = (view.nodeDOM(unit.pos) as HTMLElement | null)?.getBoundingClientRect().top ?? 0;
  const dom = view.nodeDOM(unit.pos);
  const floats = dom instanceof HTMLElement ? floatsIn(dom) : [];
  return unit.notes.filter((note) => {
    const coords = view.coordsAtPos(note.pos);
    const y = (natural((coords.top + coords.bottom) / 2, floats) - top) / zoomOf(view);
    const at = lines.tops.findLastIndex((lineTop) => lineTop <= y);
    return at >= from && at < to;
  });
}

/** Where a page that starts with this whole unit breaks: a block widget, or a float inside a nested paragraph. */
const startOf = (
  unit: Unit,
  index: number,
): { unit: number; line: number; pos: number; kind: Page["kind"]; extra?: number[] } =>
  unit.rowCells && unit.rowCells.length > 0
    ? {
        unit: index,
        line: 0,
        pos: unit.rowCells[0] as number,
        kind: "row",
        extra: unit.rowCells.slice(1),
      }
    : // A float inside the paragraph, not a widget between top-level Blocks: ProseMirror scans
      // the top level's decorations once per top-level Block on every update.
      // An empty paragraph can't hold one: with a float as its only content it has no line box.
      unit.textblock && unit.node.content.size > 0
      ? { unit: index, line: 0, pos: unit.pos + 1, kind: "float" as const }
      : { unit: index, line: 0, pos: unit.pos, kind: "block" as const };

/** The index of the unit holding a position: the last that starts at or before it. */
function unitAt(units: Unit[], pos: number): number {
  let low = 0;
  let high = units.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if ((units[middle]?.pos ?? 0) <= pos) low = middle;
    else high = middle - 1;
  }
  return low;
}

/** Each page's top, from the editor's top, and its sheet's size. */
export function pageGeometry(look: SpikeLook, pages: Page[]) {
  let top = 0;
  return pages.map((page) => {
    const section = sectionOf(look, page.section);
    const geometry = { top, section };
    top += section.height + (spikeConfig()?.gap ?? PAGE_GAP);
    return geometry;
  });
}

function decorationsOf(state: EditorState, look: SpikeLook, pages: Page[]): DecorationSet {
  const geometry = pageGeometry(look, pages);
  const decorations: Decoration[] = [];
  pages.forEach((page, index) => {
    if (index === 0) return;
    const before = pages[index - 1] as Page;
    const beforeGeometry = geometry[index - 1];
    const own = geometry[index];
    if (!beforeGeometry || !own) return;
    const from = beforeGeometry.top + beforeGeometry.section.top + before.endY;
    const height = Math.max(0, own.top + own.section.top - from);
    const rounded = Math.round(height * 100) / 100;
    if (page.kind === "block") {
      decorations.push(
        Decoration.widget(page.pos, () => spacer("div", rounded), {
          side: -1,
          key: `gap-b-${rounded}`,
          ignoreSelection: true,
        }),
      );
    } else {
      for (const at of [page.pos, ...(page.extra ?? [])]) {
        decorations.push(
          Decoration.widget(at, () => spacer("span", rounded), {
            side: -1,
            key: `gap-f-${rounded}`,
            ignoreSelection: true,
          }),
        );
      }
    }
  });
  const last = pages.at(-1);
  const lastGeometry = geometry.at(-1);
  if (last && lastGeometry) {
    const end = lastGeometry.top + lastGeometry.section.top + last.endY;
    const height = Math.max(
      0,
      lastGeometry.top + lastGeometry.section.height - lastGeometry.section.top - end,
    );
    const rounded = Math.round(height * 100) / 100;
    decorations.push(
      Decoration.widget(state.doc.content.size, () => spacer("div", rounded), {
        side: 1,
        key: `gap-end-${rounded}`,
        ignoreSelection: true,
      }),
    );
  }
  return DecorationSet.create(state.doc, decorations);
}

function spacer(tag: "div" | "span", height: number): HTMLElement {
  const element = document.createElement(tag);
  element.contentEditable = "false";
  if (tag === "span") {
    element.dataset.spikeFloat = "";
    // Wider than the paragraph on both sides: a hanging indent starts lines left of the content
    // edge, and they'd fit into whatever a 100% float leaves.
    element.style.cssText = `float:left;width:calc(100% + 20000px);margin-left:-10000px;height:${height}px;display:block`;
  } else {
    element.dataset.spikeGap = "";
    element.style.cssText = `height:${height}px`;
  }
  return element;
}

/** The sheets, headers, footers, footnotes and page numbers, outside the editable DOM. */
function drawOverlay(
  overlay: HTMLElement,
  overlayTop: HTMLElement,
  look: SpikeLook,
  pages: Page[],
  editorTop: number,
  editorLeft: number,
) {
  const geometry = pageGeometry(look, pages);
  const config = spikeConfig();
  const fontMap = config?.fontMap ?? {};
  const furnitureFace = facesOf(look.footnote.family, fontMap).css;
  const total = pages.length;
  const existing = [...overlay.children] as HTMLElement[];
  const existingTop = [...overlayTop.children] as HTMLElement[];
  const gap = spikeConfig()?.gap ?? PAGE_GAP;
  const fill = (text: string, number: number) =>
    text.replaceAll("{page}", String(number)).replaceAll("{pages}", String(total));
  pages.forEach((page, index) => {
    const at = geometry[index];
    if (!at) return;
    const { section, top } = at;
    const number = index + 1;
    const last = index === pages.length - 1;
    const signature = `${top}|${editorTop}|${editorLeft}|${page.section}|${total}|${page.notes.map((note) => note.label).join(",")}|${page.notesHeight}|${Math.round(page.endY)}|${last}|${gap}`;
    const current = existing[index];
    if (current && current.dataset.signature === signature) return;
    // Above the text: the margins, what is under the page's last line, and the gap, so nothing
    // crosses them (a split table's borders, a selection); the header, notes and footer on them.
    const over = document.createElement("div");
    over.className = "spike-furniture";
    over.style.cssText = `top:${editorTop + top}px;left:${editorLeft}px;width:${section.width}px;height:${section.height + (last ? 0 : gap)}px`;
    const mask = (from: number, to: number, className: string) => {
      const element = document.createElement("div");
      element.className = className;
      element.style.cssText = `top:${from}px;height:${Math.max(0, to - from)}px`;
      return element;
    };
    over.append(
      mask(0, section.top, "spike-mask"),
      mask(Math.min(section.height, section.top + page.endY), section.height, "spike-mask"),
    );
    if (!last)
      over.append(mask(section.height, section.height + gap, "spike-mask spike-mask--gap"));
    const sheet = document.createElement("div");
    sheet.dataset.signature = signature;
    sheet.className = "spike-sheet";
    sheet.dataset.page = String(number);
    sheet.style.cssText = `top:${editorTop + top}px;left:${editorLeft}px;width:${section.width}px;height:${section.height}px`;
    const firstOfSection = index === 0 || pages[index - 1]?.section !== page.section;
    const variant =
      firstOfSection && section.titlePg
        ? "first"
        : number % 2 === 0 && section.headers.even !== undefined
          ? "even"
          : "default";
    const header = section.headers[variant] ?? "";
    const footer = section.footers[variant] ?? "";
    const text = (content: string, css: string, className: string) => {
      const element = document.createElement("div");
      element.className = className;
      element.style.cssText = `left:${section.left}px;right:${section.right}px;font-family:${furnitureFace};${css}`;
      const [left = "", ...rest] = content.split("\t");
      const right = rest.filter(Boolean).at(-1) ?? "";
      if (right) {
        const start = document.createElement("span");
        start.textContent = fill(left, number);
        const end = document.createElement("span");
        end.textContent = fill(right, number);
        element.append(start, end);
        element.style.display = "flex";
        element.style.justifyContent = "space-between";
      } else {
        element.textContent = fill(content, number);
      }
      return element;
    };
    if (header) over.append(text(header, `top:${section.header}px`, "spike-header"));
    if (footer)
      over.append(
        text(
          footer,
          `top:${section.height - section.footer}px;transform:translateY(-100%);text-align:center`,
          "spike-footer",
        ),
      );
    if (page.notes.length > 0) {
      const notes = document.createElement("div");
      notes.className = "spike-notes";
      const footnote = look.footnote;
      const line =
        normalRatio(facesOf(footnote.family, fontMap).latin) *
        footnote.size *
        footnote.lineMultiple;
      notes.style.cssText = `left:${section.left}px;right:${section.right}px;top:${section.height - section.bottom - page.notesHeight}px;font-family:${facesOf(footnote.family, fontMap).css};font-size:${footnote.size}px;line-height:${line}px`;
      for (const note of page.notes) {
        const row = document.createElement("div");
        row.className = "spike-note";
        fillNote(row, note);
        notes.append(row);
      }
      over.append(notes);
    }
    if (current) current.replaceWith(sheet);
    else overlay.append(sheet);
    const currentTop = existingTop[index];
    if (currentTop) currentTop.replaceWith(over);
    else overlayTop.append(over);
  });
  for (const extra of existing.slice(pages.length)) extra.remove();
  for (const extra of existingTop.slice(pages.length)) extra.remove();
}

export const SpikePages = Extension.create<{ doc: Y.Doc | null }>({
  name: "spikePages",
  addOptions() {
    return { doc: null };
  },
  addProseMirrorPlugins() {
    const ydoc = this.options.doc;
    if (!ydoc) return [];
    return [
      new Plugin<PagesState>({
        key,
        state: {
          init: () => ({ pages: [], decorations: DecorationSet.empty, dirtyFrom: 0 }),
          apply: (tr, old) => {
            const laid = tr.getMeta(key) as
              | { pages: Page[]; decorations: DecorationSet; done: boolean }
              | undefined;
            if (laid)
              return {
                pages: laid.pages,
                decorations: laid.decorations,
                dirtyFrom: laid.done ? null : old.dirtyFrom,
              };
            if (tr.getMeta("spikeRelayout")) return { ...old, pages: [], dirtyFrom: 0 };
            if (!tr.docChanged) return old;
            let from = Number.POSITIVE_INFINITY;
            tr.mapping.maps.forEach((map, index) => {
              map.forEach((start) => {
                // Back to where this step's range is in the final document.
                const mapped = tr.mapping.slice(index + 1).map(start, -1);
                from = Math.min(from, mapped);
              });
            });
            const pages = old.pages.map((page) => ({
              ...page,
              pos: tr.mapping.map(
                page.pos,
                page.kind === "block" || page.kind === "first" ? -1 : 1,
              ),
              ...(page.extra && { extra: page.extra.map((at) => tr.mapping.map(at, 1)) }),
            }));
            const dirty = Math.min(old.dirtyFrom ?? Number.POSITIVE_INFINITY, from);
            return {
              pages,
              decorations: old.decorations.map(tr.mapping, tr.doc),
              dirtyFrom: Number.isFinite(dirty) ? dirty : null,
            };
          },
        },
        props: {
          decorations: (state) => key.getState(state)?.decorations,
          attributes: () => ({ class: "spike-pages" }),
        },
        view: (view) => {
          const config = spikeConfig();
          if (config) config.stats = config.stats ?? emptyStats();
          const overlay = document.createElement("div");
          overlay.className = "spike-overlay";
          view.dom.parentElement?.prepend(overlay);
          const overlayTop = document.createElement("div");
          overlayTop.className = "spike-overlay spike-overlay--top";
          view.dom.parentElement?.append(overlayTop);
          let background: ReturnType<typeof setTimeout> | null = null;
          let scheduled = false;
          let dispatching = false;
          const onKey = () => {
            const stats = spikeConfig()?.stats;
            if (stats) stats.keydownAt = performance.now();
          };
          view.dom.addEventListener("keydown", onKey, true);
          view.dom.addEventListener("beforeinput", onKey, true);
          // ProseMirror ends a composition a little after compositionend: lay out once it has.
          const afterComposition = () => {
            if (view.composing) {
              setTimeout(afterComposition, 16);
              return;
            }
            if (key.getState(view.state)?.dirtyFrom !== null) schedule("composition");
          };
          const onCompositionEnd = () => setTimeout(afterComposition, 0);
          view.dom.addEventListener("compositionend", onCompositionEnd);

          const run = (reason: string, isBackground: boolean) => {
            scheduled = false;
            const config = spikeConfig();
            const state = key.getState(view.state);
            const look = config?.look;
            if (!config || !look || !state || state.dirtyFrom === null) return;
            if (view.composing) {
              config.stats.composingSkips++;
              return;
            }
            const started = performance.now();
            const looks = looksOf(view.state.doc, ydoc, look);
            const units = unitsOf(view.state.doc, looks);
            lastUnits = units;
            lastView = view;
            const measured = performance.now();
            const dirty = state.dirtyFrom;
            let fromPage = 0;
            for (let index = 0; index < state.pages.length; index++) {
              if ((state.pages[index]?.pos ?? 0) <= dirty) fromPage = index;
              else break;
            }
            fromPage = Math.max(0, fromPage - 1);
            // Pages starting after the edit, laid out before, can be taken as they were.
            let stableFrom = state.pages.findIndex((page) => page.pos > dirty);
            if (stableFrom < 0) stableFrom = state.pages.length;
            const limit = isBackground ? 40 : config.syncPages;
            const result = layOut(
              view,
              units,
              look,
              state.pages,
              fromPage,
              Math.max(stableFrom, fromPage + 1),
              limit,
            );
            const laidAt = performance.now();
            let pages = result.pages;
            if (!result.converged) {
              // The rest keeps its old breaks until the background reaches it.
              const lastNew = pages.at(-1);
              const tail = state.pages.filter((page) => lastNew && page.pos > lastNew.pos);
              pages = [...pages, ...tail];
            }
            const decorations = decorationsOf(view.state, look, pages);
            const tr = view.state.tr.setMeta(key, { pages, decorations, done: result.converged });
            tr.setMeta("addToHistory", false);
            dispatching = true;
            view.dispatch(tr);
            dispatching = false;
            drawOverlay(overlay, overlayTop, look, pages, view.dom.offsetTop, view.dom.offsetLeft);
            const finished = performance.now();
            config.stats.runs.push({
              reason,
              pages: pages.length,
              pagesLaidOut: result.laidOut,
              measureMs: measured - started,
              layoutMs: laidAt - measured,
              totalMs: finished - started,
              converged: result.converged,
              background: isBackground,
              at: finished,
            });
            if (!isBackground) config.stats.settledAt = finished;
            if (result.converged) {
              config.stats.backgroundDoneAt = finished;
            } else {
              const lastLaid = result.pages.at(-1);
              const continueFrom = lastLaid?.pos ?? 0;
              const current = key.getState(view.state);
              if (current) current.dirtyFrom = continueFrom;
              background = setTimeout(() => run("background", true), 0);
            }
          };
          const schedule = (reason: string) => {
            if (scheduled) return;
            scheduled = true;
            if (background) clearTimeout(background);
            background = null;
            queueMicrotask(() => run(reason, false));
          };
          return {
            update: (current) => {
              if (dispatching) return;
              const state = key.getState(current.state);
              const stats = spikeConfig()?.stats;
              if (stats) stats.updateAt = performance.now();
              if (state?.dirtyFrom !== null && state?.dirtyFrom !== undefined) schedule("edit");
            },
            destroy: () => {
              if (background) clearTimeout(background);
              overlay.remove();
              overlayTop.remove();
              view.dom.removeEventListener("keydown", onKey, true);
              view.dom.removeEventListener("beforeinput", onKey, true);
              view.dom.removeEventListener("compositionend", onCompositionEnd);
            },
          };
        },
      }),
    ];
  },
});

/** Forgets every measurement (after the look or the width changed) and lays every page out again. */
export function relayoutAll(view: EditorView): void {
  view.state.doc.descendants((node) => {
    heights.delete(node);
    lineCache.delete(node);
    return true;
  });
  noteHeights.clear();
  view.dispatch(view.state.tr.setMeta("spikeRelayout", true).setMeta("addToHistory", false));
}

export const spikePagesOf = (state: EditorState) => key.getState(state)?.pages ?? [];

let lastUnits: Unit[] = [];
let lastView: EditorView | null = null;

/** For debugging the spike: the units a page holds, with their heights. */
export function pageDetail(state: EditorState, page: number) {
  const pages = spikePagesOf(state);
  const at = pages[page - 1];
  const next = pages[page];
  if (!at) return null;
  const from = unitAt(lastUnits, at.pos);
  const to = next ? unitAt(lastUnits, next.pos) : lastUnits.length - 1;
  return {
    page: at,
    units: lastUnits.slice(from, to + 1).map((unit) => ({
      pos: unit.pos,
      top: unit.topLevel,
      type: unit.node.type.name,
      text: unit.node.textContent.slice(0, 30),
      height: Math.round((heights.get(unit.node) ?? -1) * 100) / 100,
      dom: (() => {
        const dom = lastView?.nodeDOM(unit.pos);
        if (!(dom instanceof HTMLElement)) return null;
        const rect = dom.getBoundingClientRect();
        const parent = dom.parentElement;
        return {
          tag: dom.tagName,
          h: Math.round(rect.height * 10) / 10,
          top: Math.round(rect.top),
          parent: parent?.tagName ?? "",
        };
      })(),
      look: unit.look
        ? `${unit.look.size}/${unit.look.lineMultiple}/${unit.look.before}/${unit.look.after}`
        : null,
    })),
  };
}
