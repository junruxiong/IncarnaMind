/**
 * Spike for #755, not for merging: a crude stand-in for #746's Word look, so pages can be
 * compared with LibreOffice's. Each paragraph mapped from the Word file (`wordSources`) is drawn
 * with its face, size, line and paragraph spacing, alignment and indents, as node decorations;
 * a paragraph the User made takes the look of the one before it. Space before and after is
 * padding, as Word adds them where CSS margins would collapse.
 */
import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { DOMSerializer } from "@tiptap/pm/model";
import { Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import type { NodeViewConstructor } from "@tiptap/pm/view";
import type * as Y from "yjs";
import type { SpikeBlockLook, SpikeLook } from "../../../../../spikes/755-page-view/wordLayout";
import { BLOCK_ID_ATTRIBUTE, MIND_WORD_SOURCES_FIELD } from "../../../../core/api";
import { spikeConfig } from "./spikeConfig";

const CJK_TEXT = /[　-鿿豈-﫿＀-￯]/;

const ratios = new Map<string, number>();

/** Loads every face the look names (they load lazily, and a measure before they have is wrong), then forgets old measures. */
export async function loadFaces(look: SpikeLook, fontMap: Record<string, string>): Promise<void> {
  const families = new Set<string>([look.footnote.family]);
  for (const block of Object.values(look.blocks)) families.add(block.family);
  const faces = new Set<string>();
  for (const family of families) {
    for (const name of family.split("|")) if (name) faces.add(fontMap[name] ?? name);
  }
  await Promise.all(
    [...faces].flatMap((face) =>
      ["400", "700"].map((weight) =>
        document.fonts.load(`${weight} 16px ${JSON.stringify(face)}`, "Hg中").catch(() => []),
      ),
    ),
  );
  ratios.clear();
}

/** The face's single line as a multiple of its size ("line-height: normal"), measured once. */
export function normalRatio(family: string, sample = "Hg"): number {
  const key = `${family}\u0000${sample}`;
  const known = ratios.get(key);
  if (known) return known;
  // A block's line box, which holds the line gap; a span's box is only the glyphs' extent.
  const probe = document.createElement("div");
  probe.style.cssText = `position:absolute;visibility:hidden;display:block;font-family:${family};font-size:1000px;line-height:normal;white-space:nowrap`;
  probe.textContent = sample;
  document.body.append(probe);
  const ratio = probe.getBoundingClientRect().height / 1000;
  probe.remove();
  ratios.set(key, ratio || 1.2);
  return ratio || 1.2;
}

export function facesOf(
  family: string,
  fontMap: Record<string, string>,
): {
  css: string;
  latin: string;
  east: string | null;
} {
  const [latin = "Calibri", east] = family.split("|");
  const map = (name: string) => fontMap[name] ?? name;
  const latinFace = map(latin);
  const eastFace = east && east !== latin ? map(east) : null;
  const css = [latinFace, eastFace]
    .filter((face): face is string => Boolean(face))
    .map((face) => JSON.stringify(face))
    .join(", ");
  return { css, latin: JSON.stringify(latinFace), east: eastFace && JSON.stringify(eastFace) };
}

/** The line height Word gives a paragraph, in px. */
export function lineHeightOf(
  look: SpikeBlockLook,
  grid: number | null,
  fontMap: Record<string, string>,
  hasCjk: boolean,
): number {
  const faces = facesOf(look.family, fontMap);
  const ratio = Math.max(
    normalRatio(faces.latin),
    hasCjk && faces.east ? normalRatio(faces.east, "中") : 0,
  );
  // Word and LibreOffice keep line heights in whole twips (1/20 pt; 1 px is 15 twips).
  const twips = (px: number) => Math.round(px * 15) / 15;
  const single = twips(ratio * look.size);
  if (look.lineExact !== null) return look.lineExact;
  if (look.lineAtLeast !== null) return Math.max(look.lineAtLeast, single);
  const multiple = look.lineMultiple ?? 1;
  if (grid && look.snapToGrid)
    return Math.max(1, Math.ceil(single / grid - 0.05)) * grid * multiple;
  return twips(single * multiple);
}

const ALIGN: Record<string, string> = {
  both: "justify",
  distribute: "justify",
  center: "center",
  right: "right",
  end: "right",
  left: "left",
  start: "left",
};

function styleOf(
  look: SpikeBlockLook,
  all: SpikeLook,
  fontMap: Record<string, string>,
  hasCjk: boolean,
) {
  const section = all.sections[look.section] ?? all.sections[0];
  const base = all.sections[0];
  const line = lineHeightOf(look, section?.grid ?? null, fontMap, hasCjk);
  const parts = [
    `font-family:${facesOf(look.family, fontMap).css}`,
    `font-size:${look.size}px`,
    `line-height:${line}px`,
    `padding-top:${look.before}px`,
    `padding-bottom:${look.after}px`,
    `letter-spacing:normal`,
    `font-variation-settings:normal`,
    `font-weight:${look.bold ? 700 : 400}`,
    `text-indent:${look.firstLine}px`,
  ];
  const shiftLeft = section && base ? section.left - base.left : 0;
  const shiftRight = section && base ? section.right - base.right : 0;
  if (look.indentLeft + shiftLeft) parts.push(`margin-left:${look.indentLeft + shiftLeft}px`);
  if (look.indentRight + shiftRight) parts.push(`margin-right:${look.indentRight + shiftRight}px`);
  const align = look.align ? ALIGN[look.align] : undefined;
  if (align) parts.push(`text-align:${align}`);
  return parts.join(";");
}

export interface LookedNode {
  look: SpikeBlockLook;
}

/** Each textblock's look by its id, the User's own paragraphs taking the one before's. */
export function looksOf(doc: ProseMirrorNode, ydoc: Y.Doc, look: SpikeLook) {
  const sources = ydoc.getMap(MIND_WORD_SOURCES_FIELD);
  const looks = new Map<ProseMirrorNode, SpikeBlockLook>();
  let previous: SpikeBlockLook | null = null;
  doc.descendants((node) => {
    if (!node.isTextblock) return true;
    const id = node.attrs[BLOCK_ID_ATTRIBUTE];
    const source = typeof id === "string" ? sources.get(id) : undefined;
    const place =
      source && typeof source === "object" && "n" in source && typeof source.n === "string"
        ? source.n
        : null;
    const own = place ? look.blocks[place] : undefined;
    const chosen = own ?? previous;
    if (chosen) looks.set(node, chosen);
    if (own) previous = own;
    return false;
  });
  return looks;
}

const key = new PluginKey<number>("spikeWordLook");

/** Each node's look by its id, for the node views (a node the User made: none). */
function lookOf(node: ProseMirrorNode, ydoc: Y.Doc): SpikeBlockLook | null {
  const config = spikeConfig();
  if (!config?.look) return null;
  const id = node.attrs[BLOCK_ID_ATTRIBUTE];
  const source = typeof id === "string" ? ydoc.getMap(MIND_WORD_SOURCES_FIELD).get(id) : undefined;
  const place =
    source && typeof source === "object" && "n" in source && typeof source.n === "string"
      ? source.n
      : null;
  return place ? (config.look.blocks[place] ?? null) : null;
}

const styleCache = new Map<string, string>();

function styleFor(node: ProseMirrorNode, ydoc: Y.Doc): string {
  const config = spikeConfig();
  const look = lookOf(node, ydoc);
  if (!config?.look || !look) return "";
  const cjk = CJK_TEXT.test(node.textContent);
  const cacheKey = `${JSON.stringify(look)}${cjk}`;
  let style = styleCache.get(cacheKey);
  if (!style) {
    style = styleOf(look, config.look, config.fontMap, cjk);
    styleCache.set(cacheKey, style);
  }
  return style;
}

/**
 * The look as node views for paragraphs and headings, not decorations: ProseMirror looks every
 * child's decorations up in its parent's list, so one node decoration per Block costs O(n²) on
 * each update of a long Mind.
 */
const liveViews = new Set<() => void>();

function lookView(ydoc: Y.Doc): NodeViewConstructor {
  return (initial) => {
    let node = initial;
    const spec = node.type.spec.toDOM?.(node);
    const { dom, contentDOM } = spec
      ? DOMSerializer.renderSpec(document, spec)
      : { dom: document.createElement("p"), contentDOM: undefined };
    const element = dom as HTMLElement;
    let applied = "";
    const apply = () => {
      const style = styleFor(node, ydoc);
      if (style !== applied) {
        element.style.cssText = style;
        applied = style;
      }
    };
    apply();
    const redraw = () => {
      applied = "";
      apply();
    };
    liveViews.add(redraw);
    return {
      dom: element,
      contentDOM: (contentDOM as HTMLElement | undefined) ?? element,
      update(next) {
        if (next.type !== node.type || next.attrs.level !== node.attrs.level) return false;
        node = next;
        apply();
        return true;
      },
      destroy() {
        liveViews.delete(redraw);
      },
    };
  };
}

export const SpikeWordLook = Extension.create<{ doc: Y.Doc | null }>({
  name: "spikeWordLook",
  addOptions() {
    return { doc: null };
  },
  addProseMirrorPlugins() {
    const ydoc = this.options.doc;
    if (!ydoc) return [];
    const view = lookView(ydoc);
    return [
      new Plugin<number>({
        key,
        state: {
          init: () => 0,
          apply: (tr, generation) => (tr.getMeta(key) === "refresh" ? generation + 1 : generation),
        },
        props: {
          nodeViews: { paragraph: view, heading: view },
          attributes: () => ({ class: "spike-look" }),
        },
        view: () => {
          let seen = 0;
          return {
            update: (current) => {
              const generation = key.getState(current.state) ?? 0;
              if (generation === seen) return;
              seen = generation;
              styleCache.clear();
              for (const redraw of liveViews) redraw();
            },
          };
        },
      }),
    ];
  },
});

export const refreshSpikeLook = (tr: Transaction) => tr.setMeta(key, "refresh");
