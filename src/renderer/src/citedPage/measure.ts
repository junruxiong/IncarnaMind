import type { PageBox, PageText } from "../../../shared/citedPage";
import type { CitedArea } from "../viewer/citedArea";

/** The box an area takes in the window, in CSS pixels; null while none of it is drawn. */
export function boundsOf(area: CitedArea): DOMRect | null {
  const rects = (area instanceof Range ? [area] : area)
    .map((each) => each.getBoundingClientRect())
    .filter((rect) => rect.width > 0 && rect.height > 0);
  if (rects.length === 0) return null;
  const left = Math.min(...rects.map((rect) => rect.left));
  const top = Math.min(...rects.map((rect) => rect.top));
  const right = Math.max(...rects.map((rect) => rect.right));
  const bottom = Math.max(...rects.map((rect) => rect.bottom));
  return new DOMRect(left, top, right - left, bottom - top);
}

/** The area's first element, to scroll it into view. */
export function firstOf(area: CitedArea): Element | null {
  if (!(area instanceof Range)) return area[0] ?? null;
  const node = area.startContainer.childNodes[area.startOffset] ?? area.startContainer;
  return node instanceof Element ? node : node.parentElement;
}

/** The pictures in an area, to wait for. */
export function imagesIn(area: CitedArea): HTMLImageElement[] {
  if (area instanceof Range) {
    const root = area.commonAncestorContainer;
    const scope = root instanceof Element ? root : root.parentElement;
    return [...(scope?.querySelectorAll("img") ?? [])].filter((image) =>
      area.intersectsNode(image),
    );
  }
  return area.flatMap((element) => [
    ...(element instanceof HTMLImageElement ? [element] : []),
    ...element.querySelectorAll("img"),
  ]);
}

/**
 * Whether a text node isn't part of the text quotes are found in: what the
 * viewer leaves out when it finds a quote, and a footnote's number, set as a
 * superscript in the text, which the checked text doesn't have.
 */
function leftOut(node: Text): boolean {
  const parent = node.parentElement;
  if (parent?.closest("[data-quote-skip]")) return true;
  const sup = parent?.closest("sup");
  return Boolean(sup && /^\s*\d{1,3}\s*$/.test(sup.textContent ?? ""));
}

/** The text in an area, each node with the part of it inside. */
function textNodesIn(area: CitedArea): { node: Text; start: number; end: number }[] {
  const found: { node: Text; start: number; end: number }[] = [];
  const roots = area instanceof Range ? [area.commonAncestorContainer] : area;
  for (const root of roots) {
    const walker = root.ownerDocument?.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes: Node[] = root instanceof Text ? [root] : [];
    for (let node = walker?.nextNode(); node; node = walker?.nextNode()) nodes.push(node);
    for (const node of nodes) {
      if (!(node instanceof Text) || node.data.trim() === "" || leftOut(node)) continue;
      if (area instanceof Range && !area.intersectsNode(node)) continue;
      const start = area instanceof Range && node === area.startContainer ? area.startOffset : 0;
      const end =
        area instanceof Range && node === area.endContainer ? area.endOffset : node.length;
      found.push({ node, start, end });
    }
  }
  return found;
}

/**
 * The text drawn in an area, in reading order: a run for each line of each
 * text node, boxed from `rect`'s corner. A run is followed by a break where
 * the next one starts a line or stands apart, as a sheet's next cell does, so
 * a quote is found across lines and cells as it is in a PDF's text runs.
 */
export function textIn(area: CitedArea, rect: PageBox): PageText[] {
  const runs: { node: Text; text: string; box: DOMRect }[] = [];
  const range = document.createRange();
  for (const { node, start, end } of textNodesIn(area)) {
    const words = /\S+\s*/g;
    words.lastIndex = start;
    for (let word = words.exec(node.data); word && word.index < end; word = words.exec(node.data)) {
      const to = Math.min(end, word.index + word[0].length);
      range.setStart(node, word.index);
      range.setEnd(node, to);
      const box = range.getBoundingClientRect();
      if (box.width === 0 || box.height === 0) continue;
      const last = runs.at(-1);
      const sameLine =
        last?.node === node &&
        Math.abs(box.top - last.box.top) < box.height / 2 &&
        box.left >= last.box.right - 1;
      if (last && sameLine) {
        last.text += node.data.slice(word.index, to);
        last.box = new DOMRect(
          Math.min(last.box.left, box.left),
          Math.min(last.box.top, box.top),
          Math.max(last.box.right, box.right) - Math.min(last.box.left, box.left),
          Math.max(last.box.bottom, box.bottom) - Math.min(last.box.top, box.top),
        );
      } else runs.push({ node, text: node.data.slice(word.index, to), box });
    }
  }
  return runs.flatMap(({ text, box }, index): PageText[] => {
    const inside =
      box.right > rect.x &&
      box.left < rect.x + rect.width &&
      box.bottom > rect.y &&
      box.top < rect.y + rect.height;
    if (!inside) return [];
    const next = runs[index + 1]?.box;
    const apart =
      !next ||
      next.top >= box.top + box.height / 2 ||
      next.left < box.left ||
      next.left > box.right + 1;
    return [
      {
        text,
        ...(apart && { breakAfter: true }),
        box: { x: box.left - rect.x, y: box.top - rect.y, width: box.width, height: box.height },
      },
    ];
  });
}
