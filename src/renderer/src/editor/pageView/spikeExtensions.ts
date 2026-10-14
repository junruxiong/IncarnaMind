/**
 * Spike for #755, not for merging: the extensions each approach adds to a Mind's editor, and
 * the window API its e2e spec measures through (`window.__spike755Api`).
 */
import "./spike.css";
import type { Editor, Extensions } from "@tiptap/core";
import { PaginationPlus } from "tiptap-pagination-plus";
import type * as Y from "yjs";
import {
  PAGE_GAP,
  pageDetail,
  pageGeometry,
  relayoutAll,
  SpikePages,
  spikePagesOf,
} from "./customPages";
import { emptyStats, spikeConfig, spikeMode } from "./spikeConfig";
import { loadFaces, normalRatio, refreshSpikeLook, SpikeWordLook } from "./wordLookSpike";

export function spikeExtensions(doc: Y.Doc): Extensions {
  const mode = spikeMode();
  if (mode === "off") return [];
  const look = SpikeWordLook.configure({ doc });
  if (mode === "look") return [look];
  if (mode === "custom") return [look, SpikePages.configure({ doc })];
  const section = spikeConfig()?.look?.sections[0];
  if (!section) return [look];
  const [headerLeft = "", headerRight = ""] = (section.headers.default ?? "")
    .split("\t")
    .filter(Boolean);
  const footer = (section.footers.default ?? "").replaceAll("{pages}", "{total}");
  return [
    look,
    PaginationPlus.configure({
      pageHeight: section.height,
      pageWidth: section.width,
      marginTop: section.top,
      marginBottom: section.bottom,
      marginLeft: section.left,
      marginRight: section.right,
      pageGap: PAGE_GAP,
      contentMarginTop: 0,
      contentMarginBottom: 0,
      headerLeft,
      headerRight,
      footerLeft: "",
      footerRight: footer,
    }),
  ];
}

interface Region {
  page: number;
  top: number;
  bottom: number;
}

/** Each page's content region, in client coordinates. */
function regionsOf(editor: Editor): Region[] {
  const root = editor.view.dom;
  const rootRect = root.getBoundingClientRect();
  const mode = spikeMode();
  const look = spikeConfig()?.look;
  if (mode === "custom" && look) {
    const pages = spikePagesOf(editor.state);
    const zoom =
      Number(root.closest<HTMLElement>("[data-spike-scale]")?.dataset.spikeScale ?? 0) ||
      ((root as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1);
    return pageGeometry(look, pages).map((geometry, index) => ({
      page: index + 1,
      top: rootRect.top + (geometry.top + geometry.section.top) * zoom - 1,
      bottom:
        rootRect.top +
        (geometry.top + geometry.section.height - geometry.section.bottom) * zoom +
        1,
    }));
  }
  if (mode === "plus") {
    const breakers = [...root.querySelectorAll<HTMLElement>(".rm-page-break .breaker")];
    const regions: Region[] = [];
    let top = rootRect.top;
    breakers.forEach((breaker, index) => {
      const rect = breaker.getBoundingClientRect();
      regions.push({ page: index + 1, top, bottom: rect.top });
      top = rect.bottom;
    });
    return regions;
  }
  return [{ page: 1, top: rootRect.top, bottom: rootRect.bottom }];
}

const SKIP =
  "[data-spike-float], [data-spike-gap], .word-atom-label, .rm-page-break, [data-rm-pagination], .word-atom-hidden";

/** The start of each page's first line of text, as drawn. */
function firstLines(editor: Editor, length = 60): { page: number; text: string }[] {
  const regions = regionsOf(editor);
  const root = editor.view.dom;
  const found = new Map<number, { node: Text; offset: number }>();
  const texts: Text[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement?.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const range = document.createRange();
  const regionAt = (y: number) => {
    let low = 0;
    let high = regions.length - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      const region = regions[middle] as Region;
      if (y < region.top) high = middle - 1;
      else if (y > region.bottom) low = middle + 1;
      else return region;
    }
    return null;
  };
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node as Text;
    if (!text.data.trim()) continue;
    texts.push(text);
    range.selectNodeContents(text);
    for (const rect of range.getClientRects()) {
      if (rect.height === 0) continue;
      const region = regionAt((rect.top + rect.bottom) / 2);
      if (!region || found.has(region.page)) continue;
      // The first character of this text node inside the region.
      let low = 0;
      let high = text.length - 1;
      while (low < high) {
        const middle = (low + high) >> 1;
        range.setStart(text, middle);
        range.setEnd(text, middle + 1);
        const char = range.getClientRects()[0];
        const y = char ? (char.top + char.bottom) / 2 : 0;
        if (y < region.top) low = middle + 1;
        else high = middle;
      }
      found.set(region.page, { node: text, offset: low });
      range.selectNodeContents(text);
    }
  }
  return [...found.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([page, { node, offset }]) => {
      let text = node.data.slice(offset);
      for (
        let index = texts.indexOf(node) + 1;
        text.length < length && index < texts.length;
        index++
      ) {
        text += (texts[index] as Text).data;
      }
      return { page, text: text.slice(0, length) };
    });
}

function pageCount(editor: Editor): number {
  const mode = spikeMode();
  if (mode === "custom") return spikePagesOf(editor.state).length;
  if (mode === "plus") return editor.view.dom.querySelectorAll(".rm-page-break").length;
  return 0;
}

/** A position near the start of a page's text, or in its middle. */
function positionOnPage(editor: Editor, page: number, where: "start" | "middle"): number {
  const region = regionsOf(editor).find((each) => each.page === page);
  if (!region) throw new Error(`No page ${page}.`);
  const root = editor.view.dom;
  const left = root.getBoundingClientRect().left;
  const y = where === "start" ? region.top + 30 : (region.top + region.bottom) / 2;
  const scroller = root.closest(".mind-scroller") ?? document.scrollingElement;
  // posAtCoords needs the point on screen.
  if (scroller instanceof HTMLElement) {
    scroller.scrollTop += y - scroller.getBoundingClientRect().top - 200;
  }
  const after = regionsOf(editor).find((each) => each.page === page) ?? region;
  const target = where === "start" ? after.top + 30 : (after.top + after.bottom) / 2;
  const rootRect = root.getBoundingClientRect();
  const style = getComputedStyle(root);
  const x = left + (Number.parseFloat(style.paddingLeft) || 0) + 120;
  const at = editor.view.posAtCoords({ left: Math.min(x, rootRect.right - 40), top: target });
  if (!at) throw new Error(`No position on page ${page}.`);
  return at.pos;
}

function caret(editor: Editor) {
  const coords = editor.view.coordsAtPos(editor.state.selection.head);
  const scroller = editor.view.dom.closest(".mind-scroller");
  return {
    top: coords.top,
    left: coords.left,
    scrollTop: scroller instanceof HTMLElement ? scroller.scrollTop : 0,
    head: editor.state.selection.head,
  };
}

/** Which page each footnote's mark is on, and which page its note is drawn on (custom only). */
function footnotePlaces(editor: Editor) {
  const regions = regionsOf(editor);
  const pages = spikePagesOf(editor.state);
  const pageOf = (y: number) =>
    regions.find((region) => y >= region.top && y <= region.bottom)?.page ?? null;
  const marks = [
    ...editor.view.dom.querySelectorAll<HTMLElement>('.word-atom-note[data-kind="footnote"]'),
  ].map((mark) => {
    const rect = mark.getBoundingClientRect();
    return { label: mark.textContent ?? "", page: pageOf((rect.top + rect.bottom) / 2) };
  });
  const drawn = new Map<string, number>();
  pages.forEach((page, index) => {
    for (const note of page.notes) drawn.set(note.label, index + 1);
  });
  return marks.map((mark) => ({ ...mark, notePage: drawn.get(mark.label) ?? null }));
}

/** Tracks the frames after a keystroke: when the first frame after it is drawn. */
function trackFrames(editor: Editor) {
  const config = spikeConfig();
  if (!config) return () => undefined;
  const frames: { keydownAt: number; frameAt: number }[] = [];
  let pending: number | null = null;
  const onKey = () => {
    pending = performance.now();
  };
  const onUpdate = () => {
    const started = pending;
    if (started === null) return;
    pending = null;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => frames.push({ keydownAt: started, frameAt: performance.now() }));
    });
  };
  editor.view.dom.addEventListener("keydown", onKey, true);
  editor.on("update", onUpdate);
  (config as unknown as { frames: typeof frames }).frames = frames;
  return () => {
    editor.view.dom.removeEventListener("keydown", onKey, true);
    editor.off("update", onUpdate);
  };
}

export function installSpikeHarness(editor: Editor): () => void {
  if (spikeMode() === "off" || !spikeConfig()) return () => undefined;
  const config = spikeConfig();
  if (config) config.stats ??= emptyStats();
  const stop = trackFrames(editor);
  const api = {
    async apply() {
      const config = spikeConfig();
      if (config?.look) await loadFaces(config.look, config.fontMap);
      const section = config?.look?.sections[0];
      const root = editor.view.dom;
      if (section) {
        root.style.setProperty("--spike-page-width", `${section.width}px`);
        root.style.setProperty("--spike-top", spikeMode() === "plus" ? "0px" : `${section.top}px`);
        root.style.setProperty("--spike-left", `${section.left}px`);
        root.style.setProperty("--spike-right", `${section.right}px`);
        root.style.setProperty("--spike-bottom", `${section.bottom}px`);
        root
          .closest<HTMLElement>(".mind-measure")
          ?.style.setProperty("--spike-page-width", `${section.width}px`);
      }
      const started = performance.now();
      editor.view.dispatch(refreshSpikeLook(editor.state.tr));
      if (spikeMode() === "custom") relayoutAll(editor.view);
      if (spikeMode() === "plus")
        editor.view.dispatch(editor.state.tr.setMeta("PAGE_COUNT_META_KEY", {}));
      return started;
    },
    relayout() {
      const started = performance.now();
      relayoutAll(editor.view);
      return started;
    },
    pages: () => pageCount(editor),
    scrollToPage(page: number) {
      const region = regionsOf(editor).find((each) => each.page === page);
      const scroller = editor.view.dom.closest(".mind-scroller");
      if (region && scroller instanceof HTMLElement) {
        scroller.scrollTop += region.top - scroller.getBoundingClientRect().top - 120;
      }
    },
    /** Export to PDF from the same pages: no gap between sheets, a sheet to a printed page, nothing else printed. */
    printMode(on: boolean) {
      const config = spikeConfig();
      const section = config?.look?.sections[0];
      if (!config || !section) return 0;
      config.gap = on ? 0 : PAGE_GAP;
      document.documentElement.classList.toggle("spike-printing", on);
      // The first sheet's top on the first printed page's top.
      document.documentElement.style.setProperty(
        "--spike-print-offset",
        `${-editor.view.dom.offsetTop}px`,
      );
      let style = document.getElementById("spike-page-size");
      if (!style) {
        style = document.createElement("style");
        style.id = "spike-page-size";
        document.head.append(style);
      }
      style.textContent = on
        ? `@page { size: ${section.width}px ${section.height}px; margin: 0 }`
        : "";
      const started = performance.now();
      relayoutAll(editor.view);
      return started;
    },
    pageStart: (page: number) => spikePagesOf(editor.state)[page - 1]?.pos ?? null,
    zoom(factor: number, how: "zoom" | "transform" = "zoom") {
      const measure = editor.view.dom.closest<HTMLElement>(".mind-measure");
      if (!measure) return;
      measure.style.zoom = "";
      measure.style.transform = "";
      delete measure.dataset.spikeScale;
      if (factor === 1) return;
      if (how === "zoom") measure.style.zoom = String(factor);
      else {
        measure.style.transformOrigin = "top center";
        measure.style.transform = `scale(${factor})`;
        measure.dataset.spikeScale = String(factor);
      }
    },
    /** Where each page's first position is drawn, against where its sheet's text starts: the first page off by more than 2 px. */
    alignment() {
      const pages = spikePagesOf(editor.state);
      const look = spikeConfig()?.look;
      if (!look) return null;
      const root = editor.view.dom;
      const rootTop = root.getBoundingClientRect().top;
      const zoom = (root as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1;
      const geometry = pageGeometry(look, pages);
      const out: { page: number; delta: number; kind: string }[] = [];
      pages.forEach((page, index) => {
        if (index === 0) return;
        const at = geometry[index];
        if (!at) return;
        const expected = rootTop + (at.top + at.section.top) * zoom;
        let actual: number;
        try {
          actual = editor.view.coordsAtPos(
            Math.min(page.pos + (page.kind === "block" ? 1 : 0), editor.state.doc.content.size),
          ).top;
        } catch {
          return;
        }
        out.push({
          page: index + 1,
          delta: Math.round((actual - expected) / zoom),
          kind: page.kind,
        });
      });
      const bad = out.filter((each) => Math.abs(each.delta) > 40);
      return { checked: out.length, off: bad.length, first: bad.slice(0, 5) };
    },
    docText: () =>
      editor.state.doc.textContent.length + ":" + editor.state.doc.textContent.slice(0, 50),
    firstLines: (length?: number) => firstLines(editor, length),
    positionOnPage: (page: number, where: "start" | "middle") =>
      positionOnPage(editor, page, where),
    select(pos: number) {
      editor.commands.setTextSelection(pos);
      editor.view.focus();
    },
    caret: () => caret(editor),
    footnotes: () => footnotePlaces(editor),
    stats: () => spikeConfig()?.stats,
    frames: () => (spikeConfig() as unknown as { frames?: unknown[] } | null)?.frames ?? [],
    resetStats() {
      const current = spikeConfig();
      if (!current) return;
      current.stats = emptyStats();
      const frames = (current as unknown as { frames?: unknown[] }).frames;
      if (frames) frames.length = 0;
    },
    textAround: (pos: number) =>
      editor.state.doc.textBetween(
        Math.max(0, pos - 20),
        Math.min(editor.state.doc.content.size, pos + 20),
        " ",
        " ",
      ),
    docSize: () => editor.state.doc.content.size,
    pageDetail: (page: number) => pageDetail(editor.state, page),
    lineCounts() {
      return [...editor.view.dom.querySelectorAll<HTMLElement>(":scope > p")].map((block) => {
        const style = getComputedStyle(block);
        const floats = [...block.querySelectorAll<HTMLElement>("[data-spike-float]")].reduce(
          (sum, float) => sum + float.getBoundingClientRect().height,
          0,
        );
        const content =
          block.getBoundingClientRect().height -
          floats -
          Number.parseFloat(style.paddingTop) -
          Number.parseFloat(style.paddingBottom);
        return {
          text: (block.textContent ?? "").slice(0, 14),
          lines: Math.round((content / Number.parseFloat(style.lineHeight)) * 100) / 100,
        };
      });
    },
    kerning(on: boolean) {
      editor.view.dom.style.fontKerning = on ? "" : "none";
    },
    debug() {
      const blocks = [...editor.view.dom.querySelectorAll<HTMLElement>("p, h1, h2, h3, li")].slice(
        0,
        40,
      );
      return {
        carlito: normalRatio('"Carlito"'),
        carlitoLoaded: document.fonts.check('16px "Carlito"'),
        blocks: blocks.map((block) => {
          const style = getComputedStyle(block);
          return {
            tag: block.tagName,
            text: (block.textContent ?? "").slice(0, 40),
            family: style.fontFamily,
            size: style.fontSize,
            line: style.lineHeight,
            pt: style.paddingTop,
            pb: style.paddingBottom,
            mt: style.marginTop,
            mb: style.marginBottom,
            height: block.getBoundingClientRect().height,
          };
        }),
      };
    },
    blockCount: () => editor.state.doc.childCount,
  };
  (window as unknown as { __spike755Api?: typeof api }).__spike755Api = api;
  return () => {
    stop();
    delete (window as unknown as { __spike755Api?: typeof api }).__spike755Api;
  };
}
