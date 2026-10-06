import type { PDFDocumentLoadingTask, PDFDocumentProxy, RenderTask, TextLayer } from "pdfjs-dist";
import {
  type CSSProperties,
  type KeyboardEvent,
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { Document } from "../../../core/api";
import { findQuoteInPieces, type TextPiece } from "../../../shared/quoteMatch";
import {
  ChevronDownIcon,
  ChevronUpIcon,
  FitWidthIcon,
  MinusIcon,
  PlusIcon,
} from "../components/icons";
import { useT } from "../i18n";
import type { ViewerTarget } from "../store";
import { isMissingDocumentError, loadPdfJs, openPdf, type PdfJs } from "./pdfjs";
import { DocumentRemoved, ViewerMessage } from "./ViewerMessage";

/** CSS pixels per PDF point: at 100% a page shows at its printed size. */
const CSS_UNITS = 96 / 72;
const ZOOM_STEPS = [0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4];
const MIN_ZOOM = ZOOM_STEPS[0] as number;
const MAX_ZOOM = ZOOM_STEPS.at(-1) as number;
/** Space around and between pages, in CSS pixels. */
const PAGE_GAP = 12;
/** A canvas larger than this many pixels is drawn at a lower resolution, to bound memory. */
const MAX_CANVAS_PIXELS = 2 ** 24;
/** A zoom or resize re-renders visible pages once it settles, not at every step. */
const RERENDER_DELAY = 150;
/** The most pages a quote is looked for across. */
const MAX_QUOTE_PAGES = 10;

/** A page's size at 100% scale, in PDF points. */
interface PageSize {
  width: number;
  height: number;
}

/** Part of a quote, in one of a page's text runs (`item` indexes its text layer's runs). */
interface RunHighlight {
  item: number;
  start: number;
  end: number;
}

type Zoom = { fit: true } | { fit: false; zoom: number };

type Loaded =
  | { kind: "loading" }
  | { kind: "ready"; pdfjs: PdfJs; pdf: PDFDocumentProxy; firstPage: PageSize }
  | { kind: "missing" }
  | { kind: "failed"; message: string };

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

/**
 * A PDF Document, rendered with pdf.js: pages are drawn only while near the
 * visible area, each with a text layer so its text can be selected. Opened at
 * a page range with a quote, it shows the first page of the range and
 * highlights the quote if those pages contain it.
 */
export function PdfView({ document, target }: { document: Document; target: ViewerTarget }) {
  const t = useT();
  const [loaded, setLoaded] = useState<Loaded>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let task: PDFDocumentLoadingTask | undefined;
    loadPdfJs()
      .then(async (pdfjs) => {
        if (cancelled) return;
        task = openPdf(pdfjs, document.id);
        const pdf = await task.promise;
        const first = (await pdf.getPage(1)).getViewport({ scale: 1 });
        if (!cancelled) {
          setLoaded({
            kind: "ready",
            pdfjs,
            pdf,
            firstPage: { width: first.width, height: first.height },
          });
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        if (isMissingDocumentError(error)) setLoaded({ kind: "missing" });
        else setLoaded({ kind: "failed", message: error instanceof Error ? error.message : "" });
      });
    return () => {
      cancelled = true;
      void task?.destroy();
    };
  }, [document.id]);

  switch (loaded.kind) {
    case "loading":
      return <ViewerMessage>{t("viewer.loading")}</ViewerMessage>;
    case "missing":
      return <DocumentRemoved quote={target.quote} />;
    case "failed":
      return (
        <ViewerMessage tone="error">
          {t("viewer.failed", { message: loaded.message })}
        </ViewerMessage>
      );
    case "ready":
      return (
        <PdfPages
          pdfjs={loaded.pdfjs}
          pdf={loaded.pdf}
          firstPage={loaded.firstPage}
          target={target}
        />
      );
  }
}

interface PdfPagesProps {
  pdfjs: PdfJs;
  pdf: PDFDocumentProxy;
  firstPage: PageSize;
  target: ViewerTarget;
}

function PdfPages({ pdfjs, pdf, firstPage, target }: PdfPagesProps) {
  const pageCount = pdf.numPages;
  // Every page starts at the first page's size; each is corrected once it loads.
  const [sizes, setSizes] = useState<readonly PageSize[]>(() =>
    Array.from({ length: pageCount }, () => firstPage),
  );
  const [zoom, setZoom] = useState<Zoom>({ fit: true });
  const [viewportWidth, setViewportWidth] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [nearby, setNearby] = useState<ReadonlySet<number>>(() => new Set());
  const [highlights, setHighlights] = useState<ReadonlyMap<number, RunHighlight[]>>(
    () => new Map(),
  );

  const scroller = useRef<HTMLDivElement>(null);
  const pageElements = useRef<(HTMLElement | null)[]>([]);
  /** The page at the top edge of the view, and how far into it: kept through zooms and resizes. */
  const anchor = useRef({ page: 1, fraction: 0 });
  /** Set when the view was moved to a page, so the page indicator shows that page. */
  const placed = useRef<{ page: number; scrollTop: number } | null>(null);
  /** The open request whose highlight the view should move to once it is drawn. */
  const pendingHighlight = useRef<number | null>(null);
  const frame = useRef(0);

  const scale =
    zoom.fit && viewportWidth > 0
      ? clamp(
          (viewportWidth - 2 * PAGE_GAP) / firstPage.width,
          MIN_ZOOM * CSS_UNITS,
          MAX_ZOOM * CSS_UNITS,
        )
      : zoom.fit
        ? CSS_UNITS
        : zoom.zoom * CSS_UNITS;

  const pageElement = (page: number) => pageElements.current[page - 1] ?? null;

  /** Scrolls so `page` starts at the top of the view. */
  const goToPage = useCallback(
    (page: number) => {
      const container = scroller.current;
      const element = pageElements.current[clamp(page, 1, pageCount) - 1];
      if (!container || !element) return;
      const scrollTop = Math.max(0, element.offsetTop - PAGE_GAP);
      container.scrollTop = scrollTop;
      const number = Number(element.dataset.pageNumber);
      placed.current = { page: number, scrollTop: container.scrollTop };
      anchor.current = {
        page: number,
        fraction: (container.scrollTop - element.offsetTop) / element.offsetHeight,
      };
      setCurrentPage(number);
    },
    [pageCount],
  );

  /** Works out the current page (the one taking most of the view) and the anchor. */
  const measure = useCallback(() => {
    const container = scroller.current;
    if (!container) return;
    const { scrollTop, clientHeight } = container;
    const elements = pageElements.current;
    // The first page whose bottom is below the top of the view.
    let low = 0;
    let high = elements.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      const element = elements[middle];
      if (element && element.offsetTop + element.offsetHeight <= scrollTop) low = middle + 1;
      else high = middle;
    }
    const top = elements[low];
    if (!top) return;
    anchor.current = {
      page: low + 1,
      fraction: (scrollTop - top.offsetTop) / top.offsetHeight,
    };
    if (placed.current && Math.abs(placed.current.scrollTop - scrollTop) < 2) {
      setCurrentPage(placed.current.page);
      return;
    }
    placed.current = null;
    let best = low + 1;
    let bestVisible = -1;
    for (let index = low; index < elements.length; index++) {
      const element = elements[index];
      if (!element || element.offsetTop >= scrollTop + clientHeight) break;
      const visible =
        Math.min(element.offsetTop + element.offsetHeight, scrollTop + clientHeight) -
        Math.max(element.offsetTop, scrollTop);
      if (visible > bestVisible) {
        best = index + 1;
        bestVisible = visible;
      }
    }
    setCurrentPage(best);
  }, []);

  const onScroll = () => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      measure();
    });
  };
  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  // The view's width, for fitting pages to it.
  useLayoutEffect(() => {
    const container = scroller.current;
    if (!container) return;
    setViewportWidth(container.clientWidth);
    const observer = new ResizeObserver(() => setViewportWidth(container.clientWidth));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // When pages change size (a zoom, a resize, a page's real size arriving), keep the same place in view.
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs because the layout changed
  useLayoutEffect(() => {
    const container = scroller.current;
    const element = pageElement(anchor.current.page);
    if (!container || !element) return;
    container.scrollTop = element.offsetTop + anchor.current.fraction * element.offsetHeight;
    if (placed.current) placed.current.scrollTop = container.scrollTop;
  }, [scale, sizes]);

  // Each open request goes to the first page of its range (a plain reopen keeps the place).
  const appliedRequest = useRef<number | null>(null);
  useLayoutEffect(() => {
    if (appliedRequest.current === target.request) return;
    const first = appliedRequest.current === null;
    appliedRequest.current = target.request;
    if (target.pageFrom !== undefined || target.quote || first) goToPage(target.pageFrom ?? 1);
  }, [target.request, target.pageFrom, target.quote, goToPage]);

  // Finds the quote on the pages of the range, for each page's text layer to highlight.
  useEffect(() => {
    const quote = target.quote;
    setHighlights(new Map());
    pendingHighlight.current = null;
    if (!quote) return;
    let cancelled = false;
    const from = clamp(target.pageFrom ?? 1, 1, pageCount);
    const to = clamp(target.pageTo ?? from, from, Math.min(pageCount, from + MAX_QUOTE_PAGES - 1));
    void (async () => {
      const pieces: TextPiece[] = [];
      const runs: { page: number; item: number }[] = [];
      for (let page = from; page <= to; page++) {
        const content = await (await pdf.getPage(page)).getTextContent();
        let item = 0;
        for (const entry of content.items) {
          // Text runs only, counted as the text layer counts them.
          if (!("str" in entry)) continue;
          pieces.push({ text: entry.str, breakAfter: entry.hasEOL });
          runs.push({ page, item: item++ });
        }
        const last = pieces.at(-1);
        if (last) last.breakAfter = true;
      }
      if (cancelled) return;
      const found = findQuoteInPieces(pieces, quote);
      if (!found) return;
      const byPage = new Map<number, RunHighlight[]>();
      for (const part of found) {
        const run = runs[part.piece];
        if (!run) continue;
        const list = byPage.get(run.page) ?? [];
        list.push({ item: run.item, start: part.start, end: part.end });
        byPage.set(run.page, list);
      }
      pendingHighlight.current = target.request;
      setHighlights(byPage);
    })().catch((error: unknown) => {
      if (!cancelled) console.error(error);
    });
    return () => {
      cancelled = true;
    };
  }, [pdf, pageCount, target.request, target.quote, target.pageFrom, target.pageTo]);

  // Pages within a screen of the view are drawn; the rest are released.
  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const observer = new IntersectionObserver(
      (entries) =>
        setNearby((previous) => {
          const next = new Set(previous);
          for (const entry of entries) {
            const page = Number((entry.target as HTMLElement).dataset.pageNumber);
            if (entry.isIntersecting) next.add(page);
            else next.delete(page);
          }
          return next;
        }),
      { root, rootMargin: "100% 0px" },
    );
    for (const element of pageElements.current) if (element) observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const onPageSize = useCallback((page: number, size: PageSize) => {
    setSizes((previous) => {
      const old = previous[page - 1];
      if (old && old.width === size.width && old.height === size.height) return previous;
      const next = previous.slice();
      next[page - 1] = size;
      return next;
    });
  }, []);

  const firstHighlightPage = highlights.size > 0 ? Math.min(...highlights.keys()) : null;
  const onHighlightShown = useCallback(
    (page: number, element: HTMLElement) => {
      const container = scroller.current;
      if (!container || page !== firstHighlightPage) return;
      if (pendingHighlight.current !== target.request) return;
      pendingHighlight.current = null;
      const box = container.getBoundingClientRect();
      const rect = element.getBoundingClientRect();
      if (rect.top >= box.top && rect.bottom <= box.bottom) return; // already in view
      container.scrollTop = rect.top - box.top + container.scrollTop - container.clientHeight / 3;
      placed.current = { page, scrollTop: container.scrollTop };
      setCurrentPage(page);
    },
    [firstHighlightPage, target.request],
  );

  const setPageElement = useCallback((page: number, element: HTMLElement | null) => {
    pageElements.current[page - 1] = element;
  }, []);

  const currentZoom = scale / CSS_UNITS;
  const zoomTo = (next: number) => {
    measure(); // so the anchor is current
    setZoom({ fit: false, zoom: clamp(next, MIN_ZOOM, MAX_ZOOM) });
  };

  return (
    <div className="flex h-full flex-col">
      <Toolbar
        currentPage={currentPage}
        pageCount={pageCount}
        zoomPercent={Math.round(currentZoom * 100)}
        fitWidth={zoom.fit}
        onGoToPage={goToPage}
        onZoomIn={() => zoomTo(ZOOM_STEPS.find((step) => step > currentZoom + 0.001) ?? MAX_ZOOM)}
        onZoomOut={() =>
          zoomTo(ZOOM_STEPS.findLast((step) => step < currentZoom - 0.001) ?? MIN_ZOOM)
        }
        onFitWidth={() => {
          measure();
          setZoom({ fit: true });
        }}
      />
      <div
        ref={scroller}
        data-testid="pdf-scroller"
        onScroll={onScroll}
        className="relative min-h-0 flex-1 overflow-auto bg-gray-100 [overflow-anchor:none]"
      >
        <div
          className="flex w-max min-w-full flex-col items-center"
          style={{ gap: PAGE_GAP, padding: PAGE_GAP }}
        >
          {sizes.map((size, index) => {
            const page = index + 1;
            return (
              <PdfPage
                key={page}
                pdfjs={pdfjs}
                pdf={pdf}
                page={page}
                size={size}
                scale={scale}
                nearby={nearby.has(page)}
                highlights={highlights.get(page)}
                onSize={onPageSize}
                onHighlightShown={onHighlightShown}
                setElement={setPageElement}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface ToolbarProps {
  currentPage: number;
  pageCount: number;
  zoomPercent: number;
  fitWidth: boolean;
  onGoToPage(page: number): void;
  onZoomIn(): void;
  onZoomOut(): void;
  onFitWidth(): void;
}

const toolButton =
  "rounded-[6px] p-1 text-gray-600 hover:bg-gray-200 hover:text-gray-800 disabled:pointer-events-none disabled:opacity-40";

function Toolbar(props: ToolbarProps) {
  const t = useT();
  const { currentPage, pageCount, zoomPercent, fitWidth } = props;
  return (
    <div
      data-testid="pdf-toolbar"
      className="flex h-9 shrink-0 items-center gap-1 border-b border-gray-200 px-2 text-sm text-gray-600"
    >
      <button
        type="button"
        data-testid="pdf-previous-page"
        aria-label={t("viewer.page.previous")}
        title={t("viewer.page.previous")}
        disabled={currentPage <= 1}
        onClick={() => props.onGoToPage(currentPage - 1)}
        className={toolButton}
      >
        <ChevronUpIcon className="size-4" />
      </button>
      <button
        type="button"
        data-testid="pdf-next-page"
        aria-label={t("viewer.page.next")}
        title={t("viewer.page.next")}
        disabled={currentPage >= pageCount}
        onClick={() => props.onGoToPage(currentPage + 1)}
        className={toolButton}
      >
        <ChevronDownIcon className="size-4" />
      </button>
      <PageNumberInput current={currentPage} count={pageCount} onGo={props.onGoToPage} />
      <span className="whitespace-nowrap text-gray-500">
        {t("viewer.page.count", { count: pageCount })}
      </span>
      <div className="ml-auto flex items-center gap-1">
        <button
          type="button"
          data-testid="pdf-zoom-out"
          aria-label={t("viewer.zoom.out")}
          title={t("viewer.zoom.out")}
          disabled={zoomPercent <= MIN_ZOOM * 100}
          onClick={props.onZoomOut}
          className={toolButton}
        >
          <MinusIcon className="size-4" />
        </button>
        <span
          data-testid="pdf-zoom-level"
          title={t("viewer.zoom.level", { percent: zoomPercent })}
          className="w-11 text-center text-[12px] tabular-nums"
        >
          {zoomPercent}%
        </span>
        <button
          type="button"
          data-testid="pdf-zoom-in"
          aria-label={t("viewer.zoom.in")}
          title={t("viewer.zoom.in")}
          disabled={zoomPercent >= MAX_ZOOM * 100}
          onClick={props.onZoomIn}
          className={toolButton}
        >
          <PlusIcon className="size-4" />
        </button>
        <button
          type="button"
          data-testid="pdf-fit-width"
          aria-label={t("viewer.zoom.fitWidth")}
          aria-pressed={fitWidth}
          title={t("viewer.zoom.fitWidth")}
          onClick={props.onFitWidth}
          className={`${toolButton} ${fitWidth ? "bg-gray-200 text-gray-800" : ""}`}
        >
          <FitWidthIcon className="size-4" />
        </button>
      </div>
    </div>
  );
}

/** Shows the current page; typing a number and pressing Enter goes there. */
function PageNumberInput({
  current,
  count,
  onGo,
}: {
  current: number;
  count: number;
  onGo(page: number): void;
}) {
  const t = useT();
  const [draft, setDraft] = useState<string | null>(null);
  const go = () => {
    const page = Number.parseInt(draft ?? "", 10);
    if (Number.isFinite(page)) onGo(clamp(page, 1, count));
    setDraft(null);
  };
  return (
    <input
      data-testid="pdf-page-number"
      aria-label={t("viewer.page.number")}
      inputMode="numeric"
      value={draft ?? String(current)}
      onChange={(event) => setDraft(event.target.value.replace(/\D/g, ""))}
      onFocus={(event) => event.target.select()}
      onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === "Enter") go();
        if (event.key === "Escape" && draft !== null) {
          event.preventDefault(); // keep the panel open
          setDraft(null);
        }
      }}
      onBlur={() => setDraft(null)}
      className="w-10 rounded-[6px] border border-gray-300 bg-white px-1 py-[1px] text-center text-[12px] tabular-nums outline-none focus:border-gray-400"
    />
  );
}

interface PdfPageProps {
  pdfjs: PdfJs;
  pdf: PDFDocumentProxy;
  page: number;
  size: PageSize;
  scale: number;
  /** Near the visible area: drawn. Otherwise blank, with its resources released. */
  nearby: boolean;
  highlights: readonly RunHighlight[] | undefined;
  onSize(page: number, size: PageSize): void;
  onHighlightShown(page: number, element: HTMLElement): void;
  setElement(page: number, element: HTMLElement | null): void;
}

/** One page: a canvas, and a text layer above it for selecting and highlighting text. */
const PdfPage = memo(function PdfPage(props: PdfPageProps) {
  const {
    pdfjs,
    pdf,
    page,
    size,
    scale,
    nearby,
    highlights,
    onSize,
    onHighlightShown,
    setElement,
  } = props;
  const t = useT();
  const canvasHost = useRef<HTMLDivElement>(null);
  const textHost = useRef<HTMLDivElement>(null);
  const textLayer = useRef<TextLayer | null>(null);
  const [drawn, setDrawn] = useState(false);
  const [textLayerVersion, setTextLayerVersion] = useState(0);
  const hasDrawn = useRef(false);

  useEffect(() => {
    const canvasDiv = canvasHost.current;
    const textDiv = textHost.current;
    if (!canvasDiv || !textDiv) return;
    if (!nearby) {
      if (!hasDrawn.current) return;
      hasDrawn.current = false;
      canvasDiv.replaceChildren();
      textDiv.replaceChildren();
      textLayer.current = null;
      setDrawn(false);
      pdf.getPage(page).then(
        (proxy) => proxy.cleanup(),
        () => undefined, // the PDF was closed meanwhile
      );
      return;
    }

    let cancelled = false;
    let renderTask: RenderTask | undefined;
    let layer: TextLayer | undefined;
    const draw = async () => {
      const proxy = await pdf.getPage(page);
      if (cancelled) return;
      const natural = proxy.getViewport({ scale: 1 });
      onSize(page, { width: natural.width, height: natural.height });
      const viewport = proxy.getViewport({ scale });

      const canvas = document.createElement("canvas");
      const ratio = Math.min(
        window.devicePixelRatio || 1,
        Math.sqrt(MAX_CANVAS_PIXELS / (viewport.width * viewport.height)),
      );
      canvas.width = Math.floor(viewport.width * ratio);
      canvas.height = Math.floor(viewport.height * ratio);
      canvas.className = "absolute inset-0 size-full";
      renderTask = proxy.render({
        canvas,
        viewport,
        transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
      });
      await renderTask.promise;
      if (cancelled) return;
      canvasDiv.replaceChildren(canvas);
      hasDrawn.current = true;
      setDrawn(true);

      const content = await proxy.getTextContent();
      if (cancelled) return;
      const container = document.createElement("div");
      container.className = "textLayer";
      layer = new pdfjs.TextLayer({ textContentSource: content, container, viewport });
      await layer.render();
      if (cancelled) return;
      textDiv.replaceChildren(container);
      textLayer.current = layer;
      setTextLayerVersion((version) => version + 1);
    };
    // The first draw is immediate; redraws for a new scale wait until it settles.
    const timer = setTimeout(
      () =>
        draw().catch((error: unknown) => {
          if (!cancelled && !(error instanceof pdfjs.RenderingCancelledException)) {
            console.error(error);
          }
        }),
      hasDrawn.current ? RERENDER_DELAY : 0,
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
      renderTask?.cancel();
      layer?.cancel();
    };
  }, [pdfjs, pdf, page, scale, nearby, onSize]);

  // Marks the quote in the text layer, and lets the view move to it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-runs for each new text layer
  useEffect(() => {
    const layer = textLayer.current;
    if (!layer) return;
    const divs = layer.textDivs;
    const strings = layer.textContentItemsStr;
    let first: HTMLElement | null = null;
    const marked: number[] = [];
    for (const { item, start, end } of highlights ?? []) {
      const div = divs[item];
      const text = strings[item];
      if (!div || text === undefined) continue;
      marked.push(item);
      let mark: HTMLElement;
      if (start === 0 && end === text.length) {
        mark = div;
        div.classList.add("highlight");
      } else {
        mark = document.createElement("span");
        mark.className = "highlight appended";
        mark.textContent = text.slice(start, end);
        div.replaceChildren(text.slice(0, start), mark, text.slice(end));
      }
      mark.dataset.quoteHighlight = "";
      first ??= mark;
    }
    if (first) onHighlightShown(page, first);
    return () => {
      for (const item of marked) {
        const div = divs[item];
        if (!div) continue;
        div.classList.remove("highlight");
        delete div.dataset.quoteHighlight;
        div.textContent = strings[item] ?? "";
      }
    };
  }, [textLayerVersion, highlights, page, onHighlightShown]);

  return (
    <section
      ref={(element) => setElement(page, element)}
      data-page-number={page}
      data-drawn={drawn ? "true" : "false"}
      aria-label={t("viewer.page.label", { number: page })}
      className="pdf-page relative shrink-0 bg-white shadow-custom-unfocus"
      style={
        {
          width: Math.floor(size.width * scale),
          height: Math.floor(size.height * scale),
          "--scale-factor": scale,
        } as CSSProperties
      }
    >
      <div ref={canvasHost} className="absolute inset-0" />
      <div ref={textHost} className="absolute inset-0" />
    </section>
  );
});
