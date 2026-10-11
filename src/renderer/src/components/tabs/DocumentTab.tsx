import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { Document, DocumentPlace, LayoutTab } from "../../../../core/api";
import { citedTextKept } from "../../../../shared/citations";
import { useT } from "../../i18n";
import { useAppStore, type ViewerTarget } from "../../store";
import { type PlaceMemory, PlaceMemoryContext, type ViewPlace } from "../../viewer/place";
import { type ViewerFrame, ViewerFrameContext, ViewerHeader } from "../../viewer/ViewerHeader";
import { DocumentRemoved } from "../../viewer/ViewerMessage";

// Each kind's view loads the first time a Document of that kind opens, with its libraries
// (pdf.js, docx-preview, JSZip, the Office formats), rather than with the app.
const PdfView = lazy(() => import("../../viewer/PdfView").then((m) => ({ default: m.PdfView })));
const TextView = lazy(() => import("../../viewer/TextView").then((m) => ({ default: m.TextView })));
const DocxView = lazy(() => import("../../viewer/DocxView").then((m) => ({ default: m.DocxView })));
const SlidesView = lazy(() =>
  import("../../viewer/SlidesView").then((m) => ({ default: m.SlidesView })),
);
const SheetView = lazy(() =>
  import("../../viewer/SheetView").then((m) => ({ default: m.SheetView })),
);
const WebView = lazy(() => import("../../viewer/WebView").then((m) => ({ default: m.WebView })));

/** How far the User scrolls a preview, at least, to keep it: the jump to the quote doesn't count. */
const KEEP_AFTER_SCROLL_PX = 120;

/**
 * What the viewer shows of a Document's file, as a key that changes when
 * there is something new to show, so the view reads the file again: a new
 * version, once it is indexed (its `contentHash`), or the file back after it
 * went missing or out of reach. A file that goes while it is shown leaves
 * what was read of it on screen.
 */
function useFileKey(document: Document | undefined): string {
  const id = document?.id;
  const available = document?.fileStatus === "available";
  const [returns, setReturns] = useState(0);
  const last = useRef({ id, available });
  useEffect(() => {
    const before = last.current;
    if (before.id === id && available && !before.available) setReturns((count) => count + 1);
    last.current = { id, available };
  }, [id, available]);
  return `${id}:${document?.contentHash}:${returns}`;
}

/** Keeps a view's place for the open request, while its file is read again (see `PlaceMemory`). */
function usePlaceMemory(target: ViewerTarget): PlaceMemory {
  const kept = useRef<{ key: string; place: ViewPlace } | null>(null);
  const key = `${target.documentId}#${target.request}`;
  return useMemo(
    () => ({
      recall: () => (kept.current?.key === key ? kept.current.place : null),
      remember: (place) => {
        kept.current = { key, place };
      },
    }),
    [key],
  );
}

/** The view for a kind of Document (ADR-0011). */
function DocumentView({ document, target }: { document: Document; target: ViewerTarget }) {
  switch (document.kind) {
    case "pdf":
      return <PdfView document={document} target={target} />;
    case "text":
    case "markdown":
      return <TextView document={document} target={target} />;
    case "docx":
      return <DocxView document={document} target={target} />;
    case "pptx":
      return <SlidesView document={document} target={target} />;
    case "xlsx":
    case "csv":
      return <SheetView document={document} target={target} />;
    case "web":
      return <WebView document={document} target={target} />;
  }
}

/** Keys that scroll a page, not type in it. */
const SCROLL_KEYS = ["PageDown", "PageUp", "ArrowDown", "ArrowUp", "Home", "End", " "];
/** How far in from a scroller's right edge a press is on its scrollbar, overlaid or not. */
const SCROLLBAR_PX = 16;

/**
 * A preview is kept once the User reads it (VS Code's preview tabs): a scroll
 * of at least `KEEP_AFTER_SCROLL_PX` with the wheel, the keys or the
 * scrollbar (not the jump to the quote), or a selection of some text. Native
 * listeners, in the capture phase: scroll events don't bubble.
 */
function useKeepOnReading(section: HTMLElement | null, tabId: string, preview: boolean) {
  useEffect(() => {
    if (!section || !preview) return;
    /** When the User last moved the page themselves; a scrollbar held down counts until let go. */
    let intent = 0;
    let scrolled = 0;
    const lastTop = new WeakMap<EventTarget, number>();
    const keep = () => {
      void useAppStore.getState().changeLayout({ type: "keep", tabId });
    };
    const moves = () => {
      intent = Date.now();
    };
    const onKey = (event: KeyboardEvent) => {
      if (SCROLL_KEYS.includes(event.key)) moves();
    };
    // A press on a scrollbar (a scroller's edge), not on a button that moves the page.
    const onDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof HTMLElement) || target.scrollHeight <= target.clientHeight) return;
      if (event.offsetX >= target.clientWidth - SCROLLBAR_PX) intent = Number.POSITIVE_INFINITY;
    };
    const onUp = () => {
      if (intent === Number.POSITIVE_INFINITY) intent = Date.now();
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed && selection.toString().trim().length >= 2) keep();
    };
    const onScroll = (event: Event) => {
      const element = event.target;
      if (!(element instanceof HTMLElement)) return;
      const before = lastTop.get(element) ?? element.scrollTop;
      lastTop.set(element, element.scrollTop);
      if (Date.now() - intent > 400) return;
      scrolled += Math.abs(element.scrollTop - before);
      if (scrolled >= KEEP_AFTER_SCROLL_PX) keep();
    };
    const capture = { capture: true, passive: true };
    section.addEventListener("wheel", moves, capture);
    section.addEventListener("keydown", onKey, capture);
    section.addEventListener("pointerdown", onDown, capture);
    section.addEventListener("pointerup", onUp, capture);
    section.addEventListener("scroll", onScroll, capture);
    return () => {
      section.removeEventListener("wheel", moves, capture);
      section.removeEventListener("keydown", onKey, capture);
      section.removeEventListener("pointerdown", onDown, capture);
      section.removeEventListener("pointerup", onUp, capture);
      section.removeEventListener("scroll", onScroll, capture);
    };
  }, [section, tabId, preview]);
}

/**
 * A Document's tab: its toolbar and the view for its kind (PDF, Word,
 * PowerPoint, Excel, CSV, Markdown, text, web page), at the place it was opened, a
 * Citation's quote washed. It follows the Document's file on disk: a new
 * version is shown once it is indexed, and a file that went missing is shown
 * again when it comes back, at the same place.
 */
export function DocumentTab({ tab }: { tab: LayoutTab }) {
  const t = useT();
  const documentId = tab.item.kind === "document" ? tab.item.documentId : "";
  // The same target until the tab is opened at another place: each new layout is a new object.
  const placeKey = JSON.stringify(tab.place ?? {});
  const target = useMemo<ViewerTarget>(
    () => ({ documentId, ...(JSON.parse(placeKey) as DocumentPlace), request: tab.opened }),
    [documentId, placeKey, tab.opened],
  );
  // Deleted Documents leave the list, so a target that isn't in it has been removed.
  const document = useAppStore((state) => state.documents.find((each) => each.id === documentId));
  // A Citation's Document that went with its Linked folder, whose quoted text was kept.
  const unlinked = useAppStore((state) => {
    if (!target.citation) return false;
    return citedTextKept(
      {
        documentId,
        contentHash: target.citation.contentHash ?? null,
        pageFrom: target.pageFrom ?? null,
        pageTo: target.pageTo ?? null,
      },
      state.keptCitationTexts,
    );
  });
  const preview = tab.preview;
  const tabId = tab.id;
  const frame = useMemo<ViewerFrame>(
    () => ({
      document,
      preview,
      keep: () => void useAppStore.getState().changeLayout({ type: "keep", tabId }),
    }),
    [document, preview, tabId],
  );
  const fileKey = useFileKey(document);
  const places = usePlaceMemory(target);
  const [section, setSection] = useState<HTMLElement | null>(null);
  useKeepOnReading(section, tab.id, tab.preview);

  return (
    <section
      ref={setSection}
      data-testid="viewer"
      data-surface="viewer"
      data-document-id={documentId}
      aria-label={t("viewer.label")}
      className="viewer-pane flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-sheet"
    >
      <ViewerFrameContext.Provider value={frame}>
        <PlaceMemoryContext.Provider value={places}>
          {!document ? (
            <>
              <ViewerHeader />
              <DocumentRemoved quote={target.quote} unlinked={unlinked} />
            </>
          ) : (
            // Keyed, so another Document, or a new read of its file, starts afresh (the last one
            // goes at once). While its view loads, the header alone: the page comes a moment later.
            <Suspense key={fileKey} fallback={<ViewerHeader />}>
              <DocumentView document={document} target={target} />
            </Suspense>
          )}
        </PlaceMemoryContext.Provider>
      </ViewerFrameContext.Provider>
    </section>
  );
}
