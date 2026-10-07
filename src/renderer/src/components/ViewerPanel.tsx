import { useEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { citedTextKept } from "../../../shared/citations";
import { useT } from "../i18n";
import { useAppStore, type ViewerTarget } from "../store";
import { DocxView } from "../viewer/DocxView";
import { PdfView } from "../viewer/PdfView";
import { type PlaceMemory, PlaceMemoryContext, type ViewPlace } from "../viewer/place";
import { SheetView } from "../viewer/SheetView";
import { SlidesView } from "../viewer/SlidesView";
import { TextView } from "../viewer/TextView";
import { type ViewerFrame, ViewerFrameContext, ViewerHeader } from "../viewer/ViewerHeader";
import { DocumentRemoved, ViewerMessage } from "../viewer/ViewerMessage";

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
function usePlaceMemory(target: ViewerTarget | null): PlaceMemory {
  const kept = useRef<{ key: string; place: ViewPlace } | null>(null);
  const key = target ? `${target.documentId}#${target.request}` : "";
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
  }
}

/**
 * The Document viewer: a panel on the right that is closed by default, like an
 * artifact panel. It renders only while open, showing one Document at a time:
 * clicking a Document in the sidebar, or a Citation, opens it here, replacing
 * whatever was shown. It slides in as it opens (viewer.css). It follows the
 * Document's file on disk: a new version is shown once it is indexed, and a
 * file that went missing is shown again when it comes back, at the same place.
 */
export function ViewerPanel({ width, onClose }: { width: number; onClose(): void }) {
  const t = useT();
  const target = useAppStore((state) => state.viewerTarget);
  // Deleted Documents leave the list, so a target that isn't in it has been removed.
  const document = useAppStore((state) =>
    state.viewerTarget
      ? state.documents.find((each) => each.id === state.viewerTarget?.documentId)
      : undefined,
  );
  // A Citation's Document that went with its Linked folder, whose quoted text was kept.
  const unlinked = useAppStore((state) => {
    const opened = state.viewerTarget;
    if (!opened?.citation) return false;
    return citedTextKept(
      {
        documentId: opened.documentId,
        contentHash: opened.citation.contentHash ?? null,
        pageFrom: opened.pageFrom ?? null,
        pageTo: opened.pageTo ?? null,
      },
      state.keptCitationTexts,
    );
  });
  const frame = useMemo<ViewerFrame>(() => ({ document, onClose }), [document, onClose]);
  const fileKey = useFileKey(document);
  const places = usePlaceMemory(target);

  // Esc closes the panel, unless it is closing a dialog.
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (event.target instanceof Element && event.target.closest("dialog")) return;
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <section
      data-testid="viewer"
      data-document-id={document?.id}
      aria-label={t("viewer.label")}
      className="viewer-pane flex min-w-[220px] shrink flex-col overflow-hidden bg-sheet"
      style={{ flexBasis: width }}
    >
      <ViewerFrameContext.Provider value={frame}>
        <PlaceMemoryContext.Provider value={places}>
          {!target ? (
            <>
              <ViewerHeader />
              <ViewerMessage testId="viewer-empty">{t("viewer.empty")}</ViewerMessage>
            </>
          ) : !document ? (
            <>
              <ViewerHeader />
              <DocumentRemoved quote={target.quote} unlinked={unlinked} />
            </>
          ) : (
            // Keyed, so another Document, or a new read of its file, starts afresh.
            <DocumentView key={fileKey} document={document} target={target} />
          )}
        </PlaceMemoryContext.Provider>
      </ViewerFrameContext.Provider>
    </section>
  );
}
