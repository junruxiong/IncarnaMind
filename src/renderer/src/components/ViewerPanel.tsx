import { useEffect, useMemo } from "react";
import type { Document } from "../../../core/api";
import { citedTextKept } from "../../../shared/citations";
import { useT } from "../i18n";
import { useAppStore, type ViewerTarget } from "../store";
import { DocxView } from "../viewer/DocxView";
import { PdfView } from "../viewer/PdfView";
import { SheetView } from "../viewer/SheetView";
import { SlidesView } from "../viewer/SlidesView";
import { TextView } from "../viewer/TextView";
import { type ViewerFrame, ViewerFrameContext, ViewerHeader } from "../viewer/ViewerHeader";
import { DocumentRemoved, ViewerMessage } from "../viewer/ViewerMessage";

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
 * whatever was shown. It slides in as it opens (viewer.css).
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
          // Keyed, so another Document starts afresh.
          <DocumentView key={document.id} document={document} target={target} />
        )}
      </ViewerFrameContext.Provider>
    </section>
  );
}
