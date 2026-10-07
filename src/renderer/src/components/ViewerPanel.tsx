import { useEffect, useMemo } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { PdfView } from "../viewer/PdfView";
import { TextView } from "../viewer/TextView";
import { type ViewerFrame, ViewerFrameContext, ViewerHeader } from "../viewer/ViewerHeader";
import { DocumentRemoved, ViewerMessage } from "../viewer/ViewerMessage";

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
            <DocumentRemoved quote={target.quote} />
          </>
        ) : document.kind === "pdf" ? (
          // Keyed, so another Document starts afresh.
          <PdfView key={document.id} document={document} target={target} />
        ) : (
          <TextView key={document.id} document={document} target={target} />
        )}
      </ViewerFrameContext.Provider>
    </section>
  );
}
