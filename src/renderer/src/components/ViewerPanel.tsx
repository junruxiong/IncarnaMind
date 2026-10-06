import { useEffect } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { PdfView } from "../viewer/PdfView";
import { TextView } from "../viewer/TextView";
import { DocumentRemoved, ViewerMessage } from "../viewer/ViewerMessage";
import { CloseIcon, DocumentIcon } from "./icons";

/**
 * The Document viewer: a panel on the right that is closed by default, like an
 * artifact panel. It renders only while open, showing one Document at a time:
 * clicking a Document in the sidebar (or, later, a Citation) opens it here,
 * replacing whatever was shown.
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
      className="flex min-w-[220px] shrink flex-col"
      style={{ flexBasis: width }}
    >
      <div className="flex h-10 shrink-0 items-end justify-between gap-2 pr-2">
        {document ? (
          <div
            data-testid="viewer-title"
            title={document.name}
            className="relative flex h-8 min-w-0 items-center rounded-t-[9px] bg-white pr-4 pl-8 text-sm text-gray-700"
          >
            <DocumentIcon kind={document.kind} className="absolute left-[10px] size-4" />
            <span className="truncate">{document.name}</span>
          </div>
        ) : (
          <span />
        )}
        <button
          type="button"
          data-testid="viewer-close"
          aria-label={t("viewer.close")}
          title={t("viewer.close")}
          onClick={onClose}
          className="mb-1 shrink-0 rounded-[9px] p-1 text-gray-500 hover:bg-gray-300 hover:text-gray-700"
        >
          <CloseIcon className="size-4" />
        </button>
      </div>
      <div className="min-h-0 flex-grow overflow-hidden rounded-tl-[6px] bg-white">
        {!target ? (
          <ViewerMessage testId="viewer-empty">{t("viewer.empty")}</ViewerMessage>
        ) : !document ? (
          <DocumentRemoved quote={target.quote} />
        ) : document.kind === "pdf" ? (
          // Keyed, so another Document starts afresh.
          <PdfView key={document.id} document={document} target={target} />
        ) : (
          <TextView key={document.id} document={document} target={target} />
        )}
      </div>
    </section>
  );
}
