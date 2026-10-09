import { createContext, type ReactNode, useContext } from "react";
import type { Document } from "../../../core/api";
import { DocumentTagChips } from "../components/TagChips";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { CloseViewerIcon, OpenExternallyIcon } from "./icons";

/** What every view's header needs from the panel: the Document shown, and how to close. */
export interface ViewerFrame {
  /** Undefined when the viewer is empty or its Document was deleted. */
  document: Document | undefined;
  onClose(): void;
}

export const ViewerFrameContext = createContext<ViewerFrame>({
  document: undefined,
  onClose: () => undefined,
});

/** A thin vertical rule between groups of the header's controls. */
export function HeaderDivider() {
  return <span aria-hidden="true" className="viewer-divider" />;
}

/**
 * The viewer's one header, 44px like every pane header: the outline toggle
 * (`leading`), the Document's name, the view's own controls (`children`: page
 * navigation and zoom for a PDF), then "Open in default app" and close.
 */
export function ViewerHeader({
  leading,
  children,
  openable = true,
}: {
  leading?: ReactNode;
  children?: ReactNode;
  /**
   * False when the Document's file couldn't be read, so there is nothing to
   * open. Nor is there while the Document says its file is missing or out of reach.
   */
  openable?: boolean;
}) {
  const t = useT();
  const { document, onClose } = useContext(ViewerFrameContext);

  // A copy named after the Document opens, as from the sidebar's file menu.
  const openExternally = () => {
    if (!document) return;
    files.openDocumentExternally(document.id).catch((failure: unknown) => {
      useAppStore.setState({ actionError: errorMessage(failure) });
    });
  };

  return (
    <header data-testid="viewer-header" className="viewer-header">
      {leading}
      {document ? (
        <span
          data-testid="viewer-title"
          title={document.name}
          className={`viewer-title ${leading ? "ml-1" : "pl-2"}`}
        >
          {document.name}
        </span>
      ) : (
        <span className="flex-1" />
      )}
      {document && <DocumentTagChips document={document} variant="header" />}
      {children}
      {children && <HeaderDivider />}
      {document && openable && document.fileStatus === "available" && (
        <button
          type="button"
          data-testid="viewer-open-externally"
          aria-label={t("documents.copy.open")}
          title={t("documents.copy.open")}
          onClick={openExternally}
          className="viewer-icon-button"
        >
          <OpenExternallyIcon className="size-4" />
        </button>
      )}
      <button
        type="button"
        data-testid="viewer-close"
        aria-label={t("viewer.close")}
        title={t("viewer.close")}
        onClick={onClose}
        className="viewer-icon-button"
      >
        <CloseViewerIcon className="size-4" />
      </button>
    </header>
  );
}
