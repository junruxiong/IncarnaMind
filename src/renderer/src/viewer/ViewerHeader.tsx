import { createContext, type ReactNode, useContext, useState } from "react";
import type { Document } from "../../../core/api";
import { displayName, fileNameOf } from "../../../shared/documentNames";
import { DocumentLineIcon, PinLineIcon } from "../components/lineIcons";
import { ProblemLine } from "../components/ProblemLine";
import { DocumentTagChips } from "../components/TagChips";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { OpenExternallyIcon } from "./icons";

/** What every view's toolbar needs from its tab: the Document shown, and whether it is a preview. */
export interface ViewerFrame {
  /** Undefined when its Document was deleted. */
  document: Document | undefined;
  /** The tab is a preview: the toolbar offers to keep it open. */
  preview: boolean;
  keep(): void;
}

export const ViewerFrameContext = createContext<ViewerFrame>({
  document: undefined,
  preview: false,
  keep: () => undefined,
});

/** A thin vertical rule between groups of the header's controls. */
export function HeaderDivider() {
  return <span aria-hidden="true" className="viewer-divider" />;
}

/**
 * A Document tab's 36px toolbar, under its group's tab strip: the outline
 * toggle (`leading`), the Document's kind and name, its Tags, the view's own
 * controls (`children`: where it is, page navigation and zoom for a PDF),
 * then Keep open for a preview, and "Open in default app".
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
  const { document, preview, keep } = useContext(ViewerFrameContext);
  /** Why the Document's own app couldn't open it, until it is tried again. */
  const [problem, setProblem] = useState<string | null>(null);

  // A copy named after the Document opens, as from the sidebar's file menu.
  const openExternally = () => {
    if (!document) return;
    setProblem(null);
    files.openDocumentExternally(document.id).catch((failure: unknown) => {
      setProblem(
        t("viewer.file.openFailed", { name: document.name, reason: errorMessage(failure) }),
      );
    });
  };

  return (
    <>
      <header data-testid="viewer-header" className="viewer-header">
        {leading}
        {document && (
          <DocumentLineIcon
            kind={document.kind}
            className={`viewer-kind ${leading ? "ml-1" : "ml-1.5"}`}
          />
        )}
        {document ? (
          <span
            data-testid="viewer-title"
            title={
              displayName(document) === fileNameOf(document.path)
                ? displayName(document)
                : `${displayName(document)}
${fileNameOf(document.path)}`
            }
            className="viewer-title"
          >
            {displayName(document)}
          </span>
        ) : null}
        {document && <DocumentTagChips document={document} variant="header" />}
        <span className="flex-1" />
        {children}
        {preview && (
          <button
            type="button"
            data-testid="viewer-keep"
            title={t("viewer.keepTip")}
            onClick={keep}
            className="viewer-keep"
          >
            <PinLineIcon className="size-[13px]" />
            <span>{t("viewer.keep")}</span>
          </button>
        )}
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
      </header>
      {problem && (
        <ProblemLine testId="viewer-open-problem" className="shrink-0 px-4 py-2">
          {problem}
        </ProblemLine>
      )}
    </>
  );
}
