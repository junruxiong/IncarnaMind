import { Suspense, useMemo } from "react";
import type { Document, DocumentKind } from "../../../core/api";
import { useT } from "../i18n";
import { fileStatusLabel } from "../linkedFolders";
import { PEEK_PADDING, PEEK_WIDTH } from "../peekPlace";
import { DocumentView } from "../viewer/DocumentView";

/** The well a peek draws the Document's first page in: the card's width less its padding, and the height the design has. */
const WELL_PX = { width: PEEK_WIDTH - 2 * PEEK_PADDING, height: 150 } as const;

/**
 * The width each kind is laid out at before it is scaled down to the well: a
 * page's own, or a grid's and a text's, narrower so their words stay legible.
 */
const LAID_OUT_AT: Record<DocumentKind, number> = {
  pdf: 584,
  docx: 584,
  pptx: 584,
  xlsx: 420,
  csv: 420,
  markdown: 420,
  text: 420,
  web: 420,
  transcript: 420,
};

/**
 * A Document's first page, slide or sheet, drawn small by the viewer's own view
 * for its kind: laid out at its own width, scaled down, with no toolbar and
 * nothing to click. It waits until the peek has settled on this Document, so
 * stepping through a list doesn't draw every one on the way. A Document whose
 * file isn't there says so instead.
 */
export function PeekPage({ document, settled }: { document: Document; settled: boolean }) {
  const t = useT();
  const fileState = fileStatusLabel(document.fileStatus);
  const laidOutAt = LAID_OUT_AT[document.kind];
  const scale = WELL_PX.width / laidOutAt;
  const target = useMemo(() => ({ documentId: document.id, request: 1 }), [document.id]);
  return (
    <div
      data-testid="peek-page"
      data-state={fileState ? "unreadable" : settled ? "drawing" : "waiting"}
      className="relative overflow-hidden rounded-md bg-chip"
      style={{ width: WELL_PX.width, height: WELL_PX.height }}
    >
      {fileState ? (
        <p
          data-testid="peek-file-state"
          title={t(fileState.full)}
          className="absolute inset-0 flex items-center justify-center text-label font-semibold text-ink-meta"
        >
          {t(fileState.short)}
        </p>
      ) : (
        settled && (
          <div
            inert
            aria-hidden="true"
            className="peek-view viewer-pane absolute top-0 left-0 flex origin-top-left flex-col overflow-hidden"
            style={{
              width: laidOutAt,
              height: WELL_PX.height / scale,
              transform: `scale(${scale})`,
            }}
          >
            <Suspense fallback={null}>
              <DocumentView key={document.contentHash} document={document} target={target} />
            </Suspense>
          </div>
        )
      )}
    </div>
  );
}
