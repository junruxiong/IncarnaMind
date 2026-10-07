import { type ReactNode, useContext } from "react";
import { useT } from "../i18n";
import { ViewerFrameContext } from "./ViewerHeader";

/** A short message filling the viewer under its header: loading, empty, or a failure. */
export function ViewerMessage({
  children,
  testId,
  tone = "muted",
}: {
  children: ReactNode;
  testId?: string;
  tone?: "muted" | "error";
}) {
  return (
    <div
      data-testid={testId}
      role={tone === "error" ? "alert" : "status"}
      className="flex min-h-0 flex-1 items-center justify-center px-6 text-center text-ui"
    >
      <p className={`max-w-sm break-words ${tone === "error" ? "text-danger" : "text-ink-meta"}`}>
        {children}
      </p>
    </div>
  );
}

/**
 * Why a Document can't be shown: it was deleted, it went with its unlinked
 * Linked folder (the text its Citations quote was kept), or it is still there
 * but its file is missing or can't be reached.
 */
type GoneReason = "deleted" | "unlinked" | "missing" | "unavailable";

const GONE_TEXT = {
  deleted: { title: "viewer.removed.title", body: "viewer.removed.body" },
  unlinked: { title: "viewer.removed.title", body: "viewer.removed.unlinked" },
  missing: { title: "viewer.file.missing.title", body: "viewer.file.missing" },
  unavailable: { title: "viewer.file.unavailable.title", body: "viewer.file.unavailable" },
} as const;

/**
 * Shown instead of a Document that can't be shown (e.g. a Citation's), with
 * the quoted text when there is one, so the quote can still be read.
 * `unlinked`: the Citation's Document went with its Linked folder, and the
 * text it quotes was kept, so it says that instead of "deleted".
 */
export function DocumentRemoved({
  quote,
  unlinked = false,
}: {
  quote?: string | undefined;
  unlinked?: boolean;
}) {
  return <DocumentGone quote={quote} reason={unlinked ? "unlinked" : "deleted"} />;
}

/**
 * Shown instead of a Document whose file couldn't be read: it says whether
 * the file is missing or can't be reached, as the Document does. The viewer
 * shows the file again once it is back.
 */
export function FileGone({ quote }: { quote?: string | undefined }) {
  const { document } = useContext(ViewerFrameContext);
  return (
    <DocumentGone
      quote={quote}
      reason={document?.fileStatus === "unavailable" ? "unavailable" : "missing"}
    />
  );
}

function DocumentGone({ quote, reason }: { quote?: string | undefined; reason: GoneReason }) {
  const t = useT();
  const text = GONE_TEXT[reason];
  return (
    <div
      data-testid="viewer-removed"
      data-reason={reason}
      role="status"
      className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-6 py-8 text-center"
    >
      <h2 className="text-ui font-semibold text-ink">{t(text.title)}</h2>
      <p className="mt-1 max-w-sm text-ui text-ink-meta">{t(text.body)}</p>
      {quote && (
        <figure className="mt-6 w-full max-w-sm text-left">
          <figcaption className="text-label text-ink-meta">{t("viewer.removed.quote")}</figcaption>
          <blockquote className="mt-2 rounded-lg bg-frame px-4 py-3 font-serif text-[15px] leading-6 break-words whitespace-pre-wrap text-ink-answer">
            {quote}
          </blockquote>
        </figure>
      )}
    </div>
  );
}
