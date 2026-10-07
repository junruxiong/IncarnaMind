import type { ReactNode } from "react";
import { useT } from "../i18n";

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
 * Shown instead of a Document that has been deleted (e.g. a Citation's), with
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
  const t = useT();
  return (
    <div
      data-testid="viewer-removed"
      role="status"
      className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-6 py-8 text-center"
    >
      <h2 className="text-ui font-semibold text-ink">{t("viewer.removed.title")}</h2>
      <p className="mt-1 max-w-sm text-ui text-ink-meta">
        {t(unlinked ? "viewer.removed.unlinked" : "viewer.removed.body")}
      </p>
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
