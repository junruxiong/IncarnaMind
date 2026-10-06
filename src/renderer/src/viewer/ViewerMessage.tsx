import type { ReactNode } from "react";
import { useT } from "../i18n";

/** A short message filling the viewer: loading, empty, or a failure. */
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
      className={`flex h-full items-center justify-center px-6 text-center text-sm ${
        tone === "error" ? "text-red-700" : "text-gray-500"
      }`}
    >
      <p className="max-w-sm break-words">{children}</p>
    </div>
  );
}

/**
 * Shown instead of a Document that has been deleted (e.g. a Citation's), with
 * the quoted text when there is one, so the quote can still be read.
 */
export function DocumentRemoved({ quote }: { quote?: string }) {
  const t = useT();
  return (
    <div
      data-testid="viewer-removed"
      role="status"
      className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center"
    >
      <h2 className="text-base font-medium text-gray-700">{t("viewer.removed.title")}</h2>
      <p className="max-w-sm text-sm text-gray-500">{t("viewer.removed.body")}</p>
      {quote && (
        <figure className="mt-2 max-w-sm text-left text-sm">
          <figcaption className="text-[11px] text-gray-400">{t("viewer.removed.quote")}</figcaption>
          <blockquote className="mt-1 border-l-2 border-gray-200 pl-3 break-words whitespace-pre-wrap text-gray-700">
            {quote}
          </blockquote>
        </figure>
      )}
    </div>
  );
}
