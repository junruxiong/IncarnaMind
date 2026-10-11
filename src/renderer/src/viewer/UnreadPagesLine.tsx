import type { Document } from "../../../core/api";
import { ProblemLine } from "../components/ProblemLine";
import { useLanguage, useT } from "../i18n";
import { unreadPages } from "../scanStatus";
import { useAppStore } from "../store";

/**
 * A PDF whose scanned pages couldn't all be read (#397): which, where the
 * User is looking at them, with Retry, which reads only those pages again.
 */
export function UnreadPagesLine({ document }: { document: Document }) {
  const t = useT();
  const language = useLanguage();
  const retryDocument = useAppStore((state) => state.retryDocument);
  const { failure } = document;
  if (document.status !== "failed" || failure?.reason !== "pages-unread") return null;
  return (
    <div className="shrink-0 border-b border-rule px-4 py-1.5">
      <ProblemLine
        testId="scan-unread"
        title={failure.message}
        action={{
          label: t("documents.retryAction"),
          testId: "scan-retry",
          onClick: () => void retryDocument(document.id),
        }}
      >
        {unreadPages(failure, t, language, "sentence")}
      </ProblemLine>
    </div>
  );
}
