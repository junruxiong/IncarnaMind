import { useDismissOnEscape } from "../escapeDismiss";
import { useT } from "../i18n";
import { type ErrorNotice as ErrorNoticeState, useNotices } from "../notices";
import { ProblemMark } from "./ProblemLine";
import { buttonStyle } from "./ui";

/**
 * What failed where the action had no line of its own, in plain words, bottom right (DESIGN.md,
 * Problems). Each stays until the User dismisses it, with its button or Esc: it never goes by
 * itself, as an error the User didn't read would be lost. The technical text is one click away
 * under Details; nothing here shows Electron's wrapping or a stack.
 */
export function ErrorNotices() {
  const errors = useNotices((state) => state.errors);
  if (errors.length === 0) return null;
  return (
    <div className="fixed right-4 bottom-4 z-20 flex max-w-sm flex-col gap-2">
      {errors.map((notice) => (
        <Notice key={notice.id} notice={notice} />
      ))}
    </div>
  );
}

function Notice({ notice }: { notice: ErrorNoticeState }) {
  const t = useT();
  const dismiss = () => useNotices.getState().dismissError(notice.id);
  useDismissOnEscape(true, dismiss);
  const line = t("error.action", { message: notice.reason ? t(notice.reason) : notice.message });
  return (
    <div
      role="alert"
      data-testid="error-notice"
      className="flex items-start gap-3 rounded-lg bg-sheet py-2.5 pr-2.5 pl-4 text-[13px] leading-5 text-ink shadow-popover"
    >
      <ProblemMark className="mt-2 -mr-1.5" />
      <div className="min-w-0 flex-1 py-1">
        {/* A long one is cut here and kept whole under Details. */}
        <p data-testid="error-notice-message" title={line} className="line-clamp-4 break-words">
          {line}
        </p>
        <details data-testid="error-notice-details" className="mt-1 text-ink-meta">
          <summary className="w-fit cursor-default rounded-sm">{t("error.details")}</summary>
          <p className="mt-1 max-h-40 overflow-y-auto font-mono text-[12px] leading-4 break-words whitespace-pre-wrap">
            {notice.detail}
          </p>
        </details>
      </div>
      <button
        type="button"
        data-testid="error-notice-dismiss"
        onClick={dismiss}
        className={buttonStyle("ghost", "sm")}
      >
        {t("error.dismiss")}
      </button>
    </div>
  );
}
