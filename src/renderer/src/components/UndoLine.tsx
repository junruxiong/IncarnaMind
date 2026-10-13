import { useEffect, useRef } from "react";
import { shortcutLabel } from "../commands/keymap";
import { useDismissOnEscape } from "../escapeDismiss";
import { useT } from "../i18n";
import { useFade } from "../motion";
import { useAppStore } from "../store";
import { UNDO_SHOWN_MS, type UndoLineState, useUndo } from "../undo";

const lineClass =
  "mx-2 flex min-w-0 shrink-0 gap-1 pt-0.5 pb-1 pl-2 text-[12px] leading-4 text-ink-secondary";
const actionClass =
  "h-6 shrink-0 rounded-md px-1 text-[12px] leading-5 font-semibold text-ink-secondary hover:bg-hover hover:text-ink";

/**
 * The undo line (CONTEXT.md) at the foot of the sidebar, above Settings: what
 * the User just changed, then Undo, which does what ⌘Z does. One at a time; it
 * goes after a few seconds, with Esc or with the next change, while ⌘Z still
 * reaches the change. An undo the core refuses shows here as one problem line
 * with one action, which stays until dismissed. Read out politely.
 */
export function UndoLine() {
  const line = useUndo((state) => state.line);
  return (
    <div role="status" aria-live="polite" data-testid="undo-line" className="flex flex-col">
      {/* A change shown where the User made it, under an Answer in the assistant, isn't shown twice. */}
      {line && !line.change?.shownAt && <Line key={line.id} line={line} />}
    </div>
  );
}

const report = (error: unknown) => useAppStore.getState().reportError(error);

function Line({ line }: { line: UndoLineState }) {
  const t = useT();
  const shown = useRef<HTMLDivElement>(null);
  useFade(shown);

  useGoesBy(line.change ? line.id : null);
  useDismissOnEscape(true, () => useUndo.getState().dismiss());

  if (line.problem) {
    const { step, reason, fix } = line.problem;
    return (
      <div ref={shown} role="alert" data-testid="undo-problem" className={`${lineClass} items-end`}>
        <span aria-hidden="true" className="mb-[9px] size-1.5 shrink-0 rounded-full bg-attention" />
        <span className="line-clamp-2 min-w-0 mb-1 break-words">
          {t(step === "undo" ? "undo.refused" : "redo.refused", { reason })}
        </span>
        <span aria-hidden="true" className="mb-1">
          ·
        </span>
        <button
          type="button"
          data-testid="undo-fix"
          onClick={() => (fix ? void fix.run().catch(report) : useUndo.getState().dismiss())}
          className={actionClass}
        >
          {fix ? fix.label : t("move.dismiss")}
        </button>
      </div>
    );
  }

  const { message } = line.change;
  return (
    <div ref={shown} className={`${lineClass} items-center`}>
      <span data-testid="undo-message" title={message} className="min-w-0 truncate">
        {message}
      </span>
      <span aria-hidden="true">·</span>
      <button
        type="button"
        data-testid="undo-action"
        aria-label={t("move.undoLabel", { message })}
        title={t("undo.title", { shortcut: shortcutLabel("edit.undo") })}
        onClick={() => void useUndo.getState().undo()}
        className={actionClass}
      >
        {t("move.undo")}
      </button>
    </div>
  );
}

/** A change's line goes after a few seconds, unless a newer line took its place; ⌘Z still reaches it. */
export function useGoesBy(lineId: number | null): void {
  useEffect(() => {
    if (lineId === null) return;
    const timer = setTimeout(() => {
      if (useUndo.getState().line?.id === lineId) useUndo.getState().dismiss();
    }, UNDO_SHOWN_MS);
    return () => clearTimeout(timer);
  }, [lineId]);
}
