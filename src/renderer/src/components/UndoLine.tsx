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

/** The line of a change made in a place of its own, shown there instead of the sidebar's foot (`Undoable.shownAt`). */
export function UndoLineAt({ place }: { place: string }) {
  const line = useUndo((state) => (state.line?.change?.shownAt === place ? state.line : null));
  return (
    <div aria-live="polite" data-testid="undo-line-here">
      {line && <Line key={line.id} line={line} />}
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
    const { step, wording, fix } = line.problem;
    const reason = wording ? t(wording.key, wording.values) : line.problem.reason;
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

  const { message, also } = line.change;
  const undo = (
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
  );
  if (also) {
    // A longer line, said for the User rather than by them: all of it, its actions under it.
    return (
      <div ref={shown} className={`${lineClass} flex-col`}>
        <span data-testid="undo-message" className="min-w-0 break-words">
          {message}
        </span>
        <span className="-ml-1 flex items-center gap-1">
          <button
            type="button"
            data-testid={also.testId}
            onClick={() => {
              useUndo.getState().dismiss();
              also.run();
            }}
            className={actionClass}
          >
            {also.label}
          </button>
          <span aria-hidden="true">·</span>
          {undo}
        </span>
      </div>
    );
  }
  return (
    <div ref={shown} className={`${lineClass} items-center`}>
      <span data-testid="undo-message" title={message} className="min-w-0 truncate">
        {message}
      </span>
      <span aria-hidden="true">·</span>
      {undo}
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
