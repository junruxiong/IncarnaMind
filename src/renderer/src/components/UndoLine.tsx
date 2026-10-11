import { isMacOS } from "@tiptap/core";
import { useEffect } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { typesText, UNDO_SHOWN_MS, useUndo } from "../undo";
import { CloseLineIcon } from "./lineIcons";

/**
 * The last action that can be taken back, as one short line floating at the
 * foot of the sidebar's tree, so nothing under it moves: what happened, then
 * "Undo" (also ⌘Z / Ctrl+Z, outside a text field) and a × to dismiss it. It
 * goes by itself after a few seconds, and is read out politely.
 */
export function UndoLine() {
  const t = useT();
  const last = useUndo((state) => state.last);

  const undo = () =>
    useUndo
      .getState()
      .undo()
      .catch((error: unknown) => useAppStore.getState().reportError(error));

  // Goes by itself, unless a newer action replaced it meanwhile.
  useEffect(() => {
    if (!last) return;
    const timer = setTimeout(() => {
      if (useUndo.getState().last?.id === last.id) useUndo.getState().dismiss();
    }, UNDO_SHOWN_MS);
    return () => clearTimeout(timer);
  }, [last]);

  // ⌘Z / Ctrl+Z while the line shows, unless the key goes to text (the note's own undo).
  useEffect(() => {
    if (!last) return;
    const onKey = (event: KeyboardEvent) => {
      const command = isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
      if (event.defaultPrevented || !command || event.shiftKey || event.altKey) return;
      if (event.code !== "KeyZ" || typesText(event.target)) return;
      if (document.querySelector("dialog[open]")) return;
      event.preventDefault();
      void undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="undo-line"
      className="pointer-events-none absolute inset-x-2 bottom-2 z-10"
    >
      {last && (
        <div className="pointer-events-auto flex min-h-8 items-center gap-2 rounded-lg bg-sheet py-1 pr-1 pl-3 text-[13px] leading-5 text-ink-secondary shadow-popover">
          <span data-testid="undo-message" className="min-w-0 flex-1 break-words">
            {last.message}
          </span>
          <button
            type="button"
            data-testid="undo-action"
            aria-label={t("move.undoLabel", { message: last.message })}
            onClick={() => void undo()}
            className="shrink-0 rounded-sm px-1 font-semibold text-accent hover:underline"
          >
            {t("move.undo")}
          </button>
          <button
            type="button"
            aria-label={t("move.dismiss")}
            title={t("move.dismiss")}
            onClick={() => useUndo.getState().dismiss()}
            className="inline-flex size-6 shrink-0 items-center justify-center rounded-md text-ink-meta hover:bg-hover hover:text-ink"
          >
            <CloseLineIcon className="size-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
