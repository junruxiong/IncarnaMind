import { type ReactNode, useEffect, useRef, useState } from "react";
import type { ConversationAnswer } from "../../../../core/api";
import { copyAnswer } from "../../answerEditors";
import { shortcutLabel } from "../../commands/keymap";
import { stopAnswer } from "../../conversations";
import { useDismissOnEscape } from "../../escapeDismiss";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { answerUndoPlace, useUndo } from "../../undo";
import { StopIcon } from "../icons";
import { CheckLineIcon, ChevronDownLineIcon, CopyLineIcon, PlusLineIcon } from "../lineIcons";
import { useGoesBy } from "../UndoLine";

/** How long "Copied" says so. */
const COPIED_MS = 1500;

/**
 * Under an Answer in the assistant (#201, #391; DESIGN.md, Answer): only
 * "Copy · Add to note ▾", after "Stopped" for one stopped; on hover or focus,
 * and always on the newest. Its model, scope and searches are in its steps.
 * While it is written, Stop. Once added, "Added to “{Mind}” · Undo" (⌘Z
 * too), while that can be taken back. Tab reaches each by its name.
 */
export function AnswerActions({
  answer,
  streaming,
  canKeep,
  focusedMindName,
  onAdd,
  onPick,
}: {
  answer: ConversationAnswer;
  streaming: boolean;
  /** It has something to keep: Copy and Add to note. */
  canKeep: boolean;
  /** The Mind in focus, which Add to note adds to; null opens the ▾ instead. */
  focusedMindName: string | null;
  onAdd(): void;
  onPick(anchor: HTMLElement): void;
}) {
  const t = useT();
  const pick = useRef<HTMLButtonElement>(null);
  const undoButton = useRef<HTMLButtonElement>(null);
  const [copied, setCopied] = useState(false);
  const addedLine = useUndo((state) =>
    state.line?.change?.shownAt === answerUndoPlace(answer.id) ? state.line : null,
  );
  const added = addedLine?.change ?? null;
  useGoesBy(addedLine?.id ?? null);
  useDismissOnEscape(added !== null, () => useUndo.getState().dismiss());
  // Added from its own buttons (which are gone now): the focus goes to its Undo, so the keyboard
  // can take it back at once. Added from elsewhere (⌘⇧A in the note), the focus stays there.
  useEffect(() => {
    const active = document.activeElement;
    if (added && (active === document.body || active?.closest(".assistant-answer"))) {
      undoButton.current?.focus();
    }
  }, [added]);
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), COPIED_MS);
    return () => clearTimeout(timer);
  }, [copied]);

  if (streaming) return <WritingLine onStop={() => stopAnswer(answer.id)} />;

  if (added) {
    return (
      <div
        data-testid="answer-added"
        className="assistant-answer-actions assistant-answer-actions--added"
      >
        <CheckLineIcon aria-hidden="true" className="size-3.5 shrink-0 text-ink-secondary" />
        <span role="status" className="assistant-answer-added">
          {added.message}
        </span>
        <span aria-hidden="true">·</span>
        <button
          ref={undoButton}
          type="button"
          data-testid="answer-added-undo"
          aria-label={t("move.undoLabel", { message: added.message })}
          onClick={() => {
            useUndo
              .getState()
              .undo()
              .catch((error: unknown) => useAppStore.getState().reportError(error));
          }}
          className="assistant-answer-action"
        >
          {t("move.undo")}
        </button>
      </div>
    );
  }

  const addTitle = focusedMindName
    ? t("answer.add.to", { mind: focusedMindName, shortcut: shortcutLabel("answer.addToNote") })
    : t("answer.add.choose");
  return (
    // biome-ignore lint/a11y/useSemanticElements: an Answer's buttons, named together; a fieldset is for a form's.
    <div
      role="group"
      aria-label={t("answer.actions")}
      data-testid="answer-actions"
      className="assistant-answer-actions"
    >
      {answer.status === "stopped" && (
        <span data-testid="answer-stopped" className="assistant-answer-status">
          {t("answer.status.stopped")}
        </span>
      )}
      {canKeep && (
        <>
          <button
            type="button"
            data-testid="answer-copy"
            aria-label={
              copied
                ? t("answer.copied")
                : t("answer.copy.label", { shortcut: shortcutLabel("assistant.copyAnswer") })
            }
            title={t("answer.copy.label", { shortcut: shortcutLabel("assistant.copyAnswer") })}
            // A selection in the Answer stays as it is: Copy copies just that.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => {
              void copyAnswer(answer.id).then((done) => setCopied(done));
            }}
            className="assistant-answer-action"
          >
            <CopyLineIcon className="size-3.5" />
            {copied ? t("answer.copied") : t("answer.copy")}
          </button>
          <span className="assistant-answer-add">
            <button
              type="button"
              data-testid="answer-add"
              aria-label={addTitle}
              title={addTitle}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => (focusedMindName ? onAdd() : pick.current && onPick(pick.current))}
              className="assistant-answer-action assistant-answer-add-main"
            >
              <PlusLineIcon className="size-3.5" />
              {t("answer.add")}
            </button>
            <button
              ref={pick}
              type="button"
              data-testid="answer-add-to"
              aria-haspopup="dialog"
              aria-label={t("answer.addTo")}
              title={t("answer.addTo")}
              onMouseDown={(event) => event.preventDefault()}
              onClick={(event) => onPick(event.currentTarget)}
              className="assistant-answer-action assistant-answer-add-more"
            >
              <ChevronDownLineIcon className="size-3" />
            </button>
          </span>
        </>
      )}
    </div>
  );
}

/**
 * Under an Answer being written, or on its way: Stop, or what it says
 * instead (that it stopped before it started). Its step is on its steps' line.
 */
export function WritingLine({ status, onStop }: { status?: ReactNode; onStop?: () => void }) {
  const t = useT();
  return (
    <div data-testid="answer-actions" className="assistant-answer-actions">
      {status}
      {onStop && (
        <button
          type="button"
          data-testid="answer-stop"
          onClick={onStop}
          className="assistant-answer-action"
        >
          <StopIcon className="size-3" />
          {t("answer.stop")}
        </button>
      )}
    </div>
  );
}
