import { isMacOS } from "@tiptap/core";
import { createContext, Fragment, type ReactNode, useContext, useRef, useState } from "react";
import { StopIcon } from "../components/icons";
import { useT } from "../i18n";
import type { AddedToNote } from "../threads";
import { RegenerateSmallIcon } from "./icons";

/**
 * What the thread does for the Answers in it (#201), for their node views:
 * which Mind and chat they are in, and Copy, Add to note (here, at the
 * cursor, or through the ▾ to another Mind) and Undo.
 */
export interface ThreadActions {
  mindId: string;
  /** The chat shown; null when the thread is the Mind's own Blocks (a chat on its own, #203). */
  chatId: string | null;
  /** The last "Add to note" from this thread, which its Answer shows with Undo. */
  added: AddedToNote | null;
  /** Copies the Answer, Citations and all; resolves with whether it was copied. */
  copy(answerId: string): Promise<boolean>;
  /** Adds the Answer, or the part of it selected, to this Mind's note at the cursor. */
  add(answerId: string): void;
  /** Opens the ▾ picker of the Mind to add it to, under `anchor`. */
  pick(answerId: string, anchor: HTMLElement): void;
  undo(): void;
}

export const ThreadContext = createContext<ThreadActions | null>(null);

/** The thread an Answer is drawn in, or null for an Answer in a note. */
export const useThread = () => useContext(ThreadContext);

const MOD = isMacOS() ? "⌘" : "Ctrl+";
export const COPY_SHORTCUT = isMacOS() ? "⌘⇧C" : "Ctrl+Shift+C";
export const ADD_SHORTCUT = isMacOS() ? "⌘⇧A" : "Ctrl+Shift+A";
export const UNDO_SHORTCUT = `${MOD}Z`;

/** Items of the meta line, with a dot between each two. */
function Dotted({ items }: { items: ReactNode[] }) {
  return items.map((item, index) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: the items are fixed slots of the meta line.
    <Fragment key={index}>
      {index > 0 && <span aria-hidden="true">·</span>}
      {item}
    </Fragment>
  ));
}

const CopyIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.75}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="size-3.5"
  >
    <rect x="8.5" y="8.5" width="11.5" height="11.5" rx="2" />
    <path d="M15.5 8.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5" />
  </svg>
);

const PlusIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.75}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="size-3.5"
  >
    <path d="M12 5v14M5 12h14" />
  </svg>
);

const ChevronIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2.25}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="size-3"
  >
    <path d="m6 9 6 6 6-6" />
  </svg>
);

const TickIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2.75}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="size-3"
  >
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </svg>
);

/**
 * Under an Answer in the thread (DESIGN.md D124; boards Chat-B1, B2, C2): on
 * hover or focus, and always on the newest, "Copy · Add to note ▾ ·
 * Regenerate" and its model and what it searched. While it is written, what it
 * is doing and Stop; once added, "Added to note · Undo".
 */
export function ThreadAnswerActions({
  answerId,
  canKeep,
  canRegenerate,
  streaming,
  meta,
  writing,
  onStop,
  onRegenerate,
}: {
  answerId: string | null;
  /** It has something to keep: Copy and Add to note. Not a failed Answer. */
  canKeep: boolean;
  canRegenerate: boolean;
  streaming: boolean;
  /** The model, the Search scope and what it searched. */
  meta: ReactNode[];
  /** What it is doing while it is written. */
  writing: ReactNode;
  onStop(): void;
  onRegenerate(): void;
}) {
  const t = useT();
  const thread = useThread();
  const pickButton = useRef<HTMLButtonElement>(null);
  const [copied, setCopied] = useState(false);
  if (!thread || !answerId) return null;
  const added = thread.added?.answerId === answerId ? thread.added : null;

  if (streaming) {
    return (
      <div contentEditable={false} data-testid="answer-meta" className="thread-answer-actions">
        <span className="thread-answer-meta">{writing}</span>
        <button
          type="button"
          data-testid="answer-stop"
          onClick={onStop}
          className="thread-answer-action"
        >
          <StopIcon className="size-3" />
          {t("answer.stop")}
        </button>
      </div>
    );
  }

  if (added) {
    const where = added.target.here
      ? t("thread.added.here")
      : t("thread.added.there", { title: added.target.title || t("mind.untitled") });
    return (
      <div
        contentEditable={false}
        data-testid="thread-added"
        className="thread-answer-actions thread-answer-actions--added"
      >
        <span aria-hidden="true" className="thread-added-icon">
          <TickIcon />
        </span>
        <span role="status" className="thread-added-text">
          {where}
        </span>
        <span aria-hidden="true">·</span>
        <button
          type="button"
          data-testid="thread-added-undo"
          title={t("thread.added.undoTitle", { shortcut: UNDO_SHORTCUT })}
          aria-label={t("thread.added.undoLabel", { where })}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => thread.undo()}
          className="thread-undo"
        >
          {t("thread.added.undo")}
        </button>
      </div>
    );
  }

  return (
    // biome-ignore lint/a11y/useSemanticElements: an Answer's buttons, named together; a fieldset is for a form's.
    <div
      role="group"
      aria-label={t("thread.answer.actions")}
      contentEditable={false}
      data-testid="answer-meta"
      className="thread-answer-actions"
    >
      {canKeep && (
        <>
          <button
            type="button"
            data-testid="answer-copy"
            aria-label={t(copied ? "thread.answer.copied" : "thread.answer.copyLabel", {
              shortcut: COPY_SHORTCUT,
            })}
            title={t("thread.answer.copyTitle", { shortcut: COPY_SHORTCUT })}
            // A selection in the Answer stays as it is.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => {
              void thread.copy(answerId).then((done) => {
                if (!done) return;
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              });
            }}
            className="thread-answer-action"
          >
            <CopyIcon />
            {copied ? t("thread.answer.copied") : t("thread.answer.copy")}
          </button>
          <span className="thread-add">
            <button
              type="button"
              data-testid="answer-add"
              aria-label={t("thread.answer.addLabel", { shortcut: ADD_SHORTCUT })}
              title={t("thread.answer.addTitle", { shortcut: ADD_SHORTCUT })}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => thread.add(answerId)}
              className="thread-answer-action thread-add-main"
            >
              <PlusIcon />
              {t("thread.answer.add")}
            </button>
            <button
              ref={pickButton}
              type="button"
              data-testid="answer-add-to"
              aria-haspopup="dialog"
              aria-label={t("thread.answer.addTo")}
              title={t("thread.answer.addToTitle")}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                if (pickButton.current) thread.pick(answerId, pickButton.current);
              }}
              className="thread-answer-action thread-add-more"
            >
              <ChevronIcon />
            </button>
          </span>
        </>
      )}
      {canRegenerate && (
        <button
          type="button"
          data-testid="answer-regenerate"
          aria-label={t("thread.answer.regenerateLabel")}
          title={t("answer.regenerate")}
          onMouseDown={(event) => event.preventDefault()}
          onClick={onRegenerate}
          className="thread-answer-action"
        >
          <RegenerateSmallIcon className="size-3.5" />
          {t("answer.regenerate")}
        </button>
      )}
      {meta.length > 0 && (
        <span className="thread-answer-meta">
          <Dotted items={meta} />
        </span>
      )}
    </div>
  );
}
