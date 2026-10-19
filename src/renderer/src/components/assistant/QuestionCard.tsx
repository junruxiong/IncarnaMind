import { isMacOS } from "@tiptap/core";
import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";
import type { UserQuestionRequest } from "../../../../core/api";
import { keysLabel } from "../../commands/keys";
import { useT } from "../../i18n";
import { useQuestions } from "../../questions";
import { typesText } from "../../undo";
import { CheckLineIcon } from "../lineIcons";

const CONTINUE_KEYS = "CmdOrCtrl+Enter";

const isCommand = (event: { metaKey: boolean; ctrlKey: boolean }) =>
  isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;

/**
 * The assistant's question in the tray (#1374; DESIGN.md, the assistant can ask with options):
 * the question, a line under it, the options as rows with the recommended one first, and Other…
 * when the User may write their own answer. One choice goes on at once; several, or words, go on
 * with Continue (⌘↵). Esc skips. A number picks a row while the focus isn't in a field.
 */
export function QuestionCard({ request }: { request: UserQuestionRequest }) {
  const t = useT();
  const answer = useQuestions((state) => state.answer);
  const { question } = request;
  const { options, multiple, other } = question;
  const [ticked, setTicked] = useState<ReadonlySet<number>>(
    () => new Set(options.flatMap((option, index) => (option.preselected ? [index] : []))),
  );
  const [wrote, setWrote] = useState("");
  const card = useRef<HTMLElement>(null);
  const titleId = useId();
  const keys = keysLabel(CONTINUE_KEYS, isMacOS());

  const go = (chosen: ReadonlySet<number>) => {
    const words = wrote.trim();
    if (chosen.size === 0 && !words) return;
    answer(request.requestId, {
      skipped: false,
      chosen: options.flatMap((option, index) => (chosen.has(index) ? [option.label] : [])),
      wrote: words || null,
    });
  };
  const pick = (index: number) => {
    if (!multiple) {
      go(new Set([index]));
      return;
    }
    const next = new Set(ticked);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    setTicked(next);
  };
  const skip = () => answer(request.requestId, { skipped: true });
  const rows = () => [
    ...(card.current?.querySelectorAll<HTMLElement>("[data-question-row]") ?? []),
  ];

  // The question takes the keys, its first row the focus, unless the User has words typed somewhere.
  useEffect(() => {
    const active = document.activeElement;
    const words =
      active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement
        ? active.value
        : (active?.textContent ?? "");
    if (typesText(active) && words.trim() !== "") return;
    card.current?.querySelector<HTMLElement>("[data-question-row]")?.focus();
  }, []);

  // A number picks a row from anywhere but a field, as the tray's other asks take ⌘↵.
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || typesText(event.target)) return;
      const number = Number(event.key);
      if (!Number.isInteger(number) || number < 1) return;
      if (number <= options.length) {
        event.preventDefault();
        pick(number - 1);
      } else if (other && number === options.length + 1) {
        event.preventDefault();
        card.current?.querySelector<HTMLInputElement>("[data-question-other]")?.focus();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      skip();
      return;
    }
    const onRow = (event.target as HTMLElement).closest("button[data-question-row]") !== null;
    // ⌘↵ goes on from anywhere in it; with several to tick, Enter on a row does too, and Space ticks.
    if (
      event.key === "Enter" &&
      ((isCommand(event) && (multiple || other)) || (multiple && onRow && !isCommand(event)))
    ) {
      event.preventDefault();
      go(ticked);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const all = rows();
    const at = all.indexOf(document.activeElement as HTMLElement);
    if (at < 0) return;
    event.preventDefault();
    const next = all[(at + (event.key === "ArrowDown" ? 1 : all.length - 1)) % all.length];
    next?.focus();
  };

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: the question's keys, heard while the focus is in it.
    <section
      ref={card}
      aria-labelledby={titleId}
      data-testid="question-card"
      data-multiple={multiple || undefined}
      className="approval-card choice-card"
      onKeyDown={onKeyDown}
    >
      <div className="approval-head">
        <p id={titleId} className="approval-title">
          <span aria-hidden="true" className="approval-dot" />
          <span className="approval-title-text">{question.question}</span>
        </p>
        {question.detail && <p className="approval-explanation">{question.detail}</p>}
      </div>
      <div role={multiple ? "group" : "radiogroup"} className="choice-options">
        {options.map((option, index) => {
          const on = ticked.has(index);
          return (
            <button
              // biome-ignore lint/suspicious/noArrayIndexKey: an option is its place in the question.
              key={index}
              type="button"
              {...(multiple
                ? { role: "checkbox", "aria-checked": on }
                : { role: "radio", "aria-checked": on })}
              data-question-row
              data-testid="question-option"
              data-recommended={option.recommended || undefined}
              className="choice-option"
              onClick={() => pick(index)}
            >
              <span aria-hidden="true" className="choice-key">
                {index + 1}
              </span>
              {multiple ? (
                <span aria-hidden="true" className="choice-box" data-on={on || undefined}>
                  {on && <CheckLineIcon className="size-3" />}
                </span>
              ) : (
                <span aria-hidden="true" className="choice-radio" data-on={on || undefined} />
              )}
              <span className="choice-label">
                <span className="choice-label-line">
                  <span className="choice-label-text" title={option.label}>
                    {option.label}
                  </span>
                  {option.recommended && (
                    <span data-testid="question-recommended" className="choice-recommended">
                      {t("asks.recommended")}
                    </span>
                  )}
                </span>
                {option.description && (
                  <span className="choice-description">{option.description}</span>
                )}
              </span>
            </button>
          );
        })}
        {other && (
          <label className="choice-option choice-other">
            <span aria-hidden="true" className="choice-key">
              {options.length + 1}
            </span>
            <input
              data-question-row
              data-question-other
              data-testid="question-other"
              value={wrote}
              placeholder={t("asks.other")}
              spellCheck={false}
              onChange={(event) => setWrote(event.target.value)}
              onKeyDown={(event) => {
                // Enter while an input method composes picks its word; it never sends.
                if (event.key !== "Enter" || event.nativeEvent.isComposing || isCommand(event))
                  return;
                event.preventDefault();
                go(ticked);
              }}
            />
          </label>
        )}
      </div>
      <div className="approval-actions">
        {(multiple || other) && (
          <button
            type="button"
            data-testid="question-continue"
            onClick={() => go(ticked)}
            className="approval-button approval-button--primary"
          >
            {t("asks.continue")}
            <span className="approval-keys">{keys}</span>
          </button>
        )}
        <button
          type="button"
          data-testid="question-skip"
          onClick={skip}
          className="approval-button choice-skip"
        >
          {t("asks.skip")}
          <span className="approval-keys">Esc</span>
        </button>
        <span className="choice-hint">{t(multiple ? "asks.hint.many" : "asks.hint.one")}</span>
      </div>
    </section>
  );
}
