import { useId, useState } from "react";
import { ISSUES_URL, mailDraft, useFeedback } from "../feedback";
import { useGap, useT } from "../i18n";
import { TypingTextarea } from "./TypingTextarea";
import {
  buttonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  fieldLabelClass,
  inputClass,
  primaryButtonClass,
} from "./ui";
import { useModal } from "./useModal";

/**
 * Help › Send feedback: the User writes it here, and it opens in their mail
 * app as a draft, so they see exactly what goes before sending. The app sends
 * nothing itself. Bugs link to the issue tracker.
 */
export function FeedbackDialog() {
  const t = useT();
  const gap = useGap();
  const id = useId();
  const open = useFeedback((state) => state.open);
  const close = useFeedback((state) => state.close);
  const dialog = useModal(open);
  const [text, setText] = useState("");

  const finish = () => {
    setText("");
    close();
  };
  const openDraft = () => {
    window.open(mailDraft(t("feedback.subject"), text), "_blank");
    finish();
  };

  return (
    <dialog
      ref={dialog}
      onClose={finish}
      data-testid="feedback-dialog"
      aria-labelledby={`${id}-title`}
      className={`${dialogClass} w-[30rem]`}
    >
      <form
        className={dialogBodyClass}
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim() !== "") openDraft();
        }}
      >
        <h2 id={`${id}-title`} className={dialogTitleClass}>
          {t("feedback.title")}
        </h2>
        <p className={dialogTextClass}>{t("feedback.note")}</p>
        <div className="flex flex-col gap-1.5">
          <label htmlFor={`${id}-text`} className={fieldLabelClass}>
            {t("feedback.label")}
          </label>
          <TypingTextarea
            id={`${id}-text`}
            data-testid="feedback-text"
            value={text}
            rows={6}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && text.trim() !== "") {
                openDraft();
              }
            }}
            className={`${inputClass} resize-y`}
          />
        </div>
        <p className={dialogTextClass}>
          {t("feedback.bug.lead")}
          {gap(t("feedback.bug.lead"), t("feedback.bug.link"))}
          <a
            href={ISSUES_URL}
            target="_blank"
            rel="noreferrer"
            className="text-accent underline-offset-2 hover:text-accent-strong hover:underline"
          >
            {t("feedback.bug.link")}
          </a>
        </p>
        <div className={dialogActionsClass}>
          <button type="button" onClick={finish} className={buttonClass}>
            {t("feedback.cancel")}
          </button>
          <button
            type="submit"
            data-testid="feedback-open-draft"
            disabled={text.trim() === ""}
            className={primaryButtonClass}
          >
            {t("feedback.open")}
          </button>
        </div>
      </form>
    </dialog>
  );
}
