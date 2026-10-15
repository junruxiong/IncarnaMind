import { useId } from "react";
import { files } from "../core";
import { feedbackUrl, ISSUES_URL, useFeedback } from "../feedback";
import { useGap, useLanguage, useT } from "../i18n";
import {
  buttonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  primaryButtonClass,
} from "./ui";
import { useModal } from "./useModal";

export function FeedbackDialog() {
  const t = useT();
  const gap = useGap();
  const language = useLanguage();
  const id = useId();
  const open = useFeedback((state) => state.open);
  const close = useFeedback((state) => state.close);
  const dialog = useModal(open);

  return (
    <dialog
      ref={dialog}
      onClose={close}
      data-testid="feedback-dialog"
      aria-labelledby={`${id}-title`}
      className={`${dialogClass} w-[30rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id={`${id}-title`} className={dialogTitleClass}>
          {t("feedback.title")}
        </h2>
        <p className={dialogTextClass}>
          {t("feedback.note", {
            version: __APP_VERSION__,
            system: t(`feedback.system.${files.titleBar}`),
          })}
        </p>
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
          <button type="button" onClick={close} className={buttonClass}>
            {t("feedback.cancel")}
          </button>
          <button
            type="button"
            data-testid="feedback-open"
            onClick={() => {
              window.open(feedbackUrl(language, __APP_VERSION__, files.titleBar), "_blank");
              close();
            }}
            className={primaryButtonClass}
          >
            {t("feedback.open")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
