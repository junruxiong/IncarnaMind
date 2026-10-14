import { useId } from "react";
import { FEEDBACK_URL, useFeedback } from "../feedback";
import { useT } from "../i18n";
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

/**
 * Help › Send feedback: opens the issue tracker's new-issue page in the
 * browser, and says first that it is public. The app sends nothing itself.
 */
export function FeedbackDialog() {
  const t = useT();
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
        <p className={dialogTextClass}>{t("feedback.note")}</p>
        <div className={dialogActionsClass}>
          <button type="button" onClick={close} className={buttonClass}>
            {t("feedback.cancel")}
          </button>
          <button
            type="button"
            data-testid="feedback-open"
            onClick={() => {
              window.open(FEEDBACK_URL, "_blank");
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
