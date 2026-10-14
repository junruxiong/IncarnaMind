import { useId } from "react";
import {
  buttonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  primaryButtonClass,
} from "../components/ui";
import { useModal } from "../components/useModal";
import { useT } from "../i18n";

/**
 * Asks before a change deletes a section break of a Word file (#104): the
 * pages after it would lose their own layout. Open while `retry` is set;
 * Delete makes the change, Keep it or Esc leaves the text as it was.
 */
export function SectionBreakDialog({
  retry,
  onClose,
}: {
  retry: (() => void) | null;
  onClose(): void;
}) {
  const t = useT();
  const dialog = useModal(retry !== null);
  const id = useId();
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      data-testid="section-break-dialog"
      aria-labelledby={`${id}-title`}
      className={`${dialogClass} w-[26rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id={`${id}-title`} className={dialogTitleClass}>
          {t("wordImport.sectionBreak.title")}
        </h2>
        <p className={dialogTextClass}>{t("wordImport.sectionBreak.body")}</p>
        <div className={dialogActionsClass}>
          <button type="button" onClick={onClose} className={buttonClass}>
            {t("wordImport.sectionBreak.keep")}
          </button>
          <button
            type="button"
            data-testid="section-break-delete"
            onClick={() => {
              const run = retry;
              onClose();
              run?.();
            }}
            className={primaryButtonClass}
          >
            {t("wordImport.sectionBreak.delete")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
