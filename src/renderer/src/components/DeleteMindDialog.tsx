import type { Mind } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  buttonClass,
  dangerButtonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
} from "./ui";
import { useModal } from "./useModal";

/** Asks before deleting a Mind. Open while `mind` is set. */
export function DeleteMindDialog({ mind, onClose }: { mind: Mind | null; onClose(): void }) {
  const t = useT();
  const dialog = useModal(mind !== null);
  const deleteMind = useAppStore((state) => state.deleteMind);

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="delete-mind-title"
      className={`${dialogClass} w-[26rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="delete-mind-title" className={dialogTitleClass}>
          {t("mind.delete")}
        </h2>
        <p className={dialogTextClass}>
          {t("mind.delete.body", { title: mind?.title || t("mind.untitled") })}
        </p>
        <div className={dialogActionsClass}>
          <button type="button" onClick={onClose} className={buttonClass}>
            {t("mind.delete.cancel")}
          </button>
          <button
            type="button"
            data-testid="confirm-delete-mind"
            onClick={() => {
              if (mind) void deleteMind(mind.id);
              onClose();
            }}
            className={dangerButtonClass}
          >
            {t("mind.delete.confirm")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
