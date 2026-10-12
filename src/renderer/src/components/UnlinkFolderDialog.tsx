import type { LinkedFolder } from "../../../core/api";
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

/**
 * Asks before unlinking a folder, and says what it does and doesn't do: the
 * folder and its files stay as they are on disk, and what becomes of the
 * Citations that quote its Documents.
 */
export function UnlinkFolderDialog({
  target,
  onClose,
}: {
  target: { linked: LinkedFolder; name: string } | null;
  onClose(): void;
}) {
  const t = useT();
  const dialog = useModal(target !== null);
  const removeLinkedFolder = useAppStore((state) => state.removeLinkedFolder);

  const confirm = () => {
    if (target) void removeLinkedFolder(target.linked.id);
    onClose();
  };

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      data-testid="unlink-folder-dialog"
      aria-labelledby="unlink-folder-title"
      aria-describedby="unlink-folder-body"
      className={`${dialogClass} w-[28rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="unlink-folder-title" className={`${dialogTitleClass} break-words`}>
          {t("linkedFolders.unlink.title", { name: target?.name ?? "" })}
        </h2>
        <div id="unlink-folder-body" className="flex flex-col gap-2">
          <p className={dialogTextClass}>{t("linkedFolders.unlink.body")}</p>
          <p className={dialogTextClass}>{t("linkedFolders.unlink.citations")}</p>
        </div>
        <div className={dialogActionsClass}>
          <button type="button" onClick={onClose} className={buttonClass}>
            {t("linkedFolders.unlink.cancel")}
          </button>
          <button
            type="button"
            data-testid="confirm-unlink-folder"
            onClick={confirm}
            className={dangerButtonClass}
          >
            {t("linkedFolders.unlink.confirm")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
