import type { LinkedFolder } from "../../../core/api";
import { useLanguage, useT } from "../i18n";
import { formatCount, revealLabelKey } from "../linkedFolders";
import { useAppStore } from "../store";
import { MoreLineIcon } from "./lineIcons";
import {
  buttonClass,
  dangerButtonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  menuClass,
  menuItemClass,
  menuItemUnavailableClass,
  menuRuleClass,
} from "./ui";
import { useModal } from "./useModal";
import { usePopoverMenu } from "./usePopoverMenu";

/** "Show in Finder", "Show in Explorer" or "Show in file manager", for this system. */
const REVEAL_LABEL = revealLabelKey(navigator.platform);

/**
 * A Linked folder's "More" menu, on its row: pause or resume its indexing,
 * show it as Folders or as a flat list, download and index its online-only
 * files (when it has some), show it in the system's file manager, and unlink it.
 */
export function LinkedFolderMenu({
  linked,
  name,
  buttonClassName,
  onUnlink,
}: {
  linked: LinkedFolder;
  name: string;
  buttonClassName: string;
  onUnlink(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const setPaused = useAppStore((state) => state.setLinkedFolderPaused);
  const setLayout = useAppStore((state) => state.setLinkedFolderLayout);
  const download = useAppStore((state) => state.downloadOnlineOnlyFiles);
  const reveal = useAppStore((state) => state.showLinkedFolder);
  const menu = usePopoverMenu();
  const paused = linked.status === "paused";
  const unreachable = linked.status === "unavailable";
  const onlineOnly = linked.onlineOnly.downloading ? 0 : linked.onlineOnly.files;
  const label = t("linkedFolders.more", { name });

  /** Closes the menu, then does it. */
  const choose = (action: () => void) => () => {
    menu.close();
    action();
  };

  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="linked-folder-menu"
        aria-label={label}
        title={label}
        className={buttonClassName}
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={label}
        data-testid="linked-folder-actions"
        className={menuClass}
      >
        <button
          type="button"
          role="menuitem"
          data-testid="linked-folder-pause"
          onClick={choose(() => void setPaused(linked.id, !paused))}
          className={`${menuItemClass} pl-2`}
        >
          {t(paused ? "linkedFolders.resume" : "linkedFolders.pause")}
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="linked-folder-layout"
          onClick={choose(
            () => void setLayout(linked.id, linked.layout === "flat" ? "tree" : "flat"),
          )}
          className={`${menuItemClass} pl-2`}
        >
          {t(linked.layout === "flat" ? "linkedFolders.layout.tree" : "linkedFolders.layout.flat")}
        </button>
        {onlineOnly > 0 && (
          <button
            type="button"
            role="menuitem"
            data-testid="linked-folder-download"
            onClick={choose(() => void download(linked.id))}
            className={`${menuItemClass} pl-2`}
          >
            {onlineOnly === 1
              ? t("linkedFolders.downloadOnlineOnly.one")
              : t("linkedFolders.downloadOnlineOnly.other", {
                  count: formatCount(onlineOnly, language),
                })}
          </button>
        )}
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="linked-folder-reveal"
          aria-disabled={unreachable ? true : undefined}
          title={unreachable ? t("linkedFolders.reveal.unavailableReason") : linked.path}
          onClick={() => {
            if (!unreachable) choose(() => void reveal(linked.id))();
          }}
          className={`${menuItemClass} ${menuItemUnavailableClass} pl-2`}
        >
          {t(REVEAL_LABEL)}
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="unlink-folder"
          onClick={choose(onUnlink)}
          className={`${menuItemClass} pl-2 text-danger`}
        >
          {t("linkedFolders.unlinkAction")}
        </button>
      </div>
    </>
  );
}

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
