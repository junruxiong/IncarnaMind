import type { Document } from "../../../core/api";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { fileStatusLabel } from "../linkedFolders";
import { useAppStore } from "../store";
import { MoreLineIcon } from "./lineIcons";
import {
  menuClass,
  menuItemClass,
  menuItemUnavailableClass,
  menuRuleClass,
  menuTitleClass,
} from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * A Document's "More" menu: rename it; its file, where the User keeps it
 * ("Open in default app" and "Show in folder", which the main process does
 * for live Documents whose file is there, and which say why not for a file
 * that is missing or can't be reached); and delete it, or, for a missing
 * one, remove it from IncarnaMind.
 */
export function DocumentFileMenu({
  item,
  buttonClassName,
  onRename,
  onDelete,
}: {
  item: Document;
  buttonClassName: string;
  onRename(): void;
  onDelete(): void;
}) {
  const t = useT();
  const menu = usePopoverMenu();
  const fileState = fileStatusLabel(item.fileStatus);
  const reason = fileState ? t(fileState.reason) : undefined;

  const run = (action: (documentId: string) => Promise<unknown>) => {
    if (fileState) return; // its file isn't there: the item says why
    menu.close();
    action(item.id).catch((failure: unknown) => {
      useAppStore.setState({ actionError: errorMessage(failure) });
    });
  };

  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="document-file-menu"
        aria-label={t("documents.more", { name: item.name })}
        title={t("documents.more", { name: item.name })}
        className={buttonClassName}
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("documents.more", { name: item.name })}
        data-testid="document-file-actions"
        className={menuClass}
      >
        <button
          type="button"
          role="menuitem"
          data-testid="rename-document"
          onClick={() => {
            menu.close();
            onRename();
          }}
          className={`${menuItemClass} pl-2`}
        >
          {t("documents.renameAction")}
        </button>
        <div className={menuRuleClass} />
        <p className={menuTitleClass}>{t("documents.copy.menuTitle")}</p>
        <button
          type="button"
          role="menuitem"
          data-testid="document-open-externally"
          aria-disabled={fileState ? true : undefined}
          title={reason}
          onClick={() => run(files.openDocumentExternally)}
          className={`${menuItemClass} ${menuItemUnavailableClass} pl-2`}
        >
          {t("documents.copy.open")}
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="document-show-in-folder"
          aria-disabled={fileState ? true : undefined}
          title={reason}
          onClick={() => run(files.showDocumentInFolder)}
          className={`${menuItemClass} ${menuItemUnavailableClass} pl-2`}
        >
          {t("documents.copy.showInFolder")}
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid={item.fileStatus === "missing" ? "remove-document" : "delete-document"}
          onClick={() => {
            menu.close();
            onDelete();
          }}
          className={`${menuItemClass} pl-2 text-danger`}
        >
          {t(item.fileStatus === "missing" ? "documents.removeAction" : "documents.deleteAction")}
        </button>
      </div>
    </>
  );
}
