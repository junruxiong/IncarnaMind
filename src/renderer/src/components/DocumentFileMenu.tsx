import type { Document } from "../../../core/api";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { MoreLineIcon } from "./lineIcons";
import { menuClass, menuItemClass, menuRuleClass, menuTitleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * A Document's "More" menu: rename it; its original file, whose stored copy
 * is named by a hash ("Open in default app" opens a temporary copy named
 * after the Document, and "Save a copy…" saves one where the User picks; the
 * main process does both, for live Documents only); and delete it.
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

  const run = (action: (documentId: string) => Promise<unknown>) => {
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
          onClick={() => run(files.openDocumentExternally)}
          className={`${menuItemClass} pl-2`}
        >
          {t("documents.copy.open")}
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="document-save-copy"
          onClick={() => run(files.saveDocumentCopy)}
          className={`${menuItemClass} pl-2`}
        >
          {t("documents.copy.save")}
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="delete-document"
          onClick={() => {
            menu.close();
            onDelete();
          }}
          className={`${menuItemClass} pl-2 text-danger`}
        >
          {t("documents.deleteAction")}
        </button>
      </div>
    </>
  );
}
