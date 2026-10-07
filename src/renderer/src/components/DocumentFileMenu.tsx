import type { Document } from "../../../core/api";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { MoreIcon } from "./icons";
import { menuClass, menuItemClass, menuTitleClass, usePopoverMenu } from "./usePopoverMenu";

/**
 * A Document's menu for its file, where the User keeps it: "Open in default
 * app" and "Show in folder". The main process does both, for live Documents
 * whose file is there.
 */
export function DocumentFileMenu({
  item,
  buttonClassName,
}: {
  item: Document;
  buttonClassName: string;
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
        aria-label={t("documents.copy.menu", { name: item.name })}
        title={t("documents.copy.menu", { name: item.name })}
        className={buttonClassName}
      >
        <MoreIcon className="size-[14px]" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("documents.copy.menuTitle")}
        data-testid="document-file-actions"
        className={menuClass}
      >
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
          data-testid="document-show-in-folder"
          onClick={() => run(files.showDocumentInFolder)}
          className={`${menuItemClass} pl-2`}
        >
          {t("documents.copy.showInFolder")}
        </button>
      </div>
    </>
  );
}
