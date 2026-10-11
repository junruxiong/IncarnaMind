import { useFindSources } from "../findSources";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { FindSourcesLineIcon, PlusLineIcon } from "./lineIcons";
import { buttonStyle, menuClass, menuItemClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * The Library's "Add Documents ▾": files from this computer, or Find sources
 * (#296), papers found on OpenAlex, arXiv and PubMed. A menu button, its
 * items reached with the arrow keys, as the sidebar's "+" is.
 */
export function AddDocumentsMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const picking = useAppStore((state) => state.pickingDocuments);

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
        data-testid="library-add-documents"
        className={buttonStyle("secondary")}
      >
        {t("documents.add")}
        <span aria-hidden="true" className="text-[11px] text-ink-meta">
          ▾
        </span>
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("documents.add")}
        data-testid="library-add-documents-items"
        className={menuClass}
      >
        <button
          type="button"
          role="menuitem"
          data-testid="library-add-files"
          disabled={picking}
          onClick={choose(() => void useAppStore.getState().pickDocuments())}
          className={`${menuItemClass} pl-2`}
        >
          <PlusLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("documents.add.files")}</span>
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="library-find-sources"
          onClick={choose(() => useFindSources.getState().show())}
          className={`${menuItemClass} pl-2`}
        >
          <FindSourcesLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("documents.add.findSources")}</span>
        </button>
      </div>
    </>
  );
}
