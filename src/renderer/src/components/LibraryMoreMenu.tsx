import { useT } from "../i18n";
import { importReferences, useReferenceImport } from "../referenceImport";
import { MoreLineIcon } from "./lineIcons";
import { buttonStyle, menuClass, menuItemClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * The Library's "More" menu, beside Add Documents: what is done to the
 * Library's references rather than to one Document. Today that is
 * "Import references…" (#287), which is also in the command palette.
 */
export function LibraryMoreMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const running = useReferenceImport((state) => state.phase === "running");
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="library-more"
        aria-label={t("library.more")}
        title={t("library.more")}
        className={buttonStyle("ghost")}
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("library.more")}
        data-testid="library-more-items"
        className={menuClass}
      >
        <button
          type="button"
          role="menuitem"
          data-testid="import-references"
          disabled={running}
          onClick={() => {
            menu.close();
            void importReferences();
          }}
          className={`${menuItemClass} pl-2`}
        >
          {t("references.import.menu")}
        </button>
      </div>
    </>
  );
}
