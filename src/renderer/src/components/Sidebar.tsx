import type { ReactNode } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { DocumentsSection } from "./DocumentsSection";
import { GetStartedCard } from "./GettingStarted";
import {
  AppMark,
  FolderLineIcon,
  FolderPlusLineIcon,
  MindLineIcon,
  PlusLineIcon,
  SettingsLineIcon,
} from "./lineIcons";
import { NEW_MIND_SHORTCUT } from "./MindTabs";
import { SidebarStatus } from "./SidebarStatus";
import { SidebarToggle } from "./SidebarToggle";
import { rowButtonClass, rowClass, rowIconClass } from "./sidebarRows";
import { UndoLine } from "./UndoLine";
import { iconButtonClass, menuClass, menuItemClass, menuRuleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * The sidebar, on the frame: a 44px header like every pane's (the window's
 * title bar starts there, with the app's mark or macOS's traffic lights) with
 * the one "+" menu, then the tree in 28px rows: the Folders, each with its
 * Minds and Documents, and Not in a Folder (see `DocumentsSection`), and a
 * footer with the tagging and processing status above Settings. Every row
 * puts its icon at x 16 and its text at x 40; section labels share the icon
 * column. "Moved … · Undo" floats at the tree's foot (see `UndoLine`).
 */
export function Sidebar({ width, onOpenSettings }: { width: number; onOpenSettings(): void }) {
  const t = useT();

  return (
    <aside
      aria-label={t("sidebar.label")}
      id="sidebar"
      data-testid="sidebar"
      className="flex min-w-[165px] shrink flex-col bg-frame"
      style={{ flexBasis: width }}
    >
      <SidebarHeader>
        <PlusMenu />
      </SidebarHeader>

      <div className="relative flex min-h-0 flex-1 flex-col">
        <div
          data-testid="sidebar-tree"
          className="thin-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto px-2 pt-2 pb-3"
        >
          <DocumentsSection />
        </div>
        <UndoLine />
      </div>

      <GetStartedCard />
      <footer
        data-testid="sidebar-footer"
        className="flex shrink-0 flex-col border-t border-rule p-2"
      >
        <SidebarStatus />
        <div className={rowClass(false)}>
          <button type="button" onClick={onOpenSettings} className={rowButtonClass}>
            <SettingsLineIcon className={rowIconClass(false)} />
            <span className="truncate">{t("sidebar.settings")}</span>
          </button>
        </div>
      </footer>
    </aside>
  );
}

/**
 * The sidebar's 44px header, the start of the window's title bar: macOS's
 * traffic lights sit here, in place of the app's mark (styles.css, "The title
 * bar"). The "+" menu sits at its right end. The startup shell draws it too.
 */
export function SidebarHeader({ children }: { children?: ReactNode }) {
  return (
    <header
      data-testid="sidebar-header"
      // Its rule is inset 8px each side, like the rows: it never meets the sidebar's edge.
      className="title-bar sidebar-header relative flex h-11 shrink-0 items-center pr-2 after:absolute after:inset-x-2 after:bottom-0 after:h-px after:bg-rule after:content-['']"
    >
      <AppMark />
      <SidebarToggle place="header" />
      <span className="flex-1" />
      {children}
    </header>
  );
}

/**
 * The "+" at the header's right end, drawn alone while the app loads (nothing
 * in the startup shell can be used yet).
 */
export function PlusMenuPlaceholder() {
  return (
    <span aria-hidden="true" className={`${iconButtonClass} text-ink-secondary`}>
      <PlusLineIcon className="size-4" />
    </span>
  );
}

/**
 * The one "+" menu, the one place to add: New Mind, New Folder…, Add
 * Documents… and Link a folder…. Opened by keyboard or mouse, its items are
 * menu items reached with the arrow keys.
 */
function PlusMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const createMind = useAppStore((state) => state.createMind);
  const pickDocuments = useAppStore((state) => state.pickDocuments);
  const picking = useAppStore((state) => state.pickingDocuments);
  const addLinkedFolder = useAppStore((state) => state.addLinkedFolder);

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
        data-testid="plus-menu"
        aria-label={t("sidebar.plus")}
        title={t("sidebar.plus")}
        className={`${iconButtonClass} text-ink-secondary`}
      >
        <PlusLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("sidebar.plus")}
        data-testid="plus-menu-items"
        className={menuClass}
      >
        <button
          type="button"
          role="menuitem"
          data-testid="plus-new-mind"
          onClick={choose(() => void createMind())}
          className={`${menuItemClass} pl-2`}
        >
          <MindLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("sidebar.newMind")}</span>
          <span className="pl-4 text-[13px] text-ink-meta">{NEW_MIND_SHORTCUT}</span>
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="plus-new-folder"
          onClick={choose(() => useAppStore.getState().openLibrary("new"))}
          className={`${menuItemClass} pl-2`}
        >
          <FolderLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("sidebar.plus.newFolder")}</span>
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="plus-add-documents"
          disabled={picking}
          onClick={choose(() => void pickDocuments())}
          className={`${menuItemClass} pl-2`}
        >
          <PlusLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("sidebar.plus.addDocuments")}</span>
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="plus-link-folder"
          onClick={choose(() => void addLinkedFolder())}
          className={`${menuItemClass} pl-2`}
        >
          <FolderPlusLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("sidebar.plus.linkFolder")}</span>
        </button>
      </div>
    </>
  );
}
