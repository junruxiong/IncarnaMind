import type { ClipboardEvent, ReactNode } from "react";
import { shortcutLabel } from "../commands/keymap";
import { commands } from "../commands/registry";
import { TableIcon } from "../editor/icons";
import { useFindSources } from "../findSources";
import { useT } from "../i18n";
import { canRecordMeetings } from "../meetings";
import { usePruneSelection } from "../selection";
import { useAppStore } from "../store";
import { linksIn, useWebPages } from "../webPages";
import { CopiedLine } from "./CopiedLine";
import { DocumentsSection } from "./DocumentsSection";
import { GetStartedCard } from "./GettingStarted";
import {
  AppMark,
  DocumentLineIcon,
  FindSourcesLineIcon,
  FolderLineIcon,
  FolderPlusLineIcon,
  MindLineIcon,
  PlusLineIcon,
  RecordingLineIcon,
  SettingsLineIcon,
} from "./lineIcons";
import { SidebarSelection } from "./SidebarSelection";
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
 * column. The undo line, "Moved … · Undo", sits above the footer (see `UndoLine`).
 */
export function Sidebar({ width, onOpenSettings }: { width: number; onOpenSettings(): void }) {
  const t = useT();
  usePruneSelection();

  return (
    <aside
      aria-label={t("sidebar.label")}
      id="sidebar"
      data-surface="sidebar"
      data-testid="sidebar"
      className="flex min-w-[165px] shrink flex-col bg-frame"
      style={{ flexBasis: width }}
      onPaste={pasteLinks}
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
      </div>

      <GetStartedCard />
      <UndoLine />
      <CopiedLine />
      <footer
        data-testid="sidebar-footer"
        // Its rule is inset 8px each side, like the header's.
        className="relative flex shrink-0 flex-col p-2 before:absolute before:inset-x-2 before:top-0 before:h-px before:bg-rule before:content-['']"
      >
        <SidebarSelection />
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
      // Its rule is inset 8px each side, like the rows: it never meets the sidebar's edge. The
      // 8px on top put its buttons on the card's band, y 8 to 44, so every header shares a line.
      className="title-bar sidebar-header relative flex h-11 shrink-0 items-center pt-2 pr-2 after:absolute after:inset-x-2 after:bottom-0 after:h-px after:bg-rule after:content-['']"
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
 * Links pasted on the sidebar are saved as web pages: in the Folder whose
 * row has the focus, if one has.
 */
function pasteLinks(event: ClipboardEvent<HTMLElement>) {
  const target = event.target instanceof HTMLElement ? event.target : null;
  if (target?.isContentEditable || target?.closest("input, textarea")) return;
  const addresses = linksIn(event.clipboardData);
  if (addresses.length === 0) return;
  event.preventDefault();
  const folder = target?.closest('[data-testid="library-folder"]')?.getAttribute("data-folder-id");
  void useWebPages.getState().add(addresses, folder ?? null);
}

/**
 * The one "+" menu, the one place to add: New Mind, Import a Word file…,
 * Record a meeting (once Meetings are turned on), Compare Documents…, New Folder,
 * Add Documents…, Add a web page…,
 * Find sources… and Link a folder…. Opened by keyboard or mouse, its items
 * are menu items reached with the arrow keys.
 */
function PlusMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const createMind = useAppStore((state) => state.createMind);
  const pickDocuments = useAppStore((state) => state.pickDocuments);
  const picking = useAppStore((state) => state.pickingDocuments);
  const addLinkedFolder = useAppStore((state) => state.addLinkedFolder);
  const meetingsOn = useAppStore((state) => canRecordMeetings(state.settings));
  const reportError = useAppStore((state) => state.reportError);

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
          <span className="pl-4 text-[13px] text-ink-meta">{shortcutLabel("tab.new")}</span>
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="plus-import-word"
          onClick={choose(() => void commands.run("word.import").catch(reportError))}
          className={`${menuItemClass} pl-2`}
        >
          <DocumentLineIcon kind="docx" className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("wordImport.command")}</span>
        </button>
        {meetingsOn && (
          <button
            type="button"
            role="menuitem"
            data-testid="plus-record-meeting"
            onClick={choose(() => void commands.run("meeting.record").catch(reportError))}
            className={`${menuItemClass} pl-2`}
          >
            <RecordingLineIcon className="size-4 shrink-0 text-ink-meta" />
            <span className="flex-1">{t("meetings.record")}</span>
          </button>
        )}
        <button
          type="button"
          role="menuitem"
          data-testid="plus-compare-documents"
          onClick={choose(() => void commands.run("comparison.start").catch(reportError))}
          className={`${menuItemClass} pl-2`}
        >
          <TableIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("command.compareDocuments")}</span>
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="plus-new-folder"
          onClick={choose(() => void commands.run("folder.new").catch(reportError))}
          className={`${menuItemClass} pl-2`}
        >
          <FolderLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("menu.newFolder")}</span>
          <span className="pl-4 text-[13px] text-ink-meta">{shortcutLabel("folder.new")}</span>
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
          data-testid="plus-add-web-page"
          onClick={choose(() => void commands.run("webPage.add").catch(reportError))}
          className={`${menuItemClass} pl-2`}
        >
          <DocumentLineIcon kind="web" className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("sidebar.plus.addWebPage")}</span>
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="plus-find-sources"
          onClick={choose(() => useFindSources.getState().show())}
          className={`${menuItemClass} pl-2`}
        >
          <FindSourcesLineIcon className="size-4 shrink-0 text-ink-meta" />
          <span className="flex-1">{t("sidebar.plus.findSources")}</span>
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
