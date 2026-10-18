import type { ComponentType } from "react";
import type { CommandId } from "../../../shared/commands";
import type { MessageKey } from "../../../shared/i18n";
import { menuKeysLabel } from "../commands/keymap";
import { type CommandContext, commands, labelOf, offered } from "../commands/registry";
import { TableIcon } from "../editor/icons";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  ChevronRightLineIcon,
  DocumentLineIcon,
  FindSourcesLineIcon,
  FolderLineIcon,
  FolderPlusLineIcon,
  MindLineIcon,
  MoreLineIcon,
  PlusLineIcon,
  RecordingLineIcon,
  VerifyLineIcon,
} from "./lineIcons";
import { iconButtonClass, menuClass, menuItemClass } from "./ui";
import { usePopoverMenu, useSubmenu } from "./usePopoverMenu";

interface NewItem {
  id: CommandId;
  testId: string;
  icon: ComponentType<{ className?: string }>;
  /** The menu's own words, where the command's ⌘K label says more than the menu needs. */
  label?: MessageKey;
}

const WordIcon = ({ className }: { className?: string }) => (
  <DocumentLineIcon kind="docx" className={className} />
);
const WebPageIcon = ({ className }: { className?: string }) => (
  <DocumentLineIcon kind="web" className={className} />
);

/** The New menu's four rows: what most people start with. */
const NEW_ITEMS: readonly NewItem[] = [
  { id: "mind.new", testId: "plus-new-mind", icon: MindLineIcon },
  { id: "documents.add", testId: "plus-add-documents", icon: PlusLineIcon },
  { id: "folder.new", testId: "plus-new-folder", icon: FolderLineIcon },
];

/** Under More: the rest, each also in ⌘K; one whose feature is off isn't offered. */
const MORE_ITEMS: readonly NewItem[] = [
  { id: "linkedFolder.add", testId: "plus-link-folder", icon: FolderPlusLineIcon },
  { id: "meeting.record", testId: "plus-record-meeting", icon: RecordingLineIcon },
  { id: "comparison.start", testId: "plus-compare-documents", icon: TableIcon },
  { id: "verify.document", testId: "plus-verify-document", icon: VerifyLineIcon },
  { id: "webPage.add", testId: "plus-add-web-page", icon: WebPageIcon },
  { id: "word.import", testId: "plus-import-word", icon: WordIcon },
  {
    id: "folder.starters",
    testId: "plus-starter-folders",
    icon: FolderLineIcon,
    label: "folder.fromStarters",
  },
  {
    id: "sources.find",
    testId: "plus-find-sources",
    icon: FindSourcesLineIcon,
    label: "sidebar.plus.findSources",
  },
];

const WINDOW: CommandContext = { selection: [], focus: null, surface: "sidebar" };

/**
 * The sidebar header's +, the one place to start something: New Mind, Add
 * Documents…, New Folder, and More for the rest. Each item is the command ⌘K
 * lists, with its keys. Opened by keyboard or mouse, its items are menu items
 * reached with the arrow keys.
 */
export function NewMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const more = useSubmenu();

  const run = (id: CommandId) => {
    menu.close();
    commands.run(id, WINDOW).catch(useAppStore.getState().reportError);
  };

  const item = ({ id, testId, icon: Icon, label }: NewItem) => {
    const command = commands.get(id);
    if (!command || !offered(command, WINDOW)) return null;
    const keys = menuKeysLabel(command);
    return (
      <button
        key={id}
        type="button"
        role="menuitem"
        data-testid={testId}
        onClick={() => run(id)}
        className={`${menuItemClass} pl-2`}
      >
        <Icon className="size-4 shrink-0 text-ink-meta" />
        <span className="flex-1">{label ? t(label) : labelOf(command, WINDOW, t)}</span>
        {keys && <span className="pl-4 text-[13px] text-ink-meta">{keys}</span>}
      </button>
    );
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
        {/* Made as it opens: whether meetings are on, or the assistant is there to chat in, is read then. */}
        {menu.open && (
          <>
            {NEW_ITEMS.map(item)}
            <button
              {...more.itemProps}
              type="button"
              data-testid="plus-more"
              className={`${menuItemClass} pl-2 aria-expanded:bg-hover`}
            >
              <MoreLineIcon className="size-4 shrink-0 text-ink-meta" />
              <span className="flex-1">{t("sidebar.plus.more")}</span>
              <ChevronRightLineIcon className="size-3.5 text-ink-meta" />
            </button>
          </>
        )}
      </div>
      <div
        {...more.menuProps}
        role="menu"
        aria-label={t("sidebar.plus.more")}
        data-testid="plus-more-menu"
        className={menuClass}
      >
        {menu.open && MORE_ITEMS.map(item)}
      </div>
    </>
  );
}
