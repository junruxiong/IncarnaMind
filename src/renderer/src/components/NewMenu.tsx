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

const ADD_DOCUMENTS: NewItem = {
  id: "documents.add",
  testId: "plus-add-documents",
  icon: PlusLineIcon,
};

const NEW_ITEMS: readonly NewItem[] = [
  { id: "mind.new", testId: "plus-new-mind", icon: MindLineIcon },
  ADD_DOCUMENTS,
  { id: "folder.new", testId: "plus-new-folder", icon: FolderLineIcon },
];

const MORE_ITEMS: readonly NewItem[] = [
  { id: "word.import", testId: "plus-import-word", icon: WordIcon },
  { id: "meeting.record", testId: "plus-record-meeting", icon: RecordingLineIcon },
  { id: "comparison.start", testId: "plus-compare-documents", icon: TableIcon },
  { id: "verify.document", testId: "plus-verify-document", icon: VerifyLineIcon },
  {
    id: "folder.starters",
    testId: "plus-starter-folders",
    icon: FolderLineIcon,
    label: "folder.fromStarters",
  },
  { id: "webPage.add", testId: "plus-add-web-page", icon: WebPageIcon },
  {
    id: "sources.find",
    testId: "plus-find-sources",
    icon: FindSourcesLineIcon,
    label: "sidebar.plus.findSources",
  },
  { id: "linkedFolder.add", testId: "plus-link-folder", icon: FolderPlusLineIcon },
];

const WINDOW: CommandContext = { selection: [], focus: null, surface: "sidebar" };

/** Each item runs the command ⌘K lists, with its keys, so the two can't drift apart. */
export function NewMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const more = useSubmenu();
  const picking = useAppStore((state) => state.pickingDocuments);

  const entryOf = (each: NewItem) => {
    const command = commands.get(each.id);
    if (!command || !offered(command, WINDOW)) return null;
    return {
      label: each.label ? t(each.label) : labelOf(command, WINDOW, t),
      keys: menuKeysLabel(command),
      run: () => void commands.run(each.id, WINDOW).catch(useAppStore.getState().reportError),
    };
  };

  const item = (each: NewItem) => {
    const entry = entryOf(each);
    if (!entry) return null;
    const { icon: Icon } = each;
    return (
      <button
        key={each.testId}
        type="button"
        role="menuitem"
        data-testid={each.testId}
        disabled={each === ADD_DOCUMENTS && picking}
        onClick={() => {
          menu.close();
          entry.run();
        }}
        className={`${menuItemClass} pl-2`}
      >
        <Icon className="size-4 shrink-0 text-ink-meta" />
        <span className="flex-1">{entry.label}</span>
        {entry.keys && <span className="pl-4 text-[13px] text-ink-meta">{entry.keys}</span>}
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
        {/* Made as it opens: what each command offers is read then. */}
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
