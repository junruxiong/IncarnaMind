import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import type { Mind } from "../../../core/api";
import { startMoveDrag } from "../folderDrag";
import { useT } from "../i18n";
import { useMindStatus } from "../mindStatus";
import { useAppStore } from "../store";
import { isMoveToKey, MOVE_TO_SHORTCUT, RENAME_SHORTCUT, TREE_ROW } from "../treeKeys";
import { ExampleChip, useIsExample } from "./GettingStarted";
import { MindLineIcon, MoreLineIcon } from "./lineIcons";
import { MoveToPicker } from "./MoveToPicker";
import {
  openRowMenu,
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowInputClass,
  rowPadding,
} from "./sidebarRows";
import { menuClass, menuItemClass, menuRuleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * A Mind in the sidebar's tree, in its Folder (`depth` 1) or Not in a Folder
 * (0): its title, and an amber dot while one of its Answers waits for the
 * User's approval. Clicking opens it (⌘-click or a middle click in a new
 * tab). It renames in place: double-click it, press Enter on it while it is
 * open, or F2; Enter saves, Esc cancels, and a blank title changes nothing.
 * It drags onto a Folder; its menu (⋯, or a right-click) and ⇧⌘M offer the
 * same as "Move to…".
 */
export function MindRow({
  mind,
  depth,
  onDelete,
}: {
  mind: Mind;
  depth: number;
  /** Asks before deleting it. */
  onDelete(mind: Mind): void;
}) {
  const t = useT();
  const isOpen = useAppStore((state) => state.openMindId === mind.id && !state.libraryOpen);
  const openMind = useAppStore((state) => state.openMind);
  const waiting = useMindStatus(mind.id) === "waiting-for-approval";
  const isExample = useIsExample(mind.id);
  const [renaming, setRenaming] = useState(false);
  const [moving, setMoving] = useState<HTMLElement | null>(null);
  const button = useRef<HTMLButtonElement>(null);
  const title = mind.title || t("mind.untitled");

  if (renaming) {
    return (
      <li className={rowClass(isOpen)} data-testid="mind-row" data-mind-id={mind.id}>
        <span
          className="flex h-full w-full min-w-0 items-center gap-2 pr-1"
          style={rowPadding(depth)}
        >
          <MindLineIcon className={rowIconClass(isOpen)} />
          <RenameMind
            mind={mind}
            onDone={() => {
              setRenaming(false);
              // Back on its row, so the keyboard carries on from there.
              requestAnimationFrame(() => button.current?.focus());
            }}
          />
        </span>
      </li>
    );
  }

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    // F2, or Enter on the Mind already open (selected): its title, in place.
    if (event.key === "F2" || (event.key === "Enter" && isOpen && !event.metaKey)) {
      event.preventDefault();
      setRenaming(true);
    } else if (isMoveToKey(event)) {
      event.preventDefault();
      setMoving(event.currentTarget);
    }
  };

  return (
    <li
      data-testid="mind-row"
      data-mind-id={mind.id}
      data-folder-id={mind.folderId ?? ""}
      draggable
      onDragStart={(event) => startMoveDrag(event, { mindIds: [mind.id] })}
      onContextMenu={(event) => openRowMenu(event, "mind-menu")}
      className={rowClass(isOpen)}
    >
      <button
        ref={button}
        type="button"
        {...TREE_ROW}
        data-testid="mind-list-item"
        data-mind-id={mind.id}
        aria-current={isOpen ? "page" : undefined}
        title={title}
        // ⌘-click (Ctrl-click) or a middle click opens it in a new tab, as in a browser.
        onClick={(event) => openMind(mind.id, { newTab: event.metaKey || event.ctrlKey })}
        onMouseDown={(event) => {
          if (event.button === 1) event.preventDefault(); // no autoscroll
        }}
        onAuxClick={(event) => {
          if (event.button === 1) openMind(mind.id, { newTab: true });
        }}
        onDoubleClick={() => setRenaming(true)}
        onKeyDown={onKeyDown}
        className={rowButtonClass}
        style={rowPadding(depth)}
      >
        <MindLineIcon className={rowIconClass(isOpen)} />
        <span
          data-testid="row-text"
          className={`min-w-0 flex-1 truncate ${mind.title ? "" : "text-ink-meta"}`}
        >
          {title}
        </span>
        {isExample && <ExampleChip />}
        {waiting && (
          <span
            role="img"
            data-testid="mind-approval"
            aria-label={t("approvals.answer.waiting")}
            title={t("approvals.answer.waiting")}
            className="size-1.5 shrink-0 rounded-full bg-attention"
          />
        )}
      </button>
      <div className={rowActionsClass}>
        <MindMenu
          mind={mind}
          onMove={(anchor) => setMoving(anchor)}
          onRename={() => setRenaming(true)}
          onDelete={() => onDelete(mind)}
        />
      </div>
      <MoveToPicker
        anchor={moving}
        name={title}
        current={mind.folderId}
        onChoose={(folderId) => {
          if (folderId !== mind.folderId) {
            void useAppStore.getState().moveToFolder({ mindIds: [mind.id] }, folderId);
          }
        }}
        onClose={() => setMoving(null)}
      />
    </li>
  );
}

/**
 * A Mind's menu, from its ⋯ or a right-click, in the shared order: Open,
 * Open in a new tab; Move to…; Rename; Delete….
 */
function MindMenu({
  mind,
  onMove,
  onRename,
  onDelete,
}: {
  mind: Mind;
  /** Opens "Move to…" beside this element. */
  onMove(anchor: HTMLElement): void;
  onRename(): void;
  onDelete(): void;
}) {
  const t = useT();
  const menu = usePopoverMenu();
  const openMind = useAppStore((state) => state.openMind);
  const label = t("mind.more", { title: mind.title || t("mind.untitled") });
  const anchor = useRef<HTMLButtonElement>(null);

  /** Closes the menu, then does it. */
  const choose = (action: () => void) => () => {
    menu.close();
    action();
  };

  return (
    <>
      <button
        {...menu.buttonProps}
        ref={(element) => {
          menu.buttonProps.ref.current = element;
          anchor.current = element;
        }}
        type="button"
        data-testid="mind-menu"
        aria-label={label}
        title={label}
        className={rowActionButtonClass}
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={label}
        data-testid="mind-actions"
        className={menuClass}
      >
        <button
          type="button"
          role="menuitem"
          data-testid="mind-open"
          onClick={choose(() => openMind(mind.id))}
          className={`${menuItemClass} pl-2`}
        >
          {t("mind.open")}
        </button>
        <button
          type="button"
          role="menuitem"
          data-testid="mind-open-new-tab"
          onClick={choose(() => openMind(mind.id, { newTab: true }))}
          className={`${menuItemClass} pl-2`}
        >
          {t("mind.openNewTab")}
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="mind-move-to"
          onClick={choose(() => {
            // Once the menu has closed and given the focus back to its button.
            requestAnimationFrame(() => {
              if (anchor.current) onMove(anchor.current);
            });
          })}
          className={`${menuItemClass} pl-2`}
        >
          <span className="flex-1">{t("mind.moveTo")}</span>
          <span className="pl-4 text-[13px] text-ink-meta">{MOVE_TO_SHORTCUT}</span>
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="rename-mind"
          onClick={choose(onRename)}
          className={`${menuItemClass} pl-2`}
        >
          <span className="flex-1">{t("mind.rename")}</span>
          <span className="pl-4 text-[13px] text-ink-meta">{RENAME_SHORTCUT}</span>
        </button>
        <div className={menuRuleClass} />
        <button
          type="button"
          role="menuitem"
          data-testid="delete-mind"
          onClick={choose(onDelete)}
          className={`${menuItemClass} pl-2 text-danger`}
        >
          {t("mind.deleteAction")}
        </button>
      </div>
    </>
  );
}

/** A Mind's title typed in its row: Enter or leaving the field saves, Esc cancels. */
function RenameMind({ mind, onDone }: { mind: Mind; onDone(): void }) {
  const t = useT();
  const renameMind = useAppStore((state) => state.renameMind);
  const [value, setValue] = useState(mind.title);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);

  useEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);

  const finish = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    // A blank title changes nothing.
    if (save && value.trim() !== "" && value.trim() !== mind.title) void renameMind(mind.id, value);
    onDone();
  };

  return (
    <input
      ref={field}
      value={value}
      data-testid="mind-rename"
      aria-label={t("mind.renameLabel", { title: mind.title || t("mind.untitled") })}
      placeholder={t("mind.untitled")}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter" || event.key === "Escape") {
          event.preventDefault();
          // Esc cancels here, and closes nothing else.
          event.stopPropagation();
          finish(event.key === "Enter");
        }
      }}
      onBlur={() => finish(true)}
      className={rowInputClass}
    />
  );
}
