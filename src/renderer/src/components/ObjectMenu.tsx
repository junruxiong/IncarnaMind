import type { KeyboardEvent, MouseEvent } from "react";
import { menuKeysLabel } from "../commands/keymap";
import { buildMenu, type MenuItem } from "../commands/menus";
import {
  type CommandContext,
  commands,
  menuLabelOf,
  SURFACES,
  type Surface,
} from "../commands/registry";
import { useT } from "../i18n";
import type { ObjectKind, ObjectRef } from "../objects";
import { useAppStore } from "../store";
import { MoreLineIcon } from "./lineIcons";
import { rowActionButtonClass } from "./sidebarRows";
import { menuClass, menuItemClass, menuItemUnavailableClass, menuRuleClass } from "./ui";
import { openMenuFromRightClick, usePopoverMenu } from "./usePopoverMenu";

const TEST_ID_PREFIX: Record<ObjectKind, string> = {
  mind: "mind",
  conversation: "conversation",
  document: "document-file",
  folder: "folder",
  tag: "tag",
  linkedFolder: "linked-folder",
  citation: "citation",
  answer: "answer",
  block: "block",
  tab: "tab",
};

const keyOf = (target: ObjectRef) => `${target.kind}:${target.id}`;

/** A key that opens the focused row's menu: the context-menu key, or Shift+F10. */
const opensMenu = (event: KeyboardEvent) =>
  event.key === "ContextMenu" || (event.shiftKey && event.key === "F10");

const ROW = "[data-object-row]";

function focusRow(from: HTMLElement | null): void {
  const row = from?.closest<HTMLElement>(ROW);
  if (!row) return;
  const own = row.getAttribute("tabindex") !== null;
  (own
    ? row
    : row.querySelector<HTMLElement>(
        "[data-row-main], [data-tree-row], button:not([data-object-menu]), a[href], [tabindex]",
      )
  )?.focus();
}

/**
 * What a row needs for its object's menu: `buttonProps` for the ⋯ button (so
 * the row finds it), and `rowProps` for the row, so a right-click, the
 * context-menu key or Shift+F10 on it opens the menu the ⋯ button opens.
 */
export function useObjectMenu(target: ObjectRef) {
  const key = keyOf(target);
  const open = (row: HTMLElement, buttons: number): boolean => {
    const button = row.querySelector<HTMLButtonElement>(`[data-object-menu="${CSS.escape(key)}"]`);
    if (!button) return false;
    openMenuFromRightClick(button, buttons);
    return true;
  };
  return {
    buttonProps: { "data-object-menu": key },
    rowProps: {
      "data-object-row": "",
      onContextMenu(event: MouseEvent<HTMLElement>) {
        const row = event.currentTarget;
        // Not from a menu or picker drawn from this row but sitting elsewhere in the page, nor an inner row's.
        if (!(event.target instanceof Node) || !row.contains(event.target)) return;
        if (event.defaultPrevented || (event.target as Element).closest('[role="menu"]')) return;
        if (open(row, event.buttons)) {
          event.preventDefault();
          // The row's own menu only: not that of a row that holds this one.
          event.stopPropagation();
        }
      },
      onKeyDown(event: KeyboardEvent<HTMLElement>) {
        if (!opensMenu(event) || event.defaultPrevented) return;
        const field = event.target;
        if (field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement) return;
        if (open(event.currentTarget, 0)) {
          event.preventDefault();
          event.stopPropagation();
        }
      },
    },
  };
}

const surfaceOf = (element: Element | null): Surface => {
  const marked = element?.closest<HTMLElement>("[data-surface]")?.dataset.surface;
  return SURFACES.find((each) => each === marked) ?? "sidebar";
};

/**
 * The ⋯ button of an object's row, and its menu. The menu acts on `target`, or,
 * when `selection` holds it, on the whole selection (#225). `here` holds what
 * only the row showing the object can do (see `CommandContext.here`).
 */
export function ObjectMenu({
  target,
  label,
  selection,
  here,
  className = rowActionButtonClass,
  tabIndex,
}: {
  target: ObjectRef;
  /** What the button and the menu are called: "More for Board minutes". */
  label: string;
  selection?: readonly ObjectRef[];
  here?: CommandContext["here"];
  className?: string;
  tabIndex?: number;
}) {
  const t = useT();
  const menu = usePopoverMenu();
  const { buttonProps } = useObjectMenu(target);
  const prefix = TEST_ID_PREFIX[target.kind];
  const button = menu.buttonProps.ref;

  const ctx: CommandContext = {
    selection: selection?.some((each) => each.kind === target.kind && each.id === target.id)
      ? selection
      : [],
    focus: target,
    surface: surfaceOf(button.current),
    ...(here && { here }),
  };
  const entries = menu.open ? buildMenu(target.kind, ctx) : [];

  const run = (item: MenuItem) => {
    if (item.enabled !== true) return;
    menu.close();
    commands.run(item.command.id, ctx).catch(useAppStore.getState().reportError);
  };

  return (
    <>
      <button
        {...menu.buttonProps}
        {...buttonProps}
        type="button"
        data-testid={`${prefix}-menu`}
        aria-label={label}
        title={label}
        {...(tabIndex !== undefined && { tabIndex })}
        className={className}
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={label}
        data-testid={`${prefix}-actions`}
        title=""
        className={menuClass}
        onKeyDown={(event) => {
          menu.menuProps.onKeyDown(event);
          // The ⋯ button can be out of sight, so Esc returns to the row.
          if (event.key === "Escape") focusRow(button.current);
        }}
      >
        {entries.map((entry, index) => {
          if (entry === "rule") {
            // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place.
            return <div key={`rule-${index}`} className={menuRuleClass} />;
          }
          const { command, enabled } = entry;
          const keys = menuKeysLabel(command);
          const reason = enabled === true ? undefined : t(enabled.reason);
          return (
            <button
              key={command.id}
              type="button"
              role="menuitem"
              data-testid={command.id}
              aria-disabled={enabled === true ? undefined : true}
              title={reason}
              onClick={() => run(entry)}
              className={`${menuItemClass} ${menuItemUnavailableClass} pl-2 ${
                command.destructive ? "text-danger" : ""
              }`}
            >
              <span className="flex-1">{menuLabelOf(command, ctx, t)}</span>
              {keys && <span className="pl-4 text-[13px] text-ink-meta">{keys}</span>}
            </button>
          );
        })}
      </div>
    </>
  );
}
