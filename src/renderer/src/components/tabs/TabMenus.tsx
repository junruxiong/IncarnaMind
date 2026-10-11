import type { ReactNode } from "react";
import { EXPORT_FORMATS, type LayoutTab, MAX_TAB_GROUPS } from "../../../../core/api";
import type { CommandId } from "../../../../shared/commands";
import { shortcutLabel } from "../../commands/keymap";
import { type CommandContext, commands, isEnabled, offered } from "../../commands/registry";
import { FORMAT_LABELS } from "../../exports";
import { useT } from "../../i18n";
import { groupOfTab } from "../../layoutSlice";
import { useAppStore } from "../../store";
import { MoreLineIcon } from "../lineIcons";
import { menuClass, menuItemClass, menuRuleClass } from "../ui";
import { usePopoverMenu } from "../usePopoverMenu";
import { focusTab } from "./tabActions";

/** One item of a tab's or group's menu, which runs a command, or a rule between them. */
type Item =
  | {
      /** Its test ID, after "tab-menu-". */
      id: string;
      command: CommandId;
      /** Shorter than the command's label: the menu says what it acts on. */
      label: string;
      disabled?: boolean;
    }
  | "rule";

const onTab = (tabId: string | undefined): CommandContext => ({
  selection: [],
  focus: tabId ? { kind: "tab", id: tabId } : null,
  surface: "mind",
});

function MenuItems({
  items,
  ctx,
  close,
}: {
  items: (Item | false)[];
  ctx: CommandContext;
  close(): void;
}) {
  return items.map((item, index) => {
    if (item === false) return null;
    if (item === "rule") {
      // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place.
      return <div key={`rule-${index}`} className={menuRuleClass} />;
    }
    const command = commands.get(item.command);
    const usable = command !== undefined && offered(command, ctx) && isEnabled(command, ctx);
    return (
      <button
        key={item.id}
        type="button"
        role="menuitem"
        data-testid={`tab-menu-${item.id}`}
        disabled={item.disabled || !usable}
        onClick={() => {
          close();
          commands.run(item.command, ctx).catch(useAppStore.getState().reportError);
        }}
        className={`${menuItemClass} pl-2`}
      >
        <span className="flex-1">{item.label}</span>
        {command?.shortcut && (
          <span className="pl-4 text-[13px] text-ink-meta">{shortcutLabel(item.command)}</span>
        )}
      </button>
    );
  });
}

/**
 * A tab's menu: its right-click menu, and what Shift+F10 or the context-menu
 * key open on it. Its button is out of sight in the tab, so the menu opens under the tab.
 */
export function TabMenu({ tab, title }: { tab: LayoutTab; title: string }) {
  const t = useT();
  const menu = usePopoverMenu();
  const layout = useAppStore((state) => state.layout);
  const at = groupOfTab(layout, tab.id);
  const groups = layout?.groups.length ?? 1;
  const index = at?.index ?? 0;
  const moveRight = index < groups - 1;
  const items: (Item | false)[] = menu.open
    ? [
        tab.preview && { id: "keep", command: "tab.keep", label: t("tabs.menu.keep") },
        {
          id: tab.pinned ? "unpin" : "pin",
          command: "tab.pin",
          label: t(tab.pinned ? "tabs.menu.unpin" : "tabs.menu.pin"),
        },
        "rule",
        // At three groups ⌥⌘\ says why nothing happens; here the item is off.
        {
          id: "side",
          command: "tab.split",
          label: t("tabs.menu.side"),
          disabled: groups >= MAX_TAB_GROUPS,
        },
        index > 0 && { id: "move-left", command: "tab.moveLeft", label: t("tabs.menu.moveLeft") },
        {
          id: moveRight ? "move-right" : "move-new",
          command: "tab.moveRight",
          label: t(moveRight ? "tabs.menu.moveRight" : "tabs.menu.moveNew"),
        },
        tab.item.kind === "mind" && "rule",
        tab.item.kind === "mind" && {
          id: "rename",
          command: "tab.rename",
          label: t("tabs.menu.rename"),
        },
        "rule",
        { id: "close", command: "tab.close", label: t("tabs.menu.close") },
        { id: "close-others", command: "tab.closeOthers", label: t("tabs.menu.closeOthers") },
        { id: "close-right", command: "tab.closeToTheRight", label: t("tabs.menu.closeRight") },
      ]
    : [];
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        tabIndex={-1}
        data-testid="tab-menu"
        aria-label={t("tabs.tabMenu", { title })}
        className="tab-menu-anchor"
      />
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("tabs.tabMenu", { title })}
        data-testid="tab-menu-items"
        // Not the tab's tooltip over its menu: the menu sits in the tab.
        title=""
        className={menuClass}
        onKeyDown={(event) => {
          menu.menuProps.onKeyDown(event);
          // Esc gives the focus back to the tab, not to the button out of sight.
          if (event.key === "Escape") focusTab(tab.id);
        }}
      >
        <MenuItems items={items} ctx={onTab(tab.id)} close={menu.close} />
      </div>
    </>
  );
}

/** The ⋯ at a group strip's end: what applies to the group and its shown tab. */
export function GroupMenu({ groupId, merged }: { groupId: string; merged: boolean }) {
  const t = useT();
  const menu = usePopoverMenu();
  const layout = useAppStore((state) => state.layout);
  const group = layout?.groups.find((each) => each.id === groupId);
  const full = (layout?.groups.length ?? 0) >= MAX_TAB_GROUPS;
  const last = useAppStore((state) => state.settings?.user.exportOptions.format ?? "docx");
  const mindShown = group?.tabs.find((tab) => tab.id === group.activeTabId)?.item.kind === "mind";
  // The shown Mind's exports first, as the design's "More for this Mind" has them: each format,
  // the last one with ⇧⌘E, then the options sheet.
  const exports: Item[] = mindShown
    ? [
        ...EXPORT_FORMATS.map(
          (format): Item => ({
            id: `export-${format}`,
            command: format === last ? "mind.export" : `mind.exportAs.${format}`,
            label: t("export.as", { format: t(FORMAT_LABELS[format]) }),
          }),
        ),
        { id: "export-options", command: "mind.exportOptions", label: t("export.options") },
        "rule",
      ]
    : [];
  const items: (Item | false)[] =
    menu.open && group
      ? [
          ...exports,
          {
            id: "split",
            command: "tab.split",
            label: t("tabs.menu.split"),
            disabled: merged || full,
          },
          { id: "equal", command: "group.equalize", label: t("tabs.menu.equal") },
          "rule",
          { id: "reopen", command: "tab.reopen", label: t("tabs.menu.reopen") },
          { id: "close-others", command: "tab.closeOthers", label: t("tabs.menu.closeOthers") },
          { id: "close-group", command: "group.close", label: t("tabs.menu.closeGroup") },
        ]
      : [];
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="group-menu"
        aria-label={t("tabs.groupMenu")}
        title={t("tabs.groupMenu")}
        className="tab-strip-button"
      >
        <MoreLineIcon />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("tabs.groupMenu")}
        data-testid="group-menu-items"
        className={menuClass}
      >
        <MenuItems
          items={items}
          // An empty group has no tab to act on.
          ctx={onTab(group?.activeTabId ?? undefined)}
          close={menu.close}
        />
      </div>
    </>
  );
}

/** A strip button with an icon, such as Split. */
export function StripButton({
  testId,
  label,
  onClick,
  children,
}: {
  testId: string;
  label: string;
  onClick(): void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      aria-label={label}
      title={label}
      onClick={onClick}
      className="tab-strip-button"
    >
      {children}
    </button>
  );
}
