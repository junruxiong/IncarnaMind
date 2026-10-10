import type { ReactNode } from "react";
import { type LayoutChange, type LayoutTab, MAX_TAB_GROUPS } from "../../../../core/api";
import { useT } from "../../i18n";
import { groupOfTab } from "../../layoutSlice";
import { useAppStore } from "../../store";
import { MoreLineIcon } from "../lineIcons";
import { menuClass, menuItemClass, menuRuleClass } from "../ui";
import { usePopoverMenu } from "../usePopoverMenu";
import {
  CLOSE_SHORTCUT,
  focusTab,
  moveToGroup,
  REOPEN_SHORTCUT,
  SPLIT_SHORTCUT,
  splitToTheRight,
} from "./tabActions";

/** One item of a tab's or group's menu, or a rule between them. */
type Item =
  | {
      id: string;
      label: string;
      shortcut?: string;
      disabled?: boolean;
      run(): void;
    }
  | "rule";

function MenuItems({ items, close }: { items: (Item | false)[]; close(): void }) {
  return items.map((item, index) => {
    if (item === false) return null;
    if (item === "rule") {
      // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place.
      return <div key={`rule-${index}`} className={menuRuleClass} />;
    }
    return (
      <button
        key={item.id}
        type="button"
        role="menuitem"
        data-testid={`tab-menu-${item.id}`}
        disabled={item.disabled}
        onClick={() => {
          close();
          item.run();
        }}
        className={`${menuItemClass} pl-2`}
      >
        <span className="flex-1">{item.label}</span>
        {item.shortcut && <span className="pl-4 text-[13px] text-ink-meta">{item.shortcut}</span>}
      </button>
    );
  });
}

/** Closes tabs, then says how many, with Reopen. */
async function closeSome(change: LayoutChange, tab: LayoutTab): Promise<void> {
  const store = useAppStore.getState();
  const before = store.layout?.reopenable ?? 0;
  const next = await store.changeLayout(change);
  const count = next ? next.reopenable - before : 0;
  if (count > 0) store.showTabNotice({ kind: "closed", item: tab.item, count });
}

/**
 * A tab's menu: its right-click menu, and what Shift+F10 or the context-menu
 * key open on it. Its button is out of sight in the tab, so the menu opens under the tab.
 */
export function TabMenu({
  tab,
  title,
  onRename,
}: {
  tab: LayoutTab;
  title: string;
  onRename(): void;
}) {
  const t = useT();
  const menu = usePopoverMenu();
  const layout = useAppStore((state) => state.layout);
  const at = groupOfTab(layout, tab.id);
  const { changeLayout, closeTab } = useAppStore.getState();
  const groups = layout?.groups.length ?? 1;
  const group = at?.group;
  const index = at?.index ?? 0;
  const position = group?.tabs.findIndex((each) => each.id === tab.id) ?? 0;
  const items: (Item | false)[] = menu.open
    ? [
        tab.preview && {
          id: "keep",
          label: t("tabs.menu.keep"),
          run: () => void changeLayout({ type: "keep", tabId: tab.id }),
        },
        {
          id: tab.pinned ? "unpin" : "pin",
          label: t(tab.pinned ? "tabs.menu.unpin" : "tabs.menu.pin"),
          run: () => void changeLayout({ type: "pin", tabId: tab.id, pinned: !tab.pinned }),
        },
        "rule",
        {
          id: "side",
          label: t("tabs.menu.side"),
          shortcut: SPLIT_SHORTCUT,
          disabled: groups >= MAX_TAB_GROUPS,
          run: () => void splitToTheRight(tab.id),
        },
        index > 0 && {
          id: "move-left",
          label: t("tabs.menu.moveLeft"),
          run: () => void moveToGroup(tab.id, -1),
        },
        index < groups - 1
          ? {
              id: "move-right",
              label: t("tabs.menu.moveRight"),
              run: () => void moveToGroup(tab.id, 1),
            }
          : groups < MAX_TAB_GROUPS &&
            (group?.tabs.length ?? 0) > 1 && {
              id: "move-new",
              label: t("tabs.menu.moveNew"),
              run: () => void moveToGroup(tab.id, 1),
            },
        tab.item.kind === "mind" && "rule",
        tab.item.kind === "mind" && {
          id: "rename",
          label: t("tabs.menu.rename"),
          shortcut: "F2",
          run: onRename,
        },
        "rule",
        {
          id: "close",
          label: t("tabs.menu.close"),
          shortcut: CLOSE_SHORTCUT,
          run: () => closeTab(tab.id),
        },
        {
          id: "close-others",
          label: t("tabs.menu.closeOthers"),
          disabled: (group?.tabs.length ?? 0) < 2,
          run: () => void closeSome({ type: "closeOthers", tabId: tab.id }, tab),
        },
        {
          id: "close-right",
          label: t("tabs.menu.closeRight"),
          disabled: position >= (group?.tabs.length ?? 0) - 1,
          run: () => void closeSome({ type: "closeToTheRight", tabId: tab.id }, tab),
        },
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
        <MenuItems items={items} close={menu.close} />
      </div>
    </>
  );
}

/** The ⋯ at a group strip's end: what applies to the group and its tabs. */
export function GroupMenu({ groupId, merged }: { groupId: string; merged: boolean }) {
  const t = useT();
  const menu = usePopoverMenu();
  const layout = useAppStore((state) => state.layout);
  const group = layout?.groups.find((each) => each.id === groupId);
  const groups = layout?.groups.length ?? 1;
  const store = useAppStore.getState();
  const shown = group?.tabs.find((tab) => tab.id === group.activeTabId);
  const items: (Item | false)[] =
    menu.open && group
      ? [
          {
            id: "split",
            label: t("tabs.menu.split"),
            shortcut: SPLIT_SHORTCUT,
            disabled: !shown || groups >= MAX_TAB_GROUPS || merged,
            run: () => void splitToTheRight(shown?.id),
          },
          {
            id: "equal",
            label: t("tabs.menu.equal"),
            disabled: groups < 2 || merged,
            run: () => void store.changeLayout({ type: "equalize" }),
          },
          "rule",
          {
            id: "reopen",
            label: t("tabs.menu.reopen"),
            shortcut: REOPEN_SHORTCUT,
            disabled: !layout?.reopenable,
            run: () => store.reopenTab(),
          },
          {
            id: "close-others",
            label: t("tabs.menu.closeOthers"),
            disabled: !shown || group.tabs.length < 2,
            run: () => shown && void closeSome({ type: "closeOthers", tabId: shown.id }, shown),
          },
          {
            id: "close-group",
            label: t("tabs.menu.closeGroup"),
            disabled: group.tabs.length === 0,
            run: () => shown && void closeSome({ type: "closeGroup", groupId: group.id }, shown),
          },
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
        <MenuItems items={items} close={menu.close} />
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
