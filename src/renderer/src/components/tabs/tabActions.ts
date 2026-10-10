import { MAX_TAB_GROUPS } from "../../../../core/api";
import { activeTabOf, focusedGroupOf, groupOfTab, groupsMerged } from "../../layoutSlice";
import { useAppStore } from "../../store";

/*
 * What the tab shortcuts, menus and buttons do beyond one change to the
 * layout: they say why nothing happened, and move the keyboard's focus with
 * the tab.
 */

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** ⌘\ as this system writes it. */
export const SPLIT_SHORTCUT = isMac ? "⌘\\" : "Ctrl+\\";
/** ⌘W, ⇧⌘T and ⌘T as this system writes them. */
export const CLOSE_SHORTCUT = isMac ? "⌘W" : "Ctrl+W";
export const REOPEN_SHORTCUT = isMac ? "⇧⌘T" : "Ctrl+Shift+T";

/** Moves the keyboard's focus to a tab once it is drawn. */
export function focusTab(tabId: string | null | undefined): void {
  if (!tabId) return;
  requestAnimationFrame(() =>
    document.querySelector<HTMLElement>(`[data-tab-id="${CSS.escape(tabId)}"]`)?.focus(),
  );
}

/** ⌘\: a copy of a tab (the shown one by default) in a new group on the right. */
export async function splitToTheRight(tabId?: string): Promise<void> {
  const store = useAppStore.getState();
  const { layout } = store;
  if (!layout) return;
  const tab = tabId ?? activeTabOf(focusedGroupOf(layout))?.id;
  if (!tab) return;
  if (layout.groups.length >= MAX_TAB_GROUPS) {
    store.showTabNotice({ kind: "maxGroups" });
    return;
  }
  const next = await store.changeLayout({ type: "split", tabId: tab, side: "right", copy: true });
  if (!next) return;
  if (groupsMerged()) store.showTabNotice({ kind: "merged" });
  focusTab(activeTabOf(focusedGroupOf(next))?.id);
}

/** ⌘1–⌘3: the keyboard's focus to a group's shown tab. */
export async function focusGroup(number: number): Promise<void> {
  const store = useAppStore.getState();
  const group = store.layout?.groups[number - 1];
  if (!group) {
    store.showTabNotice({ kind: "noGroup", group: number });
    return;
  }
  await store.changeLayout({ type: "focus", groupId: group.id });
  focusTab(group.activeTabId);
}

/** Moves a tab to the group on its left or right, or into a new group on the right. */
export async function moveToGroup(tabId: string, direction: -1 | 1): Promise<void> {
  const store = useAppStore.getState();
  const at = groupOfTab(store.layout, tabId);
  if (!at || !store.layout) return;
  const target = store.layout.groups[at.index + direction];
  const next = target
    ? await store.changeLayout({ type: "move", tabId, groupId: target.id, before: null })
    : await store.changeLayout({ type: "split", tabId, side: "right", copy: false });
  if (next) focusTab(tabId);
}

/** Shows the next or previous tab of the focused group (Ctrl+Tab, Ctrl+Shift+Tab). */
export function cycleTabs(step: 1 | -1): void {
  const store = useAppStore.getState();
  if (!store.layout) return;
  const group = focusedGroupOf(store.layout);
  if (group.tabs.length === 0) return;
  const index = group.tabs.findIndex((tab) => tab.id === group.activeTabId);
  const next = group.tabs[(index + step + group.tabs.length) % group.tabs.length];
  if (next) void store.changeLayout({ type: "show", tabId: next.id });
}
