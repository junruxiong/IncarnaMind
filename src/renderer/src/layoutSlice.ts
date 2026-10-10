import type { LayoutChange, LayoutGroup, LayoutTab, TabItem, WindowLayout } from "../../core/api";
import { core } from "./core";

/*
 * The window's tabs and groups, as the core keeps them (see src/core/layout.ts):
 * the store holds the core's last word on them, and changes them only
 * through the bridge.
 */

/** Which window this is: the main process names it in the page's address. */
export const WINDOW_ID =
  new URLSearchParams(globalThis.location?.search ?? "").get("window") ?? "main";

/** Below this window width the groups merge into one strip, and come back when it is wider. */
export const MERGE_BELOW = 1000;

/** Whether the window is too narrow for groups side by side right now. */
export const groupsMerged = () => window.innerWidth < MERGE_BELOW;

/** A line about the tabs, shown for a moment where the sidebar ends. */
export type TabNotice =
  | { kind: "closed"; item: TabItem; count: number }
  | { kind: "maxGroups" }
  | { kind: "noGroup"; group: number }
  | { kind: "pinned"; item: TabItem }
  | { kind: "merged" };

export interface LayoutSlice {
  /** Null until loaded. */
  layout: WindowLayout | null;
  tabNotice: TabNotice | null;
  /** Reads the window's layout from the core, as it was left. */
  loadLayout(): Promise<void>;
  /** Changes the layout through the core; resolves with it once the store has it (null if refused). */
  changeLayout(change: LayoutChange): Promise<WindowLayout | null>;
  showTabNotice(notice: TabNotice | null): void;
}

type Set = (partial: Partial<LayoutSlice>) => void;

export function createLayoutSlice(set: Set, report: (error: unknown) => void): LayoutSlice {
  return {
    layout: null,
    tabNotice: null,
    async loadLayout() {
      set({ layout: await core.getLayout(WINDOW_ID) });
    },
    async changeLayout(change) {
      try {
        const layout = await core.changeLayout(WINDOW_ID, change);
        set({ layout });
        return layout;
      } catch (error) {
        report(error);
        return null;
      }
    },
    showTabNotice(notice) {
      set({ tabNotice: notice });
    },
  };
}

/** Follows the core's word on this window's layout, e.g. when a deleted Mind's tab closes. */
export function followLayout(set: Set): void {
  core.on("layout.changed", ({ windowId, layout }) => {
    if (windowId === WINDOW_ID) set({ layout });
  });
}

// ---------------------------------------------------------------------------
// Reading a layout.

export const focusedGroupOf = (layout: WindowLayout): LayoutGroup =>
  (layout.groups.find((group) => group.id === layout.focusedGroupId) ??
    layout.groups[0]) as LayoutGroup;

export const activeTabOf = (group: LayoutGroup | undefined): LayoutTab | null =>
  group?.tabs.find((tab) => tab.id === group.activeTabId) ?? null;

/** The tab the User is in: the focused group's shown tab. */
export const currentTabOf = (layout: WindowLayout | null): LayoutTab | null =>
  layout ? activeTabOf(focusedGroupOf(layout)) : null;

/** The groups on screen: all of them, or only the focused one while they are merged. */
export const visibleGroupsOf = (layout: WindowLayout, merged: boolean): LayoutGroup[] =>
  merged ? [focusedGroupOf(layout)] : layout.groups;

/**
 * The Mind the User is in: the focused group's, else the first Mind shown in
 * a group. Null when no group shows a Mind.
 */
export function currentMindIdOf(layout: WindowLayout | null): string | null {
  if (!layout) return null;
  const shown = [focusedGroupOf(layout), ...layout.groups].map((group) => activeTabOf(group));
  const mind = shown.find((tab) => tab?.item.kind === "mind")?.item;
  return mind?.kind === "mind" ? mind.mindId : null;
}

/** Whether a group shows this, in its front tab. */
export function isShown(layout: WindowLayout | null, item: TabItem, merged = false): boolean {
  if (!layout) return false;
  return visibleGroupsOf(layout, merged).some((group) => {
    const tab = activeTabOf(group)?.item;
    if (!tab || tab.kind !== item.kind) return false;
    if (tab.kind === "mind") return item.kind === "mind" && tab.mindId === item.mindId;
    if (tab.kind === "document") {
      return item.kind === "document" && tab.documentId === item.documentId;
    }
    return true;
  });
}

/** Where a tab is: its group, and that group's place from the left. */
export function groupOfTab(
  layout: WindowLayout | null,
  tabId: string,
): { group: LayoutGroup; index: number } | null {
  if (!layout) return null;
  const index = layout.groups.findIndex((group) => group.tabs.some((tab) => tab.id === tabId));
  return index === -1 ? null : { group: layout.groups[index] as LayoutGroup, index };
}
