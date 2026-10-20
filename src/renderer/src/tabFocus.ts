import type { WindowLayout } from "../../core/api";
import { focusedGroupOf, groupOfTab, groupsMerged } from "./layoutSlice";

function focusOnceDrawn(find: () => HTMLElement | null, onlyIfLost: boolean): void {
  requestAnimationFrame(() => {
    if (onlyIfLost && document.activeElement !== document.body) return;
    find()?.focus();
  });
}

/**
 * Moves the keyboard's focus to a tab once it is drawn. With `onlyIfLost`, only if the focus
 * is nowhere by then: a field the User has moved to since keeps it.
 */
export function focusTab(
  tabId: string | null | undefined,
  { onlyIfLost = false }: { onlyIfLost?: boolean } = {},
): void {
  if (!tabId) return;
  focusOnceDrawn(
    () => document.querySelector<HTMLElement>(`[data-tab-id="${CSS.escape(tabId)}"]`),
    onlyIfLost,
  );
}

/** Captures a closing tab's focus before its layout change removes the focused element (#1309). */
export function returnFocusAfterTabClose(
  tabId: string,
  before: WindowLayout | null,
): (after: WindowLayout) => void {
  const group = groupOfTab(before, tabId)?.group;
  const origin = document.activeElement;
  const onTab = origin?.closest<HTMLElement>("[data-tab-id]")?.dataset.tabId === tabId;
  const inPage =
    group?.activeTabId === tabId &&
    origin?.closest<HTMLElement>("[data-group-body]")?.dataset.groupBody === group.id;
  const inEditor = inPage && origin?.closest('[data-testid="mind-editor"]') !== null;
  return (after) => {
    if (!group || (!onTab && !inPage)) return;
    focusOnceDrawn(() => {
      const next =
        (!groupsMerged() && after.groups.find((each) => each.id === group.id)) ||
        focusedGroupOf(after);
      const pane = document.querySelector<HTMLElement>(
        `[data-testid="tab-group"][data-group-id="${CSS.escape(next.id)}"]`,
      );
      return (
        (inEditor ? pane?.querySelector<HTMLElement>('[data-testid="mind-editor"]') : null) ??
        (next.activeTabId
          ? document.querySelector<HTMLElement>(`[data-tab-id="${CSS.escape(next.activeTabId)}"]`)
          : pane?.querySelector<HTMLElement>('[data-testid="new-tab"]')) ??
        null
      );
    }, true);
  };
}
