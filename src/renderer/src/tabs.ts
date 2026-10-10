import type { Translate } from "./objects";
import { useAppStore } from "./store";

type AppState = ReturnType<typeof useAppStore.getState>;

/** The Library's place in the tab strip, which no Mind's ID is. */
export const LIBRARY_TAB = "\u0000library";

/** The tabs in the strip's order, the Library's first while it is open, and the one shown. */
export function tabStrip(state: AppState = useAppStore.getState()) {
  const order = state.libraryTab ? [LIBRARY_TAB, ...state.tabs] : state.tabs;
  const shown = state.libraryOpen ? LIBRARY_TAB : state.openMindId;
  return { order, shown };
}

export function showTab(id: string): void {
  const { showLibrary, openMind } = useAppStore.getState();
  if (id === LIBRARY_TAB) showLibrary();
  else openMind(id);
}

/** What the Library's tab is named after: the Folder it shows, or "All Documents". */
export function libraryTitle(state: AppState, t: Translate): string {
  const filter = state.libraryFilter;
  if (filter === "unsorted") return t("library.unsorted");
  const folder = state.library?.groups.find((group) => group.id === filter);
  return folder?.name ?? t("library.all");
}
