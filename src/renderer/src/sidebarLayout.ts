/**
 * When the sidebar is out of the way. The User's choice (hidden, remembered)
 * always hides it. Besides that, a window too narrow to hold the sidebar,
 * the Mind and the Document viewer hides it on its own, and it comes back
 * when there is room again, unless the User showed it since (`forcedShown`).
 */

/** What the Mind and the viewer each need beside the sidebar before it gives way. */
export const PANE_COMFORT = 360;
/** The card's 8px gaps and the 1px rule between the Mind and the viewer. */
const CHROME = 2 * 8 + 1;

/** The viewer is open and the window has no room for the sidebar and two readable panes. */
export function isNarrow(viewerOpen: boolean, windowWidth: number, sidebarWidth: number): boolean {
  return viewerOpen && windowWidth < sidebarWidth + CHROME + 2 * PANE_COMFORT;
}

export function sidebarIsHidden(state: {
  userHidden: boolean;
  narrow: boolean;
  forcedShown: boolean;
}): boolean {
  return state.userHidden || (state.narrow && !state.forcedShown);
}
