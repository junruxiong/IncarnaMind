/**
 * Test hooks, present only when the app was launched with
 * INCARNAMIND_TEST_HOOKS=1 (the main process then adds `?testHooks` to the
 * page URL): open a Document at a page range and quote, as a Citation does
 * from the group the User is in, and, not about the viewer, answer the Skill
 * picker without the system's dialog, or add files and folders by path as a drop would.
 */
import type { TestHooks } from "../../shared/testHooks";
import { interceptSkillPicker } from "./skills";
import { useAppStore } from "./store";

declare global {
  interface Window {
    incarnamindTestHooks?: TestHooks;
  }
}

export function installTestHooks(): void {
  if (!new URLSearchParams(window.location.search).has("testHooks")) return;
  window.incarnamindTestHooks = {
    openDocument: (location) => {
      const { layout, openDocument } = useAppStore.getState();
      openDocument(location, { as: "citation", ...(layout && { from: layout.focusedGroupId }) });
    },
    interceptSkillPicker,
    addPaths: (paths) => useAppStore.getState().addPaths(paths),
  };
}
