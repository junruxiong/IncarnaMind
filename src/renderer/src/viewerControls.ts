/**
 * Test hooks, present only when the app was launched with
 * INCARNAMIND_TEST_HOOKS=1 (the main process then adds `?testHooks` to the
 * page URL): open a Document at a page range and quote, as a Citation does
 * from the group the User is in, and, not about the viewer, answer the Skill
 * picker without the system's dialog, add files and folders by path as a drop
 * would, or wait for, slow down and look at what moves.
 */
import type { TestHooks } from "../../shared/testHooks";
import { currentTabOf } from "./layoutSlice";
import { leavingCopies, motionRunning, setMotionSpeed } from "./motion";
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
      // From the Mind the User is in, as a Citation in it; with no Mind in front, as from a list.
      if (currentTabOf(layout)?.item.kind === "mind" && layout) {
        openDocument(location, { as: "citation", from: layout.focusedGroupId });
      } else openDocument(location, { as: "list" });
    },
    interceptSkillPicker,
    addPaths: (paths) => useAppStore.getState().addPaths(paths),
    motionRunning,
    setMotionSpeed,
    leavingCopies,
  };
}
