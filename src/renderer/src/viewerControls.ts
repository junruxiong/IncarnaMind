/**
 * Ways to drive the Document viewer that the UI doesn't offer (yet). Neither is visible:
 * - test hooks, present only when the app was launched with
 *   INCARNAMIND_TEST_HOOKS=1 (the main process then adds `?testHooks` to the page URL):
 *   open the empty panel, or open a Document at a page range and quote, as a Citation will
 *   (and, not about the viewer, answer the Skill picker without the system's dialog);
 * - a development-only shortcut, Cmd/Ctrl+Shift+D, that toggles the panel.
 */
import { useEffect } from "react";
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
  const { openViewer, closeViewer, openDocument } = useAppStore.getState();
  window.incarnamindTestHooks = { openViewer, closeViewer, openDocument, interceptSkillPicker };
}

export function useDevViewerShortcut(): void {
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const toggleOnShortcut = (event: KeyboardEvent) => {
      const modifier = event.metaKey || event.ctrlKey;
      if (modifier && event.shiftKey && event.key.toLowerCase() === "d") {
        event.preventDefault();
        useAppStore.getState().toggleViewer();
      }
    };
    window.addEventListener("keydown", toggleOnShortcut);
    return () => window.removeEventListener("keydown", toggleOnShortcut);
  }, []);
}
