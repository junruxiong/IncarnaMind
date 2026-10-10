import { useEffect } from "react";
import { files } from "../../core";
import { useAppStore } from "../../store";
import { cycleTabs, focusGroup, splitToTheRight } from "./tabActions";

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/**
 * The window's tab shortcuts, as in VS Code and a browser: ⌘T (Ctrl+T) a new
 * Mind in a new tab, ⌘W closes the shown tab (not a pinned one), ⇧⌘T reopens
 * the tab closed last, ⌥⌘\ (Ctrl+Alt+\; ⌘\ is the sidebar's) splits the shown
 * tab to the right, ⌘1–⌘3 go to a group, and Ctrl+Tab or Ctrl+Shift+Tab move through the group's tabs. The
 * application menu's Close Tab, Reopen Closed Tab and Split to the Right do the
 * same. Not while a dialog is open, and not when something else used the key.
 */
export function useTabShortcuts(): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      const cycle = event.key === "Tab" && event.ctrlKey && !event.metaKey && !event.altKey;
      const command = isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
      // Typed keys leave here, before the search for an open dialog through the whole page.
      if (!cycle && !command) return;
      if (document.querySelector("dialog[open]")) return;
      const store = useAppStore.getState();
      if (event.altKey) {
        if (event.code !== "Backslash" || event.shiftKey) return;
        event.preventDefault();
        void splitToTheRight();
        return;
      }
      if (cycle) {
        event.preventDefault();
        cycleTabs(event.shiftKey ? -1 : 1);
        return;
      }
      if (event.shiftKey) {
        if (event.code !== "KeyT") return;
        event.preventDefault();
        store.reopenTab();
        return;
      }
      if (event.code === "KeyT") {
        event.preventDefault();
        void store.createMind();
      } else if (event.code === "KeyW") {
        event.preventDefault();
        store.closeActiveTab();
      } else if (/^Digit[1-3]$/.test(event.code)) {
        event.preventDefault();
        void focusGroup(Number(event.code.slice("Digit".length)));
      }
    };
    window.addEventListener("keydown", onKey);
    const stopMenu = files.onMenuCommand((command) => {
      if (document.querySelector("dialog[open]")) return;
      const store = useAppStore.getState();
      if (command === "close-tab") store.closeActiveTab();
      else if (command === "reopen-tab") store.reopenTab();
      else if (command === "split-right") void splitToTheRight();
    });
    return () => {
      window.removeEventListener("keydown", onKey);
      stopMenu();
    };
  }, []);
}
