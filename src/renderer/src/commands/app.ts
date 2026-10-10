import { MENU_COMMANDS, type MenuCommandId } from "../../../shared/commands";
import { noteHolding } from "../editor/noteKeymap";
import { useAppStore } from "../store";
import { showTab, tabStrip } from "../tabs";
import { composerHandle, sidebarHandle } from "./handles";
import { type Command, commands } from "./registry";

/** A menu item's command takes its label and keys from the menu's list, so they agree. */
const fromMenu = (id: MenuCommandId) => ({
  id,
  label: MENU_COMMANDS[id].label,
  shortcut: { keys: MENU_COMMANDS[id].keys },
});

function cycleTab(step: 1 | -1): void {
  const { order, shown } = tabStrip();
  if (order.length === 0) return;
  const current = shown === null ? -1 : order.indexOf(shown);
  const next = order[(current + step + order.length) % order.length];
  if (next !== undefined) showTab(next);
}

/** ⌘1–⌘8 show that Mind's tab, as in a browser; the Library's tab isn't counted. */
const showTabCommands = [1, 2, 3, 4, 5, 6, 7, 8].map(
  (number): Command => ({
    id: `tab.show${number}`,
    label: (_ctx, t) => t("command.tab.show", { number }),
    shortcut: { keys: `CmdOrCtrl+${number}` },
    palette: false,
    group: "open",
    reversible: true,
    run() {
      const id = useAppStore.getState().tabs[number - 1];
      if (id) useAppStore.getState().openMind(id);
    },
  }),
);

const appCommands: readonly Command[] = [
  {
    ...fromMenu("mind.new"),
    group: "main",
    reversible: true,
    run: () => useAppStore.getState().createMind(),
  },
  {
    id: "tab.new",
    label: "command.tab.new",
    shortcut: { keys: "CmdOrCtrl+T" },
    // The same as New Mind, which ⌘K offers.
    palette: false,
    group: "main",
    reversible: true,
    run: () => useAppStore.getState().createMind(),
  },
  {
    ...fromMenu("tab.close"),
    group: "main",
    reversible: true,
    when: () => tabStrip().shown !== null,
    run() {
      const { libraryOpen, closeLibrary, openMindId, closeTab } = useAppStore.getState();
      if (libraryOpen) closeLibrary();
      else if (openMindId) closeTab(openMindId);
    },
  },
  {
    id: "tab.next",
    label: "command.tab.next",
    shortcut: { keys: "Ctrl+Tab" },
    group: "open",
    reversible: true,
    run: () => cycleTab(1),
  },
  {
    id: "tab.previous",
    label: "command.tab.previous",
    shortcut: { keys: "Ctrl+Shift+Tab" },
    group: "open",
    reversible: true,
    run: () => cycleTab(-1),
  },
  ...showTabCommands,
  {
    id: "tab.showLast",
    label: "command.tab.showLast",
    shortcut: { keys: "CmdOrCtrl+9" },
    palette: false,
    group: "open",
    reversible: true,
    run() {
      const id = useAppStore.getState().tabs.at(-1);
      if (id) useAppStore.getState().openMind(id);
    },
  },
  {
    ...fromMenu("settings.open"),
    group: "open",
    reversible: true,
    run: () => useAppStore.getState().openSettings(),
  },
  {
    ...fromMenu("sidebar.toggle"),
    label: (_ctx, t) => t(sidebarHandle.current?.hidden ? "menu.showSidebar" : "menu.hideSidebar"),
    group: "main",
    reversible: true,
    when: () => sidebarHandle.current !== null,
    run: () => sidebarHandle.current?.toggle(),
  },
  {
    id: "composer.focus",
    label: "command.composer.focus",
    shortcut: { keys: "CmdOrCtrl+J" },
    group: "main",
    reversible: true,
    when: () => composerHandle.current !== null,
    run: () => composerHandle.current?.focus(),
  },
  {
    id: "block.delete",
    label: "editor.block.delete",
    shortcut: { keys: "CmdOrCtrl+Shift+Backspace" },
    group: "danger",
    appliesTo: ["block"],
    // The note's own undo (⌘Z) brings it back.
    reversible: true,
    run(ctx) {
      // The Blocks the note's selection touches, the focused one among them.
      if (ctx.focus) noteHolding(ctx.focus.id)?.commands.deleteBlock();
    },
  },
  {
    id: "viewer.toggle",
    label: "command.viewer.toggle",
    shortcut: { keys: "CmdOrCtrl+Shift+D" },
    group: "main",
    reversible: true,
    // A way into the viewer while developing; the User opens it from a Document or a Citation.
    when: () => import.meta.env.DEV,
    run: () => useAppStore.getState().toggleViewer(),
  },
];

for (const command of appCommands) commands.register(command);
