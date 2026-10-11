import { noteHolding } from "../editor/noteKeymap";
import { useReferenceImport } from "../referenceImport";
import { useAppStore } from "../store";
import { citationHandle, composerHandle, sidebarHandle } from "./handles";
import { meetingCommands } from "./meetings";
import { type Command, commands, fromMenu } from "./registry";
import { tableCommands } from "./tables";
import { tabCommands } from "./tabs";

const appCommands: readonly Command[] = [
  {
    ...fromMenu("mind.new"),
    group: "main",
    reversible: true,
    run: () => useAppStore.getState().createMind(),
  },
  ...tabCommands,
  ...meetingCommands,
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
    id: "citation.open",
    label: "command.citation.open",
    shortcut: { keys: "CmdOrCtrl+Alt+O" },
    group: "open",
    reversible: true,
    // While a Citation's card shows (hovered, focused or clicked).
    when: () => citationHandle.current !== null,
    run: () => citationHandle.current?.open(),
  },
  {
    id: "references.import",
    label: "references.import.menu",
    group: "file",
    // The import is taken back with Undo.
    reversible: true,
    enabled: () =>
      useReferenceImport.getState().phase === "running"
        ? { reason: "references.import.running.reason" }
        : true,
    run: () => useReferenceImport.getState().start(),
  },
  {
    // Suggest as I write (#282): on asks for consent first, as the switch in Settings does.
    id: "suggestions.toggle",
    label: (_ctx, t) =>
      t(
        useAppStore.getState().settings?.user.suggestWhileWriting
          ? "command.suggestions.off"
          : "command.suggestions.on",
      ),
    keywords: ["suggestions.settings.switch"],
    group: "main",
    reversible: true,
    when: () => useAppStore.getState().settings !== null,
    run: () => {
      const on = useAppStore.getState().settings?.user.suggestWhileWriting ?? false;
      return useAppStore.getState().updateSettings({ user: { suggestWhileWriting: !on } });
    },
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
  ...tableCommands,
];

for (const command of appCommands) commands.register(command);
