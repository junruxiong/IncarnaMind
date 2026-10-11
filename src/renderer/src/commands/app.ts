import { blockMenuOpeners } from "../editor/blockMenuOpeners";
import { noteHolding } from "../editor/noteKeymap";
import { useFindSources } from "../findSources";
import { useReferenceImport } from "../referenceImport";
import { useAppStore } from "../store";
import { assistantCommands } from "./assistant";
import { evidenceCommands } from "./evidence";
import { footnoteCommands } from "./footnotes";
import { citationHandle, sidebarHandle } from "./handles";
import { meetingCommands } from "./meetings";
import { proposalCommands } from "./proposals";
import { type Command, commands, fromMenu } from "./registry";
import { tableCommands } from "./tables";
import { tabCommands } from "./tabs";
import { webPageCommands } from "./webPages";

/** What opens the menu of the Block with this ID, if its Mind is open. */
function blockMenuFor(blockId: string) {
  const editor = noteHolding(blockId);
  return (editor && blockMenuOpeners.get(editor)) ?? null;
}

const appCommands: readonly Command[] = [
  {
    ...fromMenu("mind.new"),
    group: "main",
    reversible: true,
    run: () => useAppStore.getState().createMind(),
  },
  ...tabCommands,
  ...meetingCommands,
  ...webPageCommands,
  ...evidenceCommands,
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
    id: "sources.find",
    label: "command.sources.find",
    keywords: ["command.sources.find.keywords"],
    group: "main",
    reversible: true,
    run: () => useFindSources.getState().show(),
  },
  ...assistantCommands,
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
  {
    id: "block.menu",
    label: "editor.block.menu",
    shortcut: { keys: "Shift+F10" },
    group: "main",
    appliesTo: ["block"],
    reversible: true,
    when: (ctx) => ctx.focus?.kind === "block" && blockMenuFor(ctx.focus.id) !== null,
    run(ctx) {
      if (ctx.focus?.kind === "block") blockMenuFor(ctx.focus.id)?.(ctx.focus.id);
    },
  },
  ...footnoteCommands,
  ...tableCommands,
  ...proposalCommands,
];

for (const command of appCommands) commands.register(command);
