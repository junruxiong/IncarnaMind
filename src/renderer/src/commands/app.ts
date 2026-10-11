import { noteHolding } from "../editor/noteKeymap";
import { useFindSources } from "../findSources";
import { useReferenceImport } from "../referenceImport";
import { useAppStore } from "../store";
import { assistantCommands } from "./assistant";
import { evidenceCommands } from "./evidence";
import { footnoteCommands } from "./footnotes";
import { citationHandle, sidebarHandle } from "./handles";
import { meetingCommands } from "./meetings";
import {
  answerCommands,
  citationCommands,
  documentCommands,
  folderCommands,
  linkedFolderCommands,
  mindCommands,
  tagCommands,
} from "./objects";
import { proposalCommands } from "./proposals";
import "./objectMenus";
import { type Command, commands, fromMenu } from "./registry";
import { tableCommands } from "./tables";
import { tabCommands } from "./tabs";
import { webPageCommands } from "./webPages";

const appCommands: readonly Command[] = [
  {
    ...fromMenu("mind.new"),
    group: "main",
    reversible: true,
    run: () => useAppStore.getState().createMind(),
  },
  ...mindCommands,
  ...documentCommands,
  ...folderCommands,
  ...tagCommands,
  ...linkedFolderCommands,
  ...answerCommands,
  ...citationCommands,
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
    // While a Citation's card shows (hovered, focused or clicked), or from its menu.
    when: (ctx) => ctx.here?.["citation.open"] !== undefined || citationHandle.current !== null,
    run: (ctx) => (ctx.here?.["citation.open"] ?? citationHandle.current?.open)?.(),
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
    destructive: true,
    appliesTo: ["block", "answer"],
    // The note's own undo (⌘Z) brings it back.
    reversible: true,
    run(ctx) {
      // An Answer's menu deletes that Answer, wherever the cursor is.
      if (ctx.here?.["block.delete"]) return ctx.here["block.delete"]();
      // The Blocks the note's selection touches, the focused one among them.
      if (ctx.focus) noteHolding(ctx.focus.id)?.commands.deleteBlock();
    },
  },
  ...footnoteCommands,
  ...tableCommands,
  ...proposalCommands,
];

for (const command of appCommands) commands.register(command);
