import { useCitationStylePicker } from "../citationStylePicker";
import { noteHolding } from "../editor/noteKeymap";
import { currentTabOf } from "../layoutSlice";
import { useReferenceImport } from "../referenceImport";
import { useAppStore } from "../store";
import { footnoteCommands } from "./footnotes";
import { citationHandle, composerHandle, sidebarHandle } from "./handles";
import { meetingCommands } from "./meetings";
import { type Command, commands, fromMenu } from "./registry";
import { tableCommands } from "./tables";
import { tabCommands } from "./tabs";
import { webPageCommands } from "./webPages";

/** The Mind in the tab the User is in, if that tab holds one. */
function mindInFront(): string | null {
  const item = currentTabOf(useAppStore.getState().layout)?.item;
  return item?.kind === "mind" ? item.mindId : null;
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
    // The Mind in front's citation style (#289): the same picker as its References' button.
    id: "mind.citationStyle",
    label: "citationStyle.command",
    keywords: ["citationStyle.keywords", "references.title"],
    group: "main",
    // The picker's choice is taken back with Undo.
    reversible: true,
    when: () => mindInFront() !== null,
    run: () => {
      const mindId = mindInFront();
      if (!mindId) return;
      const button = document.querySelector<HTMLElement>(
        `article[data-mind-id="${mindId}"] [data-testid="references-style"]`,
      );
      useCitationStylePicker.getState().show(mindId, button);
    },
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
  ...footnoteCommands,
  ...tableCommands,
];

for (const command of appCommands) commands.register(command);
