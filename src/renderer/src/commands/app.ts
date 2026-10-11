import { signInAgain } from "../chatGptSignIn";
import { useCitationStylePicker } from "../citationStylePicker";
import { noteHolding } from "../editor/noteKeymap";
import { useFindSources } from "../findSources";
import { currentTabOf } from "../layoutSlice";
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
  ...evidenceCommands,
  {
    ...fromMenu("settings.open"),
    group: "open",
    reversible: true,
    run: () => useAppStore.getState().openSettings(),
  },
  {
    id: "chatgpt.continue",
    label: "chatgpt.continue",
    group: "main",
    reversible: true,
    // Signs in, then shows the plan in Settings → Models, where its model is chosen; a failure says why there.
    run: async () => {
      const { openSettings } = useAppStore.getState();
      if (await signInAgain(openSettings)) openSettings("models");
    },
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
  ...proposalCommands,
];

for (const command of appCommands) commands.register(command);
