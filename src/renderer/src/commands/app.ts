import { signInAgain } from "../chatGptSignIn";
import { useCitationStylePicker } from "../citationStylePicker";
import { blockMenuOpeners } from "../editor/blockMenuOpeners";
import { noteHolding } from "../editor/noteKeymap";
import { useFindSources } from "../findSources";
import { currentTabOf } from "../layoutSlice";
import { startSkillFromMind } from "../makeSkill";
import { useReferenceImport } from "../referenceImport";
import { useAppStore } from "../store";
import { assistantCommands } from "./assistant";
import { deletedCommands } from "./deleted";
import { editCommands } from "./edit";
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
import { type Command, type CommandContext, commands, fromMenu } from "./registry";
import { tableCommands } from "./tables";
import { tabCommands } from "./tabs";
import { webPageCommands } from "./webPages";

/** What opens the menu of the Block with this ID, if its Mind is open. */
function blockMenuFor(blockId: string) {
  const editor = noteHolding(blockId);
  return (editor && blockMenuOpeners.get(editor)) ?? null;
}

/** The Mind in the tab the User is in, if that tab holds one. */
function mindInFront(): string | null {
  const item = currentTabOf(useAppStore.getState().layout)?.item;
  return item?.kind === "mind" ? item.mindId : null;
}

/** The Mind a command acts on: the one a menu names, else the one in front (⌘K, the assistant). */
function mindOf(ctx: CommandContext): string | null {
  const targets = ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];
  return targets.find((each) => each.kind === "mind")?.id ?? mindInFront();
}

const appCommands: readonly Command[] = [
  {
    ...fromMenu("mind.new"),
    group: "main",
    reversible: true,
    run: () => useAppStore.getState().createMind(),
  },
  ...editCommands,
  ...mindCommands,
  ...documentCommands,
  ...folderCommands,
  ...tagCommands,
  ...linkedFolderCommands,
  ...deletedCommands,
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
    // While a Citation's card shows (hovered, focused or clicked), or from its menu.
    when: (ctx) => ctx.here?.["citation.open"] !== undefined || citationHandle.current !== null,
    run: (ctx) => (ctx.here?.["citation.open"] ?? citationHandle.current?.open)?.(),
  },
  {
    // The Mind in front's citation style (#289): the same picker as its References' button.
    id: "mind.citationStyle",
    label: "citationStyle.command",
    keywords: ["citationStyle.keywords", "references.title"],
    group: "main",
    // The picker's choice is taken back with Undo.
    reversible: true,
    // A Mind's row in the sidebar picks for that Mind.
    when: (ctx) => ctx.here?.["mind.citationStyle"] !== undefined || mindInFront() !== null,
    run: (ctx) => {
      if (ctx.here?.["mind.citationStyle"]) return ctx.here["mind.citationStyle"]();
      const mindId = mindInFront();
      if (!mindId) return;
      const button = document.querySelector<HTMLElement>(
        `article[data-mind-id="${mindId}"] [data-testid="references-style"]`,
      );
      useCitationStylePicker.getState().show(mindId, button);
    },
  },
  {
    // The AI drafts it in the Mind's card; nothing is saved until the User saves it.
    id: "skill.fromMind",
    label: "skills.make.command",
    keywords: ["skills.settings.title"],
    group: "main",
    reversible: true,
    when: (ctx) => mindOf(ctx) !== null,
    run: (ctx) => {
      const mindId = mindOf(ctx);
      if (mindId) return startSkillFromMind(mindId);
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
