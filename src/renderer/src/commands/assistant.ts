import type { AssistantDock } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { isSide, SIDE_DOCK_FROM } from "../assistantPlace";
import { useComposer } from "../composer";
import { useAppStore } from "../store";
import { assistantHandle } from "./handles";
import type { Command } from "./registry";

/*
 * The assistant's commands (#253): where it sits, folding it, ⌘J to its
 * composer, New chat and History. Its ⋯ menu, its header and rail, and ⌘K
 * run these, so each has a keyboard and a menu way beside the drag.
 */

const store = () => useAppStore.getState();
const isThere = () => assistantHandle.current !== null;
const folded = () => store().layout?.assistant.folded ?? false;

/** The dock commands, in the order the ⋯ menu lists them. */
export const DOCKS: readonly { dock: AssistantDock; label: MessageKey }[] = [
  { dock: "right", label: "assistant.dock.right" },
  { dock: "left", label: "assistant.dock.left" },
  { dock: "bottom", label: "assistant.dock.bottom" },
  { dock: "float", label: "assistant.dock.float" },
];

export const dockCommandId = (dock: AssistantDock) =>
  dock === "float" ? "assistant.float" : (`assistant.dock-${dock}` as const);

/**
 * Docks it, or floats it, unfolded. A side dock chosen in a narrow window is
 * kept for when it is wider; meanwhile a line says why it stays at the bottom.
 */
export function placeAt(dock: AssistantDock): Promise<void> {
  if (isSide(dock) && window.innerWidth < SIDE_DOCK_FROM) {
    store().showTabNotice({ kind: "assistantNarrow" });
  }
  return store().placeAssistant({ dock, folded: false });
}

/** Puts the cursor in the composer, unfolding the assistant for it. */
export function focusComposer(): void {
  if (folded()) void store().placeAssistant({ folded: false });
  useComposer.getState().requestFocus();
}

export const assistantCommands: readonly Command[] = [
  ...DOCKS.map(
    ({ dock, label }): Command => ({
      id: dockCommandId(dock),
      label: (_ctx, t) => t("command.assistant.dock", { dock: t(label) }),
      group: "main",
      // Another place puts it back.
      reversible: true,
      when: isThere,
      run: () => placeAt(dock),
    }),
  ),
  {
    id: "assistant.fold",
    label: (_ctx, t) => t(folded() ? "command.assistant.unfold" : "command.assistant.fold"),
    group: "main",
    reversible: true,
    when: isThere,
    run: () => store().placeAssistant({ folded: !folded() }),
  },
  {
    // ⌘J, as in Cursor: the assistant's composer, wherever the User is.
    id: "composer.focus",
    label: "command.composer.focus",
    shortcut: { keys: "CmdOrCtrl+J" },
    group: "main",
    reversible: true,
    when: isThere,
    run: focusComposer,
  },
  {
    // An empty conversation is reused, so with no conversations kept yet (#258),
    // New chat is the empty one, ready to ask in.
    id: "assistant.newChat",
    label: "assistant.newChat",
    shortcut: { keys: "CmdOrCtrl+Shift+O" },
    group: "main",
    reversible: true,
    when: isThere,
    run: focusComposer,
  },
  {
    // No shortcut: ⌘Y, the browsers' History key on a Mac, is Redo on Windows.
    id: "assistant.history",
    label: "command.assistant.history",
    group: "open",
    reversible: true,
    when: isThere,
    run: () => assistantHandle.current?.openHistory(),
  },
];
