import { useEffect } from "react";
import type { EditMenuState } from "../../../shared/bridge";
import { files } from "../core";
import { noteAround } from "../editor/noteKeymap";
import type { Translate } from "../objects";
import { typesText, type Undoable, useUndo } from "../undo";
import { type Command, type CommandContext, fromMenu } from "./registry";

/**
 * Where ⌘Z and ⇧⌘Z go from the focus (docs/designs/interaction-system.md, C.6):
 * the note's own undo in the note, a field's own in a field, and the app's
 * changes everywhere else. ⌘K runs them on the app's changes.
 */
function placeOf(ctx: CommandContext): EditMenuState["text"] {
  if (ctx.surface === "palette") return null;
  const focus = document.activeElement;
  if (!typesText(focus)) return null;
  return noteAround(focus) ? "note" : "field";
}

const nameOf = (change: Undoable | undefined) => (change ? (change.name ?? "") : null);

const labelFor = (step: "undo" | "redo", t: Translate) => {
  const { done, undone } = useUndo.getState();
  const name = nameOf((step === "undo" ? done : undone).at(-1));
  if (!name) return t(step === "undo" ? "menu.undo" : "menu.redo");
  return t(step === "undo" ? "menu.undo.named" : "menu.redo.named", { action: name });
};

export const editCommands: readonly Command[] = [
  {
    ...fromMenu("edit.undo"),
    label: (_ctx, t) => labelFor("undo", t),
    shortcut: { keys: "CmdOrCtrl+Z", outsideText: true },
    group: "edit",
    reversible: true,
    enabled: (ctx) =>
      placeOf(ctx) !== null || useUndo.getState().done.length > 0
        ? true
        : { reason: "undo.nothing" },
    run(ctx) {
      const place = placeOf(ctx);
      if (place === "note") noteAround(document.activeElement)?.commands.undo();
      // A field's own undo, the system Edit menu's editing command, which only execCommand reaches.
      else if (place === "field") document.execCommand("undo");
      else return useUndo.getState().undo();
    },
  },
  {
    ...fromMenu("edit.redo"),
    label: (_ctx, t) => labelFor("redo", t),
    shortcut: { keys: "CmdOrCtrl+Shift+Z", alsoOther: "Ctrl+Y", outsideText: true },
    group: "edit",
    reversible: true,
    enabled: (ctx) =>
      placeOf(ctx) !== null || useUndo.getState().undone.length > 0
        ? true
        : { reason: "redo.nothing" },
    run(ctx) {
      const place = placeOf(ctx);
      if (place === "note") noteAround(document.activeElement)?.commands.redo();
      else if (place === "field") document.execCommand("redo");
      else return useUndo.getState().redo();
    },
  },
];

/** The Edit menu's state here now: where its Undo and Redo go, and what they name. */
function editMenuState(): EditMenuState {
  const { done, undone } = useUndo.getState();
  return {
    text: placeOf({ selection: [], focus: null, surface: "mind" }),
    undo: nameOf(done.at(-1)),
    redo: nameOf(undone.at(-1)),
  };
}

/**
 * Keeps the application menu's Undo and Redo told where the focus is and what
 * they would take back or do again, as the focus moves and changes are made.
 */
export function useEditMenu(): void {
  useEffect(() => {
    let sent = "";
    let pending = 0;
    const report = () => {
      pending = 0;
      const state = editMenuState();
      const key = JSON.stringify(state);
      if (key === sent) return;
      sent = key;
      files.setEditMenu(state);
    };
    // Leaving one field for another blurs then focuses: one report for both.
    const soon = () => {
      if (!pending) pending = window.setTimeout(report, 0);
    };
    // Another window may have told the menu its own state meanwhile.
    const again = () => {
      sent = "";
      report();
    };
    report();
    document.addEventListener("focusin", soon);
    document.addEventListener("focusout", soon);
    window.addEventListener("focus", again);
    const stop = useUndo.subscribe(soon);
    return () => {
      window.clearTimeout(pending);
      document.removeEventListener("focusin", soon);
      document.removeEventListener("focusout", soon);
      window.removeEventListener("focus", again);
      stop();
    };
  }, []);
}
