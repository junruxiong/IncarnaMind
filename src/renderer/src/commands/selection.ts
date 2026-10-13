import { type SelectSurface, selectedOn, useSelection } from "../selection";
import { typesText } from "../undo";
import { type Command, type CommandContext, fromMenu } from "./registry";

const surfaceOf = (ctx: CommandContext): SelectSurface | null =>
  ctx.surface === "sidebar" || ctx.surface === "library" ? ctx.surface : null;

/** Select All and Esc, the keys a list takes for its rows (docs/agents/interaction.md, rule 6). */
export const selectionCommands: readonly Command[] = [
  {
    ...fromMenu("edit.selectAll"),
    // Only while the focus is in a list: in the note, the composer or a field, ⌘A selects text.
    shortcut: { keys: "CmdOrCtrl+A", list: true },
    palette: false,
    group: "edit",
    reversible: true,
    run(ctx) {
      const surface = surfaceOf(ctx);
      const focus = document.activeElement;
      // The Edit menu's Select All comes here too, from wherever the focus is.
      if (surface && !typesText(focus) && focus?.closest("[aria-multiselectable]")) {
        useSelection.getState().all(surface);
      } else {
        document.execCommand("selectAll");
      }
    },
  },
  {
    id: "selection.clear",
    label: "command.selection.clear",
    shortcut: { keys: "Escape", list: true },
    palette: false,
    group: "edit",
    reversible: true,
    when: (ctx) => {
      const surface = surfaceOf(ctx);
      return surface !== null && selectedOn(surface).length > 0;
    },
    run(ctx) {
      useSelection.getState().clear(surfaceOf(ctx) ?? undefined);
    },
  },
];
