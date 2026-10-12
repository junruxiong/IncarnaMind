import { showBibliography } from "../editor/bibliographyCommands";
import { noteHolding } from "../editor/noteKeymap";
import type { Command } from "./registry";

/** Insert Bibliography (#102): the "/" menu's Bibliography, from ⌘K, at the Block the cursor is in. */
export const bibliographyCommands: readonly Command[] = [
  {
    id: "bibliography.insert",
    label: "editor.bibliography.insert",
    group: "main",
    appliesTo: ["block"],
    reversible: true,
    when: (ctx) => ctx.focus?.kind === "block" && noteHolding(ctx.focus.id) !== null,
    run(ctx) {
      const editor = ctx.focus?.kind === "block" ? noteHolding(ctx.focus.id) : null;
      if (editor && !editor.isDestroyed && editor.isEditable) showBibliography(editor);
    },
  },
];
