import { peekAt } from "../peek";
import type { Command } from "./registry";
import { documentsOf, idsOf, onlyOne } from "./targets";

/** Peek: a quick look at a Document beside the list, as Space on its row asks for it (see `usePeekKey`). */
export const peekCommands: readonly Command[] = [
  {
    id: "document.peek",
    label: "document.peek",
    rowKeys: "Space",
    group: "open",
    appliesTo: ["document"],
    when: (ctx) =>
      (ctx.surface === "sidebar" || ctx.surface === "library") &&
      onlyOne(ctx) &&
      documentsOf(ctx).length === 1,
    reversible: true,
    run(ctx) {
      const id = idsOf(ctx, "document")[0];
      const surface = ctx.surface;
      // Once the menu has closed and given the focus back.
      if (id && (surface === "sidebar" || surface === "library")) {
        requestAnimationFrame(() => peekAt(id, surface));
      }
    },
  },
];
