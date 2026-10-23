import { useReviewables } from "../reviewPane";
import type { Command, CommandContext } from "./registry";
import { mindShown, onlyOne } from "./targets";

function reviewableShown(ctx: CommandContext) {
  const mindId = onlyOne(ctx) ? mindShown(ctx) : null;
  const held = mindId ? useReviewables.getState()[mindId] : undefined;
  return held?.has() ? held : null;
}

/** The reviewing pane of someone's changes in a Word file (#697), from the keyboard and ⌘K. */
export const reviewCommands: readonly Command[] = [
  {
    id: "review.open",
    label: "review.title",
    keywords: ["review.keywords"],
    group: "main",
    // It only shows the pane; Esc or its × closes it.
    reversible: true,
    when: (ctx) => reviewableShown(ctx) !== null,
    run: (ctx) => reviewableShown(ctx)?.open(),
  },
];
