import { countShownOf, setCountShown } from "../../../shared/mindWordCount";
import { useShownPages } from "../shownPages";
import type { Command, CommandContext } from "./registry";
import { mindShown, onlyOne } from "./targets";

const pageShown = (ctx: CommandContext) => {
  const mindId = mindShown(ctx);
  return mindId ? (useShownPages.getState()[mindId] ?? null) : null;
};

export const pageCommands: readonly Command[] = [
  {
    id: "mind.countShown",
    label: (ctx, t) => {
      const doc = pageShown(ctx);
      return t(doc && countShownOf(doc) ? "page.countHidden.command" : "page.countShown.command");
    },
    keywords: ["page.countShown.keywords"],
    group: "main",
    // Run again, it switches back; the Mind's text doesn't change.
    reversible: true,
    when: (ctx) => onlyOne(ctx) && pageShown(ctx) !== null,
    run: (ctx) => {
      const doc = pageShown(ctx);
      if (doc) setCountShown(doc, !countShownOf(doc));
    },
  },
];
