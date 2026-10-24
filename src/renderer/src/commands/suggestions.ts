import { acceptSuggestions, rejectSuggestions, waitingIds } from "../suggestionReview";
import type { Command, CommandContext } from "./registry";
import { idsOf } from "./targets";

const waitingIn = (ctx: CommandContext): string[] => {
  const waiting = new Set(waitingIds());
  return idsOf(ctx, "document").filter((id) => waiting.has(id));
};

export const suggestionCommands: readonly Command[] = [
  {
    id: "document.accept",
    label: (ctx, t) => {
      const count = waitingIn(ctx).length;
      return count > 1 ? t("document.accept.many", { count }) : t("document.accept");
    },
    group: "main",
    appliesTo: ["document"],
    when: (ctx) => waitingIn(ctx).length > 0,
    // ⌘Z puts the suggestion back.
    reversible: true,
    async run(ctx) {
      return (await acceptSuggestions(waitingIn(ctx))) ?? undefined;
    },
  },
  {
    id: "document.reject",
    label: (ctx, t) => {
      const count = waitingIn(ctx).length;
      return count > 1 ? t("document.reject.many", { count }) : t("document.reject");
    },
    group: "main",
    appliesTo: ["document"],
    when: (ctx) => waitingIn(ctx).length > 0,
    reversible: true,
    async run(ctx) {
      return (await rejectSuggestions(waitingIn(ctx))) ?? undefined;
    },
  },
  {
    id: "library.acceptSuggestions",
    label: "library.acceptSuggestions",
    // In the Library's lists only: ⌘↵ sends a Question or accepts a change where those are.
    shortcut: { keys: "CmdOrCtrl+Enter", list: true },
    group: "main",
    when: (ctx) =>
      (ctx.surface === "library" || ctx.surface === "palette") && waitingIds().length > 0,
    reversible: true,
    async run() {
      return (await acceptSuggestions(waitingIds())) ?? undefined;
    },
  },
];
