import type { MindLook } from "../../../shared/wordLook";
import { useWordLooks } from "../wordLooks";
import type { Command, CommandContext } from "./registry";
import { mindShown, onlyOne } from "./targets";

const lookShown = (ctx: CommandContext) => {
  const mindId = mindShown(ctx);
  return mindId ? (useWordLooks.getState()[mindId] ?? null) : null;
};

const switchTo = (look: MindLook): Pick<Command, "when" | "run"> => ({
  when: (ctx) => onlyOne(ctx) && lookShown(ctx)?.shown === (look === "word" ? "plain" : "word"),
  run: (ctx) => lookShown(ctx)?.choose(look),
});

export const wordLookCommands: readonly Command[] = [
  {
    id: "mind.look.word",
    label: "look.command.word",
    keywords: ["look.keywords"],
    group: "main",
    // Plain Mind switches back; nothing in the Mind changes either way.
    reversible: true,
    ...switchTo("word"),
  },
  {
    id: "mind.look.plain",
    label: "look.command.plain",
    keywords: ["look.keywords"],
    group: "main",
    reversible: true,
    ...switchTo("plain"),
  },
  {
    id: "mind.openOriginal",
    label: "wordImport.openOriginal",
    keywords: ["look.keywords"],
    group: "open",
    reversible: true,
    when: (ctx) => onlyOne(ctx) && Boolean(lookShown(ctx)?.openOriginal),
    run: (ctx) => lookShown(ctx)?.openOriginal?.(),
  },
];
