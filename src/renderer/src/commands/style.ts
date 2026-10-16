import { useMindStylePanel } from "../mindStylePanel";
import type { Command, CommandContext } from "./registry";
import { mindShown, onlyOne } from "./targets";

const buttonAt = (selector: string) => document.querySelector<HTMLButtonElement>(selector);

/** The ⋯ of the object the command was asked on, else of the Mind's shown tab; none: the window's top. */
function anchorFor(ctx: CommandContext, mindId: string): HTMLButtonElement | null {
  const target = ctx.selection[0] ?? ctx.focus;
  const own =
    target && buttonAt(`[data-object-menu="${CSS.escape(`${target.kind}:${target.id}`)}"]`);
  return (
    own ||
    buttonAt(
      `[role="tab"][aria-selected="true"][data-mind-id="${CSS.escape(mindId)}"] [data-object-menu]`,
    )
  );
}

export const styleCommands: readonly Command[] = [
  {
    id: "mind.style",
    label: "mindStyle.command",
    keywords: ["mindStyle.keywords"],
    group: "main",
    // Opening it changes nothing; each choice in it is taken back with Undo.
    reversible: true,
    when: (ctx) => onlyOne(ctx) && mindShown(ctx) !== null,
    run: (ctx) => {
      const mindId = mindShown(ctx);
      if (!mindId) return;
      // Once the menu has closed and given the focus back, so Style opens beside its button.
      requestAnimationFrame(() =>
        useMindStylePanel.getState().show(mindId, anchorFor(ctx, mindId)),
      );
    },
  },
];
