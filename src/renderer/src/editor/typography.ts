import { InputRule } from "@tiptap/core";
import Typography from "@tiptap/extension-typography";

/**
 * Whether the text typed so far in a textblock is inside LaTeX that isn't
 * closed yet: after a `$$` (or the `$$$` of block math) with no closing one.
 */
function inOpenLatex(textBefore: string): boolean {
  return (textBefore.match(/\$\$/g)?.length ?? 0) % 2 === 1;
}

/**
 * A rule's pattern that matches only outside LaTeX being typed. It stays a
 * RegExp, so Tiptap still hands the rule its capture groups.
 */
class OutsideLatex extends RegExp {
  override exec(text: string): RegExpExecArray | null {
    return inOpenLatex(text) ? null : super.exec(text);
  }
}

/**
 * The old editor's smart typography, as the User types: “quotes”, ’ for
 * apostrophes, — for `--`, … for `...`, arrows, © and the like. Backspace
 * right after a replacement undoes it.
 *
 * Only typed text changes (these are input rules). What the core writes into
 * the Mind's Yjs document, an Answer's text and its Citations' quotes, comes
 * in as a remote change and is left exactly as written; so is pasted text.
 * Code (blocks and inline) is left alone by Tiptap, and so is LaTeX typed
 * between `$$` before it becomes a formula, where `^2` or `->` must stay as typed.
 */
export const SmartTypography = Typography.extend({
  addInputRules() {
    return (this.parent?.() ?? []).map((rule) =>
      rule.find instanceof RegExp
        ? new InputRule({
            find: new OutsideLatex(rule.find),
            handler: rule.handler,
            undoable: rule.undoable,
          })
        : rule,
    );
  },
});
