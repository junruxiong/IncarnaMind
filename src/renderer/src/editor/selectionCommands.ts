import type { Editor } from "@tiptap/core";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import { NOTE_BLOCK_TYPES } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";

/** What the selection menu's "Text ›" turns the selected Blocks into. */
export interface TurnInto {
  id: string;
  label: MessageKey;
  /** Whether the selection already is this. */
  active(editor: Editor): boolean;
  run(editor: Editor): boolean;
}

/** Lifts the selection out of lists and quotes first, so a list item turned into a heading isn't nested. */
const fresh = (editor: Editor) => editor.chain().focus().clearNodes();

const BLOCK_KINDS: readonly TurnInto[] = [
  ...([1, 2, 3] as const).map(
    (level): TurnInto => ({
      id: `heading-${level}`,
      label: `editor.slash.heading${level}`,
      active: (editor) => editor.isActive("heading", { level }),
      run: (editor) => fresh(editor).setHeading({ level }).run(),
    }),
  ),
  {
    id: "bullet-list",
    label: "editor.turnInto.bulletList",
    active: (editor) => editor.isActive("bulletList"),
    run: (editor) => fresh(editor).toggleBulletList().run(),
  },
  {
    id: "ordered-list",
    label: "editor.turnInto.orderedList",
    active: (editor) => editor.isActive("orderedList"),
    run: (editor) => fresh(editor).toggleOrderedList().run(),
  },
  {
    id: "quote",
    label: "editor.turnInto.quote",
    active: (editor) => editor.isActive("blockquote"),
    run: (editor) => fresh(editor).setBlockquote().run(),
  },
];

const TEXT: TurnInto = {
  id: "text",
  label: "editor.slash.text",
  active: (editor) => !BLOCK_KINDS.some((kind) => kind.active(editor)),
  run: (editor) => fresh(editor).run(),
};

export const TURN_INTO: readonly TurnInto[] = [TEXT, ...BLOCK_KINDS];

/** What the selection is now. */
export const currentTurnInto = (editor: Editor): TurnInto =>
  BLOCK_KINDS.find((kind) => kind.active(editor)) ?? TEXT;

const NOT_TURNED: readonly string[] = ["codeBlock", "blockMath"];

/**
 * Whether the selection lies in Notes of text: Questions and Answers are not
 * turned into anything else, and neither are code blocks and formulas.
 */
export function canTurnInto(editor: Editor): boolean {
  const { doc, selection } = editor.state;
  let ok = true;
  doc.nodesBetween(selection.from, selection.to, (node, _pos, parent) => {
    if (parent === doc) {
      const name = node.type.name;
      ok =
        ok && (NOTE_BLOCK_TYPES as readonly string[]).includes(name) && !NOT_TURNED.includes(name);
    }
    return ok;
  });
  return ok;
}

/** Whether the selection sits inside one line of text, which a formula can replace. */
export function inOneTextblock(editor: Editor): boolean {
  const { $from, $to } = editor.state.selection;
  return $from.sameParent($to) && $from.parent.isTextblock;
}

/** The selected text replaced by an inline formula written in it; where the formula is, for its LaTeX field to open at, or null. */
export function selectionToMath(editor: Editor): number | null {
  if (!inOneTextblock(editor)) return null;
  const { from, to } = editor.state.selection;
  const latex = editor.state.doc.textBetween(from, to);
  const inserted = editor.commands.insertContentAt(
    { from, to },
    { type: "inlineMath", attrs: { latex } },
  );
  return inserted ? from : null;
}

/** A typed address as a link target: "example.com" is taken as https. */
export function linkTarget(typed: string): string {
  const value = typed.trim();
  return /^[a-z][a-z0-9+.-]*:/i.test(value) ? value : `https://${value}`;
}

/** Runs a change as an Undo step of its own, rather than merged with the typing just before or after it. */
export function asOwnUndoStep<T>(editor: Editor, run: () => T): T {
  const undo = yUndoPluginKey.getState(editor.state)?.undoManager;
  undo?.stopCapturing();
  const result = run();
  undo?.stopCapturing();
  return result;
}
