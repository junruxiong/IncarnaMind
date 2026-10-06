import type { VirtualElement } from "@floating-ui/dom";
import { type Editor, Extension } from "@tiptap/core";
import { isChangeOrigin } from "@tiptap/extension-collaboration";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { NodeSelection, Selection, type Transaction } from "@tiptap/pm/state";
import {
  absolutePositionToRelativePosition,
  relativePositionToAbsolutePosition,
  ySyncPluginKey,
} from "@tiptap/y-tiptap";
import { useEffect, useMemo, useRef, useState } from "react";
import { useT } from "../i18n";
import { MATH_TYPES } from "./noteSchema";
import { Popover } from "./Popover";

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    mathEditing: {
      /** Opens the LaTeX field of the formula at `pos`. */
      editMath: (pos: number) => ReturnType;
    };
  }
}

/**
 * Opening a formula's LaTeX field: `editMath`, and Enter on a formula selected
 * with the arrow keys. Clicking a formula opens it too (see `noteExtensions`).
 */
export const MathEditing = Extension.create<{ onEdit: (pos: number) => void }>({
  name: "mathEditing",
  // Before the Enter handlers that would split the paragraph around the formula.
  priority: 1000,

  addOptions() {
    return { onEdit: () => undefined };
  },

  addCommands() {
    return {
      editMath:
        (pos) =>
        ({ state, dispatch }) => {
          if (!mathAt(state.doc, pos)) return false;
          if (dispatch) this.options.onEdit(pos);
          return true;
        },
    };
  },

  addKeyboardShortcuts() {
    return {
      Enter: ({ editor }) => {
        const { selection } = editor.state;
        return selection instanceof NodeSelection && editor.commands.editMath(selection.from);
      },
    };
  },
});

function mathAt(doc: ProseMirrorNode, pos: number): ProseMirrorNode | null {
  const node = pos >= 0 && pos < doc.content.size ? doc.nodeAt(pos) : null;
  return node && MATH_TYPES.includes(node.type.name) ? node : null;
}

/**
 * A document position that stays on the same spot through edits: this
 * editor's own, mapped step by step, and those the core pushes, which replace
 * the whole document and are followed through Yjs instead.
 */
class TrackedPosition {
  private relative: ReturnType<typeof absolutePositionToRelativePosition> | null = null;

  constructor(
    private readonly editor: Editor,
    public pos: number,
  ) {
    this.remember();
  }

  follow(transaction: Transaction): void {
    if (!transaction.docChanged) return;
    const sync = ySyncPluginKey.getState(this.editor.state);
    if (isChangeOrigin(transaction) && sync?.binding && this.relative) {
      this.pos =
        relativePositionToAbsolutePosition(
          sync.doc,
          sync.type,
          this.relative,
          sync.binding.mapping,
        ) ?? -1;
    } else {
      this.pos = transaction.mapping.map(this.pos);
    }
    this.remember();
  }

  private remember(): void {
    const sync = ySyncPluginKey.getState(this.editor.state);
    this.relative = sync?.binding
      ? absolutePositionToRelativePosition(this.pos, sync.type, sync.binding.mapping)
      : null;
  }
}

/**
 * The LaTeX field of one formula, under it. What is typed shows in the formula
 * straight away. Enter or a press elsewhere keeps it, Esc puts back what was
 * there, and a formula left empty is removed.
 */
export function MathEditor({
  editor,
  pos,
  onClose,
}: {
  editor: Editor;
  pos: number;
  onClose(): void;
}) {
  const t = useT();
  const field = useRef<HTMLTextAreaElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  const position = useMemo(() => new TrackedPosition(editor, pos), [editor, pos]);
  const original = useMemo(
    () => String(mathAt(editor.state.doc, pos)?.attrs.latex ?? ""),
    [editor, pos],
  );
  const [latex, setLatex] = useState(original);
  const anchor = useMemo<VirtualElement>(
    () => ({
      getBoundingClientRect: () => {
        const dom = editor.view.nodeDOM(position.pos);
        return dom instanceof Element ? dom.getBoundingClientRect() : new DOMRect();
      },
      contextElement: editor.view.dom,
    }),
    [editor, position],
  );

  // A frame later, after any focus the editor still has queued (Tiptap's `focus()` waits a frame).
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      field.current?.focus();
      field.current?.select();
    });
    return () => cancelAnimationFrame(frame);
  }, []);

  // Follow the formula through edits; close if it is deleted.
  useEffect(() => {
    const follow = ({ transaction }: { transaction: Transaction }) => {
      position.follow(transaction);
      if (!mathAt(editor.state.doc, position.pos)) close.current();
    };
    editor.on("transaction", follow);
    return () => {
      editor.off("transaction", follow);
    };
  }, [editor, position]);

  const write = (value: string) => {
    if (!mathAt(editor.state.doc, position.pos)) return;
    editor.commands.command(({ tr }) => {
      tr.setNodeAttribute(position.pos, "latex", value);
      return true;
    });
  };

  const finish = (value: string, refocus: boolean) => {
    const node = mathAt(editor.state.doc, position.pos);
    close.current();
    if (!node) return;
    const at = position.pos;
    const chain = refocus ? editor.chain().focus() : editor.chain();
    if (!value.trim()) {
      if (node.type.name === "blockMath") chain.deleteBlock(at).run();
      else chain.deleteInlineMath({ pos: at }).run();
      return;
    }
    if (value !== node.attrs.latex) write(value);
    if (refocus) {
      chain
        .command(({ tr }) => {
          tr.setSelection(Selection.near(tr.doc.resolve(at + node.nodeSize)));
          return true;
        })
        .run();
    }
  };

  return (
    <Popover
      anchor={anchor}
      onDismiss={() => finish(latex, false)}
      aria-label={t("editor.math.label")}
      className="editor-menu w-96 max-w-[90vw] p-2"
    >
      <textarea
        ref={field}
        data-testid="math-editor"
        aria-label={t("editor.math.label")}
        placeholder={t("editor.math.placeholder")}
        value={latex}
        rows={Math.min(6, latex.split("\n").length)}
        spellCheck={false}
        onChange={(event) => {
          setLatex(event.target.value);
          write(event.target.value);
        }}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing) return;
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            finish(latex, true);
          } else if (event.key === "Escape") {
            event.preventDefault();
            finish(original, true);
          }
        }}
        className="math-input"
      />
      <p className="mt-1 px-1 text-custom-xs text-gray-500">{t("editor.math.hint")}</p>
    </Popover>
  );
}
