import { TextSelection } from "@tiptap/pm/state";
import { type Editor, useEditorState } from "@tiptap/react";
import { BubbleMenu, type BubbleMenuProps } from "@tiptap/react/menus";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";

type Mark = "bold" | "italic" | "strike";

const MARKS: readonly { mark: Mark; label: MessageKey; className: string }[] = [
  { mark: "bold", label: "editor.format.bold", className: "font-bold" },
  { mark: "italic", label: "editor.format.italic", className: "italic" },
  { mark: "strike", label: "editor.format.strike", className: "line-through" },
];

/** Shown over selected text, not over code or a selected Block. */
const showOverText: NonNullable<BubbleMenuProps["shouldShow"]> = ({
  editor,
  element,
  view,
  state,
  from,
  to,
}) => {
  const { selection } = state;
  if (selection.empty || !(selection instanceof TextSelection) || !editor.isEditable) return false;
  if (editor.isActive("codeBlock") || !state.doc.textBetween(from, to).length) return false;
  return view.hasFocus() || element.contains(document.activeElement);
};

/** The old editor's bubble menu: bold, italic and strikethrough for the selected text. */
export function FormatMenu({ editor }: { editor: Editor }) {
  const t = useT();
  const active = useEditorState({
    editor,
    selector: ({ editor: current }) => ({
      bold: current.isActive("bold"),
      italic: current.isActive("italic"),
      strike: current.isActive("strike"),
    }),
  });

  return (
    <BubbleMenu
      editor={editor}
      shouldShow={showOverText}
      role="toolbar"
      aria-label={t("editor.format.label")}
      data-testid="format-menu"
      className="format-menu"
    >
      {MARKS.map(({ mark, label, className }) => (
        <button
          key={mark}
          type="button"
          aria-pressed={active[mark]}
          // Keep the selection in the editor.
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => editor.chain().focus().toggleMark(mark).run()}
          className={className}
        >
          {t(label)}
        </button>
      ))}
    </BubbleMenu>
  );
}
