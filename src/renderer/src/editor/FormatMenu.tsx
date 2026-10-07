import { isMacOS } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { type Editor, useEditorState } from "@tiptap/react";
import { BubbleMenu, type BubbleMenuProps } from "@tiptap/react/menus";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";

type Mark = "bold" | "italic" | "strike" | "highlight";

/** A shortcut as the menu shows it: ⌘⇧H on macOS, Ctrl+Shift+H elsewhere. */
const shortcut = (key: string, shift = false) =>
  isMacOS() ? `⌘${shift ? "⇧" : ""}${key}` : `Ctrl+${shift ? "Shift+" : ""}${key}`;

/** Each mark's button, with the shortcut Tiptap binds it to. */
const MARKS: readonly { mark: Mark; label: MessageKey; className: string; keys: string }[] = [
  { mark: "bold", label: "editor.format.bold", className: "font-bold", keys: shortcut("B") },
  { mark: "italic", label: "editor.format.italic", className: "italic", keys: shortcut("I") },
  {
    mark: "strike",
    label: "editor.format.strike",
    className: "line-through",
    keys: shortcut("S", true),
  },
  {
    mark: "highlight",
    label: "editor.format.highlight",
    className: "format-highlight",
    keys: shortcut("H", true),
  },
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

/** The bubble menu over selected text: bold, italic, strikethrough and highlight. */
export function FormatMenu({ editor }: { editor: Editor }) {
  const t = useT();
  const active = useEditorState({
    editor,
    selector: ({ editor: current }) => ({
      bold: current.isActive("bold"),
      italic: current.isActive("italic"),
      strike: current.isActive("strike"),
      highlight: current.isActive("highlight"),
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
      {MARKS.map(({ mark, label, className, keys }) => (
        <button
          key={mark}
          type="button"
          aria-pressed={active[mark]}
          title={`${t(label)} (${keys})`}
          data-testid={`format-${mark}`}
          // Keep the selection in the editor.
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => editor.chain().focus().toggleMark(mark).run()}
          data-mark={mark}
        >
          {/* The label shows what the mark does: bold, italic, struck through, highlighted. */}
          <span className={className}>{t(label)}</span>
        </button>
      ))}
    </BubbleMenu>
  );
}
