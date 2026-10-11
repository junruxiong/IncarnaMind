import type { VirtualElement } from "@floating-ui/dom";
import type { Editor } from "@tiptap/core";
import { useRef, useState } from "react";
import { useT } from "../i18n";
import { Popover } from "./Popover";

/**
 * The field a link's address is typed in, next to `anchor`: Enter applies it,
 * an empty one takes the link off, Esc cancels.
 */
export function LinkField({
  anchor,
  current,
  onApply,
  onClose,
}: {
  anchor: Element | VirtualElement;
  current: string | undefined;
  onApply(typed: string): void;
  onClose(): void;
}) {
  const t = useT();
  const field = useRef<HTMLInputElement>(null);
  const [typed, setTyped] = useState(current ?? "");

  return (
    <Popover
      anchor={anchor}
      initialFocus={field}
      aria-label={t("editor.link.label")}
      onDismiss={onClose}
      className="editor-menu link-editor"
    >
      <input
        ref={field}
        type="text"
        inputMode="url"
        data-testid="link-field"
        aria-label={t("editor.link.label")}
        placeholder={t("editor.link.placeholder")}
        value={typed}
        spellCheck={false}
        onChange={(event) => setTyped(event.target.value)}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing || event.key === "Process") return;
          if (event.key === "Enter") {
            event.preventDefault();
            onApply(typed);
          }
        }}
        className="math-input"
      />
      <p className="math-hint">{t(current ? "editor.link.hintEdit" : "editor.link.hint")}</p>
    </Popover>
  );
}

/** Where the selection is on screen, for a field to sit under when there is no button to anchor to. */
export function caretAnchor(editor: Editor): VirtualElement {
  const { from, to } = editor.state.selection;
  return {
    getBoundingClientRect() {
      const start = editor.view.coordsAtPos(from);
      const end = editor.view.coordsAtPos(to);
      const left = Math.min(start.left, end.left);
      const top = Math.min(start.top, end.top);
      return new DOMRect(left, top, Math.max(end.right, start.right) - left, end.bottom - top);
    },
  };
}
