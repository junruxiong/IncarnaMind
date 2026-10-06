import { type ComputePositionConfig, offset } from "@floating-ui/dom";
import { isMacOS } from "@tiptap/core";
import { DragHandle } from "@tiptap/extension-drag-handle-react";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { Editor } from "@tiptap/react";
import { useEffect, useRef, useState } from "react";
import { GripIcon, TrashIcon } from "../components/icons";
import { useT } from "../i18n";
import { findBlock } from "./blockCommands";
import { BLOCK_ID_ATTRIBUTE } from "./noteSchema";
import { Popover } from "./Popover";

/** Left of the Block's first line, clear of its focus outline. */
const HANDLE_POSITION: ComputePositionConfig = {
  placement: "left-start",
  middleware: [offset({ mainAxis: 10, crossAxis: 2 })],
};

const DELETE_SHORTCUT_LABEL = isMacOS() ? "⌘⇧⌫" : "Ctrl+Shift+Backspace";

/**
 * The handle left of the Block under the mouse. Dragging it moves the Block;
 * clicking it opens the Block's menu.
 */
export function BlockHandle({ editor }: { editor: Editor }) {
  const t = useT();
  const hovered = useRef<{ node: ProseMirrorNode; pos: number } | null>(null);
  const grip = useRef<HTMLButtonElement>(null);
  /** The Block whose menu is open: its ID, and where it was when the menu opened. */
  const [menuFor, setMenuFor] = useState<{ id: unknown; pos: number } | null>(null);

  // While the menu is open, the handle stays on its Block.
  const lockHandle = (locked: boolean) => {
    if (!editor.isDestroyed)
      editor.view.dispatch(editor.state.tr.setMeta("lockDragHandle", locked));
  };

  const openMenu = () => {
    const block = hovered.current;
    if (!block) return;
    lockHandle(true);
    setMenuFor({ id: block.node.attrs[BLOCK_ID_ATTRIBUTE], pos: block.pos });
  };

  const closeMenu = (refocus: boolean) => {
    setMenuFor(null);
    lockHandle(false);
    if (refocus) editor.commands.focus();
  };

  const deleteBlock = () => {
    if (!menuFor) return;
    // Found by ID, which stays right even if the Mind changed while the menu was open.
    const pos =
      typeof menuFor.id === "string" ? findBlock(editor.state.doc, menuFor.id)?.pos : menuFor.pos;
    closeMenu(false);
    if (pos !== undefined) editor.chain().focus().deleteBlock(pos).run();
  };

  return (
    <>
      <DragHandle
        editor={editor}
        className="block-handle"
        computePositionConfig={HANDLE_POSITION}
        onNodeChange={({ node, pos }) => {
          hovered.current = node ? { node, pos } : null;
        }}
      >
        <button
          ref={grip}
          type="button"
          data-testid="block-handle"
          aria-label={t("editor.block.handle")}
          title={t("editor.block.handle")}
          aria-haspopup="menu"
          aria-expanded={menuFor !== null}
          onClick={openMenu}
        >
          <GripIcon className="size-5" />
        </button>
      </DragHandle>
      {menuFor && grip.current && (
        <BlockMenu
          anchor={grip.current}
          onDelete={deleteBlock}
          onClose={closeMenu}
          label={t("editor.block.menu")}
          deleteLabel={t("editor.block.delete")}
        />
      )}
    </>
  );
}

function BlockMenu({
  anchor,
  onDelete,
  onClose,
  label,
  deleteLabel,
}: {
  anchor: Element;
  onDelete(): void;
  onClose(refocus: boolean): void;
  label: string;
  deleteLabel: string;
}) {
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => first.current?.focus(), []);

  return (
    <Popover
      anchor={anchor}
      role="menu"
      aria-label={label}
      data-testid="block-menu"
      onDismiss={() => onClose(false)}
      onKeyDown={(event) => {
        // Esc returns to writing.
        if (event.key !== "Escape") return;
        event.preventDefault();
        onClose(true);
      }}
      className="editor-menu w-[240px]"
    >
      <button
        ref={first}
        type="button"
        role="menuitem"
        data-testid="block-menu-delete"
        onClick={onDelete}
        className="editor-menu-item"
      >
        <span className="editor-menu-icon">
          <TrashIcon className="size-4" />
        </span>
        <span className="flex-1">{deleteLabel}</span>
        <kbd className="font-sans text-custom-xs text-gray-400">{DELETE_SHORTCUT_LABEL}</kbd>
      </button>
    </Popover>
  );
}
