import { type ComputePositionConfig, offset } from "@floating-ui/dom";
import { isMacOS } from "@tiptap/core";
import { DragHandle } from "@tiptap/extension-drag-handle-react";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { Editor } from "@tiptap/react";
import { useEffect, useRef, useState } from "react";
import { INCLUDE_IN_CONTEXT_ATTRIBUTE, NOTE_BLOCK_TYPES } from "../../../core/api";
import { CheckIcon, GripIcon, TrashIcon } from "../components/icons";
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

const isNote = (node: ProseMirrorNode) =>
  (NOTE_BLOCK_TYPES as readonly string[]).includes(node.type.name);

/** The Block whose menu is open: its ID, where it was when the menu opened, and what it is. */
interface MenuTarget {
  id: unknown;
  pos: number;
  /** Null for Blocks that aren't Notes (Questions, Answers), which have no context flag. */
  included: boolean | null;
}

/**
 * The handle left of the Block under the mouse. Dragging it moves the Block;
 * clicking it opens the Block's menu.
 */
export function BlockHandle({ editor }: { editor: Editor }) {
  const t = useT();
  const hovered = useRef<{ node: ProseMirrorNode; pos: number } | null>(null);
  const grip = useRef<HTMLButtonElement>(null);
  const [menuFor, setMenuFor] = useState<MenuTarget | null>(null);

  // While the menu is open, the handle stays on its Block.
  const lockHandle = (locked: boolean) => {
    if (!editor.isDestroyed)
      editor.view.dispatch(editor.state.tr.setMeta("lockDragHandle", locked));
  };

  const openMenu = () => {
    const block = hovered.current;
    if (!block) return;
    lockHandle(true);
    setMenuFor({
      id: block.node.attrs[BLOCK_ID_ATTRIBUTE],
      pos: block.pos,
      included: isNote(block.node)
        ? block.node.attrs[INCLUDE_IN_CONTEXT_ATTRIBUTE] !== false
        : null,
    });
  };

  const closeMenu = (refocus: boolean) => {
    setMenuFor(null);
    lockHandle(false);
    if (refocus) editor.commands.focus();
  };

  /** Where the menu's Block is now: found by ID, which stays right even if the Mind changed meanwhile. */
  const targetPos = (target: MenuTarget) =>
    typeof target.id === "string" ? findBlock(editor.state.doc, target.id)?.pos : target.pos;

  const deleteBlock = () => {
    if (!menuFor) return;
    const pos = targetPos(menuFor);
    closeMenu(false);
    if (pos !== undefined) editor.chain().focus().deleteBlock(pos).run();
  };

  const toggleContext = () => {
    if (!menuFor || menuFor.included === null) return;
    const pos = targetPos(menuFor);
    const include = !menuFor.included;
    closeMenu(true);
    if (pos === undefined) return;
    editor
      .chain()
      .command(({ tr }) => {
        // On is the default, so it isn't stored.
        tr.setNodeAttribute(pos, INCLUDE_IN_CONTEXT_ATTRIBUTE, include ? null : false);
        return true;
      })
      .run();
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
          included={menuFor.included}
          onDelete={deleteBlock}
          onToggleContext={toggleContext}
          onClose={closeMenu}
        />
      )}
    </>
  );
}

function BlockMenu({
  anchor,
  included,
  onDelete,
  onToggleContext,
  onClose,
}: {
  anchor: Element;
  /** The Note's context flag, or null when the Block isn't a Note. */
  included: boolean | null;
  onDelete(): void;
  onToggleContext(): void;
  onClose(refocus: boolean): void;
}) {
  const t = useT();
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => first.current?.focus(), []);

  return (
    <Popover
      anchor={anchor}
      role="menu"
      aria-label={t("editor.block.menu")}
      data-testid="block-menu"
      onDismiss={() => onClose(false)}
      onKeyDown={(event) => {
        // Esc returns to writing.
        if (event.key !== "Escape") return;
        event.preventDefault();
        onClose(true);
      }}
      className="editor-menu w-[260px]"
    >
      {included !== null && (
        <button
          ref={first}
          type="button"
          role="menuitemcheckbox"
          aria-checked={included}
          data-testid="block-menu-context"
          onClick={onToggleContext}
          className="editor-menu-item"
        >
          <span className="editor-menu-icon">{included && <CheckIcon className="size-4" />}</span>
          <span className="flex-1">{t("question.context.include")}</span>
        </button>
      )}
      <button
        ref={included === null ? first : undefined}
        type="button"
        role="menuitem"
        data-testid="block-menu-delete"
        onClick={onDelete}
        className="editor-menu-item"
      >
        <span className="editor-menu-icon">
          <TrashIcon className="size-4" />
        </span>
        <span className="flex-1">{t("editor.block.delete")}</span>
        <kbd className="font-sans text-custom-xs text-gray-400">{DELETE_SHORTCUT_LABEL}</kbd>
      </button>
    </Popover>
  );
}
