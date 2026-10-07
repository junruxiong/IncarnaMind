import type { ComputePositionConfig, VirtualElement } from "@floating-ui/dom";
import { isMacOS } from "@tiptap/core";
import { DragHandle } from "@tiptap/extension-drag-handle-react";
import { NodeRangeSelection } from "@tiptap/extension-node-range";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { Editor } from "@tiptap/react";
import { useEffect, useRef, useState } from "react";
import { INCLUDE_IN_CONTEXT_ATTRIBUTE, NOTE_BLOCK_TYPES, QUESTION_BLOCK } from "../../../core/api";
import { CheckIcon, GripIcon, TrashIcon } from "../components/icons";
import { useT } from "../i18n";
import { findBlock, questionWithAnswer } from "./blockCommands";
import { BLOCK_ID_ATTRIBUTE } from "./noteSchema";
import { Popover } from "./Popover";

/** Centred on the point `handleAnchor` gives, to its left. */
const HANDLE_POSITION: ComputePositionConfig = { placement: "left", middleware: [] };

/** The position just inside a node's first textblock, if it has one. */
function firstTextPos(node: ProseMirrorNode, pos: number): number | null {
  if (node.isTextblock) return pos + 1;
  let found: number | null = null;
  node.descendants((child, offset) => {
    if (found !== null) return false;
    if (child.isTextblock) found = pos + 1 + offset + 1;
    return found === null;
  });
  return found;
}

/**
 * Where the handle goes: in the left margin, a fixed gap left of the Mind's
 * text edge (the CSS sets `--handle-gap`, wider for a Question, whose band
 * and Ask button take the margin), level with the Block's first line of text.
 * An Answer's first line is its first paragraph, under the "Answer" label.
 */
function handleAnchor(
  editor: Editor,
  block: { node: ProseMirrorNode; pos: number } | null,
): VirtualElement | null {
  if (!block || editor.isDestroyed) return null;
  const { view } = editor;
  return {
    contextElement: view.dom,
    getBoundingClientRect: () => {
      const style = getComputedStyle(view.dom);
      const gapName =
        block.node.type.name === QUESTION_BLOCK ? "--handle-gap-question" : "--handle-gap";
      const gap = Number.parseFloat(style.getPropertyValue(gapName)) || 0;
      // The text edge: the editor's box reaches over the margin (styles.css, `--editor-gutter`).
      const edge = view.dom.getBoundingClientRect().left + Number.parseFloat(style.paddingLeft);
      const x = edge - gap;
      let y: number;
      const textPos = firstTextPos(block.node, block.pos);
      if (textPos !== null) {
        const line = view.coordsAtPos(textPos);
        y = (line.top + line.bottom) / 2;
      } else {
        // A formula or a rule: the middle of its first line.
        const dom = view.nodeDOM(block.pos);
        const rect = dom instanceof Element ? dom.getBoundingClientRect() : new DOMRect();
        y = rect.top + Math.min(rect.height, Number.parseFloat(style.lineHeight) || 28) / 2;
      }
      return new DOMRect(x, y, 0, 0);
    },
  };
}

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

  // A Question and the Answer right after it are dragged as one. This runs
  // after the handle's own dragstart (it listens on the document), so it
  // replaces what Tiptap set up: the selection a move deletes, what is dropped,
  // and the picture under the pointer.
  useEffect(() => {
    const dragWithAnswer = (event: DragEvent) => {
      const block = hovered.current;
      if (!(event.target instanceof Element) || !event.target.closest(".block-handle")) return;
      if (!block || editor.isDestroyed) return;
      const pair = questionWithAnswer(editor.state.doc, block.pos);
      if (!pair) return;
      const selection = NodeRangeSelection.create(editor.state.doc, pair.from, pair.to);
      editor.view.dispatch(editor.state.tr.setSelection(selection));
      editor.view.dragging = { slice: selection.content(), move: true };

      const picture = document.createElement("div");
      // Drawn as in the editor, with room on the left for the Question band.
      picture.className = "mind-editor";
      Object.assign(picture.style, {
        position: "absolute",
        top: "-10000px",
        marginLeft: "0",
        paddingLeft: "56px",
        width: `${editor.view.dom.clientWidth}px`,
      });
      for (const pos of [
        pair.from,
        pair.from + (editor.state.doc.nodeAt(pair.from)?.nodeSize ?? 0),
      ]) {
        const dom = editor.view.nodeDOM(pos);
        if (dom instanceof Element) picture.append(dom.cloneNode(true));
      }
      document.body.append(picture);
      event.dataTransfer?.setDragImage(picture, 0, 0);
      // The browser takes its picture once this event is over.
      setTimeout(() => picture.remove(), 0);
    };
    document.addEventListener("dragstart", dragWithAnswer);
    return () => document.removeEventListener("dragstart", dragWithAnswer);
  }, [editor]);

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
        getReferencedVirtualElement={() => handleAnchor(editor, hovered.current)}
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
          <GripIcon className="size-4" />
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

  return (
    <Popover
      anchor={anchor}
      initialFocus={first}
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
      className="editor-menu w-[240px]"
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
        <kbd className="editor-menu-shortcut">{DELETE_SHORTCUT_LABEL}</kbd>
      </button>
    </Popover>
  );
}
