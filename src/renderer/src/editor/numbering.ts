import { type Editor, Extension, wrappingInputRule } from "@tiptap/core";
import { Fragment, type Node as ProseMirrorNode, type ResolvedPos } from "@tiptap/pm/model";
import { Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { ySyncPluginKey } from "@tiptap/y-tiptap";
import {
  DEEPEST_LEVEL,
  isNumberingScheme,
  itemLevel,
  type NumberingScheme,
  numberingMarker,
} from "../../../shared/numbering";
import { asOneStep, undoStackOf } from "./editorUndo";

/**
 * A numbered list's scheme (#1419; DESIGN.md › Lists › Numbering styles), the level its outermost
 * list starts at, and whether it goes on from the list before. All three are optional, so a Mind
 * saved before reads as a "1. a. i." list from level 0, starting again.
 */
export const NUMBERING_ATTRIBUTE = "numbering";
export const START_LEVEL_ATTRIBUTE = "startLevel";
export const CONTINUES_ATTRIBUTE = "continues";

/** The gutter a level's text starts after, and the step it moves right by for a wide marker: in px. */
const LEVEL_INDENT = 48;
const WIDE_STEP = 12;
const MARKER_GAP = 8;

export const schemeOf = (list: ProseMirrorNode): NumberingScheme => {
  const scheme = list.attrs[NUMBERING_ATTRIBUTE];
  return isNumberingScheme(scheme) ? scheme : "default";
};
export const startLevelOf = (list: ProseMirrorNode): number => {
  const level = Number(list.attrs[START_LEVEL_ATTRIBUTE]);
  return Number.isInteger(level) && level > 0 ? level : 0;
};

const KEY = new PluginKey<DecorationSet>("numbering");

export interface NumberingStorage {
  /**
   * The marker typed just now (`text`) while it becomes a list ("marker"), then while that list
   * is the last change ("list").
   */
  typed: { step: "marker" | "list"; text: string } | null;
  /** Opens a list item's right-click menu at the pointer (see `ListMenu`, which sets it). */
  openMenu: ((at: { x: number; y: number }) => void) | null;
}

declare module "@tiptap/core" {
  interface Storage {
    numbering: NumberingStorage;
  }
}

/** Opens the Numbering menu under the paragraph the caret is in, from ⌘K. */
export const NUMBERING_OPEN = "numbering-open";

export const requestNumbering = (editor: Editor): boolean =>
  editor.view.dom.dispatchEvent(new CustomEvent(NUMBERING_OPEN));

let measure: ((text: string) => number) | null = null;
const markerWidth = (text: string) => measure?.(text) ?? 0;

const isList = (node: ProseMirrorNode) =>
  node.type.name === "orderedList" || node.type.name === "bulletList";

/**
 * Each numbered item's marker, on its first paragraph, and the room its text needs when the marker
 * is wider than its gutter. A list that goes on takes up the numbers of the numbered list before it.
 */
function markers(doc: ProseMirrorNode): DecorationSet {
  return DecorationSet.create(doc, markerDecorations(doc));
}

/** Each numbered item's marker, by the position of its first paragraph, in order. */
export function drawnMarkers(doc: ProseMirrorNode): { pos: number; marker: string }[] {
  return markerDecorations(doc).flatMap((decoration) => {
    const marker: unknown = decoration.spec.marker;
    return typeof marker === "string" ? [{ pos: decoration.from, marker }] : [];
  });
}

function markerDecorations(doc: ProseMirrorNode): Decoration[] {
  const decorations: Decoration[] = [];
  let before: number[] | null = null;
  const draw = (
    list: ProseMirrorNode,
    at: number,
    depth: number,
    outer: { scheme: NumberingScheme; startLevel: number },
    numbers: number[],
  ) => {
    const level = itemLevel(depth, outer.startLevel);
    const ordered = list.type.name === "orderedList";
    if (ordered) {
      const start = Number(list.attrs.start);
      numbers[level] = (Number.isInteger(start) && start > 0 ? start : 1) - 1;
      numbers.length = level + 1;
      for (let parent = 0; parent < level; parent++) numbers[parent] ||= 1;
    }
    list.forEach((item, offset) => {
      const itemAt = at + 1 + offset;
      if (ordered) {
        numbers[level] = (numbers[level] ?? 0) + 1;
        numbers.length = level + 1;
        const first = item.firstChild;
        if (first?.isTextblock) {
          const marker = numberingMarker(outer.scheme, level, numbers);
          const gutter = LEVEL_INDENT * (level + 1);
          const over = markerWidth(marker) + MARKER_GAP - gutter;
          const extra = over > 0 ? WIDE_STEP * Math.ceil(over / WIDE_STEP) : 0;
          decorations.push(
            Decoration.node(
              itemAt + 1,
              itemAt + 1 + first.nodeSize,
              { "data-marker": marker },
              { marker },
            ),
          );
          if (extra > 0) {
            decorations.push(
              Decoration.node(itemAt, itemAt + item.nodeSize, {
                style: `--marker-room: ${extra}px`,
              }),
            );
          }
        }
      }
      item.forEach((child, childOffset) => {
        if (isList(child)) draw(child, itemAt + 1 + childOffset, depth + 1, outer, [...numbers]);
      });
    });
  };
  doc.descendants((node, pos, parent) => {
    if (!isList(node)) return true;
    if (parent?.type.name === "listItem" || parent?.type.name === "taskItem") return false;
    if (node.type.name !== "orderedList") return true;
    const continues = node.attrs[CONTINUES_ATTRIBUTE] === true && before !== null;
    const numbers: number[] = continues && before ? [...before] : [];
    const outer = { scheme: schemeOf(node), startLevel: startLevelOf(node) };
    if (continues && before) {
      // It goes on from the list before: its first item takes the next number.
      const level = itemLevel(0, outer.startLevel);
      const shifted = { ...node.attrs, start: (before[level] ?? 0) + 1 };
      draw(node.type.create(shifted, node.content), pos, 0, outer, numbers);
    } else {
      draw(node, pos, 0, outer, numbers);
    }
    before = numbers.length > 0 ? numbers : before;
    return false;
  });
  return decorations;
}

/**
 * The first markers that begin a list when typed at the start of an empty line, as Word's
 * AutoFormat reads them, and the scheme and level each draws in. Chinese ones end in their own
 * punctuation, so they need no space. Any "N. " begins a default list at N, as before.
 */
const FIRST_MARKERS: Readonly<Record<string, { scheme: NumberingScheme; level: number }>> = {
  "a. ": { scheme: "default", level: 1 },
  "i. ": { scheme: "default", level: 2 },
  "1.1 ": { scheme: "outline", level: 1 },
  "1.1. ": { scheme: "outline", level: 1 },
  "(a) ": { scheme: "clauses", level: 0 },
  "(i) ": { scheme: "clauses", level: 1 },
  "(A) ": { scheme: "clauses", level: 2 },
  "一、": { scheme: "chinese", level: 0 },
  "（一）": { scheme: "chinese", level: 1 },
  "（1）": { scheme: "chinese", level: 3 },
};
const TYPED_PATTERN = /^(?:\d{1,4}\. |1\.1\.? |[ai]\. |\([aiA]\) |一、|（一）|（1）)$/;

function typedMarker(text: string): { scheme: NumberingScheme; level: number; start: number } {
  const first = FIRST_MARKERS[text];
  return first
    ? { ...first, start: 1 }
    : { scheme: "default", level: 0, start: Number.parseInt(text, 10) };
}

export const Numbering = Extension.create<object, NumberingStorage>({
  name: "numbering",
  // Ahead of the ordered list's own "1. ", which knows no scheme.
  priority: 1000,

  addGlobalAttributes() {
    return [
      {
        types: ["orderedList"],
        attributes: {
          [NUMBERING_ATTRIBUTE]: {
            default: null,
            parseHTML: (element: HTMLElement) => {
              const value = element.getAttribute("data-numbering");
              return isNumberingScheme(value) ? value : null;
            },
            renderHTML: (attributes: Record<string, unknown>) =>
              isNumberingScheme(attributes[NUMBERING_ATTRIBUTE])
                ? { "data-numbering": attributes[NUMBERING_ATTRIBUTE] }
                : {},
          },
          [START_LEVEL_ATTRIBUTE]: {
            default: 0,
            parseHTML: (element: HTMLElement) =>
              Number(element.getAttribute("data-start-level")) || 0,
            renderHTML: (attributes: Record<string, unknown>) => {
              const level = Number(attributes[START_LEVEL_ATTRIBUTE]) || 0;
              return level > 0
                ? { "data-start-level": String(level), style: `--start-level: ${level}` }
                : {};
            },
          },
          [CONTINUES_ATTRIBUTE]: {
            default: false,
            parseHTML: (element: HTMLElement) => element.hasAttribute("data-continues"),
            renderHTML: (attributes: Record<string, unknown>) =>
              attributes[CONTINUES_ATTRIBUTE] === true ? { "data-continues": "" } : {},
          },
        },
      },
    ];
  },

  addStorage() {
    return { typed: null, openMenu: null };
  },

  addKeyboardShortcuts() {
    // Enter on an empty item, or Backspace at its start, steps out one level: in a list that
    // starts deeper, there is a level to step out to before the list ends.
    const stepOut = () =>
      emptyItemAt(this.editor) && asOneStep(this.editor, () => raiseFromStartLevel(this.editor));
    // Before the list item's own Tab and Shift-Tab, which can't nest a first item or lift an item
    // of a list that starts deeper, as ⌘] and ⌘[ (`changeIndent`) do too.
    return {
      Enter: stepOut,
      Backspace: stepOut,
      Tab: () => asOneStep(this.editor, () => deepenList(this.editor)),
      "Shift-Tab": () => asOneStep(this.editor, () => raiseFromStartLevel(this.editor)),
    };
  },

  addInputRules() {
    const editor = this.editor;
    const storage = this.storage;
    const type = editor.schema.nodes.orderedList;
    if (!type) return [];
    return [
      wrappingInputRule({
        find: TYPED_PATTERN,
        type,
        getAttributes: (match) => {
          const typed = typedMarker(match[0]);
          storage.typed = { step: "marker", text: match[0] };
          // In a list, a nested list takes its level and its scheme from the list it is in.
          const { $from } = editor.state.selection;
          for (let depth = $from.depth; depth > 0; depth--) {
            if ($from.node(depth).type.name === "listItem") return { start: typed.start };
          }
          return {
            start: typed.start,
            [NUMBERING_ATTRIBUTE]: typed.scheme === "default" ? null : typed.scheme,
            [START_LEVEL_ATTRIBUTE]: typed.level,
          };
        },
        joinPredicate: (match, node) => {
          // A typed first marker means that number, so under another list it begins a new one.
          const typed = typedMarker(match[0]);
          return (
            schemeOf(node) === typed.scheme &&
            startLevelOf(node) === typed.level &&
            node.childCount + (Number(node.attrs.start) || 1) === typed.start
          );
        },
      }),
    ];
  },

  addProseMirrorPlugins() {
    const editor = this.editor;
    const storage = this.storage;
    return [
      new Plugin<DecorationSet>({
        key: KEY,
        // Tiptap dispatches the typed text, then the list. Each change before it reaches the Mind
        // starts a step of its own: the list, and the typing after it. So ⌘Z straight after puts
        // the typed marker back, and takes back no more.
        appendTransaction: (transactions, _before, state) => {
          const typed = storage.typed;
          if (!typed || !transactions.some((tr) => tr.docChanged)) return null;
          if (typed.step === "marker") {
            if (!transactions.some((tr) => tr.getMeta("inputRule"))) return null;
            undoStackOf(editor)?.stopCapturing();
            storage.typed = { ...typed, step: "list" };
            return null;
          }
          storage.typed = null;
          undoStackOf(editor)?.stopCapturing();
          // ⌘Z straight after puts the marker back as text, with the caret after it, where it was.
          const undone = transactions.some(
            (tr) => tr.getMeta(ySyncPluginKey)?.isUndoRedoOperation === true,
          );
          const { selection } = state;
          const { $from } = selection;
          if (!undone || !selection.empty || !$from.parent.textContent.startsWith(typed.text)) {
            return null;
          }
          return state.tr.setSelection(
            TextSelection.create(state.doc, $from.start() + typed.text.length),
          );
        },
        state: {
          init: (_, state) => markers(state.doc),
          apply: (tr, old) => (tr.docChanged || tr.getMeta(KEY) ? markers(tr.doc) : old),
        },
        props: {
          decorations: (state) => KEY.getState(state),
          handleDOMEvents: {
            contextmenu: (view, event) => {
              const open = storage.openMenu;
              if (!open || !view.editable) return false;
              const at = view.posAtCoords({ left: event.clientX, top: event.clientY });
              const $at = at && view.state.doc.resolve(at.pos);
              const inItem = $at && listItemDepth($at) > 0;
              if (!$at || !inItem) return false;
              event.preventDefault();
              const { from, to } = view.state.selection;
              // A right-click on the selection keeps it, for Cut and Copy; elsewhere it moves the caret.
              if ($at.pos < from || $at.pos > to) {
                view.dispatch(view.state.tr.setSelection(TextSelection.near($at)));
              }
              view.focus();
              open({ x: event.clientX, y: event.clientY });
              return true;
            },
          },
        },
        view: (view) => {
          const canvas = document.createElement("canvas").getContext("2d");
          let font = "";
          measure = (text) => {
            if (!canvas) return 0;
            if (canvas.font !== font) canvas.font = font;
            return canvas.measureText(text).width;
          };
          // Drawn again once the page's font is known, as markers are measured in it.
          if (typeof requestAnimationFrame !== "function") return {};
          const frame = requestAnimationFrame(() => {
            const style = getComputedStyle(view.dom);
            font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            if (!view.isDestroyed) view.dispatch(view.state.tr.setMeta(KEY, true));
          });
          return { destroy: () => cancelAnimationFrame(frame) };
        },
      }),
    ];
  },
});

/** The outermost numbered list the selection is in, with where it starts; null outside one. */
export function numberedListAt(editor: Editor): { node: ProseMirrorNode; pos: number } | null {
  const { $from } = editor.state.selection;
  let found: { node: ProseMirrorNode; pos: number } | null = null;
  for (let depth = $from.depth; depth > 0; depth--) {
    const node = $from.node(depth);
    if (node.type.name === "orderedList") found = { node, pos: $from.before(depth) };
    else if (node.type.name === "bulletList" && found) break;
  }
  return found;
}

const setListAttributes = (editor: Editor, attributes: Record<string, unknown>): boolean => {
  const list = numberedListAt(editor);
  if (!list) return false;
  const tr = editor.state.tr.setNodeMarkup(list.pos, undefined, {
    ...list.node.attrs,
    ...attributes,
  });
  editor.view.dispatch(tr);
  return true;
};

/**
 * Numbers the list the caret is in, or the selected paragraphs, in `scheme`: the whole list takes
 * it, nested levels too. One Undo step.
 */
export function setNumbering(editor: Editor, scheme: NumberingScheme): boolean {
  return asOneStep(editor, () => {
    if (!numberedListAt(editor)) editor.chain().focus().toggleOrderedList().run();
    return setListAttributes(editor, { [NUMBERING_ATTRIBUTE]: scheme });
  });
}

export const restartNumbering = (editor: Editor): boolean =>
  asOneStep(editor, () => setListAttributes(editor, { start: 1, [CONTINUES_ATTRIBUTE]: false }));

/** Whether the list the caret is in can go on from a numbered list before it, and doesn't yet. */
export function canContinueNumbering(editor: Editor): boolean {
  const list = numberedListAt(editor);
  if (!list || list.node.attrs[CONTINUES_ATTRIBUTE] === true) return false;
  let earlier = false;
  editor.state.doc.nodesBetween(0, list.pos, (node, pos, parent) => {
    if (earlier || pos >= list.pos) return false;
    if (node.type.name === "orderedList" && parent?.type.name !== "listItem") earlier = true;
    return !earlier;
  });
  return earlier;
}

export const continueNumbering = (editor: Editor): boolean =>
  canContinueNumbering(editor) &&
  asOneStep(editor, () => setListAttributes(editor, { [CONTINUES_ATTRIBUTE]: true }));

/**
 * What the Numbering menus show at the caret: its list's scheme, null outside a numbered list, and
 * whether Restart at 1 and Continue numbering would change anything.
 */
export function numberingState(editor: Editor): {
  scheme: NumberingScheme | null;
  canRestart: boolean;
  canContinue: boolean;
} {
  const list = numberedListAt(editor);
  return {
    scheme: list ? schemeOf(list.node) : null,
    canRestart:
      list !== null &&
      (Number(list.node.attrs.start) !== 1 || list.node.attrs[CONTINUES_ATTRIBUTE] === true),
    canContinue: canContinueNumbering(editor),
  };
}

/** How deep the innermost bulleted or numbered list item around `$at` is; 0 outside one. */
function listItemDepth($at: ResolvedPos): number {
  for (let depth = $at.depth; depth > 1; depth--) {
    if ($at.node(depth).type.name === "listItem" && isList($at.node(depth - 1))) return depth;
  }
  return 0;
}

function emptyItemAt(editor: Editor): boolean {
  const { selection } = editor.state;
  const { $from } = selection;
  return (
    selection.empty &&
    $from.parent.content.size === 0 &&
    $from.depth > 1 &&
    $from.node(-1).type.name === "listItem" &&
    $from.node(-1).childCount === 1
  );
}

/** The outermost numbered list the caret is in, when the caret is in one of its own items, not a nested list's. */
function outerItemAt(editor: Editor) {
  const { $from } = editor.state.selection;
  for (let depth = $from.depth; depth > 1; depth--) {
    if ($from.node(depth).type.name !== "listItem") continue;
    const list = $from.node(depth - 1);
    const outer =
      list.type.name === "orderedList" && $from.node(depth - 2).type.name !== "listItem";
    if (!outer) return null;
    return { list, listPos: $from.before(depth - 1), index: $from.index(depth - 1) };
  }
  return null;
}

/**
 * Tab where an item can't nest (the first item of a list): only that item goes a level deeper, as
 * in Word, in a one-item list of its own that starts a level deeper. The items after it stay a
 * list where they were. False when the item nests as usual.
 */
export function deepenList(editor: Editor): boolean {
  const at = outerItemAt(editor);
  if (!at || at.index > 0) return false;
  const { list, listPos } = at;
  const level = startLevelOf(list);
  if (level >= DEEPEST_LEVEL) return true;
  const deeper = { ...list.attrs, [START_LEVEL_ATTRIBUTE]: level + 1 };
  const { from } = editor.state.selection;
  const first = list.firstChild;
  if (!first) return false;
  const rest = list.content.cut(first.nodeSize);
  const tr =
    rest.size === 0
      ? editor.state.tr.setNodeMarkup(listPos, undefined, deeper)
      : editor.state.tr.replaceWith(listPos, listPos + list.nodeSize, [
          list.type.create({ ...deeper, start: 1, [CONTINUES_ATTRIBUTE]: false }, first),
          list.type.create(list.attrs, rest),
        ]);
  tr.setSelection(TextSelection.create(tr.doc, from));
  editor.view.dispatch(tr);
  return true;
}

/**
 * Shift-Tab on an item of a list that starts deeper than the first level: the item comes up one
 * level, as lifting an item out of a nested list does. The items before it stay where they were,
 * and those after it nest under it, so each keeps its level. False at the first level, where the
 * item leaves the list as usual. Lists at different levels are never joined.
 */
export function raiseFromStartLevel(editor: Editor): boolean {
  const at = outerItemAt(editor);
  if (!at) return false;
  const level = startLevelOf(at.list);
  if (level === 0) return false;
  const { list, listPos, index } = at;
  const items: ProseMirrorNode[] = [];
  list.forEach((item) => {
    items.push(item);
  });
  const before = items.slice(0, index);
  const item = items[index];
  const after = items.slice(index + 1);
  if (!item) return false;
  const nested = after.length
    ? [
        list.type.create(
          { ...list.attrs, start: 1, [START_LEVEL_ATTRIBUTE]: 0, [CONTINUES_ATTRIBUTE]: false },
          after,
        ),
      ]
    : [];
  const raised = item.type.create(item.attrs, item.content.append(Fragment.from(nested)));
  const lists = [
    ...(before.length ? [list.type.create(list.attrs, before)] : []),
    list.type.create(
      {
        ...list.attrs,
        start: 1,
        [START_LEVEL_ATTRIBUTE]: level - 1,
        [CONTINUES_ATTRIBUTE]: false,
      },
      [raised],
    ),
  ];
  // Up beside the list after it, at its level and in its scheme, it goes on in that list: a first
  // item taken a level deeper (`deepenList`) and brought back is one list again.
  const next = editor.state.doc.resolve(listPos + list.nodeSize).nodeAfter;
  const joins =
    next?.type === list.type &&
    startLevelOf(next) === level - 1 &&
    schemeOf(next) === schemeOf(list) &&
    next.attrs[CONTINUES_ATTRIBUTE] !== true;
  const last = lists.at(-1);
  if (joins && next && last) {
    lists[lists.length - 1] = list.type.create(last.attrs, last.content.append(next.content));
  }
  const offsetInItem =
    editor.state.selection.from -
    (listPos + 1 + items.slice(0, index).reduce((sum, each) => sum + each.nodeSize, 0));
  const tr = editor.state.tr.replaceWith(
    listPos,
    listPos + list.nodeSize + (joins && next ? next.nodeSize : 0),
    lists,
  );
  const raisedAt = listPos + (before.length ? (lists[0]?.nodeSize ?? 0) : 0) + 1;
  tr.setSelection(
    TextSelection.near(tr.doc.resolve(Math.min(raisedAt + offsetInItem, tr.doc.content.size))),
  );
  editor.view.dispatch(tr.scrollIntoView());
  return true;
}
