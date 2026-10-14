/**
 * "Writing…" where the assistant writes next while its changes come in
 * (#202): over the paragraph it is writing anew, which is dimmed until its
 * change is in, or after the Block it wrote last, or at the Mind's end. Only
 * a mark on the screen: nothing is in the Mind until a Block is written.
 */

import type { Editor } from "@tiptap/core";
import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { BLOCK_ID_ATTRIBUTE, type WritingPlace as Place } from "../../../core/api";

const writingPlaceKey = new PluginKey<Place | null>("writingPlace");

/** Where the Block with this id is, at any depth; null when it isn't in the Mind. */
function blockAt(doc: ProseMirrorNode, id: string): { pos: number; node: ProseMirrorNode } | null {
  let found: { pos: number; node: ProseMirrorNode } | null = null;
  doc.descendants((node, pos) => {
    if (found) return false;
    if (node.attrs[BLOCK_ID_ATTRIBUTE] === id) found = { pos, node };
    return node.isBlock && !found;
  });
  return found;
}

function decorations(doc: ProseMirrorNode, place: Place, label: string): Decoration[] {
  const widget = () => {
    const line = document.createElement("div");
    line.contentEditable = "false";
    line.dataset.testid = "mind-writing";
    line.className = "mind-writing";
    const text = document.createElement("span");
    text.className = "answer-writing";
    text.textContent = label;
    line.append(text);
    return line;
  };
  if ("over" in place) {
    const block = blockAt(doc, place.over);
    if (!block) return [];
    return [
      Decoration.widget(block.pos, widget, { side: -1, key: `writing:over:${place.over}` }),
      Decoration.node(block.pos, block.pos + block.node.nodeSize, {
        class: "mind-rewriting",
        "data-testid": "mind-rewriting",
      }),
    ];
  }
  const block = place.after === null ? null : blockAt(doc, place.after);
  const at = block ? block.pos + block.node.nodeSize : doc.content.size;
  return [Decoration.widget(at, widget, { side: 1, key: `writing:${place.after ?? "end"}` })];
}

export const WritingPlace = Extension.create<{ label(): string }>({
  name: "writingPlace",

  addOptions() {
    return { label: () => "" };
  },

  addProseMirrorPlugins() {
    const { label } = this.options;
    return [
      new Plugin<Place | null>({
        key: writingPlaceKey,
        state: {
          init: () => null,
          apply: (tr, place) => {
            const meta = tr.getMeta(writingPlaceKey) as Place | null | undefined;
            return meta === undefined ? place : meta;
          },
        },
        props: {
          decorations(state) {
            const place = writingPlaceKey.getState(state);
            if (!place) return null;
            return DecorationSet.create(state.doc, decorations(state.doc, place, label()));
          },
        },
      }),
    ];
  },
});

/** Shows "Writing…" where the assistant writes next, or takes it away (null). */
export function setWritingPlace(editor: Editor, place: Place | null): void {
  if (editor.isDestroyed) return;
  editor.view.dispatch(editor.state.tr.setMeta(writingPlaceKey, place));
}
