/**
 * "Writing…" where the assistant's next Block goes, while it writes new
 * Blocks into the Mind as they come (#202): after the Block it wrote last,
 * or at the Mind's end. Only a mark on the screen: nothing is in the Mind
 * until a Block is written.
 */

import type { Editor } from "@tiptap/core";
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { BLOCK_ID_ATTRIBUTE } from "../../../core/api";

/** Where the next Block goes: after the top-level Block with this id, or at the end (null). */
type Place = { after: string | null } | null;

const writingPlaceKey = new PluginKey<Place>("writingPlace");

/** The position just after the top-level Block `after` names; the end when it names none there. */
function positionAfter(doc: Editor["state"]["doc"], after: string | null): number {
  let at = doc.content.size;
  if (after === null) return at;
  doc.forEach((node, offset) => {
    if (node.attrs[BLOCK_ID_ATTRIBUTE] === after) at = offset + node.nodeSize;
  });
  return at;
}

export const WritingPlace = Extension.create<{ label(): string }>({
  name: "writingPlace",

  addOptions() {
    return { label: () => "" };
  },

  addProseMirrorPlugins() {
    const { label } = this.options;
    return [
      new Plugin<Place>({
        key: writingPlaceKey,
        state: {
          init: () => null,
          apply: (tr, place) => {
            const meta = tr.getMeta(writingPlaceKey) as Place | undefined;
            return meta === undefined ? place : meta;
          },
        },
        props: {
          decorations(state) {
            const place = writingPlaceKey.getState(state);
            if (!place) return null;
            const widget = () => {
              const line = document.createElement("div");
              line.contentEditable = "false";
              line.dataset.testid = "mind-writing";
              line.className = "mind-writing";
              const text = document.createElement("span");
              text.className = "answer-writing";
              text.textContent = label();
              line.append(text);
              return line;
            };
            return DecorationSet.create(state.doc, [
              Decoration.widget(positionAfter(state.doc, place.after), widget, {
                side: 1,
                key: `writing:${place.after ?? "end"}`,
              }),
            ]);
          },
        },
      }),
    ];
  },
});

/** Shows "Writing…" where the next Block goes, or takes it away (null). */
export function setWritingPlace(editor: Editor, place: Place): void {
  if (editor.isDestroyed) return;
  editor.view.dispatch(editor.state.tr.setMeta(writingPlaceKey, place));
}
