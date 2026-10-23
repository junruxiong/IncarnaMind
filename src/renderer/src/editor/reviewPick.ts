import { type Editor, Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";

/** A part of what the reviewing pane's selected card is: a run of words, or a node (a Block, a comment's mark). */
export interface PickedRange {
  from: number;
  to: number;
  node?: true;
}

const reviewPickKey = new PluginKey<DecorationSet>("reviewPick");

/**
 * The change or comment the reviewing pane's selected card is (#697), washed
 * in the text as the change the cursor is in is, without moving the cursor:
 * the focus stays in the pane while ↑ and ↓ go through it.
 */
export const ReviewPick = Extension.create({
  name: "reviewPick",

  addProseMirrorPlugins() {
    return [
      new Plugin<DecorationSet>({
        key: reviewPickKey,
        state: {
          init: () => DecorationSet.empty,
          apply(tr, set) {
            const picked = tr.getMeta(reviewPickKey) as readonly PickedRange[] | undefined;
            if (!picked) return set.map(tr.mapping, tr.doc);
            const size = tr.doc.content.size;
            return DecorationSet.create(
              tr.doc,
              picked
                .filter((range) => range.from < range.to && range.to <= size)
                .map((range) =>
                  range.node
                    ? Decoration.node(range.from, range.to, { class: "review-picked" })
                    : Decoration.inline(range.from, range.to, { class: "review-picked" }),
                ),
            );
          },
        },
        props: { decorations: (state) => reviewPickKey.getState(state) ?? null },
      }),
    ];
  },
});

export function pickInMind(editor: Editor, picked: readonly PickedRange[]): void {
  if (editor.isDestroyed) return;
  editor.view.dispatch(editor.state.tr.setMeta(reviewPickKey, picked));
}

/**
 * Scrolls the Mind so a place shows, a third of the way down, unless it
 * already does: at once, as the cursor's own scrolling is.
 */
export function scrollMindTo(editor: Editor, pos: number): void {
  if (editor.isDestroyed) return;
  const scroller = editor.view.dom.closest<HTMLElement>(".mind-scroller");
  if (!scroller) return;
  const at = Math.min(Math.max(pos, 0), editor.state.doc.content.size);
  const coords = editor.view.coordsAtPos(at);
  const box = scroller.getBoundingClientRect();
  // The bar of changes stays at the Mind's top as it scrolls: a change under it doesn't show.
  const covered =
    editor.view.dom.closest(".mind-editor-frame")?.querySelector(".proposals-bar")?.clientHeight ??
    0;
  if (coords.top >= box.top + covered && coords.bottom <= box.bottom - 48) return;
  scroller.scrollTop += coords.top - box.top - box.height / 3;
}
