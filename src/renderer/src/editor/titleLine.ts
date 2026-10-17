import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { isTitleLine } from "./EndHint";

/**
 * Text pasted into a new Mind keeps its formatting (#1088): pasted over the
 * empty title its first line starts with, anything but one line of text
 * makes that line body text first, so the first pasted paragraph isn't made
 * the title. One line of text pasted there is the title.
 */
export const TitleLine = Extension.create({
  name: "titleLine",

  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: new PluginKey("titleLine"),
        props: {
          handlePaste(view, _event, slice) {
            const { doc, schema } = view.state;
            const paragraph = schema.nodes.paragraph;
            const pasted = slice.content;
            const oneLine = pasted.childCount === 1 && pasted.firstChild?.isTextblock === true;
            if (!paragraph || doc.childCount !== 1 || !isTitleLine(doc.firstChild) || oneLine) {
              return false;
            }
            view.dispatch(view.state.tr.setNodeMarkup(0, paragraph));
            // The paste goes on as it would into an empty Mind.
            return false;
          },
        },
      }),
    ];
  },
});
