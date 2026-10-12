/**
 * A note's copy, with the Citations in it as footnotes (see `copyNotes`): HTML Word pastes as
 * footnotes, and text with markers and Sources. Added to every editor that shows Notes.
 */
import { Extension } from "@tiptap/core";
import { DOMSerializer } from "@tiptap/pm/model";
import { Plugin } from "@tiptap/pm/state";
import type { SourceContext } from "../../../shared/sources";
import { languageNow } from "../i18n";
import { useAppStore } from "../store";
import { addFootnotes, citationsIn, textWithSources } from "./copyNotes";

function sourceContext(): SourceContext {
  return { language: languageNow(), documents: useAppStore.getState().documents };
}

export const CopyCitations = Extension.create({
  name: "copyCitations",
  // Before the editor's own plain-text copy, which writes a Citation as its reference.
  priority: 1000,

  addProseMirrorPlugins() {
    return [
      new Plugin({
        props: {
          clipboardSerializer: (() => {
            const base = DOMSerializer.fromSchema(this.editor.schema);
            const serializer = new DOMSerializer(base.nodes, base.marks);
            serializer.serializeFragment = (fragment, options, target) => {
              const out = base.serializeFragment(fragment, options, target);
              addFootnotes(out, citationsIn(fragment), sourceContext());
              return out;
            };
            return serializer;
          })(),
          clipboardTextSerializer: (slice, view) =>
            textWithSources(slice.content, view.state.schema, sourceContext()),
        },
      }),
    ];
  },
});
