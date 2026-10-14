import type { Editor } from "@tiptap/core";
import { MIND_SETTINGS_FIELD } from "../../../core/api";
import { setBibliographyShown } from "../../../shared/mindBibliography";
import { editorDoc } from "./proposalReview";

/** The Mind's own settings, in the Yjs document the editor is bound to; null before it is bound. */
const settingsOf = (editor: Editor) => editorDoc(editor)?.getMap<unknown>(MIND_SETTINGS_FIELD);

/**
 * "/" Bibliography: shows the Mind's Bibliography at its end (a Numbered Mind has none until
 * asked; one in a citation style always has it) and brings it into view. The Mind's own undo
 * doesn't cover settings, so the block's Hide takes it back.
 */
export function showBibliography(editor: Editor): void {
  const settings = settingsOf(editor);
  if (settings) setBibliographyShown(settings, true);
  // After the block is drawn.
  requestAnimationFrame(() =>
    editor.view.dom.ownerDocument
      .querySelector('[data-testid="references"]')
      ?.scrollIntoView({ block: "nearest" }),
  );
}

/** Takes the Bibliography away from a Numbered Mind again. */
export function hideBibliography(editor: Editor): void {
  const settings = settingsOf(editor);
  if (settings) setBibliographyShown(settings, false);
}
