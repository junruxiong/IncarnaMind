/**
 * Whether a Numbered Mind shows its Bibliography (#102). A Mind in another citation style always
 * does (its References, #289); a Numbered one, the default, has none until the User asks for it
 * with "/" Bibliography. Like the citation style, it lives in the Mind's Yjs settings
 * (`MIND_SETTINGS_FIELD`), so it goes wherever the Mind goes and its content schema stays as it is.
 */
import type * as Y from "yjs";

const KEY = "bibliography";

export const bibliographyShown = (settings: Y.Map<unknown>): boolean =>
  settings.get(KEY) === "shown";

/** Shows the Bibliography in a Numbered Mind, or (false) takes it away again. */
export function setBibliographyShown(settings: Y.Map<unknown>, shown: boolean): void {
  if (shown) {
    if (settings.get(KEY) !== "shown") settings.set(KEY, "shown");
  } else if (settings.has(KEY)) {
    settings.delete(KEY);
  }
}
