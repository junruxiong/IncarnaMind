/**
 * The citation style a Mind is written in (#289). Like its model (`mindModel.ts`), it lives in
 * the Mind's Yjs settings (`MIND_SETTINGS_FIELD`), not in its Blocks, so it goes wherever the
 * Mind goes and its content schema stays as it is. A Mind with no choice is Numbered
 * (`DEFAULT_CITATION_STYLE`).
 */
import type * as Y from "yjs";
import { type CitationStyleId, DEFAULT_CITATION_STYLE, isCitationStyleId } from "./citationStyles";

const STYLE_KEY = "citationStyle";
/** The User's "not for this Mind" to the prompt that offers a style (`CitationStylePrompt`). */
const PROMPT_KEY = "citationStylePrompt";

/** The style chosen for the Mind, or null when it follows the default. Anything malformed counts as none. */
export function chosenCitationStyle(settings: Y.Map<unknown>): CitationStyleId | null {
  const value = settings.get(STYLE_KEY);
  return isCitationStyleId(value) ? value : null;
}

/** The style the Mind's Citations and Bibliography are in. */
export function citationStyleIn(settings: Y.Map<unknown>): CitationStyleId {
  return chosenCitationStyle(settings) ?? DEFAULT_CITATION_STYLE;
}

/** Chooses a style for the Mind; null forgets the choice, so it follows the default again. */
export function setCitationStyle(settings: Y.Map<unknown>, style: CitationStyleId | null): void {
  if (style === null) {
    if (settings.has(STYLE_KEY)) settings.delete(STYLE_KEY);
  } else if (settings.get(STYLE_KEY) !== style) {
    settings.set(STYLE_KEY, style);
  }
}

/** Whether the User closed the prompt that offers a style in this Mind. */
export const citationStylePromptDismissed = (settings: Y.Map<unknown>): boolean =>
  settings.get(PROMPT_KEY) === "dismissed";

/** Closes the prompt for this Mind, or (false) shows it again. */
export function dismissCitationStylePrompt(settings: Y.Map<unknown>, dismissed = true): void {
  if (dismissed) settings.set(PROMPT_KEY, "dismissed");
  else if (settings.has(PROMPT_KEY)) settings.delete(PROMPT_KEY);
}
