/**
 * The citation style a Mind is written in (#289). Like its model (`mindModel.ts`), it lives in
 * the Mind's Yjs settings (`MIND_SETTINGS_FIELD`), not in its Blocks, so it goes wherever the
 * Mind goes and its content schema stays as it is. A Mind with no choice follows the interface
 * language (`DEFAULT_CITATION_STYLE`).
 */
import type * as Y from "yjs";
import type { Language } from "../core/language";
import { type CitationStyleId, DEFAULT_CITATION_STYLE, isCitationStyleId } from "./citationStyles";

const STYLE_KEY = "citationStyle";

/** The style chosen for the Mind, or null when it follows the default. Anything malformed counts as none. */
export function chosenCitationStyle(settings: Y.Map<unknown>): CitationStyleId | null {
  const value = settings.get(STYLE_KEY);
  return isCitationStyleId(value) ? value : null;
}

/** The style the Mind's Citations and Bibliography are in. */
export function citationStyleIn(settings: Y.Map<unknown>, language: Language): CitationStyleId {
  return chosenCitationStyle(settings) ?? DEFAULT_CITATION_STYLE[language];
}

/** Chooses a style for the Mind; null forgets the choice, so it follows the default again. */
export function setCitationStyle(settings: Y.Map<unknown>, style: CitationStyleId | null): void {
  if (style === null) {
    if (settings.has(STYLE_KEY)) settings.delete(STYLE_KEY);
  } else if (settings.get(STYLE_KEY) !== style) {
    settings.set(STYLE_KEY, style);
  }
}
