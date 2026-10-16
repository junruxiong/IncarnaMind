/**
 * The citation style a Mind is written in (#289). Like its model (`mindModel.ts`), it lives in
 * the Mind's Yjs settings (`MIND_SETTINGS_FIELD`), not in its Blocks, so it goes wherever the
 * Mind goes and its content schema stays as it is. A Mind with no choice is in the default
 * style (`defaultCitationStyle`), unless the User said "not for this Mind" to the offer of one.
 */
import type * as Y from "yjs";
import type { Language } from "../core/language";
import { type CitationStyleId, defaultCitationStyle, isCitationStyleId } from "./citationStyles";

const STYLE_KEY = "citationStyle";
/** The User's "not for this Mind" to the prompt that offers a style (`CitationStylePrompt`). */
const PROMPT_KEY = "citationStylePrompt";
/** The line that said the Mind took the Chinese default style was shown: it never shows again. */
const DEFAULT_NOTICE_KEY = "citationStyleDefaultNotice";

/** The style chosen for the Mind, or null when it follows the default. Anything malformed counts as none. */
export function chosenCitationStyle(settings: Y.Map<unknown>): CitationStyleId | null {
  const value = settings.get(STYLE_KEY);
  return isCitationStyleId(value) ? value : null;
}

/** What decides the style of a Mind that hasn't chosen one, beyond its settings. */
export interface StyleContext {
  /** The interface language. */
  language: Language;
  chineseText: boolean;
  citesWorks: boolean;
}

/**
 * The style a Mind is in: its choice; else Numbered when the User closed the offer of a style
 * there ("not for this Mind"), which no default overrides; else the default. The core and the
 * editor both work it out here, so an export and the page never disagree.
 */
export function resolvedCitationStyle(
  mind: { chosen: CitationStyleId | null; promptDismissed: boolean },
  context: StyleContext,
): CitationStyleId {
  if (mind.chosen !== null) return mind.chosen;
  return mind.promptDismissed ? "numbered" : defaultCitationStyle(context.language, context);
}

/** The style the Mind's Citations and Bibliography are in (`resolvedCitationStyle`). */
export function citationStyleIn(settings: Y.Map<unknown>, context: StyleContext): CitationStyleId {
  return resolvedCitationStyle(
    {
      chosen: chosenCitationStyle(settings),
      promptDismissed: citationStylePromptDismissed(settings),
    },
    context,
  );
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

export const defaultStyleNoticeShown = (settings: Y.Map<unknown>): boolean =>
  settings.get(DEFAULT_NOTICE_KEY) === "shown";

export function markDefaultStyleNoticeShown(settings: Y.Map<unknown>): void {
  if (!defaultStyleNoticeShown(settings)) settings.set(DEFAULT_NOTICE_KEY, "shown");
}

/** Closes the prompt for this Mind, or (false) shows it again. */
export function dismissCitationStylePrompt(settings: Y.Map<unknown>, dismissed = true): void {
  if (dismissed) settings.set(PROMPT_KEY, "dismissed");
  else if (settings.has(PROMPT_KEY)) settings.delete(PROMPT_KEY);
}
