/**
 * What a Mind remembers about its word count: the target the User set and what
 * the count shows. Like the model (`mindModel.ts`), it lives in the Mind's Yjs
 * settings, not in its Blocks, so it goes wherever the Mind goes and the
 * Blocks and the content schema stay as they are.
 */
import type * as Y from "yjs";
import { MIND_SETTINGS_FIELD } from "../core/api";
import { COUNT_BASES, type CountBasis } from "./wordCount";

const TARGET_KEY = "wordTarget";
const BASIS_KEY = "countBasis";

const settingsOf = (doc: Y.Doc): Y.Map<unknown> => doc.getMap(MIND_SETTINGS_FIELD);

/** The target set for the Mind, or null. Anything malformed counts as none. */
export function wordTargetOf(doc: Y.Doc): number | null {
  const value = settingsOf(doc).get(TARGET_KEY);
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

/** Sets the target; null removes it. */
export function setWordTarget(doc: Y.Doc, target: number | null): void {
  const settings = settingsOf(doc);
  if (target === null) {
    if (settings.has(TARGET_KEY)) settings.delete(TARGET_KEY);
  } else if (settings.get(TARGET_KEY) !== target) {
    settings.set(TARGET_KEY, target);
  }
}

/** What the Mind's count shows; words and characters unless the User chose otherwise. */
export function countBasisOf(doc: Y.Doc): CountBasis {
  const value = settingsOf(doc).get(BASIS_KEY);
  return COUNT_BASES.find((each) => each === value) ?? "words";
}

export function setCountBasis(doc: Y.Doc, basis: CountBasis): void {
  const settings = settingsOf(doc);
  if (basis === "words") {
    if (settings.has(BASIS_KEY)) settings.delete(BASIS_KEY);
  } else if (settings.get(BASIS_KEY) !== basis) {
    settings.set(BASIS_KEY, basis);
  }
}

/** Calls `listener` whenever the target or the basis may have changed, here or from elsewhere. Returns how to stop. */
export function observeWordCountSettings(doc: Y.Doc, listener: () => void): () => void {
  const settings = settingsOf(doc);
  settings.observe(listener);
  return () => settings.unobserve(listener);
}
