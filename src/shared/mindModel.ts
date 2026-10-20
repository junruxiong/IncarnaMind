/**
 * The model a Mind's next Questions are asked with, chosen in its composer and
 * remembered for that Mind (DESIGN.md, Composer › Model). It lives in the
 * Mind's Yjs document (`MIND_SETTINGS_FIELD`), not in its Blocks, so it goes
 * wherever the Mind goes and the Blocks stay as they are. A Mind with no
 * choice follows the Answers default (`UserSettings.chatModel`); a new Mind
 * is given the model last chosen on the device (`DeviceSettings.lastChatModel`).
 */
import type * as Y from "yjs";
import { type ChatModelChoice, MIND_SETTINGS_FIELD } from "../core/api";

const MODEL_KEY = "model";

const settingsOf = (doc: Y.Doc): Y.Map<unknown> => doc.getMap(MIND_SETTINGS_FIELD);

export const sameModel = (a: ChatModelChoice | null, b: ChatModelChoice | null): boolean =>
  a !== null && b !== null && a.providerId === b.providerId && a.modelId === b.modelId;

/** The model chosen for the Mind, or null when it follows the default. Anything malformed counts as none. */
export function mindModelOf(doc: Y.Doc): ChatModelChoice | null {
  const value = settingsOf(doc).get(MODEL_KEY);
  if (typeof value !== "object" || value === null) return null;
  const { providerId, modelId } = value as Record<string, unknown>;
  return typeof providerId === "string" &&
    providerId !== "" &&
    typeof modelId === "string" &&
    modelId !== ""
    ? { providerId, modelId }
    : null;
}

/** Remembers a model for the Mind; null forgets it, so the Mind follows the default again. */
export function setMindModel(doc: Y.Doc, choice: ChatModelChoice | null): void {
  const settings = settingsOf(doc);
  if (choice === null) {
    if (settings.has(MODEL_KEY)) settings.delete(MODEL_KEY);
    return;
  }
  if (sameModel(mindModelOf(doc), choice)) return;
  settings.set(MODEL_KEY, { providerId: choice.providerId, modelId: choice.modelId });
}

/** Calls `listener` whenever the Mind's choice may have changed, here or from elsewhere. Returns how to stop. */
export function observeMindModel(doc: Y.Doc, listener: () => void): () => void {
  const settings = settingsOf(doc);
  settings.observe(listener);
  return () => settings.unobserve(listener);
}
