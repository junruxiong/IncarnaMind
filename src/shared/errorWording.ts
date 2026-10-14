/**
 * How an error's wording (src/core/errors.ts) crosses from the main process to the window (#627).
 * Electron hands the window only the error's text, so the wording rides after its sentence as one
 * tagged line of JSON. The log keeps a message's first line, so it keeps the sentence alone.
 */
import { type ErrorWording, isRecord, TranslatableError } from "../core/errors";
import { en } from "./i18n/en";
import type { MessageKey } from "./i18n/types";

const TAG = "\n⁣wording:";

/** The error to throw to the window: one with a wording carries it in its message, any other is as it was. */
export function errorForWindow(error: unknown): unknown {
  if (!(error instanceof TranslatableError) || !error.wording) return error;
  const told = new Error(`${error.message}${TAG}${JSON.stringify(error.wording)}`);
  told.name = error.name;
  if (error.stack) told.stack = error.stack;
  return told;
}

/** Takes the wording off an error's text. A key this dictionary doesn't know leaves the sentence alone. */
export function takeWording(text: string): { sentence: string; wording: ErrorWording | null } {
  const at = text.lastIndexOf(TAG);
  if (at < 0) return { sentence: text, wording: null };
  return { sentence: text.slice(0, at), wording: wordingOf(text.slice(at + TAG.length)) };
}

function wordingOf(json: string): ErrorWording | null {
  let value: unknown;
  try {
    value = JSON.parse(json);
  } catch {
    return null;
  }
  if (!isRecord(value) || typeof value.key !== "string" || !isMessageKey(value.key)) return null;
  if (value.values === undefined) return { key: value.key };
  if (!isRecord(value.values)) return null;
  const values: Record<string, string | number> = {};
  for (const [name, each] of Object.entries(value.values)) {
    if (typeof each !== "string" && typeof each !== "number") return null;
    values[name] = each;
  }
  return { key: value.key, values };
}

const isMessageKey = (key: string): key is MessageKey => Object.hasOwn(en, key);
