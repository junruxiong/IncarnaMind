import type { en } from "./en";

export type MessageKey = keyof typeof en;

/** A complete translation: every key, and nothing else. */
export type Dictionary = { readonly [K in MessageKey]: string };

export type MessageParams = Readonly<Record<string, string | number>>;
