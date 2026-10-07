/**
 * The in-house typed dictionary. Every UI string goes through `translate`, in
 * the renderer and in the main process (e.g. error dialogs).
 */
import type { Language } from "../../core/language";
import { en } from "./en";
import type { Dictionary, MessageKey, MessageParams } from "./types";
import { zhCN } from "./zh-CN";

export type { MessageKey, MessageParams } from "./types";

const dictionaries: { readonly [L in Language]: Dictionary } = {
  en,
  "zh-CN": zhCN,
};

export function translate(language: Language, key: MessageKey, params?: MessageParams): string {
  const template = dictionaries[language][key];
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (placeholder, name: string) =>
    Object.hasOwn(params, name) ? String(params[name]) : placeholder,
  );
}
