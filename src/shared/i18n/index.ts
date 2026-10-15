/**
 * The in-house typed dictionary. Every UI string goes through `translate`, in
 * the renderer and in the main process (e.g. error dialogs).
 */
import type { Language } from "../../core/language";
import { en } from "./en";
import { gap } from "./join";
import type { Dictionary, MessageKey, MessageParams } from "./types";
import { zhCN } from "./zh-CN";

export type { MessageKey, MessageParams } from "./types";
export { gap };

const dictionaries: { readonly [L in Language]: Dictionary } = {
  en,
  "zh-CN": zhCN,
};

/**
 * Chinese templates put no space around an agent's name or an export format's: whether one goes
 * there depends on the name, Latin or Chinese (docs/agents/zh-CN.md).
 */
const SPACED_BY_NAME = ["agent", "format"] as const;

export function translate(language: Language, key: MessageKey, params?: MessageParams): string {
  const template = dictionaries[language][key];
  if (!params) return template;
  const fill = (text: string) =>
    text.replace(/\{(\w+)\}/g, (placeholder, name: string) =>
      Object.hasOwn(params, name) ? String(params[name]) : placeholder,
    );
  const spaced =
    language === "zh-CN"
      ? SPACED_BY_NAME.find((name) => Object.hasOwn(params, name) && template.includes(`{${name}}`))
      : undefined;
  if (spaced === undefined) return fill(template);
  const value = String(params[spaced]);
  return template
    .split(`{${spaced}}`)
    .map(fill)
    .reduce(
      (text, part) => text + gap(language, text, value) + value + gap(language, value, part) + part,
    );
}
