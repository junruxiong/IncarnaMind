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

const AGENT_PLACEHOLDER = "{agent}";

export function translate(language: Language, key: MessageKey, params?: MessageParams): string {
  const template = dictionaries[language][key];
  if (!params) return template;
  const fill = (text: string) =>
    text.replace(/\{(\w+)\}/g, (placeholder, name: string) =>
      Object.hasOwn(params, name) ? String(params[name]) : placeholder,
    );
  // Chinese templates put no space around an agent's name: it depends on the name (docs/agents/zh-CN.md).
  if (
    language !== "zh-CN" ||
    !Object.hasOwn(params, "agent") ||
    !template.includes(AGENT_PLACEHOLDER)
  ) {
    return fill(template);
  }
  const agent = String(params.agent);
  return template
    .split(AGENT_PLACEHOLDER)
    .map(fill)
    .reduce(
      (text, part) => text + gap(language, text, agent) + agent + gap(language, agent, part) + part,
    );
}
