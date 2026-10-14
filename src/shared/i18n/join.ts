import type { Language } from "../../core/language";

const FULL_WIDTH_PUNCTUATION = /[，。：；？！、（）“”‘’…—《》【】]$/u;
const FULL_WIDTH_PUNCTUATION_START = /^[，。：；？！、（）“”‘’…—《》【】]/u;
const HAN_END = /\p{Script=Han}$/u;
const HAN_START = /^\p{Script=Han}/u;

/**
 * What goes between two pieces of text a screen puts side by side: a space in
 * English; in Chinese none after or before full-width punctuation or between
 * Chinese characters, else a space (docs/agents/zh-CN.md).
 */
export function gap(language: Language, before: string, after: string): string {
  if (language !== "zh-CN") return " ";
  if (before === "" || after === "") return "";
  if (FULL_WIDTH_PUNCTUATION.test(before) || FULL_WIDTH_PUNCTUATION_START.test(after)) return "";
  return HAN_END.test(before) && HAN_START.test(after) ? "" : " ";
}
