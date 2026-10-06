/** Interface languages. Pure module: the renderer and the main process import it too. */

export const languages = ["en", "zh-CN"] as const;

export type Language = (typeof languages)[number];

/** "system" follows the OS language. */
export type LanguagePreference = Language | "system";

export function isLanguagePreference(value: unknown): value is LanguagePreference {
  return value === "system" || languages.some((language) => language === value);
}

/**
 * Picks the interface language. "system" takes the first OS language IncarnaMind
 * supports; any Chinese variant maps to Simplified Chinese. Falls back to English.
 */
export function resolveLanguage(
  preference: LanguagePreference,
  systemLanguages: readonly string[],
): Language {
  if (preference !== "system") return preference;
  for (const tag of systemLanguages) {
    const base = tag.toLowerCase().split(/[-_]/)[0];
    if (base === "zh") return "zh-CN";
    if (base === "en") return "en";
  }
  return "en";
}
