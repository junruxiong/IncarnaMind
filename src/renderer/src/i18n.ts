import { useCallback } from "react";
import { type Language, resolveLanguage } from "../../core/language";
import { type MessageKey, type MessageParams, translate } from "../../shared/i18n";
import { useAppStore } from "./store";

/** The interface language: from settings once loaded, the OS language before that. */
export function useLanguage(): Language {
  const language = useAppStore((state) => state.settings?.language);
  return language ?? resolveLanguage("system", navigator.languages);
}

export function languageNow(): Language {
  return (
    useAppStore.getState().settings?.language ?? resolveLanguage("system", navigator.languages)
  );
}

/** A message in the interface language now, outside React: e.g. an Undo line's. */
export function translateNow(key: MessageKey, params?: MessageParams): string {
  return translate(languageNow(), key, params);
}

/** Returns `t(key, params)` for the current interface language. */
export function useT(): (key: MessageKey, params?: MessageParams) => string {
  const language = useLanguage();
  return useCallback((key, params) => translate(language, key, params), [language]);
}
