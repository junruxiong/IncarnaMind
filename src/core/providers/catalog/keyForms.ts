/**
 * What each provider's API keys look like, where the provider says so or its keys show it: the
 * first-run card and the key form use it to catch a key pasted for another service before
 * anything is sent (#1384). Only forms that tell their service apart are here. A provider
 * missing from the table (Mistral, DeepSeek, Qwen, GLM, SiliconFlow, a server's own keys) has
 * keys that look like others' or like nothing in particular, so its keys are never flagged and
 * never recognised.
 */
const KEY_FORMS: Readonly<Record<string, readonly RegExp[]>> = {
  anthropic: [/^sk-ant-/],
  openai: [/^sk-(?:proj|svcacct|admin)-/, /^sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}$/],
  google: [/^AIza[\w-]{30,}$/],
  openrouter: [/^sk-or-/],
  xai: [/^xai-/],
  // The same length as the older OpenAI form, which has its marker where a Kimi key has none.
  kimi: [/^sk-(?!.{20}T3BlbkFJ)[A-Za-z0-9]{48}$/],
};

/** The providers whose keys have a form of their own. */
export const KEY_FORM_PROVIDERS: readonly string[] = Object.keys(KEY_FORMS);

/** Whether `key` looks like the keys of the provider `catalogId`; null where its keys have no form of their own. */
export function keyFits(catalogId: string, key: string): boolean | null {
  const forms = KEY_FORMS[catalogId];
  return forms ? forms.some((form) => form.test(key.trim())) : null;
}

/** The provider whose keys `key` looks like, if one does. */
export function serviceOfKey(key: string): string | null {
  const given = key.trim();
  for (const [catalogId, forms] of Object.entries(KEY_FORMS)) {
    if (forms.some((form) => form.test(given))) return catalogId;
  }
  return null;
}
