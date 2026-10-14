import type { ChatProviderKind } from "../core/api";

/**
 * The chat providers whose own web search an Answer can use (#128): Anthropic,
 * OpenAI, and the ChatGPT plan, whose route is OpenAI's. Settings names them
 * when the chosen model's provider has none.
 */
export const WEB_SEARCH_PROVIDERS: readonly ChatProviderKind[] = ["anthropic", "openai", "chatgpt"];

export const searchesTheWeb = (kind: ChatProviderKind): boolean =>
  WEB_SEARCH_PROVIDERS.includes(kind);
