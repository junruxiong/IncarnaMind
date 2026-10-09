/**
 * Whether a chat model reads images in a request. No provider's API says so
 * before a request, so it is known from the provider and the model's name,
 * as other apps do; when it can't be told, the answer is no, and nothing is
 * sent that the model would refuse.
 */
import type { ChatProviderKind } from "../api";

/** OpenAI's models that read images: GPT-4o, 4.1, 4.5, GPT-5 and later, and o1, o3 and o4 but not o1-mini or o3-mini. */
const OPENAI_IMAGES = /^(gpt-(4o|4\.1|4\.5|[5-9])|chatgpt-4o|o4|o[13](?!-mini))/i;

/** Every Claude model since Claude 3 reads images. */
const ANTHROPIC_IMAGES = /^claude-(?!2|instant)/i;

/** Gemini models all read images. */
const GOOGLE_IMAGES = /^gemini-/i;

/**
 * True for a model known to read images. An OpenAI-compatible server or a
 * model in Ollama is not known here, so it gets text only.
 */
export function chatModelReadsImages(kind: ChatProviderKind, modelId: string): boolean {
  const id = modelId.trim();
  switch (kind) {
    case "openai":
    case "chatgpt":
      return OPENAI_IMAGES.test(id);
    case "anthropic":
      return ANTHROPIC_IMAGES.test(id);
    case "google":
      return GOOGLE_IMAGES.test(id);
    case "openai-compatible":
    case "ollama":
      return false;
  }
}
