/**
 * What the model is asked for a suggestion, and how its reply is read: the
 * paragraph the User is writing, the Passages found for it, and back the
 * words to add, the Passage they rest on and a quote from it.
 */
import { jsonSchema } from "ai";
import { englishLocation } from "../../shared/locations";
import type { FoundPassage } from "../answers/citing";

/** The most words a suggestion may have, and the most CJK characters: one short sentence. */
export const MAX_WORDS = 30;
export const MAX_CJK_CHARACTERS = 60;
/** A reply longer than this, after its first sentence, is not taken: the model ignored the limit. */
const TOLERANCE = 1.5;

export const SUGGESTION_INSTRUCTIONS = [
  "You help the User write. They paused at the end of the paragraph they are writing. Suggest how it goes on, using only the Passages from their Documents below.",
  "- If its last sentence is unfinished, finish it; otherwise write the next sentence. The paragraph ends at the end of a word.",
  `- One sentence at most: no more than ${MAX_WORDS} words, or ${MAX_CJK_CHARACTERS} Chinese characters.`,
  "- Write in the paragraph's language and tone, as the User would go on. Don't repeat what the paragraph already says.",
  "- Rest it on one Passage, and copy from that Passage a short quote that supports it, character for character.",
  "- If no Passage supports a way on, give empty text.",
  'Reply with JSON only: {"text": the words to add after the paragraph, "passage": the id of the Passage, such as P1, "quote": the quote}.',
].join("\n");

export interface SuggestionReply {
  text: string;
  passage: string;
  quote: string;
}

export const suggestionSchema = jsonSchema<SuggestionReply>({
  type: "object",
  properties: {
    text: {
      type: "string",
      description: "The words to add after the paragraph; empty if no Passage supports any.",
    },
    passage: { type: "string", description: "The id of the Passage it rests on, such as P1." },
    quote: {
      type: "string",
      description: "A short quote from that Passage, copied character for character.",
    },
  },
  required: ["text", "passage", "quote"],
  additionalProperties: false,
});

const attribute = (value: string) => value.replace(/"/g, "'").replace(/\s+/g, " ");

/** The Passages' short ids, as the model sees them: P1, P2… in the order given. */
const passageHandle = (index: number) => `P${index + 1}`;

/** The request: the paragraph, then each Passage with its id, Document and Location. */
export function suggestionPrompt(paragraph: string, passages: readonly FoundPassage[]): string {
  const shown = passages.map((passage, index) => {
    const where = passage.location
      ? ` location="${attribute(englishLocation(passage.location))}"`
      : "";
    return `<passage id="${passageHandle(index)}" document="${attribute(passage.documentName)}"${where}>\n${passage.text}\n</passage>`;
  });
  return [`<paragraph>\n${paragraph}\n</paragraph>`, ...shown].join("\n\n");
}

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu;
/** Where the first sentence ends: after its full stop, in English or Chinese. */
const FIRST_SENTENCE = /^[\s\S]*?(?:[。！？]|[.!?](?=\s|$|["'”’)）]))["'”’)）」]?/u;

/** How long a text is, as a suggestion's limit counts it: words, or CJK characters. */
function tooLong(text: string): boolean {
  const cjk = text.match(CJK)?.length ?? 0;
  const words = text.replace(CJK, " ").split(/\s+/).filter(Boolean).length;
  return cjk > MAX_CJK_CHARACTERS * TOLERANCE || words > MAX_WORDS * TOLERANCE;
}

/**
 * The model's reply, read: the words (their first sentence, without quotation
 * marks around them), the index of the Passage it names among those given
 * ("P2", or "2"), and its quote. Null when it gives no words, names no
 * Passage it was given, or runs far past the limit.
 */
export function readReply(
  value: unknown,
  passageCount: number,
): { text: string; passage: number; quote: string } | null {
  if (typeof value !== "object" || value === null) return null;
  const { text, passage, quote } = value as Record<string, unknown>;
  if (typeof text !== "string" || typeof passage !== "string" || typeof quote !== "string") {
    return null;
  }
  const words = text
    .trim()
    .replace(/^["“”]+|["“”]+$/g, "")
    .trim();
  // A "sentence" shorter than this is an abbreviation's full stop ("Dr.", "e.g."): kept whole.
  const first = FIRST_SENTENCE.exec(words)?.[0]?.trim();
  const sentence = first && first.length >= 12 ? first : words;
  if (sentence === "" || tooLong(sentence) || quote.trim() === "") return null;
  const number = /^\s*\[?\s*(?:passage\s*)?P?\s*(\d{1,3})\s*\]?\s*$/i.exec(passage)?.[1];
  const index = number === undefined ? -1 : Number(number) - 1;
  if (index < 0 || index >= passageCount) return null;
  return { text: sentence, passage: index, quote };
}
