/**
 * Citations for what an Answer quotes without markers (#760). Some models
 * read Passages, then quote them word for word, or nearly, and give no
 * Citation for it: the Answer looks uncited, though everything in it can be
 * checked. So once an Answer is written and its own records placed, each of
 * its sentences with no marker is matched to the Passages it read:
 *
 * 1. a quotation in it ("…", “…”, 「…」) that the Citation check finds in a
 *    Passage, word for word;
 * 2. else the sentence itself, when it is worded closely like a sentence of
 *    a Passage that the check finds there: that sentence is the quote.
 *
 * Each match becomes a Citation record, taken as structured output's are,
 * and its marker goes after the sentence (see ./markerPlacement). Only a
 * quote the check finds is taken, so nothing is cited "not found".
 *
 * When nothing matches, the Answer has no Citation at all, and it draws on
 * the Passages (a sentence of it shares a few words with one of theirs), the
 * model is asked once more, in one request with the Answer and the Passages
 * it read, for its Citations (as ./quoteRetry asks for quotes). What it gives
 * is taken the same way. Never more than once per Answer. An Answer that
 * shares nothing with them, such as one saying the Documents don't answer
 * the Question, isn't asked about.
 */
import { jsonSchema, Output, streamText } from "ai";
import { sentenceSpans } from "../../shared/sentences";
import type { AnswerEngineEvent, AnswerRequest } from "./engine";
import {
  answerSentences,
  likeness,
  type MarkerRecord,
  textEdits,
  withMissingMarkers,
  wordsOf,
} from "./markerPlacement";
import { CITE_RETRY_INSTRUCTIONS, citeRetryPrompt } from "./prompt";
import { replyJson } from "./quoteRetry";
import type { WindowBudget } from "./window";

/** The most Citations added to one Answer. */
export const MAX_ADDED = 12;
/** Quoted text with fewer words than this (or CJK characters, by pairs) says too little to match. */
const MIN_QUOTED_WORDS = 3;
/** A sentence matches a Passage's sentence when they share this many words… */
const MIN_SHARED_WORDS = 5;
/** …and this much of their words (Dice similarity). */
const MIN_SIMILARITY = 0.6;
/** An Answer draws on the Passages when one of its sentences shares this many words with one of theirs. */
const MIN_RELATED_WORDS = 3;
/** The most Passages, and of each one's text, the one request for Citations holds without a window. */
const MAX_ASKED_PASSAGES = 12;
const MAX_ASKED_CHARACTERS = 3000;
/** The request's output cap: a sentence, an id and a quote for each Citation. */
const ASK_OUTPUT_TOKENS = 1500;

/** Text in quotation marks. */
const QUOTATION = /["“「『]([^"“”「」『』\n]+)["”」』]/gu;
const MARKER = /\[\^(\d{1,4})\]/g;

interface ShownPassage {
  id: string;
  text: string;
}

/** A Citation to add: its Passage and quote, and the Answer's sentence it goes after. */
interface Match extends MarkerRecord {
  passage: string;
  sentence: string;
}

/**
 * A match, before its marker is given: the number of a marker the model
 * wrote in the sentence with no record behind it, which the Citation takes,
 * if there is one.
 */
type Found = Omit<Match, "marker"> & { marker?: number };

/**
 * The Answer's sentences no Citation is on: those with no marker, or with
 * only markers the model wrote and gave no record for (`recorded`).
 */
function uncitedSentences(answer: string, recorded: (marker: number) => boolean) {
  return answerSentences(answer).filter((sentence) => !sentence.markers.some(recorded));
}

/** The sentences of a Passage's text, each as it is in the text. */
function passageSentences(text: string): string[] {
  return text
    .split(/\n+/)
    .flatMap((line) => sentenceSpans(line).map((span) => line.slice(span.from, span.to).trim()))
    .filter((sentence) => sentence !== "");
}

/**
 * The quotations in `answer` that say enough to match, each with where it
 * starts: a quotation may run past a sentence's end ("“…drink.” So…").
 */
function quotationsIn(answer: string): { at: number; quote: string }[] {
  return [...answer.matchAll(QUOTATION)]
    .map((found) => ({ at: found.index, quote: (found[1] as string).trim() }))
    .filter(({ quote }) => wordsOf(quote).size >= MIN_QUOTED_WORDS);
}

/** The sentence of a Passage that `words` are worded most closely like, if close enough. */
function closestSentence(
  words: ReadonlySet<string>,
  passages: readonly { id: string; sentences: { text: string; words: Set<string> }[] }[],
): { passage: string; quote: string } | null {
  let best: { passage: string; quote: string; score: number } | null = null;
  for (const passage of passages) {
    for (const candidate of passage.sentences) {
      const alike = likeness(words, candidate.words);
      if (alike.shared < MIN_SHARED_WORDS || alike.score < MIN_SIMILARITY) continue;
      if (!best || alike.score > best.score) {
        best = { passage: passage.id, quote: candidate.text, score: alike.score };
      }
    }
  }
  return best && { passage: best.passage, quote: best.quote };
}

/**
 * The Citations to add to `answer`: for each sentence without a marker, the
 * first quotation starting in it that the check finds in a Passage, else the
 * Passage sentence it is worded closely like, if the check finds that; in
 * the Answer's order, at most `MAX_ADDED`.
 */
export function matchUnmarked(
  answer: string,
  passages: readonly ShownPassage[],
  found: (passage: string, quote: string) => boolean,
  recorded: (marker: number) => boolean = () => false,
): Found[] {
  const split = passages.map((passage) => ({
    id: passage.id,
    sentences: passageSentences(passage.text).map((text) => ({ text, words: wordsOf(text) })),
  }));
  const quotations = quotationsIn(answer);
  const matches: Found[] = [];
  for (const { start, end, text: sentence, markers } of uncitedSentences(answer, recorded)) {
    const own = markers[0] === undefined ? {} : { marker: markers[0] };
    if (matches.length >= MAX_ADDED) break;
    const byQuotation = quotations
      .filter(({ at }) => at >= start && at < end)
      .flatMap(({ quote }) =>
        passages.flatMap((passage) =>
          found(passage.id, quote) ? [{ passage: passage.id, quote }] : [],
        ),
      )[0];
    if (byQuotation) {
      matches.push({ ...byQuotation, sentence, ...own });
      continue;
    }
    const words = wordsOf(sentence);
    if (words.size < MIN_SHARED_WORDS) continue;
    const closest = closestSentence(words, split);
    if (closest && found(closest.passage, closest.quote)) {
      matches.push({ ...closest, sentence, ...own });
    }
  }
  return matches;
}

/** Whether a sentence of `answer` shares a few words with a sentence of the Passages. */
function drawsOn(answer: string, passages: readonly ShownPassage[]): boolean {
  const theirs = passages.flatMap((passage) => passageSentences(passage.text).map(wordsOf));
  return answerSentences(answer).some((sentence) => {
    const words = wordsOf(sentence.text);
    return theirs.some((each) => likeness(words, each).shared >= MIN_RELATED_WORDS);
  });
}

/** The Citations in the reply to the one request, as given. */
export function parseAsked(value: unknown): Omit<Match, "marker">[] {
  const list = (value as { citations?: unknown } | undefined)?.citations;
  if (!Array.isArray(list)) return [];
  return list.flatMap((item) => {
    if (typeof item !== "object" || item === null) return [];
    const { sentence, passage, quote } = item as Record<string, unknown>;
    return typeof sentence === "string" && typeof passage === "string" && typeof quote === "string"
      ? [{ sentence: sentence.trim(), passage: passage.trim(), quote: quote.trim() }]
      : [];
  });
}

const askedSchema = jsonSchema<{ citations: unknown }>({
  type: "object",
  properties: {
    citations: {
      type: "array",
      items: {
        type: "object",
        properties: {
          sentence: { type: "string", description: "The Answer's sentence, as the Answer has it." },
          passage: { type: "string", description: "The Passage's id, e.g. P1." },
          quote: {
            type: "string",
            description:
              "A quote supporting the sentence, copied character for character from the Passage.",
          },
        },
        required: ["sentence", "passage", "quote"],
      },
    },
  },
  required: ["citations"],
});

/** The Passages the one request holds: within the window's room when there is one. */
function passagesToAsk(
  answer: string,
  passages: readonly ShownPassage[],
  budget: WindowBudget | null | undefined,
): ShownPassage[] {
  if (!budget) {
    return passages
      .slice(0, MAX_ASKED_PASSAGES)
      .map((passage) => ({ ...passage, text: passage.text.slice(0, MAX_ASKED_CHARACTERS) }));
  }
  let room = budget.roomBeside(
    [CITE_RETRY_INSTRUCTIONS, citeRetryPrompt(answer, [])],
    ASK_OUTPUT_TOKENS,
  );
  const asked: ShownPassage[] = [];
  for (const passage of passages) {
    const cost = budget.tokens(citeRetryPrompt("", [passage]));
    if (cost > room) break;
    asked.push(passage);
    room -= cost;
  }
  return asked;
}

/** The one request for an Answer's Citations; what it gives, or nothing if it fails. */
async function askForCitations(
  request: Pick<AnswerRequest, "model" | "signal">,
  answer: string,
  passages: readonly ShownPassage[],
  temperature: number | undefined,
): Promise<Omit<Match, "marker">[]> {
  const { signal } = request;
  let raw = "";
  try {
    const result = streamText({
      model: request.model,
      instructions: CITE_RETRY_INSTRUCTIONS,
      prompt: citeRetryPrompt(answer, passages),
      temperature,
      maxOutputTokens: ASK_OUTPUT_TOKENS,
      output: Output.object({ schema: askedSchema, name: "citations" }),
      // One request, never more.
      maxRetries: 0,
      abortSignal: signal,
      onError: () => undefined,
    });
    for await (const part of result.fullStream) {
      if (signal.aborted || part.type === "abort") return [];
      if (part.type === "text-delta") raw += part.text;
      else if (part.type === "error") throw part.error;
    }
  } catch (error) {
    if (signal.aborted) return [];
    console.error(
      `The Answer's Citations couldn't be asked for: ${error instanceof Error ? error.message : String(error)}`,
    );
    return [];
  }
  return parseAsked(await replyJson(raw));
}

/**
 * Once an Answer (`answer`, its own records' markers placed) is written: adds
 * the Citations for what it quotes without markers, and when nothing matches
 * and it has no Citation, asks the model for them once. Yields the edits that
 * put the markers in, then "cited-after-checking" if it added any or asked;
 * nothing for an Answer that read no Passages. Returns the text.
 */
export async function* citeUnmarked(
  request: Pick<AnswerRequest, "model" | "documents" | "signal">,
  answer: string,
  temperature: number | undefined,
  budget?: WindowBudget | null,
): AsyncGenerator<AnswerEngineEvent, string> {
  const { documents, signal } = request;
  if (!documents.shownPassages || !documents.quoteFound || !documents.hasRecord) return answer;
  const passages = documents.shownPassages();
  if (passages.length === 0 || answer.trim() === "") return answer;
  const found = (passage: string, quote: string) => documents.quoteFound?.(passage, quote) ?? false;

  const recorded = (marker: number) => documents.hasRecord?.(marker) ?? false;
  let matches = matchUnmarked(answer, passages, found, recorded);
  let asked = false;
  const cited = [...answer.matchAll(MARKER)].some((each) => recorded(Number(each[1])));
  if (matches.length === 0 && !cited && drawsOn(answer, passages)) {
    const toAsk = passagesToAsk(answer, passages, budget);
    if (toAsk.length > 0) {
      asked = true;
      yield { type: "phase", phase: "checking-quotes" };
      const given = await askForCitations(request, answer, toAsk, temperature);
      if (signal.aborted) return answer;
      matches = given.filter((each) => found(each.passage, each.quote)).slice(0, MAX_ADDED);
    }
  }
  if (matches.length === 0) {
    if (asked) yield { type: "cited-after-checking", count: 0, asked };
    return answer;
  }

  const used = [...answer.matchAll(MARKER)].map((each) => Number(each[1]));
  let next = Math.max(0, ...used) + 1;
  const records: Match[] = matches.map((match) => {
    if (match.marker !== undefined) return { ...match, marker: match.marker };
    while (recorded(next)) next++;
    const record = { ...match, marker: next };
    next++;
    return record;
  });
  try {
    documents.cite(
      records.map(({ marker, passage, quote }) => ({ marker, passage, quote })),
      { structured: true },
    );
  } catch (error) {
    console.error(
      `The Answer's added Citations couldn't be recorded: ${error instanceof Error ? error.message : String(error)}`,
    );
    return answer;
  }
  const { text } = withMissingMarkers(
    answer,
    records,
    (marker) => documents.hasRecord?.(marker) ?? false,
  );
  yield* textEdits(answer, text);
  // The Citations the Answer shows: a record whose marker didn't go in is dropped.
  const shown = new Set([...text.matchAll(MARKER)].map((each) => Number(each[1])));
  const added = new Set(records.map((record) => record.marker));
  yield {
    type: "cited-after-checking",
    count: [...added].filter((marker) => shown.has(marker) && recorded(marker)).length,
    asked,
  };
  return text;
}
