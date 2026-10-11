/**
 * The cheap judge (the evaluation's chat model, Haiku 5.5): for one run, which
 * uncited sentences make a claim drawn from the Library, which name a source
 * without a Citation, and whether the answer is correct against the expected
 * one. It proposes; a person checks a 10% sample from the reviewer sheet, as
 * the evaluation's Citation sheet works.
 */
import { generateText } from "ai";
import { createAiSdkChatModel } from "../../../src/core";
import type { ChatSettings } from "../../lib/config";
import type { Task } from "./bakeoff";
import type { Sentence } from "./score";
import { type ModelUsage, NO_USAGE } from "./usage";

export interface Verdict {
  correct: "yes" | "partly" | "no";
  /** Indexes (from 0) of uncited sentences that make a claim from the Library. */
  claims: number[];
  /** Indexes of uncited sentences that name a source (a Document, page, link, "the Library") instead of citing. */
  sources: number[];
  usage: ModelUsage;
}

const SYSTEM = `You grade one answer from an evaluation of tools that answer questions from a user's own documents. Reply with one JSON object and nothing else.`;

export async function judge(
  chat: ChatSettings,
  task: Task,
  prompt: string,
  sentences: readonly Sentence[],
): Promise<Verdict> {
  const model = createAiSdkChatModel({
    kind: chat.kind,
    modelId: chat.modelId,
    apiKey: chat.apiKey,
    baseUrl: chat.baseUrl,
  });
  const numbered = sentences
    .map((sentence, index) => `${index + 1} ${sentence.cited ? "[C]" : "[ ]"} ${sentence.text}`)
    .join("\n");
  const unanswerable = task.unanswerable
    ? `\nThe documents in scope do NOT answer this. "correct" is "yes" only if the answer says so and gives no made-up value; "no" if it states an answer.`
    : "";
  const result = await generateText({
    model,
    system: SYSTEM,
    prompt: `Task given to the tool: ${prompt}
Expected answer: ${task.answer}${unanswerable}

The answer, one sentence per line, numbered. [C] marks a sentence with a citation token; [ ] one without:
${numbered || "(empty)"}

Return {"correct": "yes" | "partly" | "no", "claims": [numbers], "sources": [numbers]}.
- "correct": does the answer give the expected answer's facts ("partly" if some are missing or slightly off)?
- "claims": the numbers of [ ] sentences that state a fact presented as coming from the user's documents, or that answer the task. Leave out headings, greetings, offers of help, statements that the documents don't say something, questions to the user, notes about how the answer was found, and sentences that only repeat a cited one.
- "sources": the numbers of [ ] sentences that name where something comes from (a document or paper title, a page or slide, a link, "the library" or "your documents") without a citation token.`,
  });
  const usage: ModelUsage = {
    ...NO_USAGE,
    calls: 1,
    inputTokens: result.usage.inputTokens ?? 0,
    cachedInputTokens: result.usage.inputTokenDetails?.cacheReadTokens ?? 0,
    outputTokens: result.usage.outputTokens ?? 0,
  };
  const json = /\{[\s\S]*\}/.exec(result.text)?.[0] ?? "{}";
  let parsed: { correct?: unknown; claims?: unknown; sources?: unknown };
  try {
    parsed = JSON.parse(json) as typeof parsed;
  } catch {
    parsed = {};
  }
  const correct =
    parsed.correct === "yes" || parsed.correct === "partly" || parsed.correct === "no"
      ? parsed.correct
      : "no";
  const uncited = (value: unknown) => [
    ...new Set(
      (Array.isArray(value) ? value : [])
        .map((each) => Number(each) - 1)
        .filter((index) => Number.isInteger(index) && sentences[index] && !sentences[index].cited),
    ),
  ];
  return { correct, claims: uncited(parsed.claims), sources: uncited(parsed.sources), usage };
}
