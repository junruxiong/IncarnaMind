/**
 * The cheap judge (the evaluation's chat model, Haiku 5.5): for one run, which
 * uncited sentences make a claim drawn from the Library, and whether the
 * answer is correct against the expected one. It proposes; the owner checks a
 * 10% sample from the reviewer sheet, as the evaluation's Citation sheet works.
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

The answer, one sentence per line, numbered. [C] marks a sentence with a citation; [ ] one without:
${numbered || "(empty)"}

Return {"correct": "yes" | "partly" | "no", "claims": [numbers]}.
- "correct": does the answer give the expected answer's facts ("partly" if some are missing or slightly off)?
- "claims": the numbers of [ ] sentences that state a fact taken from the user's documents. Leave out headings, greetings, offers of help, statements that the documents don't say something, questions to the user, notes about how the answer was found, and sentences that only repeat a cited one.`,
  });
  const usage: ModelUsage = {
    ...NO_USAGE,
    calls: 1,
    inputTokens: result.usage.inputTokens ?? 0,
    cachedInputTokens: result.usage.inputTokenDetails?.cacheReadTokens ?? 0,
    outputTokens: result.usage.outputTokens ?? 0,
  };
  const json = /\{[\s\S]*\}/.exec(result.text)?.[0] ?? "{}";
  let parsed: { correct?: unknown; claims?: unknown };
  try {
    parsed = JSON.parse(json) as { correct?: unknown; claims?: unknown };
  } catch {
    parsed = {};
  }
  const correct =
    parsed.correct === "yes" || parsed.correct === "partly" || parsed.correct === "no"
      ? parsed.correct
      : "no";
  const claims = (Array.isArray(parsed.claims) ? parsed.claims : [])
    .map((value) => Number(value) - 1)
    .filter((index) => Number.isInteger(index) && sentences[index] && !sentences[index].cited);
  return { correct, claims: [...new Set(claims)], usage };
}
