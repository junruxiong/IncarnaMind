/**
 * A scripted model that checks claims (#96, #97) as its plans say, and names
 * a verified document's claims: those it has a plan for.
 */
import { shownPassages } from "./citations";
import { type ModelCall, type ScriptedReply, scriptedModel } from "./models";

/** How the scripted model checks a sentence: what it cites, and how it replies. */
export interface Plan {
  verdict?: "supported" | "unsupported" | "contradicted";
  /** The line it cites, from the Passage that holds it. */
  quote?: string;
  says?: string;
  /** Its reply instead, once it has searched. */
  reply?: ScriptedReply;
  /** What it says when asked to confirm a contradiction it found. */
  judgement?: "conflict" | "same" | "neither";
}

const SENTENCE = /<sentence>\n([\s\S]*?)\n<\/sentence>/;

/** The sentence a check Run is asked about. */
export const sentenceOf = (call: ModelCall) => {
  const text = call.options.prompt
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => ("text" in part ? part.text : "")).join(""),
    )
    .join("\n");
  return SENTENCE.exec(text)?.[1] ?? "";
};

/** The numbers of the listed sentences ("[3] …") that have a plan. */
function plannedNumbers(prompt: string, plans: Record<string, Plan>): number[] {
  return prompt.split("\n").flatMap((line) => {
    const listed = /^\[(\d+)\] (.*)$/.exec(line);
    return listed?.[2] !== undefined && listed[2] in plans ? [Number(listed[1])] : [];
  });
}

/**
 * A model that checks as its plans say: it searches for the sentence, cites,
 * then gives its verdict. Asked once more for the deciding Passage, it gives
 * `evidence` (none by default); asked which of a document's sentences make
 * claims, it names those it has a plan for.
 */
export function checkingModel(
  plans: Record<string, Plan>,
  calls: ModelCall[] = [],
  evidence: string = JSON.stringify({ passage: null }),
) {
  return scriptedModel(
    (call) => {
      calls.push(call);
      const plan = plans[sentenceOf(call)] ?? { verdict: "unsupported" };
      const searches = call.results.filter((result) => result.tool === "search_documents");
      if (searches.length === 0) {
        return { calls: [{ tool: "search_documents", input: { query: sentenceOf(call) } }] };
      }
      if (plan.reply) return plan.reply;
      const { quote } = plan;
      if (quote && !call.results.some((result) => result.tool === "cite")) {
        const passage =
          shownPassages(searches.at(-1)?.text ?? "").find((each) => each.text.includes(quote)) ??
          shownPassages(searches.at(-1)?.text ?? "")[0];
        return {
          calls: [
            { tool: "cite", input: { citations: [{ marker: 1, passage: passage?.id, quote }] } },
          ],
        };
      }
      return {
        text: `Verdict: ${plan.verdict}${plan.says ? `\n${plan.says} [^1]` : ""}`,
      };
    },
    {
      generate: (call) => {
        if (call.system.startsWith("You sort the sentences")) {
          return { text: JSON.stringify({ claims: plannedNumbers(call.prompt, plans) }) };
        }
        // Asked to confirm a contradiction, it judges as the sentence's plan says.
        if (call.system.startsWith("Compare a sentence")) {
          const sentence = SENTENCE.exec(call.prompt)?.[1] ?? "";
          return { text: plans[sentence]?.judgement ?? "conflict" };
        }
        return { text: evidence };
      },
    },
  );
}
