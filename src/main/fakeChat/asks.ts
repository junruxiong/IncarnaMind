/**
 * In the assistant, with `ask_user` (#1374): "Summarise the main documents" (总结主要的文档), with
 * more than two Documents, asks which ones, the two newest ticked, then writes a summary of those
 * chosen into the Mind in focus. Skipped, it goes on with the two newest and says so.
 * "Summarise the main ones" asks as Claude Haiku 5.5 did: one choice, one ticked, Other… allowed.
 */
import { lastResult, type Prompt, type Reply, systemOf } from "./prompt";

const MAIN_DOCUMENTS =
  /\bsummari[sz]e the (?:main|key|important) documents\b|总结.*(?:主要|重要).*文档/i;
const MAIN_ONES = /\bsummari[sz]e the main ones\b/i;

const namesIn = (system: string): string[] => {
  const list = /names, newest first:\n((?:- .+\n?)+)/.exec(system)?.[1] ?? "";
  return [...list.matchAll(/^- (.+)$/gm)].map((match) => match[1]?.trim() ?? "");
};

const chosenIn = (result: string): string[] =>
  [...result.matchAll(/“([^”]+)”/g)].map((match) => match[1] ?? "");

export function asksReply(
  prompt: Prompt,
  question: string,
  names: readonly string[],
): Reply | null {
  const single = MAIN_ONES.test(question);
  if (!names.includes("ask_user") || !(single || MAIN_DOCUMENTS.test(question))) return null;
  const documents = namesIn(systemOf(prompt));
  const chinese = /\p{Script=Han}/u.test(question);
  const answered = lastResult(prompt, "ask_user");
  if (single) {
    if (answered !== undefined) return { text: "Going on with that." };
    return {
      tool: "ask_user",
      input: {
        question: "Which documents should I summarise as the “main” ones?",
        multiple: false,
        other: true,
        options: [
          { label: "The two newest", recommended: true, preselected: true },
          { label: `All ${documents.length} Documents` },
          { label: "Only the leases" },
        ],
      },
    };
  }
  if (answered === undefined) {
    if (documents.length <= 2) return null;
    return {
      tool: "ask_user",
      input: {
        question: chinese ? "要总结哪几份文档？" : "Which Documents should I summarise?",
        detail: chinese
          ? `资料库里有 ${documents.length} 份，已选中最新的两份。`
          : `Your Library has ${documents.length}. The two newest are picked.`,
        multiple: true,
        other: true,
        options: documents.slice(0, 4).map((name, index) => ({
          label: name,
          recommended: index === 0,
          preselected: index < 2,
        })),
      },
    };
  }
  const skipped = answered.includes("skipped");
  const chosen = skipped ? documents.slice(0, 2) : chosenIn(answered);
  const added = lastResult(prompt, "add_to_mind");
  if (added === undefined) {
    const content = [
      chinese ? "## 摘要" : "## Summary",
      ...chosen.map((name) => (chinese ? `“${name}”的要点。` : `The main points of “${name}”.`)),
    ].join("\n\n");
    return { tool: "add_to_mind", input: { content } };
  }
  return {
    text: skipped
      ? chinese
        ? "你没有选择，先总结了最新的两份文档。"
        : "Summarised the two newest Documents, as you didn't choose."
      : chinese
        ? "已写好你选的文档的摘要。"
        : "Summarised the Documents you chose.",
  };
}
