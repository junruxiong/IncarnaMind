/**
 * "Write three paragraphs" (写三段), asked beside a Mind: the assistant adds
 * three paragraphs to the Mind in focus with add_to_mind, each a change
 * waiting to be accepted (#902), then says so.
 */
import { lastResult, type Prompt, type Reply } from "./prompt";

const WRITE_THREE = /^write (?:three|3) paragraphs\b|^写三段/i;

export const PARAGRAPHS = {
  en: [
    "The tide turns at noon.",
    "Spring tides come at new and full moon.",
    "Harbours publish the times of high water.",
  ],
  zh: ["潮水每天涨落两次。", "朔望之时潮水最大。", "港口公布每日的高潮时间。"],
};

export function paragraphsReply(prompt: Prompt, asked: string): Reply | null {
  if (!WRITE_THREE.test(asked)) return null;
  if (lastResult(prompt, "add_to_mind") !== undefined) return { text: "Added three paragraphs." };
  const content = (/^写/.test(asked) ? PARAGRAPHS.zh : PARAGRAPHS.en).join("\n\n");
  return { tool: "add_to_mind", input: { content } };
}
