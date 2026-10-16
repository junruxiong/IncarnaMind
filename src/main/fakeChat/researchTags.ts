import { type GenerateOptions, systemOf, userText } from "./prompt";

/** Research's tagger (#130, src/core/research/prompt.ts). */
const RESEARCH_TAGGER = /^You tag the web pages a research Task saved/m;

/** Tags for the pages Research saved: each page's first word, the pages that share it under it. */
export function fakeResearchTags(options: GenerateOptions): string | null {
  if (!RESEARCH_TAGGER.test(systemOf(options.prompt))) return null;
  const request = userText(options.prompt.find((message) => message.role === "user"));
  const pages = new Map<string, number[]>();
  for (const [, number, name] of request.matchAll(/^(\d+)\. (.+)$/gm)) {
    const word = /\p{L}+/u.exec(name ?? "")?.[0];
    if (word) pages.set(word, [...(pages.get(word) ?? []), Number(number)]);
  }
  return [...pages].map(([word, numbers]) => `${word}: ${numbers.join(", ")}`).join("\n");
}
