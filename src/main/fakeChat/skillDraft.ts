/**
 * Make this a Skill (`fakeSkillDraft`): a draft whose "when to use it" and first
 * step quote the Mind's last Question, so a test sees that the Mind was read.
 */

import { type GenerateOptions, systemOf } from "./prompt";

/** The scripted Make this a Skill: a draft in the form the instructions ask for (see ../../core/skills/draft). */
export function fakeSkillDraft(options: GenerateOptions): string | null {
  if (!/into a Skill/.test(systemOf(options.prompt))) return null;
  const request = options.prompt
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  const questions = [...request.matchAll(/^\*\*Question:\*\* (.+)$/gm)].map((match) => match[1]);
  const last = questions.at(-1) ?? "the task";
  return [
    "NAME: scripted-review",
    `WHEN: Use when asked to do the same as "${last}".`,
    "STEPS:",
    `1. Do what the Question asks: ${last}`,
    "2. Read the Documents in scope.",
    "3. Cite each finding.",
  ].join("\n");
}
