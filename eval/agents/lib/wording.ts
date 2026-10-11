/**
 * One wording for every route (B.5 of the design): the server's own
 * instructions (src/core/agents/mcp.ts), cut to the Tools an arm has, and the
 * Agent Skill for arm C built from the same sentences and the CLI's help,
 * which repeats the MCP Tools' descriptions word for word.
 */
import { SERVER_INSTRUCTIONS } from "../../../src/core/agents/mcp";

/** The Tools each outside arm sees. */
export const FINE_TOOLS = ["search", "read_passages", "cite", "check_citations"] as const;
export const ASK_TOOLS = ["ask"] as const;

const lines = SERVER_INSTRUCTIONS.split("\n");
const general = lines.filter((line) => !line.includes("`ask`"));
const workflow = lines.find((line) => line.includes("`ask`")) ?? "";
const [askSentence, writeSentence] = workflow.split(/(?<=\.) To write your own text, /);

/** "To answer, `search`, then cite each claim…": the workflow without `ask`. */
const fineSentence = writeSentence ? `To answer, ${writeSentence}` : workflow;

export type Wording = "fine" | "ask" | "both";

/** The server's instructions for an arm that has only these Tools. */
export function instructionsFor(wording: Wording): string {
  if (wording === "both") return SERVER_INSTRUCTIONS;
  return [...general, wording === "ask" ? (askSentence ?? workflow) : fineSentence].join("\n");
}

/**
 * The Agent Skill a plugin would ship beside the MCP server (round 2, MQ): the
 * same server instructions, naming the MCP Tools, with the same description
 * as arm C's, so an agent loads it for the same questions.
 */
export function mcpSkillMarkdown(): string {
  return `---
name: incarnamind
description: Search the User's IncarnaMind Library (their own Documents) and cite it with Citations IncarnaMind checks. Use for any question about the User's documents, files, reports or notes.
---

# IncarnaMind

${lines.join("\n\n")}

The Tools are on the \`incarnamind\` MCP server: \`ask\`, \`search\`, \`read_passages\`, \`cite\` and \`check_citations\`.
`;
}

/** Arm C's Agent Skill: the same instructions, the CLI's verbs for the Tools, and its help. */
export function skillMarkdown(cliHelp: string): string {
  const verbs = fineSentence
    .replace("`search`", '`incarnamind search "<words>"`')
    .replace(
      "call `check_citations` on your draft",
      "run `incarnamind check -` with your draft on stdin",
    );
  return `---
name: incarnamind
description: Search the User's IncarnaMind Library (their own Documents) and cite it with Citations IncarnaMind checks. Use for any question about the User's documents, files, reports or notes.
---

# IncarnaMind

${[...general, verbs].join("\n\n")}

## Commands

\`\`\`
${cliHelp.trim()}
\`\`\`
`;
}
