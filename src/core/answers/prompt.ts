/** The instructions every Answer is written with. Kept short: every word is sent with every Question. */
import type { CitationSupport, SkillFile } from "../api";
import type { LoadedSkill, SkillSummary } from "../skills";

/** A language a Question is clearly written in, judged by its script alone. */
function scriptLanguage(question: string): string | null {
  if (/[぀-ヿ]/.test(question)) return "Japanese";
  if (/[가-힯]/.test(question)) return "Korean";
  if (/[一-鿿]/.test(question)) return "Chinese";
  return null;
}

/** The system prompt for answering `question`, before anything about Documents. */
export function answerInstructions(question: string): string {
  const language = scriptLanguage(question);
  return [
    "You answer Questions in IncarnaMind, a notebook where the User writes Notes and asks Questions in place.",
    "The messages are the part of the User's notebook above the Question: the User's Notes and earlier Questions as user messages, your earlier Answers as assistant messages. The last user message ends with the Question to answer now; use the rest as context.",
    "",
    "- Answer in the language the Question is written in, even when the Notes or Documents are in another language.",
    "- If you don't know the answer, or it isn't in what you were given and you aren't sure of it, say so plainly instead of guessing.",
    "- Write Markdown: paragraphs, headings, lists, fenced code blocks with a language, and LaTeX math as $…$ inline or $$…$$ on lines of its own. Don't use tables or HTML.",
    ...(language ? [`- The Question is written in ${language}, so answer in ${language}.`] : []),
  ].join("\n");
}

/**
 * What to do with the Tools of the User's Connectors, if any are offered.
 * Their results aren't Passages: they are never cited with markers. `alone`:
 * there are no Documents, so these are the only Tools. Tools that may change
 * something ask the User first, who may deny the call.
 */
export function connectorInstructions(
  tools: readonly { provider: { name: string } }[],
  alone: boolean,
): string {
  if (tools.length === 0) return "";
  const names = [...new Set(tools.map((each) => `"${each.provider.name}"`))].join(", ");
  return [
    `You can ${alone ? "" : "also "}call Tools from the User's Connectors (${names}): other services the User has connected. Each Tool's name starts with its Connector's. Use them to look things up when the Question needs what they have.`,
    "- Call a Tool that changes something (creates, sends, books, deletes…) only when the Question asks for that change. The User approves such calls first and may deny one: then carry on without it, don't call it again, and say what wasn't done.",
    "- What they return isn't from the User's Documents: never put a citation marker on it. Say in words where it came from when that helps.",
    ...(alone ? ["- Write nothing before your Tool calls: only the Answer, after them."] : []),
  ].join("\n");
}

/**
 * Connectors that are on but can't be used until the User signs in to them
 * again: their Tools aren't offered, and the Answer says so if it needed them.
 */
export function signInNeededInstructions(connectorNames: readonly string[]): string {
  if (connectorNames.length === 0) return "";
  const names = connectorNames.map((name) => `"${name}"`).join(", ");
  const plural = connectorNames.length > 1;
  return `The User's ${plural ? "Connectors" : "Connector"} ${names} ${plural ? "need" : "needs"} the User to sign in again (in Settings → Connectors), so ${plural ? "their" : "its"} Tools can't be used for this Answer. If the Question needs ${plural ? "them" : "it"}, say so in one short sentence, then answer as well as you can without ${plural ? "them" : "it"}.`;
}

const NOT_COVERED =
  "If the Documents don't cover the Question, say so plainly, then answer from your own knowledge if you can, without markers.";

const RECORD =
  'the Passage\'s id, where the quote is, and a short quote copied exactly from the Passage, without its marks. Where: the page, slide, section, or rows or lines the quote is in, or two in a row, written as the Passage names them (its pages or location, and marks where each new one starts, like [p. 4], [slide 4], [§ 2.1 Sensitivity], [Revenue, rows 2–41] or [lines 51–100]; name rows or lines more narrowly when you can, e.g. "Revenue, rows 12–14")';

/** The Citation check finds a quote with an ellipsis only part by part (see ../../shared/quoteMatch): best avoided. */
const UNBROKEN_QUOTE =
  "- Copy each quote as one unbroken stretch of the Passage: don't leave words out with an ellipsis (... or …); quote less instead.";

/** The most Document names the instructions list: the most recently added. */
export const LISTED_DOCUMENTS = 50;

/** The longest a listed Document name may be, in characters. */
const MAX_NAME_LENGTH = 200;

/** The Documents an Answer may draw on: how many, and some of their names. */
export interface ListedDocuments {
  /** How many there are. */
  total: number;
  /** The names of the most recently added, newest first: at most `LISTED_DOCUMENTS`. */
  names: readonly string[];
}

/** A Document name on one line of its own, and not too long. */
function listedName(name: string): string {
  const line = name.replace(/\s+/g, " ").trim();
  return line.length > MAX_NAME_LENGTH ? `${line.slice(0, MAX_NAME_LENGTH - 1)}…` : line;
}

/**
 * How many Documents there are and what they are called, so the model can
 * tell which ones a Question means ("compare A with B") and say which there
 * are. Past `LISTED_DOCUMENTS`, the rest are counted.
 */
function documentNames({ total, names }: ListedDocuments): string {
  const more = total - names.length;
  return [
    more > 0
      ? `The names of the ${names.length} added most recently:`
      : total === 1
        ? "Its name:"
        : "Their names, newest first:",
    ...names.map((name) => `- ${listedName(name)}`),
    ...(more > 0 ? [`- …and ${more} more`] : []),
    "Use the names to tell which Documents a Question means, and to say which Documents there are when asked; only the Passages say what is in them.",
  ].join("\n");
}

/**
 * What to do with the User's Documents, for a way of citing. `passages`: what
 * the one search found, for a model that can't call Tools. `scoped`: the User
 * limited the Question to some of their Documents (its Search scope), and
 * `listed` covers only those.
 */
export function documentInstructions(
  mode: CitationSupport | "no-documents",
  listed: ListedDocuments,
  passages = "",
  scoped = false,
): string {
  const documents = `${listed.total} Document${listed.total === 1 ? "" : "s"}`;
  const added = `${scoped ? `The User limited this Question to ${documents} of theirs` : `The User has added ${documents}`} (PDF, Word, PowerPoint, Excel, CSV, text and Markdown files).`;
  switch (mode) {
    case "no-documents":
      return "";
    case "tools":
      return [
        `${added} Search them with search_documents whenever they may help; search again with other words if the Passages don't answer the Question.`,
        documentNames(listed),
        "",
        "Cite every claim you draw from a Passage:",
        `- Before writing the Answer, call cite once with a record for each quote you will use: a marker number (1, 2, …), ${RECORD}.`,
        UNBROKEN_QUOTE,
        "- In the Answer, put the marker as [^1] right after each claim it supports. Use only markers you recorded, and add no list of sources.",
        "- Write nothing before your Tool calls: only the Answer, after them.",
        NOT_COVERED,
      ].join("\n");
    case "structured-output":
      return [
        added,
        documentNames(listed),
        "",
        "Passages found in the User's Documents for this Question:",
        passages,
        "",
        // The example's location is a placeholder: a small model copied a real-looking one ("p. 3") as its own (#67).
        'Reply with one JSON object: {"answer": "…", "citations": [{"marker": 1, "passage": "P1", "location": "…", "quote": "…"}]}.',
        "- answer: the Answer, in Markdown. Cite every claim you draw from a Passage by putting a marker such as [^1] right after it. Add no list of sources.",
        `- citations: one record for each marker: ${RECORD}.`,
        UNBROKEN_QUOTE,
        NOT_COVERED,
      ].join("\n");
    case "none":
      return [
        added,
        documentNames(listed),
        "",
        "Passages found in the User's Documents for this Question:",
        passages,
        "",
        "Use them where they help. Don't write citation markers or a list of sources.",
        "If the Passages don't cover the Question, say so plainly, then answer from your own knowledge if you can.",
      ].join("\n");
  }
}

/**
 * For a model that can't call Tools, before its one search: rewriting the
 * Question into a search query that stands on its own, from the notebook
 * above it (see `searchQuery` in ./engine).
 */
export const SEARCH_QUERY_INSTRUCTIONS = [
  "You turn a Question from the User's notebook into one query for searching the User's Documents. The search sees only the query.",
  '- Use the notebook text above the Question to work out what the Question refers to, and name it in the query: replace words such as "it", "they", "this paper" or "the second one" with what they mean.',
  "- Keep the Question's own key words, in its language. If the Question already stands on its own, give it back unchanged.",
  "- Reply with the query alone, on one line. Don't answer the Question or explain.",
].join("\n");

/** The request to rewrite `question` into a search query, with the notebook text above it. */
export function searchQueryPrompt(earlier: string, question: string): string {
  return [
    "The notebook text above the Question:",
    "<notebook>",
    earlier,
    "</notebook>",
    "",
    `The Question: ${question}`,
  ].join("\n");
}

/** XML attribute text. */
const attribute = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

/**
 * A Skill's files besides SKILL.md, for the model to read when its
 * instructions point to them, and its scripts to run when `scripts` can.
 */
function skillFiles(files: readonly SkillFile[], scripts: boolean): string {
  const others = files.filter((file) => file.path !== "SKILL.md");
  if (others.length === 0) return "";
  return [
    "Its other files, which read_skill_file can read when the instructions point to them:",
    ...others.map((file) =>
      !file.script
        ? `- ${file.path}`
        : scripts
          ? `- ${file.path} (a script: run_skill_script runs it when the instructions say to)`
          : `- ${file.path} (a script: it can't be run here, but it can be read)`,
    ),
  ].join("\n");
}

/** A Skill's instructions, as `use_skill` gives them, or as a forced Skill is loaded up front. */
export function loadedSkillText(
  skill: LoadedSkill,
  { withFiles, scripts = false }: { withFiles: boolean; scripts?: boolean },
): string {
  return [
    `<skill name="${attribute(skill.name)}">`,
    skill.instructions,
    "</skill>",
    ...(withFiles ? [skillFiles(skill.files, scripts)] : []),
  ]
    .filter(Boolean)
    .join("\n");
}

const SCRIPTS =
  "When a Skill's instructions say to run one of its scripts, call run_skill_script with the Skill's name, the script's path and its arguments. The User approves each run first and may deny it: then carry on without it, don't run it again, and say what wasn't done.";

/**
 * What the Answer may do with Skills. The enabled Skills are listed by name
 * and description only, when the model can load them with `use_skill`; a
 * forced Skill's instructions are there in full. `scripts`: the model can run
 * Skills' scripts with `run_skill_script`.
 */
export function skillInstructions(
  listed: readonly SkillSummary[],
  forced: LoadedSkill | null,
  skillTools: boolean,
  scripts = false,
): string {
  const parts: string[] = [];
  if (forced) {
    parts.push(
      `For this Question the User chose the Skill "${forced.name}": follow its instructions.`,
      loadedSkillText(forced, { withFiles: skillTools, scripts: skillTools && scripts }),
    );
  }
  if (skillTools && listed.length > 0) {
    parts.push(
      [
        forced
          ? "Other Skills, each with a name and what it is for:"
          : "Skills give instructions for particular tasks. Each has a name and what it is for:",
        "<skills>",
        ...listed.map(
          (skill) => `<skill name="${attribute(skill.name)}">${skill.description}</skill>`,
        ),
        "</skills>",
        "When the Question is one a Skill is for, call use_skill with its name before answering, then follow what it loads.",
      ].join("\n"),
    );
  }
  if (skillTools && scripts) parts.push(SCRIPTS);
  return parts.join("\n\n");
}
