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

/** The system prompt for answering Questions, before anything about Documents. The same for every Question. */
export function answerInstructions(): string {
  return [
    "You answer Questions in IncarnaMind, a notebook where the User writes Notes and asks Questions in place.",
    "The messages are the User's Notes and Questions as user messages, your Answers as assistant messages. The last user message ends with the Question to answer now; use the rest as context.",
    "It may start with the notebook page the User asks beside, in <mind> (earlier Q&A in <question> and <answer>; […] where a long page was cut), then the text they selected in it, in <selection>: the Question is about that first, the rest of the page is background.",
    "",
    "- Answer in the language the Question is written in, even when the Notes or Documents are in another language.",
    "- If you don't know the answer, or it isn't in what you were given and you aren't sure of it, say so plainly instead of guessing.",
    "- Write Markdown: paragraphs, headings, lists, tables (one row per line, with a heading row), fenced code blocks with a language, and LaTeX math as $…$ inline or $$…$$ on lines of its own. Don't use HTML.",
  ].join("\n");
}

/**
 * The one line of the instructions that depends on the Question, for a
 * Question clearly written in one script. It goes last, after everything that
 * stays the same from one Question to the next, so a change of language
 * doesn't change the start of the prompt (a provider caches that: #343).
 */
export function languageInstruction(question: string): string {
  const language = scriptLanguage(question);
  return language ? `- The Question is written in ${language}, so answer in ${language}.` : "";
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
 * Find sources' Tool, when offered (#296): when to call it, and that what it
 * finds isn't the User's Documents, so it is never cited with a marker.
 */
export function sourcesInstructions(offered: boolean): string {
  if (!offered) return "";
  return [
    "You can also call find_sources to search scholarly databases for published papers outside the User's Documents. Call it only when the Question asks you to find papers, articles, studies or other sources.",
    "- Name what it found by title, first author and year, and say which are open access; never put a citation marker on them. The User adds the ones they want from the search's results, under your Answer.",
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

/**
 * Calculated figures (ADR-0021): a model's own arithmetic looks exactly like a
 * right answer when it is wrong, so it is steered to `calculate` for every
 * number it works out, simple ones included.
 */
const CALCULATE = [
  'Never work out a number yourself, not even a simple one: for any number the Answer gives that isn\'t written in a Passage (a difference, sum, average, ratio, share, growth rate, CAGR, change in percentage points, conversion between 万 and 亿 or millions and billions, or which of two is larger), call calculate and write its marker in the number\'s place, with no number or unit beside it: "grew [=1] in 2025", not "grew 20.4% [=1]".',
  '- Name each input\'s number as it is written in its quote, labelled with what it is and its period; give a table heading\'s unit line ("单位：万元", "in millions") as unitQuote.',
  "- A number copied as it is from a Passage is cited with [^n]; calculate is only for numbers you work out.",
].join("\n");

/** Structured output's `calculations`, for a model offered them: what `CALCULATE` is to the Tool loop. */
const CALCULATIONS =
  '- calculations: one for each number the Answer works out from numbers in Passages (a difference, sum, average, ratio, share, growth rate, a change in percentage points, a conversion between 万 and 亿 or millions and billions), with [=1] in the answer where the number goes, never the number itself: its marker; its inputs, each a name for the formula, a label with its year or period, the marker of the Citation whose quote holds the number, and the number as it is written there; and a formula over the names in Excel\'s syntax, such as (b - a) / a, with "as": "percent" for a growth rate or share. When a number\'s unit or scale is only in a table\'s heading ("单位：万元"), give that line as unitQuote.';

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
 * `listed` covers only those. `calculate`: the model is offered the
 * `calculate` Tool (or `calculations` in its JSON), and steered to it.
 */
export function documentInstructions(
  mode: CitationSupport | "no-documents",
  listed: ListedDocuments,
  passages = "",
  scoped = false,
  calculate = false,
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
        "- In the Answer, put the marker as [^1] right after each claim it supports, in a table's cell too. Use only markers you recorded, and add no list of sources.",
        "- Write nothing before your Tool calls: only the Answer, after them.",
        NOT_COVERED,
        ...(calculate ? ["", CALCULATE] : []),
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
        calculate
          ? 'Reply with one JSON object: {"answer": "…", "citations": [{"marker": 1, "passage": "P1", "location": "…", "quote": "…"}], "calculations": [{"marker": 1, "label": "…", "inputs": [{"name": "a", "label": "…", "citation": 1, "number": "…"}], "formula": "…"}]}.'
          : 'Reply with one JSON object: {"answer": "…", "citations": [{"marker": 1, "passage": "P1", "location": "…", "quote": "…"}]}.',
        "- answer: the Answer, in Markdown. Cite every claim you draw from a Passage by putting a marker such as [^1] right after it, in a table's cell too. Add no list of sources.",
        `- citations: one record for each marker: ${RECORD}.`,
        UNBROKEN_QUOTE,
        ...(calculate ? [CALCULATIONS] : []),
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

/**
 * For a model that can't call Tools (or a small local one), before its second
 * search: the query translated into a language some of the Documents are in,
 * as the search Tool tells a model in the loop to search (see
 * `searchLanguage` in ./engine).
 */
export const TRANSLATE_QUERY_INSTRUCTIONS = [
  "You translate a query for searching the User's Documents into the language some of them are written in: search finds Passages best in their own language.",
  "- Keep names, numbers and technical terms as Documents in that language would write them.",
  "- Reply with the translated query alone, on one line. Don't answer it or explain.",
].join("\n");

/** The request to translate a search query into `language`. */
export function translateQueryPrompt(query: string, language: string): string {
  return `Translate this query into ${language}:\n${query}`;
}

/**
 * For a local model that answered with structured output, once its Answer is
 * written: the one request for exact quotes, for the records whose quotes the
 * check didn't find (see ./quoteRetry).
 */
export const QUOTE_RETRY_INSTRUCTIONS = [
  "You fix the quotes of an Answer's Citations. A program looks for each quote in its Passage's text, word for word, and didn't find these.",
  "- For each marker, copy from its Passage's text the words that support its sentence: one unbroken stretch, character for character, no longer than a sentence. Don't reword, translate, join sentences, change characters or leave words out.",
  "- If the Passage's text has nothing that supports the sentence, give an empty quote.",
  'Reply with one JSON object: {"quotes": [{"marker": 1, "quote": "…"}]}, with an item for each marker.',
].join("\n");

/**
 * For a local model that answered with structured output, once its Answer is
 * written: the one request to fix the figures that couldn't be worked out
 * (see ./figureRetry).
 */
export const FIGURE_RETRY_INSTRUCTIONS = [
  "You fix the calculated figures of an Answer. A program reads each input's number from its quote and does the arithmetic, and couldn't for these figures, for the reason given with each.",
  "- Give each figure again, with its marker, label, inputs and formula, fixed: copy each quote and number character for character from its Passage's text, and give unitQuote or reading where the reason asks for one.",
  "- If a Passage's text has nothing to work a figure out from, leave that figure out.",
  'Reply with one JSON object: {"calculations": [{"marker": 1, "label": "…", "inputs": [{"name": "a", "label": "…", "passage": "P1", "quote": "…", "number": "…"}], "formula": "…"}]}.',
].join("\n");

/** A figure that couldn't be worked out, as the request to fix it shows it. */
export interface FigureToRetry {
  marker: number;
  /** The figure as the model gave it, as JSON. */
  given: string;
  problems: readonly string[];
  /** The Passages its inputs name, each with its text or the part of it nearest the figure. */
  passages: readonly { id: string; text: string }[];
}

/** The request to fix figures: each one as given, why it couldn't be worked out, and its Passages. */
export function figureRetryPrompt(figures: readonly FigureToRetry[]): string {
  return figures
    .map((figure) =>
      [
        `[=${figure.marker}] as you gave it: ${figure.given}`,
        `Why it couldn't be worked out: ${figure.problems.join(" ")}`,
        ...figure.passages.map(
          (passage) => `<passage id="${attribute(passage.id)}">\n${passage.text}\n</passage>`,
        ),
      ].join("\n"),
    )
    .join("\n\n");
}

/** A record whose quote wasn't found, as the request for exact quotes shows it. */
export interface QuoteToRetry {
  marker: number;
  /** The Answer's sentence its marker is in. */
  sentence: string;
  /** The quote the model gave. */
  quote: string;
  /** Its Passage's id, e.g. "P3". */
  passage: string;
  /** Its Passage's text, or the part of it nearest the quote. */
  text: string;
}

/** The request for exact quotes: each record's marker, sentence, quote and Passage text. */
export function quoteRetryPrompt(records: readonly QuoteToRetry[]): string {
  return records
    .map((record) =>
      [
        `[^${record.marker}] is on this sentence of the Answer: ${record.sentence}`,
        `Its quote, not found word for word: ${record.quote ? `"${record.quote}"` : "(none)"}`,
        `<passage id="${attribute(record.passage)}">`,
        record.text,
        "</passage>",
      ].join("\n"),
    )
    .join("\n\n");
}

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
