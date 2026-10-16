/**
 * A Question about the User's notes or what they selected ("notes", "selected",
 * 笔记, 选中), asked beside a Mind, gets an Answer that starts by saying what it
 * read of the Mind: its name and first line, and the selected text. One about
 * what someone changed ("change", 修订) lists the changes it read marked in
 * the Mind, with who made each (#734).
 *
 * In the assistant, with the Minds' Tools (#202, `mindsReply`): a request to
 * shorten (缩短, 改短), or to "replace X with Y", calls edit_mind with the request as its instruction,
 * which Edit with AI's scripted model carries out as Shorten, and so does one
 * "with a table" (加上表格), which it carries out by adding one (#961); one
 * for a new note (新笔记, 新建) calls create_mind, titled from what follows "called"
 * (叫), its content citing the Documents when there are any; and "add X to
 * <Mind>" (把 X 添加到 <Mind>) calls add_to_mind with X, and "add X" (添加 X)
 * adds X to the Mind in focus; "add X to <Mind> and make a new note called Y"
 * calls both. Each then replies with the first sentence of what its calls
 * came to; add_to_mind's and create_mind's input streams in word by word,
 * slowly for a Question asked "slowly", as does edit_mind's edit. "Write three
 * paragraphs" is `paragraphsReply`'s.
 */

import { answerReply, CITED_ANSWER } from "./answers";
import { ASSISTANT_REPLACES } from "./appliedEdits";
import { paragraphsReply } from "./paragraphs";
import {
  lastQuestion,
  lastResult,
  lastUserText,
  type Prompt,
  type Reply,
  SHORTEN,
  toolResults,
  userText,
} from "./prompt";

/** A Question about the User's notes or what they selected in them. */
const ABOUT_THE_MIND = /\bnotes?\b|\bselect(?:ed|ion)\b|笔记|选中/i;

/** A Question about what someone changed in the Mind. */
const ABOUT_THE_CHANGES = /\bchang(?:e|es|ed)\b|修订/i;

/** What the Answer says it read of the Mind a Question about it was asked beside, if any. */
export function mindRead(prompt: Prompt): string | null {
  const question = lastQuestion(prompt);
  const aboutChanges = ABOUT_THE_CHANGES.test(question);
  if (!ABOUT_THE_MIND.test(question) && !aboutChanges) return null;
  // The Mind starts the first user message, and the selection the last (#564).
  const text = lastUserText(prompt);
  const mind = /<mind name="([^"]*)">\n([\s\S]*?)<\/mind>/.exec(
    userText(prompt.find((message) => message.role === "user")),
  );
  if (!mind) return null;
  if (aboutChanges) {
    const changes = [...(mind[2] ?? "").matchAll(/\[(?:inserted|deleted) by [^\]]*\]/g)];
    return changes.length > 0
      ? `Changes in “${mind[1]}”: ${changes.map((each) => each[0]).join("; ")}`
      : `No changes are marked in “${mind[1]}”.`;
  }
  const first = (mind[2] ?? "")
    .split("\n")
    .map((line) => line.replace(/^#+\s*/, "").trim())
    .find((line) => line !== "" && line !== "[…]")
    // A short Answer, however long the line.
    ?.slice(0, 80);
  const selection = /<selection>\n([\s\S]*?)\n<\/selection>/.exec(text)?.[1];
  return [
    first ? `Read “${mind[1]}”, which starts: ${first}` : `Read “${mind[1]}”, which is empty.`,
    ...(selection ? [`About what you selected: ${selection}`] : []),
  ].join("\n\n");
}

/** A request to change the Mind so it has a table: "…, with a table" (加上表格). */
const WITH_A_TABLE = /\bwith a table\b|加上?表格/i;

/** A request for a new note, and its title after "called" (叫). */
const NEW_NOTE = /\bnew (?:note|mind)\b|新笔记|新建/i;
const CALLED = /\b(?:called|named|titled)\s+[“"]?(.+?)[”"]?[.。]?$|叫[“"]?(.+?)[”"]?的/iu;
/** A request to add something to a Mind by name: what, and to which. */
const ADD_TO =
  /^add\s+(.+?)\s+to\s+[“"]?(.+?)[”"]?[.。]?$|^把\s*(.+?)\s*添加到\s*[“"]?(.+?)[”"]?[。]?$/iu;
/** A request to add something to the Mind in focus: "add X" (添加 X). */
const ADD_HERE = /^add\s+(.+?)[.。]?$|^添加\s*(.+?)[。]?$/iu;
/** Both at once: "add X to <Mind> and make a new note called Y". */
const ADD_AND_NEW =
  /^add\s+(.+?)\s+to\s+[“"]?(.+?)[”"]?\s+and\s+(?:make|create)\s+a\s+new\s+note\s+called\s+[“"]?(.+?)[”"]?[.。]?$/iu;

/** The first sentence of what a call came to, as the scripted assistant's reply. */
const firstSentence = (result: string) => /^[^.。]*[.。]?/u.exec(result)?.[0] ?? result;

/**
 * The scripted assistant's choice with the Minds' Tools (see the top of this
 * file): a Tool call, the reply once it has returned, or null to answer.
 */
export function mindsReply(prompt: Prompt, asked: string, names: readonly string[]): Reply | null {
  if (!names.includes("edit_mind")) return null;
  // Asked "slowly", only its pace changes.
  const question = asked.replace(/\s+slowly$/i, "");
  const paragraphs = paragraphsReply(prompt, question);
  if (paragraphs) return paragraphs;
  const both = ADD_AND_NEW.exec(question);
  if (both) {
    if (lastResult(prompt, "add_to_mind") === undefined) {
      return { tool: "add_to_mind", input: { mind: both[2], content: both[1] } };
    }
    const made = lastResult(prompt, "create_mind");
    if (made === undefined) {
      const content = "## Summary\n\nWritten by the scripted assistant.";
      return { tool: "create_mind", input: { title: both[3], content } };
    }
    return {
      text: `${firstSentence(lastResult(prompt, "add_to_mind") ?? "")} ${firstSentence(made)}`,
    };
  }
  if (SHORTEN.test(question) || ASSISTANT_REPLACES.test(question) || WITH_A_TABLE.test(question)) {
    const result = lastResult(prompt, "edit_mind");
    // Asked "slowly", the edit is written slowly too.
    if (result === undefined) return { tool: "edit_mind", input: { instruction: asked } };
    return { text: firstSentence(result) };
  }
  const add = ADD_TO.exec(question);
  if (add) {
    const result = lastResult(prompt, "add_to_mind");
    if (result === undefined) {
      return {
        tool: "add_to_mind",
        input: { mind: add[2] ?? add[4], content: add[1] ?? add[3] },
      };
    }
    return { text: firstSentence(result) };
  }
  const here = ADD_HERE.exec(question);
  if (here) {
    const result = lastResult(prompt, "add_to_mind");
    if (result === undefined)
      return { tool: "add_to_mind", input: { content: here[1] ?? here[2] } };
    return { text: firstSentence(result) };
  }
  if (!NEW_NOTE.test(question)) return null;
  const made = lastResult(prompt, "create_mind");
  if (made !== undefined) return { text: firstSentence(made) };
  const cited = toolResults(prompt, "cite").length > 0;
  if (names.includes("search_documents") && !cited) {
    const reply = answerReply(prompt, names);
    if ("tool" in reply) return reply;
  }
  const called = CALLED.exec(question);
  const title = called?.[1] ?? called?.[2] ?? "Scripted notes";
  const content = `## Summary\n\n${cited ? CITED_ANSWER : "Written by the scripted assistant."}`;
  return { tool: "create_mind", input: { title, content } };
}
