/**
 * Putting a Question asked in the composer into the thread (#201): at the end
 * of the chat the thread shows, starting one when there is none, in the
 * Mind's Yjs document, where the core then writes its Answer right after it.
 * The note is left alone. Written as the editor's binding writes Blocks
 * (see src/core/answers/blocks.ts), so the thread's editor shows it as is.
 */
import * as Y from "yjs";
import { QUESTION_BLOCK } from "../../../core/api";
import { chatBlocks, chatsOf, createChat } from "../../../shared/mindChats";
import { type QuestionToAsk, questionAttributes } from "./composerAsk";

/** A Question put in a chat, and whether asking it started the chat. */
export interface PlacedInThread {
  chatId: string;
  questionId: string;
  startedChat: boolean;
}

/** The Question's Block: its text, lines joined by line breaks, and its attributes (none null). */
function questionElement(question: QuestionToAsk): Y.XmlElement {
  const element = new Y.XmlElement(QUESTION_BLOCK);
  for (const [key, value] of Object.entries(questionAttributes(question))) {
    // biome-ignore lint/suspicious/noExplicitAny: Yjs types attribute values loosely.
    if (value !== null && value !== undefined) element.setAttribute(key, value as any);
  }
  const children: (Y.XmlText | Y.XmlElement)[] = [];
  question.text.split(/\r\n|\r|\n/).forEach((line, index) => {
    if (index > 0) children.push(new Y.XmlElement("hardBreak"));
    if (line) children.push(new Y.XmlText(line));
  });
  element.insert(0, children);
  return element;
}

/**
 * Appends the Question to the chat `chatId`, or to a new chat when it is
 * null or no longer there, and returns where it went.
 */
export function placeInThread(
  doc: Y.Doc,
  chatId: string | null,
  question: QuestionToAsk,
): PlacedInThread {
  const existing = chatId ? chatBlocks(doc, chatId) : null;
  const id = existing && chatId ? chatId : crypto.randomUUID();
  doc.transact(() => {
    const blocks = existing ?? createChat(doc, id, new Date().toISOString());
    blocks.insert(blocks.length, [questionElement(question)]);
  });
  return { chatId: id, questionId: question.id, startedChat: existing === null };
}

/**
 * Takes a Question that couldn't be asked back out of the thread (its words go
 * back to the composer), and the chat with it if asking it started the chat.
 */
export function takeBackFromThread(doc: Y.Doc, placed: PlacedInThread): void {
  const blocks = chatBlocks(doc, placed.chatId);
  if (!blocks) return;
  doc.transact(() => {
    for (let index = blocks.length - 1; index >= 0; index--) {
      const block = blocks.get(index);
      if (block instanceof Y.XmlElement && block.getAttribute("id") === placed.questionId) {
        blocks.delete(index, 1);
      }
    }
    if (placed.startedChat && blocks.length === 0) chatsOf(doc).delete(placed.chatId);
  });
}
