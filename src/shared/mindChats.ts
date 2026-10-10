/**
 * A Mind's chats (#201, DESIGN.md D124): the Questions asked in the thread
 * above its composer and their Answers, kept in the Mind's Yjs document
 * beside its Blocks (`MIND_CHATS_FIELD`), so they go wherever the Mind goes
 * and its note changes only when the User adds an Answer to it.
 *
 * Each chat holds its Questions and Answers as a Y.XmlFragment in the shape
 * the Mind's own Blocks have, so the core writes, cites and checks Answers
 * there as it does anywhere, and the thread draws them with the editor's own
 * node views. Shared by the core, the renderer and the tests.
 */
import * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  MIND_CHATS_FIELD,
  MIND_CONTENT_FIELD,
  QUESTION_BLOCK,
} from "../core/api";

const CREATED_AT = "createdAt";
const BLOCKS = "blocks";

/** A chat as the thread's history lists it. */
export interface ChatSummary {
  id: string;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** Its first Question's text, on one line; "" while it has none. */
  title: string;
  /** How many Questions it holds. */
  questions: number;
}

/** The Mind's chats, by id. */
export const chatsOf = (doc: Y.Doc): Y.Map<unknown> => doc.getMap(MIND_CHATS_FIELD);

/** A chat's Questions and Answers, or null when the Mind has no such chat. */
export function chatBlocks(doc: Y.Doc, chatId: string): Y.XmlFragment | null {
  const chat = chatsOf(doc).get(chatId);
  if (!(chat instanceof Y.Map)) return null;
  const blocks = chat.get(BLOCKS);
  return blocks instanceof Y.XmlFragment ? blocks : null;
}

/** The Mind's own Blocks, then every chat's: everywhere its Citations can be. */
export const citingFragments = (doc: Y.Doc): Y.XmlFragment[] => [
  doc.getXmlFragment(MIND_CONTENT_FIELD),
  ...chatFragments(doc),
];

/** Every chat's Questions and Answers, in no order: e.g. to find what their Citations quote. */
export function chatFragments(doc: Y.Doc): Y.XmlFragment[] {
  const fragments: Y.XmlFragment[] = [];
  chatsOf(doc).forEach((chat) => {
    const blocks = chat instanceof Y.Map ? chat.get(BLOCKS) : null;
    if (blocks instanceof Y.XmlFragment) fragments.push(blocks);
  });
  return fragments;
}

/**
 * Starts a chat in the Mind with this id (a random UUID), or returns its
 * Blocks if it is there already.
 */
export function createChat(doc: Y.Doc, chatId: string, createdAt: string): Y.XmlFragment {
  const existing = chatBlocks(doc, chatId);
  if (existing) return existing;
  let blocks: Y.XmlFragment | null = null;
  doc.transact(() => {
    const chat = new Y.Map<unknown>();
    chatsOf(doc).set(chatId, chat);
    chat.set(CREATED_AT, createdAt);
    blocks = new Y.XmlFragment();
    chat.set(BLOCKS, blocks);
  });
  return blocks as unknown as Y.XmlFragment;
}

/** A Question's text, on one line. */
const questionText = (element: Y.XmlElement): string =>
  element
    .toArray()
    .map((child) =>
      child instanceof Y.XmlText
        ? (child.toDelta() as { insert: unknown }[])
            .map((run) => (typeof run.insert === "string" ? run.insert : ""))
            .join("")
        : // A line break.
          " ",
    )
    .join("")
    .replace(/\s+/g, " ")
    .trim();

/** What the thread's history shows of a chat. */
function summaryOf(id: string, chat: Y.Map<unknown>): ChatSummary | null {
  const blocks = chat.get(BLOCKS);
  if (!(blocks instanceof Y.XmlFragment)) return null;
  const createdAt = chat.get(CREATED_AT);
  const questions = blocks
    .toArray()
    .filter(
      (block): block is Y.XmlElement =>
        block instanceof Y.XmlElement && block.nodeName === QUESTION_BLOCK,
    );
  const first = questions[0];
  return {
    id,
    createdAt: typeof createdAt === "string" ? createdAt : "",
    title: first ? questionText(first) : "",
    questions: questions.length,
  };
}

/** The Mind's chats, the newest first. */
export function listChats(doc: Y.Doc): ChatSummary[] {
  const chats: ChatSummary[] = [];
  chatsOf(doc).forEach((chat, id) => {
    const summary = chat instanceof Y.Map ? summaryOf(id, chat) : null;
    if (summary) chats.push(summary);
  });
  return chats.sort((a, b) =>
    a.createdAt === b.createdAt ? a.id.localeCompare(b.id) : b.createdAt.localeCompare(a.createdAt),
  );
}

/** A chat's summary, or null when the Mind has no such chat. */
export function chatSummary(doc: Y.Doc, chatId: string): ChatSummary | null {
  const chat = chatsOf(doc).get(chatId);
  return chat instanceof Y.Map ? summaryOf(chatId, chat) : null;
}

/** The chat whose top-level Blocks include this one (a Question or an Answer), if any. */
export function chatContaining(doc: Y.Doc, blockId: string): string | null {
  let found: string | null = null;
  chatsOf(doc).forEach((chat, id) => {
    if (found || !(chat instanceof Y.Map)) return;
    const blocks = chat.get(BLOCKS);
    if (!(blocks instanceof Y.XmlFragment)) return;
    for (const block of blocks.toArray()) {
      if (block instanceof Y.XmlElement && block.getAttribute(BLOCK_ID_ATTRIBUTE) === blockId) {
        found = id;
        return;
      }
    }
  });
  return found;
}

/** Calls `listener` whenever the Mind's chats may have changed, here or from elsewhere. Returns how to stop. */
export function observeChats(doc: Y.Doc, listener: () => void): () => void {
  const chats = chatsOf(doc);
  const onChange = () => listener();
  chats.observeDeep(onChange);
  return () => chats.unobserveDeep(onChange);
}
