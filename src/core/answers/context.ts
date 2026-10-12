/**
 * Question context (ADR-0007): what an Answer is generated from. In a Mind,
 * every Block above the Question, in order, except Notes the User switched off:
 * Notes and earlier Questions as the User's words, earlier Answers as the
 * model's. In a conversation, its earlier Questions and Answers the same way,
 * and the Mind in focus when the Question was asked, read the same way (#544),
 * with what the User selected in it first. When that is over the token
 * budget, the oldest content goes first; a long Mind keeps its headings, the
 * selection, the part around the cursor and its newest Blocks. Where the
 * third-party text in it came from is read beside it (#78): an Answer to such
 * a context is written by a Run that counts as having read that text.
 */
import { createHash } from "node:crypto";
import * as Y from "yjs";
import { parseThirdPartySource } from "../../shared/textOrigins";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  type ConversationMessage,
  INCLUDE_IN_CONTEXT_ATTRIBUTE,
  MIND_TEXT_ORIGINS_FIELD,
  type MindFocus,
  QUESTION_BLOCK,
  type ThirdPartySource,
} from "../api";
import { attribute, contentMarkdown, textAttribute, toMarkdown } from "./blocks";
import { answerMarkdown } from "./contextText";
import type { AnswerMessage } from "./engine";

/** About 12k tokens, the old app's budget for what precedes a Question. */
const QUESTION_CONTEXT_TOKEN_BUDGET = 12_000;

/**
 * About 2k tokens: how much of the Question context a Question is rewritten
 * into a search query from, the old backend's budget for its condense step.
 */
const SEARCH_QUERY_CONTEXT_TOKEN_BUDGET = 2_000;

/** A Block's share of the budget that is too small to be worth keeping the tail of. */
const MIN_TAIL_TOKENS = 64;

interface Piece {
  role: AnswerMessage["role"];
  text: string;
  /** Where the third-party text in it came from (#78); absent: none. */
  origins?: readonly ThirdPartySource[];
}

/** Characters written with one token each, roughly: CJK, kana, Hangul and full-width forms. */
function isWideChar(code: number): boolean {
  return (
    (code >= 0x2e80 && code <= 0x9fff) ||
    (code >= 0xac00 && code <= 0xd7af) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0xff00 && code <= 0xffef) ||
    (code >= 0x20000 && code <= 0x2fa1f)
  );
}

/**
 * An approximate token count, without a tokenizer: one per CJK character and
 * one per four other characters, close enough across providers for a budget.
 */
export function estimateTokens(text: string): number {
  let wide = 0;
  let other = 0;
  for (const char of text) {
    if (isWideChar(char.codePointAt(0) ?? 0)) wide++;
    else other++;
  }
  return wide + Math.ceil(other / 4);
}

/** The start of `text` that is about `tokens` estimated tokens long. */
export function head(text: string, tokens: number): string {
  let used = 0;
  let end = 0;
  for (const char of text) {
    const cost = estimateTokens(char);
    if (used + cost > tokens) break;
    used += cost;
    end += char.length;
  }
  return text.slice(0, end);
}

/** The end of `text` that fits in about `tokens` tokens, marked as cut. */
function tail(text: string, tokens: number): string {
  const chars = [...text];
  let used = 0;
  let start = chars.length;
  while (start > 0) {
    const code = chars[start - 1]?.codePointAt(0) ?? 0;
    const cost = isWideChar(code) ? 1 : 0.25;
    if (used + cost > tokens) break;
    used += cost;
    start--;
  }
  return `…${chars.slice(start).join("").trimStart()}`;
}

/** What a Block above the Question contributes, if anything. */
function pieceOf(block: Y.XmlElement): Piece | null {
  if (block.nodeName === QUESTION_BLOCK) {
    const text = toMarkdown(block);
    return text ? { role: "user", text } : null;
  }
  if (block.nodeName === ANSWER_BLOCK) {
    if (textAttribute(block, "status") === "failed") return null;
    const text = contentMarkdown(block);
    return text ? { role: "assistant", text } : null;
  }
  // A Note.
  if (attribute(block, INCLUDE_IN_CONTEXT_ATTRIBUTE) === false) return null;
  const text = toMarkdown(block);
  return text ? { role: "user", text } : null;
}

export interface QuestionContext {
  /** In order; the last one is the User's, ending with the Question. */
  messages: AnswerMessage[];
  /** The Question's own text. */
  question: string;
}

/** A Question context, and where the third-party text it holds came from. */
export interface QuestionContextWithOrigins {
  context: QuestionContext;
  /**
   * Where the third-party text in what the context keeps came from (#78),
   * each once, the newest first: the Blocks (or earlier Answers) it keeps, in
   * whole or in part, that someone else's text went into. A Run that answers
   * it counts as having read that text. Empty: none.
   */
  origins: ThirdPartySource[];
}

/**
 * The third-party sources recorded for a Block and the Blocks nested in it
 * (a quote added inside a list, say), each once, in the order found.
 */
function originsIn(block: Y.XmlElement, origins: Y.Map<unknown>): ThirdPartySource[] {
  const found = new Set<ThirdPartySource>();
  const visit = (element: Y.XmlElement) => {
    const id = textAttribute(element, BLOCK_ID_ATTRIBUTE);
    const source = id === null ? null : parseThirdPartySource(origins.get(id));
    if (source) found.add(source);
    for (const child of element.toArray()) if (child instanceof Y.XmlElement) visit(child);
  };
  visit(block);
  return [...found];
}

/** Where the third-party text of kept pieces (oldest first) came from: each source once, the newest first. */
const originsOf = (kept: readonly Piece[]): ThirdPartySource[] => [
  ...new Set(kept.toReversed().flatMap((piece) => piece.origins ?? [])),
];

/**
 * The Question context of the Question at `index` among the Mind's top-level
 * Blocks. The Question itself is always kept, even alone over the budget.
 */
export function buildQuestionContext(
  blocks: Y.XmlFragment,
  index: number,
  budget = QUESTION_CONTEXT_TOKEN_BUDGET,
): QuestionContext {
  return questionContextOf(blocks, index, budget).context;
}

/**
 * `buildQuestionContext`, with where the third-party text in the Blocks it
 * keeps came from, as the Mind's `textOrigins` records it.
 */
export function questionContextOf(
  blocks: Y.XmlFragment,
  index: number,
  budget = QUESTION_CONTEXT_TOKEN_BUDGET,
): QuestionContextWithOrigins {
  const children = blocks.toArray();
  const question = children[index];
  if (!(question instanceof Y.XmlElement)) throw new Error("No Question at that index.");
  const questionText = toMarkdown(question);
  const origins = blocks.doc?.getMap<unknown>(MIND_TEXT_ORIGINS_FIELD);
  const recorded = origins && origins.size > 0 ? origins : null;
  /** What the Blocks above contribute, newest first, read only as far as the budget goes. */
  const newestFirst = function* (): Generator<Piece> {
    for (let at = index - 1; at >= 0; at--) {
      const block = children[at];
      const piece = block instanceof Y.XmlElement ? pieceOf(block) : null;
      if (!piece) continue;
      yield recorded && block instanceof Y.XmlElement
        ? { ...piece, origins: originsIn(block, recorded) }
        : piece;
    }
  };
  const kept = keptWithinBudget(newestFirst(), budget - estimateTokens(questionText));
  return {
    context: { messages: asMessages(kept, questionText), question: questionText },
    origins: originsOf(kept),
  };
}

/** A Block of a Mind as a Question in the assistant reads it. */
interface ReadBlock {
  text: string;
  /** A heading: a long Mind keeps its headings first. */
  heading: boolean;
  origins: readonly ThirdPartySource[];
}

/** What of a Mind a request holds (see `MindReading`). */
export interface MindSection {
  /** The Mind, then what the User selected in it, as their message starts; "" when none of it fits. */
  text: string;
  /** Some of the Mind was left out to fit. */
  partial: boolean;
  /** Where the third-party text in what it keeps came from, each once. */
  origins: ThirdPartySource[];
}

/**
 * The Mind in focus, read for a Question in the assistant (#544), as plain
 * text that no longer needs its document.
 */
export interface MindReading {
  /** Its version as read: a fingerprint of its text. */
  version: string;
  /**
   * What of it fits in about `tokens` tokens (`estimateTokens`); by default, as
   * much as a Question context in a Mind holds.
   */
  section(tokens?: number): MindSection;
}

/** Where a Block was left out of a cut Mind, as the model reads it. */
const LEFT_OUT = "[…]";
/** What a kept Block adds besides its text, at most: the blank line before it, and a `LEFT_OUT` before that. */
const BLOCK_EXTRA_TOKENS = estimateTokens(`${LEFT_OUT}\n\n\n\n`);

const xmlAttribute = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

/** A Block as the Mind's text reads it: a Note as it is, and an earlier Question or Answer marked as one. */
function readText(block: Y.XmlElement, piece: Piece): string {
  if (block.nodeName === QUESTION_BLOCK) return `<question>${piece.text}</question>`;
  if (block.nodeName === ANSWER_BLOCK) return `<answer>\n${piece.text}\n</answer>`;
  return piece.text;
}

/**
 * Reads a Mind for a Question asked beside it in the assistant (#544), as a
 * Question in the Mind reads the Blocks above it (`pieceOf`): Notes the User
 * excluded stay out, earlier Answers keep their Citations as plain references,
 * and Proposals still waiting aren't read. `focus` is where the User is in it:
 * the text they selected is read first, unless it is in a Note they excluded.
 */
export function readMind(
  blocks: Y.XmlFragment,
  name: string,
  focus: MindFocus | null,
): MindReading {
  const recordedOrigins = blocks.doc?.getMap<unknown>(MIND_TEXT_ORIGINS_FIELD);
  const recorded = recordedOrigins && recordedOrigins.size > 0 ? recordedOrigins : null;
  const read: ReadBlock[] = [];
  /**
   * Each top-level Block by id: where it is in `read` (a Block that adds nothing, such as
   * an empty line, is where the one before it is), and whether the User excluded it.
   */
  const byId = new Map<string, { index: number; excluded: boolean }>();
  for (const block of blocks.toArray()) {
    if (!(block instanceof Y.XmlElement)) continue;
    const piece = pieceOf(block);
    const id = textAttribute(block, BLOCK_ID_ATTRIBUTE);
    if (id) {
      byId.set(id, {
        index: piece ? read.length : read.length - 1,
        excluded: attribute(block, INCLUDE_IN_CONTEXT_ATTRIBUTE) === false,
      });
    }
    if (!piece) continue;
    read.push({
      text: readText(block, piece),
      heading: block.nodeName === "heading",
      origins: recorded ? originsIn(block, recorded) : [],
    });
  }
  const placed = (focus?.blockIds ?? []).map((id) => byId.get(id));
  const focused = [
    ...new Set(placed.flatMap((each) => (each && each.index >= 0 ? [each.index] : []))),
  ];
  const selection =
    focus && placed.length > 0 && placed.every((each) => each !== undefined && !each.excluded)
      ? focus.selection.trim()
      : "";
  const version = createHash("sha256")
    .update(read.map((block) => block.text).join("\n\n"))
    .digest("base64url")
    .slice(0, 22);
  return {
    version,
    section: (tokens = QUESTION_CONTEXT_TOKEN_BUDGET) =>
      cutMind({ name, read, focused, selection }, tokens),
  };
}

/**
 * Indices around the cursor: the focused Blocks first, then their neighbours,
 * one before and one after in turn, outwards.
 */
function* aroundCursor(focused: readonly number[], count: number): Generator<number> {
  if (focused.length === 0) return;
  yield* focused;
  let before = Math.min(...focused) - 1;
  let after = Math.max(...focused) + 1;
  while (before >= 0 || after < count) {
    if (after < count) yield after++;
    if (before >= 0) yield before--;
  }
}

/**
 * What of a Mind fits in about `tokens` tokens: all of it when it fits. Else
 * the selection first, then its headings, then the Blocks around the cursor
 * (up to half of what is left), then its newest Blocks, the last one's end if
 * it doesn't fit whole, then more around the cursor. Kept Blocks stay in
 * order, with `LEFT_OUT` where Blocks were left out.
 */
function cutMind(
  mind: { name: string; read: readonly ReadBlock[]; focused: readonly number[]; selection: string },
  tokens: number,
): MindSection {
  const { read, focused, selection } = mind;
  const wrap = (body: string) =>
    `<mind name="${xmlAttribute(mind.name)}">\n${body}${body ? "\n" : ""}</mind>`;
  const selected = (text: string) => (text ? `\n\n<selection>\n${text}\n</selection>` : "");
  const focusOrigins = (kept: boolean) =>
    kept ? focused.flatMap((index) => read[index]?.origins ?? []) : [];
  const originsOf = (indices: Iterable<number>, withSelection: boolean) => [
    ...new Set([
      ...[...indices].flatMap((index) => read[index]?.origins ?? []),
      ...focusOrigins(withSelection),
    ]),
  ];

  const whole = wrap(read.map((block) => block.text).join("\n\n")) + selected(selection);
  if (estimateTokens(whole) <= tokens) {
    return { text: whole, partial: false, origins: originsOf(read.keys(), selection !== "") };
  }

  let left = tokens - estimateTokens(wrap(LEFT_OUT)) - BLOCK_EXTRA_TOKENS;
  if (left <= 0) return { text: "", partial: true, origins: [] };
  let selectionText = "";
  if (selection) {
    const room = left - estimateTokens(selected("…"));
    selectionText =
      estimateTokens(selected(selection)) <= left
        ? selection
        : room > 0
          ? `${head(selection, room).trimEnd()}…`
          : "";
    left -= estimateTokens(selected(selectionText));
  }

  const kept = new Map<number, string>();
  const cost = (index: number) => estimateTokens(read[index]?.text ?? "") + BLOCK_EXTRA_TOKENS;
  const take = (index: number, room = left): boolean => {
    if (kept.has(index)) return true;
    const needed = cost(index);
    if (needed > room) return false;
    kept.set(index, read[index]?.text ?? "");
    left -= needed;
    return true;
  };
  const takeAround = (share: number) => {
    let room = share;
    for (const index of aroundCursor(focused, read.length)) {
      if (kept.has(index)) continue;
      const needed = cost(index);
      if (!take(index, Math.min(room, left))) return;
      room -= needed;
    }
  };

  read.forEach((block, index) => {
    if (block.heading) take(index);
  });
  takeAround(Math.ceil(left / 2));
  for (let index = read.length - 1; index >= 0; index--) {
    if (take(index)) continue;
    // Less one for the "…" `tail` marks the cut with.
    const room = left - BLOCK_EXTRA_TOKENS - 1;
    if (room >= MIN_TAIL_TOKENS) {
      kept.set(index, tail(read[index]?.text ?? "", room));
      left = 0;
    }
    break;
  }
  takeAround(left);

  const order = [...kept.keys()].sort((a, b) => a - b);
  const parts: string[] = [];
  let next = 0;
  for (const index of order) {
    if (index > next) parts.push(LEFT_OUT);
    parts.push(kept.get(index) ?? "");
    next = index + 1;
  }
  if (next < read.length) parts.push(LEFT_OUT);
  return {
    text: wrap(parts.join("\n\n")) + selected(selectionText),
    partial: true,
    origins: originsOf(order, selectionText !== ""),
  };
}

/**
 * The Question context of a Question in one of the assistant's conversations
 * (ADR-0017): the messages before it, Questions as the User's words and Answers
 * (with their Citations as plain references) as the model's, the oldest left out
 * first, as in a Mind. A failed Answer adds nothing. `mind`, what of the Mind in
 * focus it reads (`MindSection`), comes right before the Question, in its message.
 */
export function conversationContext(
  earlier: readonly ConversationMessage[],
  question: string,
  options: { mind?: string; budget?: number } = {},
): QuestionContext {
  return conversationContextOf(earlier, question, options).context;
}

/**
 * `conversationContext`, with where the third-party text in the earlier
 * Answers it keeps came from: Answers written in a Run that had read some.
 */
export function conversationContextOf(
  earlier: readonly ConversationMessage[],
  question: string,
  { mind = "", budget = QUESTION_CONTEXT_TOKEN_BUDGET }: { mind?: string; budget?: number } = {},
): QuestionContextWithOrigins {
  const pieces: Piece[] = earlier.flatMap((message): Piece[] => {
    if (message.kind === "question")
      return message.text ? [{ role: "user", text: message.text }] : [];
    if (message.status === "failed") return [];
    const text = answerMarkdown(message.text, message.citations, message.figures);
    if (!text) return [];
    return [
      { role: "assistant", text, ...(message.textOrigin && { origins: [message.textOrigin] }) },
    ];
  });
  const kept = keptWithinBudget(pieces.reverse(), budget - estimateTokens(question));
  const asked = mind ? `${mind}\n\n${question}` : question;
  return { context: { messages: asMessages(kept, asked), question }, origins: originsOf(kept) };
}

/**
 * The newest pieces above the Question that fit in `remaining` tokens, oldest
 * first: the oldest pieces are the ones left out, and the newest piece that
 * doesn't fit whole keeps its end, marked as cut, when that is worth keeping.
 * Those before the first of the User's are left out too: a conversation
 * starts with the User.
 */
function keptWithinBudget(newestFirst: Iterable<Piece>, remaining: number): Piece[] {
  const kept: Piece[] = [];
  for (const piece of newestFirst) {
    if (remaining <= 0) break;
    const cost = estimateTokens(piece.text);
    if (cost <= remaining) {
      kept.push(piece);
      remaining -= cost;
    } else {
      if (remaining >= MIN_TAIL_TOKENS) kept.push({ ...piece, text: tail(piece.text, remaining) });
      remaining = 0;
    }
  }
  kept.reverse();
  while (kept[0]?.role === "assistant") kept.shift();
  return kept;
}

/** The kept pieces, then the Question, as messages: one per run of the same role, as providers expect roles to alternate. */
function asMessages(kept: readonly Piece[], question: string): AnswerMessage[] {
  const messages: AnswerMessage[] = [];
  for (const piece of [...kept, { role: "user", text: question } satisfies Piece]) {
    const last = messages.at(-1);
    if (last?.role === piece.role) last.content = `${last.content}\n\n${piece.text}`;
    else messages.push({ role: piece.role, content: piece.text });
  }
  return messages;
}

/**
 * The Question after the newest pieces above it that fit in `remaining`
 * tokens, as messages (see `keptWithinBudget`).
 */
function withinBudget(
  newestFirst: Iterable<Piece>,
  question: string,
  remaining: number,
): AnswerMessage[] {
  return asMessages(keptWithinBudget(newestFirst, remaining), question);
}

/**
 * The Mind read with a Question in the assistant, as its request holds it:
 * `text`, right before the Question in its message, and how to cut it to
 * about `tokens` tokens by the Mind's own rules ("" when none of it fits; see
 * `MindReading`).
 */
export interface RequestMind {
  text: string;
  cut(tokens: number): string;
}

/**
 * The Question context cut down to about `budget` tokens (by `estimateTokens`)
 * by the same rules as `buildQuestionContext`: the Question is always kept,
 * the oldest content goes first, and the newest part that doesn't fit whole
 * keeps its end. The Mind read with the Question (`mind`) comes before the
 * rest, cut by its own rules if it doesn't fit whole. For a model whose
 * context window is known only once the model is ready (see ./window). Null
 * when the Question alone is over the budget.
 */
export function fitQuestionContext(
  messages: readonly AnswerMessage[],
  question: string,
  budget: number,
  mind?: RequestMind,
): AnswerMessage[] | null {
  const above: Piece[] = messages.map((message) => ({ role: message.role, text: message.content }));
  // The last message ends with the Question, after whatever Notes (or the Mind) come right
  // before it; if it doesn't, it is kept whole as the Question.
  let asked = question;
  let section: string | null = null;
  const last = above.pop();
  if (last?.role === "user" && last.text.endsWith(question)) {
    let before = last.text.slice(0, last.text.length - question.length).trim();
    if (mind?.text && before.endsWith(mind.text)) {
      section = mind.text;
      before = before.slice(0, before.length - mind.text.length).trim();
    }
    if (before) above.push({ role: "user", text: before });
  } else if (last?.role === "user") {
    asked = last.text;
  } else if (last) {
    above.push(last);
  }
  let remaining = budget - estimateTokens(asked);
  if (remaining < 0) return null;
  if (section !== null && mind) {
    // Less one for the blank line before the Question.
    const fitted = estimateTokens(section) < remaining ? section : mind.cut(remaining - 1);
    if (fitted) {
      asked = `${fitted}\n\n${asked}`;
      remaining -= estimateTokens(fitted) + 1;
    }
  }
  return withinBudget(above.reverse(), asked, remaining);
}

/**
 * The Question context above the Question, as text to rewrite the Question
 * into a search query from: the newest of it, within about `budget` tokens,
 * each part labelled as the User's or an Answer. Empty when there is nothing
 * above the Question.
 */
export function earlierContext(
  messages: readonly AnswerMessage[],
  question: string,
  budget = SEARCH_QUERY_CONTEXT_TOKEN_BUDGET,
): string {
  const pieces: Piece[] = messages.map((message) => ({
    role: message.role,
    text: message.content,
  }));
  // The last message ends with the Question, after whatever Notes come right before it.
  const last = pieces.at(-1);
  if (last?.role === "user" && last.text.endsWith(question)) {
    last.text = last.text.slice(0, last.text.length - question.length).trim();
  }
  const kept: string[] = [];
  let remaining = budget;
  // Newest first, so the oldest are the ones left out.
  for (let at = pieces.length - 1; at >= 0 && remaining > 0; at--) {
    const piece = pieces[at] as Piece;
    if (!piece.text) continue;
    const label = piece.role === "user" ? "User" : "Answer";
    const cost = estimateTokens(piece.text);
    if (cost <= remaining) {
      kept.push(`${label}: ${piece.text}`);
      remaining -= cost;
    } else {
      if (remaining >= MIN_TAIL_TOKENS) kept.push(`${label}: ${tail(piece.text, remaining)}`);
      remaining = 0;
    }
  }
  return kept.reverse().join("\n\n");
}
