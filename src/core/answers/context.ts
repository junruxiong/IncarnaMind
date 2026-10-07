/**
 * Question context (ADR-0007): what an Answer is generated from. Every Block
 * above the Question, in order, except Notes the User switched off: Notes and
 * earlier Questions as the User's words, earlier Answers as the model's. When
 * that is over the token budget, the oldest content goes first.
 */
import * as Y from "yjs";
import { ANSWER_BLOCK, INCLUDE_IN_CONTEXT_ATTRIBUTE, QUESTION_BLOCK } from "../api";
import { attribute, contentMarkdown, textAttribute, toMarkdown } from "./blocks";
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

/**
 * The Question context of the Question at `index` among the Mind's top-level
 * Blocks. The Question itself is always kept, even alone over the budget.
 */
export function buildQuestionContext(
  blocks: Y.XmlFragment,
  index: number,
  budget = QUESTION_CONTEXT_TOKEN_BUDGET,
): QuestionContext {
  const children = blocks.toArray();
  const question = children[index];
  if (!(question instanceof Y.XmlElement)) throw new Error("No Question at that index.");
  const questionText = toMarkdown(question);
  /** What the Blocks above contribute, newest first, read only as far as the budget goes. */
  const newestFirst = function* (): Generator<Piece> {
    for (let at = index - 1; at >= 0; at--) {
      const block = children[at];
      const piece = block instanceof Y.XmlElement ? pieceOf(block) : null;
      if (piece) yield piece;
    }
  };
  return {
    messages: withinBudget(newestFirst(), questionText, budget - estimateTokens(questionText)),
    question: questionText,
  };
}

/**
 * The Question after the newest pieces above it that fit in `remaining`
 * tokens, as messages: the oldest pieces are the ones left out, and the
 * newest piece that doesn't fit whole keeps its end, marked as cut, when that
 * is worth keeping.
 */
function withinBudget(
  newestFirst: Iterable<Piece>,
  question: string,
  remaining: number,
): AnswerMessage[] {
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
  kept.push({ role: "user", text: question });

  // One message per run of the same role, as providers expect roles to alternate.
  const messages: AnswerMessage[] = [];
  for (const piece of kept) {
    const last = messages.at(-1);
    if (last?.role === piece.role) last.content = `${last.content}\n\n${piece.text}`;
    else messages.push({ role: piece.role, content: piece.text });
  }
  // A conversation starts with the User.
  while (messages[0]?.role === "assistant") messages.shift();
  return messages;
}

/**
 * The Question context cut down to about `budget` tokens (by `estimateTokens`)
 * by the same rules as `buildQuestionContext`: the Question is always kept,
 * the oldest content goes first, and the newest part that doesn't fit whole
 * keeps its end. For a local model, whose context window is known only once
 * the model is ready (see ./window). Null when the Question alone is over the budget.
 */
export function fitQuestionContext(
  messages: readonly AnswerMessage[],
  question: string,
  budget: number,
): AnswerMessage[] | null {
  const above: Piece[] = messages.map((message) => ({ role: message.role, text: message.content }));
  // The last message ends with the Question, after whatever Notes come right before it;
  // if it doesn't, it is kept whole as the Question.
  let asked = question;
  const last = above.pop();
  if (last?.role === "user" && last.text.endsWith(question)) {
    const before = last.text.slice(0, last.text.length - question.length).trim();
    if (before) above.push({ role: "user", text: before });
  } else if (last?.role === "user") {
    asked = last.text;
  } else if (last) {
    above.push(last);
  }
  const remaining = budget - estimateTokens(asked);
  if (remaining < 0) return null;
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
