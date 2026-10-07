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
export const QUESTION_CONTEXT_TOKEN_BUDGET = 12_000;

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

  let remaining = budget - estimateTokens(questionText);
  const kept: Piece[] = [];
  // Newest first, so the oldest are the ones left out.
  for (let at = index - 1; at >= 0 && remaining > 0; at--) {
    const block = children[at];
    const piece = block instanceof Y.XmlElement ? pieceOf(block) : null;
    if (!piece) continue;
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
  kept.push({ role: "user", text: questionText });

  // One message per run of the same role, as providers expect roles to alternate.
  const messages: AnswerMessage[] = [];
  for (const piece of kept) {
    const last = messages.at(-1);
    if (last?.role === piece.role) last.content = `${last.content}\n\n${piece.text}`;
    else messages.push({ role: piece.role, content: piece.text });
  }
  // A conversation starts with the User.
  while (messages[0]?.role === "assistant") messages.shift();
  return { messages, question: questionText };
}
