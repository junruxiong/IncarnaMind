/**
 * Fitting an Answer's requests into a local model's window (see
 * src/core/answers/window.ts): the Question context is cut by the same rules
 * as when it is built (src/core/answers/context.ts), and search results keep
 * whole Passages.
 */
import { describe, expect, test } from "vitest";
import { estimateTokens, fitQuestionContext } from "../../src/core/answers/context";
import type { AnswerMessage } from "../../src/core/answers/engine";
import { createWindowBudget, NO_ROOM_FOR_PASSAGES } from "../../src/core/answers/window";

/** About `tokens` tokens of text, labelled. */
const block = (label: string, tokens: number) => `${label}: ${"word ".repeat(tokens * 0.8)}`;

const passage = (id: string, tokens: number) =>
  `<passage id="${id}" document="Tides">\n${"tide ".repeat(tokens * 0.8)}\n</passage>`;

describe("Cutting the Question context to a window", () => {
  const question = "What do my Notes say?";
  const messages: AnswerMessage[] = [
    { role: "user", content: block("Oldest", 500) },
    { role: "assistant", content: block("Answer", 500) },
    { role: "user", content: `${block("Newest", 500)}\n\n${question}` },
  ];

  test("keeps the Question and the newest content, leaving the oldest out first", () => {
    const fitted = fitQuestionContext(messages, question, 1_200);
    expect(fitted?.at(-1)?.content.endsWith(question)).toBe(true);
    const text = fitted?.map((message) => message.content).join("\n") ?? "";
    expect(text).toContain("Newest:");
    expect(text).toContain("Answer:");
    expect(text).not.toContain("Oldest:");
    // Within the budget, give or take the rounding of each part.
    expect(estimateTokens(text)).toBeLessThanOrEqual(1_200 + 4);
  });

  test("the newest part that doesn't fit whole keeps its end, marked as cut", () => {
    const fitted = fitQuestionContext(messages, question, 300) ?? [];
    expect(fitted).toHaveLength(1);
    const [only] = fitted;
    // The Newest Note lost its start; the Answer before it, and the oldest Note, are gone.
    expect(only?.content.startsWith("…")).toBe(true);
    expect(only?.content.endsWith(question)).toBe(true);
    expect(only?.content).not.toContain("Newest:");
  });

  test("a conversation still starts with the User", () => {
    const fitted = fitQuestionContext(messages, question, 1_050) ?? [];
    expect(fitted[0]?.role).toBe("user");
  });

  test("with room for nothing but the Question, the Question alone; with less, none", () => {
    expect(fitQuestionContext(messages, question, estimateTokens(question))).toEqual([
      { role: "user", content: question },
    ]);
    expect(fitQuestionContext(messages, question, 2)).toBeNull();
  });
});

describe("A window's budget", () => {
  const window = { tokens: 8_192, outputTokens: 2_048 };

  test("fits a request, keeping room for a search and for the output", () => {
    const budget = createWindowBudget(window);
    const messages: AnswerMessage[] = Array.from({ length: 20 }, (_, index) => ({
      role: index % 2 === 0 ? "user" : "assistant",
      content: block(`Block ${index}`, 400),
    }));
    messages.push({ role: "user", content: "Why?" });
    const fit = budget.fit({
      instructions: "Answer well.",
      messages,
      question: "Why?",
      reserve: 4_500,
    });
    if (!fit.ok) throw new Error(fit.error.message);
    expect(fit.estimated + 4_500 + window.outputTokens).toBeLessThanOrEqual(window.tokens);
    expect(fit.messages.at(-1)?.content).toBe("Why?");
  });

  test("a request whose Question alone doesn't fit fails as too long", () => {
    const fit = createWindowBudget(window).fit({
      instructions: "Answer well.",
      messages: [{ role: "user", content: block("Huge", 9_000) }],
      question: block("Huge", 9_000),
    });
    expect(fit).toMatchObject({ ok: false, error: { kind: "too-long" } });
  });

  test("a search's Passages are kept whole, in order, while they fit", () => {
    const budget = createWindowBudget(window);
    const passages = [passage("P1", 1_500), passage("P2", 1_500), passage("P3", 1_500)].join(
      "\n\n",
    );
    const loop = budget.loop(3_800);
    const first = loop.passages(passages, 3);
    expect(first.passageCount).toBe(1);
    expect(first.text).toContain('id="P1"');
    expect(first.text).not.toContain('id="P2"');
    // The window is full now: the next search gives none, and the model is told to answer.
    expect(loop.passages(passages, 3)).toEqual({ text: NO_ROOM_FOR_PASSAGES, passageCount: 0 });
    expect(loop.canCallTools()).toBe(false);
  });

  test("Ollama's own count of the first request replaces the estimate", () => {
    const budget = createWindowBudget(window);
    const loop = budget.loop(1_000);
    // The model counted twice what was estimated: the rest of the Answer counts larger.
    loop.stepFinished({ inputTokens: 2_000, outputTokens: 20 });
    expect(budget.factor).toBeGreaterThan(2);
    // And the room left is counted from the real size.
    const result = loop.passages(passage("P1", 1_000), 1);
    expect(result.passageCount).toBe(1);
  });
});
