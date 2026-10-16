import { describe, expect, test } from "vitest";
import type { AnswerToolCall, ConversationAnswer, ConversationMessage } from "../../src/core";
import { conversationContext, estimateTokens } from "../../src/core/answers/context";
import { toolRecord, toolRecordLines } from "../../src/core/answers/toolRecord";

/**
 * The record of what an Answer's Tool calls really did, as the next
 * Question reads it: what each changed and in which Mind, or why nothing
 * changed, from the calls' cards alone.
 */

const names = (mindId: string) => (mindId === "m1" ? "Tide notes" : null);

const call = (
  fields: Partial<AnswerToolCall> & Pick<AnswerToolCall, "tool" | "source">,
): AnswerToolCall => ({ id: fields.tool, input: {}, status: "done", resultCount: null, ...fields });

describe("The record of an Answer's Tool calls", () => {
  // protects=each kind of call reads as what it really did, its Mind by its name now; fails_when=a failed call reads as done, a change loses its Mind or count, or a revision reads as an addition; no_claim=how a model reads it.
  test("each call as what it did", () => {
    const change = (fields: object) => ({
      mindId: "m1",
      created: false,
      request: "r",
      proposals: 7,
      ...fields,
    });
    expect(
      toolRecordLines(
        [
          call({
            tool: "search_documents",
            source: "documents",
            input: { query: "tides" },
            resultCount: 8,
          }),
          call({
            tool: "search_documents",
            source: "documents",
            input: { query: "moon" },
            status: "failed",
          }),
          call({
            tool: "create_mind",
            source: "minds",
            input: { title: "Tides" },
            mindChange: change({ created: true }),
          }),
          call({
            tool: "create_mind",
            source: "minds",
            input: { title: "Empty" },
            mindChange: change({ mindId: "m2", created: true, proposals: 0 }),
          }),
          call({ tool: "add_to_mind", source: "minds", mindChange: change({ proposals: 1 }) }),
          call({
            tool: "edit_mind",
            source: "minds",
            input: { part: "Summary" },
            mindChange: change({ proposals: 2 }),
          }),
          call({ tool: "edit_mind", source: "minds", mindChange: change({ revised: true }) }),
          call({ tool: "edit_mind", source: "minds", input: { mind: "Q3 review" } }),
          call({
            tool: "edit_mind",
            source: "minds",
            status: "failed",
            mindProblem: { kind: "waiting", mindId: "m1", count: 12, mixed: false },
          }),
          call({
            tool: "read_mind",
            source: "minds",
            status: "failed",
            input: { mind: "Chapter 9" },
            mindProblem: { kind: "not-found", name: "Chapter 9" },
          }),
          call({
            tool: "read_web_page",
            source: "web",
            input: { url: "https://example.com" },
            webPage: { address: "https://example.com", documentId: "d", name: "Example" },
          }),
          call({
            tool: "book_boat",
            source: "connector",
            connector: { id: "c", name: "Boats" },
            status: "failed",
            approval: "denied",
          }),
          call({ tool: "sign_in", source: "connector", status: "failed", signInRequired: true }),
        ],
        names,
      ),
    ).toEqual([
      "- search_documents “tides”: 8 Passages",
      "- search_documents “moon”: failed",
      "- create_mind: made the Mind “Tide notes” and proposed 7 changes, waiting for the User",
      "- create_mind: made the Mind “Empty”, empty",
      "- add_to_mind “Tide notes”: proposed 1 new Block, waiting for the User",
      "- edit_mind “Tide notes” (section “Summary”): proposed 2 changes, waiting for the User",
      "- edit_mind “Tide notes”: revised your own waiting text in place, and proposed 7 changes, waiting for the User",
      "- edit_mind “Q3 review”: no change was needed",
      "- edit_mind “Tide notes”: nothing changed: 12 changes the assistant can't revise waited there for the User",
      "- read_mind “Chapter 9”: couldn't read it: no Mind is called “Chapter 9”",
      "- read_web_page https://example.com: saved as the Document “Example”",
      "- book_boat (the Connector “Boats”): the User denied it, so it wasn't made",
    ]);
  });

  // protects=the record says how many Citations a change wrote and how many markers it left out, and how many the reply holds; fails_when=a page whose Citations were lost reads as cited, or the reply's are left out; no_claim=none beyond protects.
  test("the Citations a change wrote, the markers it left out, and the reply's", () => {
    const made = call({
      tool: "create_mind",
      source: "minds",
      mindChange: { mindId: "m1", created: true, request: "r", proposals: 7, cited: 6 },
    });
    const lost = call({
      tool: "add_to_mind",
      source: "minds",
      mindChange: { mindId: "m1", created: false, request: "r", proposals: 2, unrecorded: 3 },
    });
    expect(toolRecord([made, lost], names, 9).split("\n").slice(2, -1)).toEqual([
      "- create_mind: made the Mind “Tide notes” and proposed 7 changes, waiting for the User, with 6 Citations",
      "- add_to_mind “Tide notes”: proposed 2 new Blocks, waiting for the User; 3 markers had no Citation recorded with cite, so they were left out",
      "- cite: 9 Citations in your reply; each reads as its source in brackets, like [Tides, p. 3]",
    ]);
  });

  // protects=an Answer with no calls adds no record, and one with calls is wrapped for the model; fails_when=an empty record is sent, or the block isn't marked; no_claim=none beyond protects.
  test("no calls, no record; otherwise one block", () => {
    expect(toolRecord([], names)).toBe("");
    expect(
      toolRecord(
        [call({ tool: "read_mind", source: "minds", input: { mind: "Tide notes" } })],
        names,
      ),
    ).toBe(
      [
        "<tool-record>",
        "What the Tools of your Answer above really did, as IncarnaMind recorded it:",
        "- read_mind “Tide notes”: read it",
        "</tool-record>",
      ].join("\n"),
    );
  });
});

describe("A record within the context's budget", () => {
  const searches = Array.from({ length: 12 }, (_, at) =>
    call({
      tool: "search_documents",
      source: "documents",
      input: { query: `tides and the moon, part ${at + 1}` },
      resultCount: 8,
    }),
  );
  const answer = (toolCalls: AnswerToolCall[], text: string): ConversationAnswer => ({
    kind: "answer",
    id: "a1",
    conversationId: "c",
    questionId: "q1",
    text,
    citations: [],
    figures: [],
    model: null,
    status: "done",
    errorKind: null,
    errorMessage: null,
    citationSupport: null,
    toolCalls,
    textOrigin: null,
    read: null,
    steps: null,
    createdAt: "2026-10-16T00:00:00.000Z",
    updatedAt: "2026-10-16T00:00:00.000Z",
  });
  const earlier = (reply: ConversationAnswer): ConversationMessage[] => [
    {
      kind: "question",
      id: "q1",
      conversationId: "c",
      text: "write a summary page",
      askedFrom: null,
      focus: null,
      scope: null,
      model: null,
      forcedSkill: null,
      createdAt: "2026-10-16T00:00:00.000Z",
    },
    reply,
  ];
  const QUESTION = "make it longer";
  const sent = (messages: ConversationMessage[], budget: number) =>
    conversationContext(messages, QUESTION, { budget, mindNames: names })
      .messages.map((message) => message.content)
      .join("\n\n");

  // protects=a record that doesn't fit whole is left out, never cut, and nothing older than it is kept; fails_when=part of a record is sent, or the Answer before it comes without it; no_claim=none beyond protects.
  test("a record that doesn't fit whole is left out with all before it", () => {
    const reply = answer(searches, "The papers say the moon drives the tides. ".repeat(4));
    const record = toolRecord(searches, names);
    const budget = estimateTokens(QUESTION) + estimateTokens(record) - 20;

    expect(sent(earlier(reply), budget)).toBe(QUESTION);
  });

  // protects=a record whose Answer was left out goes too; fails_when=the record is sent without the Answer it is about; no_claim=none beyond protects.
  test("a record whose Answer was left out goes with it", () => {
    const made = call({
      tool: "create_mind",
      source: "minds",
      mindChange: { mindId: "m1", created: true, request: "r", proposals: 2 },
    });
    const reply = answer([made], "The page sums up both papers. ".repeat(80));
    const record = toolRecord([made], names);
    const budget = estimateTokens(QUESTION) + estimateTokens(record) + 10;

    expect(sent(earlier(reply), budget)).toBe(QUESTION);
    expect(sent(earlier(reply), 100_000)).toContain(
      `The page sums up both papers.\n\n${record}\n\n${QUESTION}`,
    );
  });
});
