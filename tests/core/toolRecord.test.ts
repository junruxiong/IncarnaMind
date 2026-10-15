import { describe, expect, test } from "vitest";
import type { AnswerToolCall } from "../../src/core";
import { toolRecord, toolRecordLines } from "../../src/core/answers/toolRecord";

/**
 * The record of what an Answer's Tool calls really did (#893), as the next
 * Question reads it: what each changed and in which Mind, or why nothing
 * changed, from the calls' cards alone.
 */

const names = (mindId: string) => (mindId === "m1" ? "Tide notes" : null);

const call = (fields: Partial<AnswerToolCall> & Pick<AnswerToolCall, "tool" | "source">) =>
  ({ id: fields.tool, input: {}, status: "done", resultCount: null, ...fields }) as AnswerToolCall;

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
      "- create_mind: made the Mind “Tide notes”, its text as 7 changes waiting for the User",
      "- create_mind: made the Mind “Empty”, empty",
      "- add_to_mind “Tide notes”: proposed 1 new Block, waiting for the User",
      "- edit_mind “Tide notes” (section “Summary”): proposed 2 changes waiting for the User",
      "- edit_mind “Tide notes”: revised your own waiting text; it is gone, and 7 changes waiting for the User in its place",
      "- edit_mind “Q3 review”: no change was needed",
      "- edit_mind “Tide notes”: nothing changed: 12 changes the assistant can't revise waited there for the User",
      "- read_mind “Chapter 9”: couldn't read it: no Mind is called “Chapter 9”",
      "- read_web_page https://example.com: saved as the Document “Example”",
      "- book_boat (the Connector “Boats”): the User denied it, so it wasn't made",
    ]);
  });

  // protects=the record says how many checked Citations a change wrote and how many markers it left out, and that the reply's Citations are real; fails_when=a page whose Citations were lost reads as cited, or the reply's read as made up; no_claim=none beyond protects.
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
      "- create_mind: made the Mind “Tide notes”, its text as 7 changes waiting for the User, with 6 checked Citations",
      "- add_to_mind “Tide notes”: proposed 2 new Blocks, waiting for the User; 3 markers had no Citation recorded with cite, so they were left out",
      "- cite: 9 Citations in your reply, checked against their Documents; each reads as its source in brackets, like [Tides, p. 3]",
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
