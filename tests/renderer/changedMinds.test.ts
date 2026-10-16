import { describe, expect, test } from "vitest";
import type { AnswerToolCall, MindChange } from "../../src/core/api";
import { changedMinds } from "../../src/renderer/src/changedMinds";

/*
 * The rows of an Answer's card of Minds changed (#202): one per Mind, in the
 * order the Answer first changed each, with every request of its changes there
 * and what its calls did: paragraphs written whole, places changed.
 */

let calls = 0;
const call = (tool: string, mindChange?: MindChange): AnswerToolCall => ({
  id: `call-${++calls}`,
  tool,
  source: "minds",
  input: {},
  status: mindChange ? "done" : "failed",
  resultCount: null,
  ...(mindChange && { mindChange }),
});

describe("The Minds an Answer changed", () => {
  // protects=one row per Mind, in order, with all its requests, made once it was made, and its counts from the calls; fails_when=a Mind shows twice, a request is lost, a made Mind reads as edited, or an edit counts as written; no_claim=how the card draws them.
  test("one row per Mind, in the order first changed, with every request there and what was done", () => {
    const rows = changedMinds({
      toolCalls: [
        call("read_mind"),
        call("edit_mind", { mindId: "a", created: false, request: "r1", proposals: 2 }),
        call("create_mind", {
          mindId: "b",
          created: true,
          request: "r2",
          proposals: 3,
          paragraphs: 2,
        }),
        call("add_to_mind", {
          mindId: "a",
          created: false,
          request: "r3",
          proposals: 1,
          paragraphs: 1,
        }),
        call("add_to_mind", {
          mindId: "b",
          created: false,
          request: "r4",
          proposals: 1,
          paragraphs: 1,
        }),
        call("edit_mind"),
      ],
    });
    expect(rows).toEqual([
      { mindId: "a", created: false, requests: ["r1", "r3"], wrote: 1, changed: 2 },
      { mindId: "b", created: true, requests: ["r2", "r4"], wrote: 3, changed: 0 },
    ]);
  });

  // protects=an empty Mind made with nothing in it is still a row, with no request; fails_when=it is left off the card; no_claim=none beyond protects.
  test("an empty Mind made is a row with no requests; an Answer that changed none has none", () => {
    expect(
      changedMinds({
        toolCalls: [call("create_mind", { mindId: "c", created: true, request: "", proposals: 0 })],
      }),
    ).toEqual([{ mindId: "c", created: true, requests: [], wrote: 0, changed: 0 }]);
    expect(changedMinds({ toolCalls: [call("read_mind")] })).toEqual([]);
  });

  // protects=a revision of the assistant's own waiting text counts its paragraphs as written, not places changed, and one that only removed text writes none; fails_when=it reads "Changed N places", or a removal reads as written; no_claim=none beyond protects.
  test("a revision of its own waiting text counts as paragraphs written, a removal as none", () => {
    expect(
      changedMinds({
        toolCalls: [
          call("edit_mind", {
            mindId: "a",
            created: false,
            request: "r1",
            proposals: 5,
            paragraphs: 3,
            revised: true,
          }),
        ],
      }),
    ).toEqual([{ mindId: "a", created: false, requests: ["r1"], wrote: 3, changed: 0 }]);
    expect(
      changedMinds({
        toolCalls: [
          call("edit_mind", {
            mindId: "a",
            created: false,
            request: "r1",
            proposals: 0,
            revised: true,
          }),
        ],
      }),
    ).toEqual([{ mindId: "a", created: false, requests: ["r1"], wrote: 0, changed: 0 }]);
  });
});
