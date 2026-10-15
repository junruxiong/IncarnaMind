/**
 * The guards every everyday request in the assistant is held to (#893,
 * eval/everyday/checks.ts), on the failures that were found: a page claimed
 * and never made, a summary written only in the panel, a page whose
 * Citations were lost, and a summary of two papers that covers one.
 */
import { describe, expect, test } from "vitest";
import { type RequestOutcome, requestFailures } from "../../eval/everyday/checks";
import type { AnswerToolCall } from "../../src/core";

const PAPERS = [
  { id: "tutorial", name: "A Tutorial about Random Neural Networks in Supervised Learning" },
  { id: "graphst", name: "2306.10683v1" },
];

const citation = (documentId: string) => ({ documentId });

const created: AnswerToolCall = {
  id: "c1",
  tool: "create_mind",
  source: "minds",
  input: { title: "Summary" },
  status: "done",
  resultCount: null,
  mindChange: { mindId: "m1", created: true, request: "r1", proposals: 7 },
};

const outcome = (fields: Partial<RequestOutcome>): RequestOutcome => ({
  expected: { write: true, cite: true, coverAll: true },
  answer: { text: "", citations: [], toolCalls: [], steps: null },
  citedInMinds: ["tutorial", "graphst"],
  scoped: PAPERS,
  minds: ["Reading group", "Summary"],
  ...fields,
});

describe("The everyday request guards", () => {
  // protects=a reply that says it made a page no call made fails twice: nothing changed, and the claim isn't real; fails_when=it passes; no_claim=none beyond protects.
  test("a page claimed and never made fails", () => {
    const failures = requestFailures(
      outcome({
        answer: {
          text: "I've created a summary page for the two papers.",
          citations: [],
          toolCalls: [],
          steps: null,
        },
        citedInMinds: [],
      }),
    );
    expect(failures.map((failure) => failure.kind)).toEqual([
      "no-change",
      "unbacked",
      "no-citations",
      "uncovered",
    ]);
  });

  // protects=a good summary written only in the panel fails a write request; fails_when=its Citations in the panel hide that nothing was written; no_claim=none beyond protects.
  test("a summary written only in the panel fails a write request", () => {
    const failures = requestFailures(
      outcome({
        answer: {
          text: "The tutorial is a practical guide [^1]. GraphST models urban regions [^2].",
          citations: [citation("tutorial"), citation("graphst")],
          toolCalls: [],
          steps: null,
        },
        citedInMinds: [],
      }),
    );
    expect(failures.map((failure) => failure.kind)).toEqual(["no-change", "no-citations"]);
  });

  // protects=a page made with no Citations in it fails though the reply carries markers; fails_when=the reply's Citations count for the page's; no_claim=none beyond protects.
  test("a page whose Citations were lost fails, whatever the reply cites", () => {
    const failures = requestFailures(
      outcome({
        answer: {
          text: "I made a new Mind, “Summary”, with a short summary of both papers.",
          citations: [citation("tutorial"), citation("graphst")],
          toolCalls: [created],
          steps: null,
        },
        citedInMinds: [],
      }),
    );
    expect(failures).toEqual([{ kind: "no-citations" }]);
  });

  // protects=a summary of two papers that cites one fails unless it names the other as not covered; fails_when=the missing paper passes unnoticed, or a named one still fails; no_claim=none beyond protects.
  test("every Document in scope is cited, or named", () => {
    const one = outcome({
      answer: { text: "Made “Summary”.", citations: [], toolCalls: [created], steps: null },
      citedInMinds: ["tutorial"],
    });
    expect(requestFailures(one)).toEqual([{ kind: "uncovered", documents: ["2306.10683v1"] }]);
    const named = outcome({
      ...one,
      answer: {
        ...one.answer,
        text: "Made “Summary”. I couldn't read 2306.10683v1, so it covers the tutorial only.",
      },
    });
    expect(requestFailures(named)).toEqual([]);
  });

  // protects=a question answered in the panel, cited, with no claim, passes; a claim the app flagged still fails, marked as flagged; fails_when=either is wrong; no_claim=none beyond protects.
  test("a cited answer passes, and a flagged claim still fails", () => {
    const answer = {
      text: "The tutorial is a practical guide [^1].",
      citations: [citation("tutorial")],
      toolCalls: [],
      steps: null,
    };
    const question = outcome({
      expected: { write: false, cite: true, coverAll: false },
      answer,
      citedInMinds: [],
    });
    expect(requestFailures(question)).toEqual([]);
    const flagged = outcome({
      ...question,
      answer: {
        ...answer,
        steps: {
          sentAt: null,
          finishedAt: null,
          phases: [],
          reasoning: null,
          unbackedClaims: ["added"],
        },
      },
    });
    expect(requestFailures(flagged)).toEqual([
      { kind: "unbacked", claims: ["added"], flagged: true },
    ]);
  });
});
