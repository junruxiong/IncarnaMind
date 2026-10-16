/**
 * The guards every everyday request in the assistant is held to
 * (eval/everyday/checks.ts), on the failures that were found, and the judge
 * of claims they read (eval/everyday/judge.ts), which is apart from the app's
 * own net.
 */
import { describe, expect, test } from "vitest";
import { type RequestOutcome, requestFailures } from "../../eval/everyday/checks";
import { judgeClaims, verdictIn } from "../../eval/everyday/judge";
import type { AnswerToolCall } from "../../src/core";
import { scriptedModel } from "../helpers/models";

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
  verdict: { unbacked: [] },
  citedInMinds: ["tutorial", "graphst"],
  scoped: PAPERS,
  ...fields,
});

describe("The everyday request guards", () => {
  // protects=a page claimed and never made fails on every guard it breaks; fails_when=it passes; no_claim=none beyond protects.
  test("a page claimed and never made fails", () => {
    const text = "I've created a summary page for the two papers.";
    const failures = requestFailures(
      outcome({
        answer: { text, citations: [], toolCalls: [], steps: null },
        verdict: { unbacked: [text] },
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

  // protects=a page made with no Citations in it fails, whatever the reply cites; fails_when=the reply's Citations count for the page's; no_claim=none beyond protects.
  test("a page whose Citations were lost fails, whatever the reply cites", () => {
    const failures = requestFailures(
      outcome({
        answer: {
          text: "Made “Summary”.",
          citations: [citation("tutorial"), citation("graphst")],
          toolCalls: [created],
          steps: null,
        },
        citedInMinds: [],
      }),
    );
    expect(failures).toEqual([{ kind: "no-citations" }]);
  });

  // protects=a Document in scope counts as covered only when cited or named as left out, not merely named; fails_when=a name alone passes; no_claim=none beyond protects.
  test("every Document in scope is cited, or named as left out", () => {
    const one = (text: string) =>
      outcome({
        answer: { text, citations: [], toolCalls: [created], steps: null },
        citedInMinds: ["tutorial"],
      });
    expect(requestFailures(one("Made “Summary”. 2306.10683v1 is the GraphST paper."))).toEqual([
      { kind: "uncovered", documents: ["2306.10683v1"] },
    ]);
    expect(
      requestFailures(
        one("Made “Summary”. I couldn't read 2306.10683v1, so it covers the tutorial."),
      ),
    ).toEqual([]);
  });

  // protects=a claim the judge finds fails, marked when the app flagged it too; fails_when=it passes; no_claim=none beyond protects.
  test("a claim the judge finds fails, flagged or not", () => {
    const flagged = outcome({
      expected: { write: false, cite: false, coverAll: false },
      answer: {
        text: "I added it.",
        citations: [],
        toolCalls: [],
        steps: {
          sentAt: null,
          finishedAt: null,
          phases: [],
          reasoning: null,
          unbackedClaims: ["added"],
        },
      },
      verdict: { unbacked: ["I added it."] },
    });
    expect(requestFailures(flagged)).toEqual([
      { kind: "unbacked", sentences: ["I added it."], flagged: true },
    ]);
  });
});

describe("The judge of claims", () => {
  // protects=the judge reads the record and the reply, and its verdict is read back; fails_when=either is left out of what it reads, or a verdict it can't read passes; no_claim=how well a real model judges.
  test("reads the record and the reply, and its verdict", async () => {
    let asked = "";
    const model = scriptedModel(() => ({ text: "" }), {
      generate: ({ prompt }) => {
        asked = prompt;
        return {
          text: '```json\n{"unbacked": ["The new page replaces the earlier version."]}\n```',
        };
      },
    });
    const verdict = await judgeClaims(
      model,
      "Made it. The new page replaces the earlier version.",
      ["- add_to_mind “Summary”: proposed 2 new Blocks, waiting for the User"],
    );
    expect(asked).toContain("add_to_mind “Summary”");
    expect(asked).toContain("The new page replaces the earlier version.");
    expect(verdict).toEqual({ unbacked: ["The new page replaces the earlier version."] });
    expect(verdictIn("no idea").unbacked).toHaveLength(1);
  });
});
