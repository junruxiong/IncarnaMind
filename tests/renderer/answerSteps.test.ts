import { describe, expect, test } from "vitest";
import type { AnswerToolCall, AskedFrom, MarkedCitation } from "../../src/core/api";
import {
  foldedLine,
  liveStep,
  type StepsInput,
  stepRows,
} from "../../src/renderer/src/answerSteps";

/*
 * An Answer's steps (#391): the folded line, the timeline's rows and the live
 * step, from what an Answer in a conversation keeps.
 */

const MIND: AskedFrom = { mindId: "m1", name: "Lit review", deleted: false };
const MODEL = { providerId: "p1", modelId: "claude-sonnet-5-5" };

const search = (
  id: string,
  query: string,
  extra: Partial<AnswerToolCall> = {},
): AnswerToolCall => ({
  id,
  tool: "search_documents",
  source: "documents",
  input: { query },
  status: "done",
  resultCount: 4,
  ...extra,
});

const citation = (
  marker: number,
  passageId: string,
  documentId: string,
  page: number,
): MarkedCitation => ({
  marker,
  passageId,
  documentId,
  documentName: `Doc ${documentId}`,
  contentHash: "h",
  pageFrom: page,
  pageTo: page,
  location: { kind: "page", from: page, to: page },
  quote: "A quote.",
  check: "found",
  checkReason: null,
});

const connectorCall: AnswerToolCall = {
  id: "c1",
  tool: "search_issues",
  source: "connector",
  connector: { id: "linear", name: "Linear" },
  input: { query: "heat" },
  status: "done",
  resultCount: null,
};

function input(
  answer: Partial<StepsInput["answer"]> = {},
  extra: Partial<StepsInput> = {},
): StepsInput {
  return {
    answer: {
      text: "",
      citations: [],
      toolCalls: [],
      model: MODEL,
      steps: {
        sentAt: "2026-10-12T09:00:00.000Z",
        finishedAt: "2026-10-12T09:00:04.200Z",
        phases: [{ phase: "writing", at: "2026-10-12T09:00:00.500Z" }],
        reasoning: null,
      },
      ...answer,
    },
    readFrom: null,
    meeting: false,
    ...extra,
  };
}

describe("an Answer's steps", () => {
  // protects=an Answer with phases only says who answered and in how long; fails_when=it invents a search or a reasoning row; no_claim=none beyond protects.
  test("a phase-only Answer: answered by its model, in whole seconds, and folded to that", () => {
    const steps = input();
    expect(stepRows(steps)).toEqual([{ kind: "answered", model: "claude-sonnet-5-5", seconds: 4 }]);
    expect(foldedLine(steps)).toEqual({ kind: "answered", model: "claude-sonnet-5-5" });
  });

  // protects=consecutive searches are one row with every query, Passages found summed and used counted once, and their sources each once; fails_when=they split, or count a Passage twice; no_claim=none beyond protects.
  test("searches in a row group into one, with what they found and used and their sources", () => {
    const steps = input({
      toolCalls: [
        search("s1", "park cooling", { passages: ["a", "b"] }),
        search("s2", "irrigated parks", { passages: ["c", "d"] }),
        search("s3", "park edge", { passages: ["e"] }),
      ],
      citations: [
        citation(1, "a", "d1", 4),
        citation(2, "c", "d2", 6),
        citation(3, "c", "d2", 6),
        citation(4, "e", "d3", 9),
      ],
    });
    const [, row] = stepRows(steps);
    expect(row).toMatchObject({ kind: "search", found: 12, used: 3 });
    expect(row?.kind === "search" && row.calls.map((call) => call.input.query)).toEqual([
      "park cooling",
      "irrigated parks",
      "park edge",
    ]);
    expect(row?.kind === "search" && row.sources.map((source) => source.marker)).toEqual([1, 2, 4]);
    expect(foldedLine(steps)).toEqual({ kind: "searched", sources: 3, extraScopes: 0 });
  });

  // protects=a failed search is its own row where it happened, never grouped; fails_when=it joins a search row or moves; no_claim=none beyond protects.
  test("a failed call sits where it happened, between the searches it parts", () => {
    const failed = search("s2", "sensor readings", {
      status: "failed",
      resultCount: null,
      error: { kind: "auth", message: "Invalid API key." },
    });
    const rows = stepRows(input({ toolCalls: [search("s1", "a"), failed, search("s3", "b")] }));
    expect(rows.map((row) => row.kind)).toEqual(["answered", "search", "failed-search", "search"]);
    expect(rows[2]).toEqual({ kind: "failed-search", call: failed });
  });

  // protects=reasoning shows as "{model} thought for Ns" with its summary, and a model without one gets "Answered by" with no summary; fails_when=a summary shows for a model that gave none, or the times aren't whole seconds of at least 1; no_claim=none beyond protects.
  test("no reasoning: answered by the model; with a summary, thought for whole seconds, at least 1", () => {
    expect(stepRows(input())[0]).toEqual({
      kind: "answered",
      model: "claude-sonnet-5-5",
      seconds: 4,
    });
    const thought = input({
      steps: {
        sentAt: null,
        finishedAt: null,
        phases: [],
        reasoning: { summary: "Look in the Folder.", ms: 300 },
      },
    });
    expect(stepRows(thought)[0]).toEqual({
      kind: "reasoning",
      model: "claude-sonnet-5-5",
      summary: "Look in the Folder.",
      seconds: 1,
    });
  });

  // protects=an older Answer with no steps still gets its folded line and "Answered by {model}" with no time; fails_when=it has no line, or a made-up time; no_claim=none beyond protects.
  test("an older Answer with no steps stored: answered by its model, with no time", () => {
    const older = input({ steps: null, toolCalls: [search("s1", "tides")] });
    expect(stepRows(older)[0]).toEqual({
      kind: "answered",
      model: "claude-sonnet-5-5",
      seconds: null,
    });
    expect(foldedLine(older)).toEqual({ kind: "searched", sources: 0, extraScopes: 0 });
  });

  // protects=a deleted Mind reads as deleted, in the row and the folded line; fails_when=its old name shows as if it were there; no_claim=none beyond protects.
  test("a deleted Mind is read as deleted", () => {
    const deleted = { ...MIND, deleted: true };
    const steps = input({}, { readFrom: deleted });
    expect(stepRows(steps)[1]).toEqual({ kind: "read", mind: deleted });
    expect(foldedLine(steps)).toEqual({ kind: "read", mind: deleted });
  });

  // protects=an Answer asked during a meeting folds to "Listened to the live transcript"; fails_when=it says Searched; no_claim=whether it was asked during the recording is the caller's.
  test("a meeting: it listened to the live transcript", () => {
    const steps = input(
      { toolCalls: [search("s1", "budget")], citations: [citation(1, "a", "t1", 1)] },
      { readFrom: MIND, meeting: true },
    );
    expect(foldedLine(steps)).toEqual({ kind: "listened", sources: 1 });
  });

  // protects=with no search the line says the Mind read, or the model with no Mind; fails_when=it says Searched; no_claim=none beyond protects.
  test("no search: read the Mind in focus, or answered by the model with none", () => {
    expect(foldedLine(input({}, { readFrom: MIND }))).toEqual({ kind: "read", mind: MIND });
    expect(foldedLine(input())).toEqual({ kind: "answered", model: "claude-sonnet-5-5" });
    expect(foldedLine(input({ model: null }))).toBeNull();
  });

  // protects=a Connector searched beside the Question's scope counts as one more scope; fails_when=it isn't counted, or a Connector that didn't run is; no_claim=none beyond protects.
  test("two scopes: a Connector's beside the Question's", () => {
    const signIn = { ...connectorCall, id: "c2", status: "failed" as const, signInRequired: true };
    expect(foldedLine(input({ toolCalls: [search("s1", "heat"), connectorCall, signIn] }))).toEqual(
      { kind: "searched", sources: 0, extraScopes: 1 },
    );
  });

  // protects=the rows' order: reasoning, Read, the calls as they started, the check last; fails_when=any row is out of order or a call waiting for approval shows; no_claim=none beyond protects.
  test("the order of the rows, and the check last with what wasn't found", () => {
    const skillUse: AnswerToolCall = {
      id: "k1",
      tool: "use_skill",
      source: "skill",
      input: { name: "lit-review" },
      status: "done",
      resultCount: null,
    };
    const waiting = {
      ...connectorCall,
      id: "c3",
      status: "running" as const,
      approval: "waiting" as const,
    };
    const rows = stepRows(
      input(
        {
          steps: {
            sentAt: null,
            finishedAt: null,
            phases: [],
            reasoning: { summary: "Plan.", ms: 2600 },
          },
          toolCalls: [skillUse, search("s1", "heat", { passages: ["a"] }), connectorCall, waiting],
          citations: [
            citation(1, "a", "d1", 4),
            { ...citation(2, "a", "d1", 5), check: "not-found" },
          ],
        },
        { readFrom: MIND },
      ),
    );
    expect(rows.map((row) => row.kind)).toEqual([
      "reasoning",
      "read",
      "skill",
      "search",
      "call",
      "check",
    ]);
    expect(rows.at(-1)).toEqual({ kind: "check", count: 2, notFound: 1 });
  });
});

describe("the live step", () => {
  const live = (
    answer: Partial<StepsInput["answer"]>,
    extra: Partial<Parameters<typeof liveStep>[0]> = {},
  ) =>
    liveStep({
      ...input(answer),
      readFrom: MIND,
      phase: undefined,
      waitingForApproval: false,
      ...extra,
    });

  // protects=the live line moves Reading → Searching → Writing → Checking from the Answer's state, and waiting for approval wins; fails_when=a step comes out of order or approval is hidden; no_claim=the 1-second hold is the component's, tested end to end.
  test("reading, searching, writing, checking; waiting for approval first", () => {
    expect(live({})).toEqual({ kind: "reading", mind: MIND });
    expect(live({}, { phase: "searching" })).toEqual({ kind: "searching" });
    expect(live({ toolCalls: [search("s1", "heat")] }, { phase: "writing" })).toEqual({
      kind: "searching",
    });
    expect(live({ text: "Parks are cooler", toolCalls: [search("s1", "heat")] })).toEqual({
      kind: "writing",
    });
    expect(
      live({ text: "Parks are cooler [^1].", citations: [citation(1, "a", "d1", 4)] }),
    ).toEqual({ kind: "checking", count: 1 });
    expect(live({}, { phase: "checking-quotes" })).toEqual({ kind: "checking", count: 0 });
    expect(live({}, { phase: "searching", waitingForApproval: true })).toEqual({ kind: "waiting" });
    expect(live({ text: "" }, { readFrom: null })).toEqual({ kind: "writing" });
  });
});
