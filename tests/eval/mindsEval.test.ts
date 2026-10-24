/**
 * The evaluation of requests that change several Minds (eval/minds): what its
 * checks catch, on scripted runs through the core, and its scoring. The real
 * models it is run on are not asked here.
 */
import { describe, expect, test } from "vitest";
import { ask } from "../../eval/everyday/asking";
import { CASES, type MindsCase, WORKSPACES } from "../../eval/minds/cases";
import { sentencesIn } from "../../eval/minds/checks";
import { examine } from "../../eval/minds/examine";
import { BAR, median, type RunResult, rowsOf } from "../../eval/minds/report";
import { readWorkspace, seedWorkspace } from "../../eval/minds/state";
import { createTempDataFolder, startCore } from "../helpers/core";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

const caseOf = (id: string): MindsCase => {
  const found = CASES.find((each) => each.id === id);
  if (!found) throw new Error(`No request ${id}.`);
  return found;
};

const rewrittenTerm = ({ prompt }: ModelCall): ScriptedReply | null => {
  const passage = /<passage>\n([\s\S]*?)\n<\/passage>/.exec(prompt)?.[1];
  return passage === undefined
    ? null
    : { text: passage.replaceAll("supplier", "vendor").replaceAll("Supplier", "Vendor") };
};

/** Asks `id` of a scripted model, and holds the run to the request, as the evaluation does. */
async function runScripted(
  id: string,
  script: (call: ModelCall) => ScriptedReply,
  unbacked: readonly string[] = [],
) {
  const each = caseOf(id);
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(scriptedModel(script)).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  await seedWorkspace(core, WORKSPACES[each.language]);
  const before = await readWorkspace(core);
  const open = before.find((mind) => mind.title === each.open);
  const answer = await ask(core, {
    text: each.request,
    mindId: open?.id ?? null,
    conversationId: null,
    focus: null,
  });
  return {
    answer,
    ...(await examine(core, each, before, answer, async () => unbacked)),
  };
}

const edits =
  (titles: readonly string[]) =>
  (call: ModelCall): ScriptedReply =>
    rewrittenTerm(call) ??
    (call.results.length === 0
      ? {
          calls: titles.map((mind) => ({
            tool: "edit_mind",
            input: { mind, instruction: "Use vendor." },
          })),
        }
      : { text: "Changed it." });

const HARBOUR = ["Harbour plan", "Harbour budget", "Harbour risks"];

describe("The evaluation of requests that change several Minds", { timeout: 60_000 }, () => {
  // protects=a run that does what was asked passes every check, and the card, Reject all, Undo and Accept all are exercised on it; fails_when=a correct run fails, or the card lists other Minds or counts; no_claim=how a real model does it.
  test("a request done right passes", async () => {
    const run = await runScripted("term-three-en", edits(HARBOUR));

    expect(run.failures).toEqual([]);
    expect(run.card.map((row) => row.mind)).toEqual(HARBOUR);
    expect(run.card.map((row) => row.changed)).toEqual([2, 3, 2]);
  });

  // protects=a Mind changed that was not asked for is named, with its card row; fails_when=Warehouse notes changes unseen; no_claim=none beyond protects.
  test("a Mind changed without being asked for fails the run", async () => {
    const run = await runScripted("term-three-en", edits([...HARBOUR, "Warehouse notes"]));

    expect(run.failures).toEqual([
      { kind: "wrong-minds", missing: [], extra: ["Warehouse notes"] },
    ]);
  });

  // protects=a Mind left alone that was asked for is named; fails_when=two of three changed passes; no_claim=none beyond protects.
  test("a Mind asked for and left alone fails the run", async () => {
    const run = await runScripted("term-three-en", edits(HARBOUR.slice(0, 2)));

    expect(run.failures).toEqual([{ kind: "wrong-minds", missing: ["Harbour risks"], extra: [] }]);
  });

  // protects=text that is not just the term swapped fails with the Block that differs; fails_when=a rewrite that changes other words passes; no_claim=none beyond protects.
  test("a rewrite that changes more than the term fails the run", async () => {
    const run = await runScripted("term-three-en", (call) => {
      const rewritten = rewrittenTerm(call);
      return rewritten && "text" in rewritten
        ? { text: (rewritten.text ?? "").replace("two batches", "three batches") }
        : edits(HARBOUR)(call);
    });

    expect(run.failures).toEqual([
      {
        kind: "wrong-text",
        mind: "Harbour plan",
        why: expect.stringContaining("three batches"),
      },
    ]);
  });

  // protects=a claim the judge flags fails the run, whatever the Minds hold; fails_when=a flagged sentence passes; no_claim=the judge itself, which the everyday checks cover.
  test("a sentence the judge flags fails the run", async () => {
    const run = await runScripted("term-three-en", edits(HARBOUR), [
      "I also fixed Warehouse notes.",
    ]);

    expect(run.failures).toEqual([
      { kind: "unbacked", sentences: ["I also fixed Warehouse notes."], flagged: false },
    ]);
  });

  // protects=a request for a Mind that is not there passes only when nothing changes and the reply names it; fails_when=a reply that never names the Mind passes, or a change to the open Mind goes unseen; no_claim=none beyond protects.
  test("a Mind that is not there is named, and nothing changes", async () => {
    const named = await runScripted("missing-mind-en", (call) =>
      call.results.length === 0
        ? {
            calls: [
              { tool: "edit_mind", input: { mind: "Budget forecast", instruction: "Shorten." } },
            ],
          }
        : { text: "I couldn't find a Mind called Budget forecast, so nothing changed." },
    );
    expect(named.failures).toEqual([]);
    expect(named.card).toEqual([]);

    const unnamed = await runScripted("missing-mind-en", () => ({ text: "Done." }));
    expect(unnamed.failures).toEqual([{ kind: "unnamed", name: "Budget forecast" }]);

    const open = await runScripted("missing-mind-en", (call) => {
      const rewritten = rewrittenTerm(call);
      if (rewritten) return { text: "Goals are short." };
      return call.results.length === 0
        ? {
            calls: [
              {
                tool: "edit_mind",
                input: { mind: "Project plan", part: "Goals", instruction: "Shorten." },
              },
            ],
          }
        : { text: "I couldn't find Budget forecast; I shortened Goals in Project plan." };
    });
    expect(open.failures).toContainEqual({
      kind: "wrong-minds",
      missing: [],
      extra: ["Project plan"],
    });
  });

  // protects=a new Mind is listed by the card as new, Reject all removes it and Undo brings it back, and Accept all keeps it; fails_when=the Mind is left behind by Reject all, not restored by Undo, or its facts are not required; no_claim=none beyond protects.
  test("a Mind made from two others passes with its card, Reject all and Undo", async () => {
    const run = await runScripted("new-from-two-en", (call) =>
      call.results.length === 0
        ? {
            calls: [
              {
                tool: "create_mind",
                input: {
                  title: "Launch brief",
                  content:
                    "# Launch brief\n\nBeta opens on 2 February with the Nordic pilot group.\n\nHire a part-time designer.",
                },
              },
            ],
          }
        : { text: "Made the Launch brief." },
    );

    expect(run.failures).toEqual([]);
    expect(run.card).toEqual([{ mind: "Launch brief", created: true, changed: 0, wrote: 2 }]);

    const thin = await runScripted("new-from-two-en", (call) =>
      call.results.length === 0
        ? {
            calls: [
              {
                tool: "create_mind",
                input: { title: "Launch brief", content: "# Launch brief\n\nSoon." },
              },
            ],
          }
        : { text: "Made the Launch brief." },
    );
    expect(thin.failures).toContainEqual({
      kind: "wrong-text",
      mind: "Launch brief",
      why: expect.stringContaining("Nordic pilot"),
    });
  });
});

describe("The requests and their scoring", () => {
  // protects=every request is asked in English and in Chinese, against a workspace that holds the Minds it names; fails_when=a request names a Mind its workspace lacks, or a language has fewer requests; no_claim=how well a model does them.
  test("each request has a pair, over a workspace with the Minds it names", () => {
    for (const each of CASES) {
      const titles = WORKSPACES[each.language].map((mind) => mind.title);
      for (const title of [each.open, ...Object.keys(each.expected.changed)]) {
        if (title !== null) expect(titles, each.id).toContain(title);
      }
    }
    for (const language of ["en", "zh-CN"] as const) {
      expect(CASES.filter((each) => each.language === language)).toHaveLength(CASES.length / 2);
    }
    expect(WORKSPACES.en).toHaveLength(WORKSPACES["zh-CN"].length);
  });

  // protects=a sentence ends at its mark, not at a decimal point or an abbreviation's dot inside a number; fails_when=4.2 counts as two sentences; no_claim=none beyond protects.
  test("sentences are counted by their closing marks", () => {
    expect(sentencesIn("Revenue was €4.2 million. It grew 6%.")).toBe(2);
    expect(sentencesIn("Revenue grew")).toBe(1);
    expect(sentencesIn("收入增长了。支出持平！")).toBe(2);
  });

  // protects=a request is under the bar below 9 of 10 runs and not at 9; fails_when=9 of 10 is under the bar, or an error counts as a pass; no_claim=none beyond protects.
  test("a request is under the bar below 9 of 10 runs", () => {
    const result = (run: number, passed: boolean): RunResult => ({
      id: "r",
      language: "en",
      request: "",
      model: "m",
      run,
      failures: passed ? [] : [{ kind: "card", why: "x" }],
      error: null,
      reply: "",
      answeredSeconds: 1,
      seconds: 1,
      chat: {
        requests: 0,
        inputTokens: 0,
        cachedInputTokens: 0,
        cacheWriteTokens: 0,
        outputTokens: 0,
      },
      judging: {
        requests: 0,
        inputTokens: 0,
        cachedInputTokens: 0,
        cacheWriteTokens: 0,
        outputTokens: 0,
      },
      retries: 0,
      calls: [],
      card: [],
      transcript: [],
    });
    const runs = (passes: number) =>
      Array.from({ length: 10 }, (_, index) => result(index + 1, index < passes));
    expect(BAR).toBe(0.9);
    expect(rowsOf(runs(9))[0]).toMatchObject({ passed: 9, runs: 10, under: false });
    expect(rowsOf(runs(8))[0]).toMatchObject({ passed: 8, runs: 10, under: true });
    expect(rowsOf([{ ...result(1, true), error: "No Answer." }])[0]).toMatchObject({ passed: 0 });
    expect(median([3, 1, 2])).toBe(2);
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });
});
