/**
 * Check this Mind (#737): every sentence of a Mind that makes a claim,
 * checked in one run; then only what changed. Headings, questions and the
 * User's own opinions are left out, as the model names the claims.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { CheckMindInput, Core } from "../../src/core";
import { updateCheck } from "../../src/core/answers/checks/sentences";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { claimCheckShown } from "../../src/shared/claimChecks";
import { mindCheckOf } from "../../src/shared/mindCheck";
import { checkGenerate, checkReply, isCheckRun, type Plan } from "../helpers/checks";
import { createTempDataFolder, manualClock, queryDatabase, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient, typeInto } from "../helpers/mindClient";
import { editMind, readMind, writeMind } from "../helpers/minds";
import {
  type GenerateCall,
  type ModelCall,
  type ReportedUsage,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

const TIDES = {
  name: "Tides.md",
  contents: "# Tides\n\nSpring tides come at new and full moon.\n",
};
const OFFICE = {
  name: "Harbour office.md",
  contents:
    "# Harbour office\n\nThe harbour office publishes the times of high water every week.\n",
};

const MOON = "Spring tides come at new and full moon.";
const YEARLY = "Harbours publish the times of high water every year.";
const PENGUINS = "Penguins nest on the cliffs above the harbour.";
const QUESTION = "Does high water come at noon on Sundays?";
const OPINION = "We think the harbour office is very helpful.";

const PLANS: Record<string, Plan> = {
  [MOON]: { verdict: "supported", quote: MOON },
  [YEARLY]: {
    verdict: "contradicted",
    quote: "The harbour office publishes the times of high water every week.",
    says: "The harbour office says every week, not every year.",
  },
  [PENGUINS]: { verdict: "unsupported" },
};

const text = (value: string): JSONContent => ({ type: "text", text: value });

/** The check Runs made: each starts with a call that has no Tool results yet. */
const runsIn = (calls: readonly ModelCall[]) =>
  calls.filter((call) => isCheckRun(call) && call.results.length === 0);

/**
 * A core with the two Documents in a Folder, a model on this computer (or,
 * `billed`, one billed per token, each request reporting its tokens), and
 * every call to it kept.
 */
async function setUp(billed?: { usage: ReportedUsage }) {
  const checks: ModelCall[] = [];
  const finds: GenerateCall[] = [];
  const model = scriptedModel(
    (call) => {
      checks.push(call);
      return checkReply(call, PLANS);
    },
    {
      generate: (call) => {
        if (call.system.startsWith("You sort the sentences")) finds.push(call);
        return checkGenerate(PLANS)(call);
      },
      ...(billed && { usage: billed.usage }),
    },
  );
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, {
    createChatModel: scriptedModels(model).createChatModel,
    now: manualClock("2026-10-02T14:20:00.000Z").now,
  });
  if (billed) {
    core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
    const provider = await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "sk-ant-test",
      modelId: "claude-haiku-5-5",
    });
    if (provider.service) await core.allowDataFlow("checks", provider.service.id);
  } else await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const sources = await createTempDataFolder();
  const documents = await addAndProcess(core, [
    await writeSourceFile(sources, TIDES.name, TIDES.contents),
    await writeSourceFile(sources, OFFICE.name, OFFICE.contents),
  ]);
  const folder = await core.createLibraryGroup({ name: "Harbours", description: "" });
  await core.moveToFolder({ documentIds: documents.map((each) => each.id), folderId: folder.id });
  const mind = await core.createMind({ title: "Harbour notes", folderId: folder.id });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [
    { type: "heading", attrs: { level: 2 }, content: [text("High water in the harbour")] },
    { type: "paragraph", attrs: { id: "one" }, content: [text(`${MOON} ${YEARLY} ${QUESTION}`)] },
    { type: "paragraph", attrs: { id: "two" }, content: [text(`${OPINION} ${PENGUINS}`)] },
  ]);
  await client.settled();
  return { core, client, dataDir, mindId: mind.id, folderId: folder.id, checks, finds };
}

const request = (mindId: string, mode: CheckMindInput["mode"], id = "run"): CheckMindInput => ({
  requestId: `${id}-${mode}`,
  mindId,
  mode,
});

/** Each check's mark in the Mind, as the editor shows it now. */
async function shownChecks(core: Core, client: MindClient, folderId: string | null) {
  await client.settled();
  const versions = new Map((await core.listDocuments()).map((each) => [each.id, each.contentHash]));
  return claimSpans(readMind(client)).map((span) => ({
    sentence: span.sentence.text.trim(),
    shown: claimCheckShown(span.check, { text: span.sentence.text, versions, folderId }),
  }));
}

describe("Check this Mind", () => {
  test("checks every claim, each marked; the heading, the question and the User's opinion aren't; the run is recorded", async () => {
    const { core, client, mindId, folderId, checks, finds } = await setUp();
    expect(await core.planMindCheck(mindId, "all")).toEqual({
      kind: "ready",
      sentences: 5,
      findsClaims: true,
      inFolder: true,
      model: null,
    });

    expect(await core.checkMind(request(mindId, "all"))).toEqual({
      kind: "checked",
      sentences: 3,
      checked: 3,
      smallModel: false,
      stop: null,
    });
    expect(
      (await shownChecks(core, client, folderId)).map(({ sentence, shown }) => [sentence, shown]),
    ).toEqual([
      [MOON, { state: "supported" }],
      [YEARLY, { state: "contradicted" }],
      [PENGUINS, { state: "unsupported" }],
    ]);
    // One request names the claims among the five sentences; one check Run each.
    expect(finds).toHaveLength(1);
    expect(finds[0]?.system).toMatch(/^You sort the sentences of the User's own draft/);
    expect(runsIn(checks)).toHaveLength(3);
    expect(mindCheckOf(client.doc)).toMatchObject({
      at: "2026-10-02T14:20:00.000Z",
      sentences: 3,
      stop: null,
    });
    expect(mindCheckOf(client.doc)?.skipped).toHaveLength(2);

    // Nothing has changed since: nothing to check.
    expect(await core.planMindCheck(mindId, "changed")).toEqual({ kind: "nothing" });
  });

  test("Check again after two claims are edited checks those two only, without naming the claims again", async () => {
    const { core, client, mindId, folderId, checks, finds } = await setUp();
    await core.checkMind(request(mindId, "all"));
    const runsBefore = runsIn(checks).length;

    typeInto(client, 1, 0, "Big ");
    typeInto(client, 2, OPINION.length + 1, "Some ");
    expect((await shownChecks(core, client, folderId)).map(({ shown }) => shown.state)).toEqual([
      "needs-recheck",
      "contradicted",
      "needs-recheck",
    ]);

    expect(await core.planMindCheck(mindId, "changed")).toEqual({
      kind: "ready",
      sentences: 2,
      findsClaims: false,
      inFolder: true,
      model: null,
    });
    expect(await core.checkMind(request(mindId, "changed"))).toMatchObject({
      kind: "checked",
      sentences: 2,
    });
    // Two check Runs, and the model wasn't asked again which sentences claim something.
    expect(runsIn(checks).length - runsBefore).toBe(2);
    expect(finds).toHaveLength(1);
    expect((await shownChecks(core, client, folderId)).map(({ shown }) => shown.state)).toEqual([
      "unsupported",
      "contradicted",
      "unsupported",
    ]);
  });

  // protects=a sentence the User keeps as their own view (#1111) isn't checked again, in any mode, until its words change; fails_when=a run sends it, or an edit of it doesn't bring it back; no_claim=how the card sets it.
  test("a sentence kept as the User's own view isn't checked again until its words change", async () => {
    const { core, client, mindId, folderId } = await setUp();
    await core.checkMind(request(mindId, "all"));
    const penguins = claimSpans(readMind(client)).find((span) =>
      span.sentence.text.startsWith(PENGUINS),
    );
    client.doc.transact(() => {
      updateCheck(client.blocks, penguins?.check.id ?? "", (check) => ({ ...check, kept: true }));
    }, "editor");
    await client.settled();
    expect((await shownChecks(core, client, folderId)).map(({ shown }) => shown.state)).toEqual([
      "supported",
      "contradicted",
      "kept",
    ]);

    expect(await core.planMindCheck(mindId, "all")).toMatchObject({ kind: "ready", sentences: 4 });
    expect(await core.planMindCheck(mindId, "changed")).toEqual({ kind: "nothing" });

    typeInto(client, 2, OPINION.length + 1, "Some ");
    expect(await core.planMindCheck(mindId, "changed")).toMatchObject({ sentences: 1 });
  });

  test("a sentence written since is found among the new ones only, and the opinion isn't sent again", async () => {
    const { core, client, mindId, finds } = await setUp();
    await core.checkMind(request(mindId, "all"));
    const added = "The harbour master retired in 2025.";
    editMind(client, (blocks) => [
      ...blocks,
      { type: "paragraph", attrs: { id: "three" }, content: [text(added)] },
    ]);
    await client.settled();

    expect(await core.planMindCheck(mindId, "changed")).toMatchObject({
      sentences: 1,
      findsClaims: true,
    });
    await core.checkMind(request(mindId, "changed"));
    expect(finds).toHaveLength(2);
    expect(finds[1]?.prompt).toBe(`[1] ${added}`);
  });

  test("the monthly cap stops the run: the claims left say “not checked: cost cap”", async () => {
    const checks: ModelCall[] = [];
    let core: Core | null = null;
    const model = scriptedModel(
      (call) => {
        checks.push(call);
        return checkReply(call, PLANS);
      },
      {
        generate: (call) => {
          // The claims are named, then the cap is reached before any is checked.
          if (call.system.startsWith("You sort the sentences")) {
            void core?.updateSettings({ user: { monthlyCap: { amount: 0, currency: "USD" } } });
          }
          return checkGenerate(PLANS)(call);
        },
      },
    );
    core = startCore(await createTempDataFolder(), { createChatModel: () => model });
    core.on("consent.requested", (asked) => void core?.respondToConsent(asked.requestId, true));
    await core.saveChatProvider({
      kind: "anthropic",
      modelId: "claude-haiku-5-5",
      apiKey: "sk-test",
    });
    const sources = await createTempDataFolder();
    await addAndProcess(core, [await writeSourceFile(sources, TIDES.name, TIDES.contents)]);
    const mind = await core.createMind({ title: "Harbour notes", folderId: null });
    const client = await connectToMind(core, mind.id);
    writeMind(client, [
      { type: "paragraph", attrs: { id: "all" }, content: [text(`${MOON} ${YEARLY} ${PENGUINS}`)] },
    ]);
    await client.settled();

    expect(await core.checkMind(request(mind.id, "all"))).toEqual({
      kind: "checked",
      sentences: 3,
      checked: 0,
      smallModel: false,
      stop: { reason: "cost-cap" },
    });
    expect((await shownChecks(core, client, null)).map(({ shown }) => shown)).toEqual(
      [MOON, YEARLY, PENGUINS].map(() => ({ state: "not-checked", reason: "cost-cap" })),
    );
    expect(runsIn(checks)).toEqual([]);
    expect(mindCheckOf(client.doc)?.stop).toEqual({ reason: "cost-cap" });
  });

  // protects=the ask counts exactly the claims a run checks, and the model sorts each sentence once (#973); fails_when=the ask counts sentences the model leaves out, Check sorts again, or declining spends more; no_claim=what a real model names.
  test("the claims named before the ask are exactly those checked, and neither Check nor a second run sorts them again; declining spends nothing more", async () => {
    const { core, client, mindId, checks, finds } = await setUp();
    const memo = "This memo sums up what the harbour office told us.";
    editMind(client, (blocks) => [
      ...blocks,
      { type: "paragraph", attrs: { id: "three" }, content: [text(memo)] },
    ]);
    await client.settled();
    expect(await core.planMindCheck(mindId, "changed")).toMatchObject({
      sentences: 6,
      findsClaims: true,
    });

    expect(await core.findMindClaims(request(mindId, "changed", "sort"))).toEqual({
      kind: "found",
      claims: 3,
    });
    expect(finds).toHaveLength(1);
    expect(checks).toHaveLength(0);
    // Declined: asked again, nothing is sent again.
    expect(await core.findMindClaims(request(mindId, "changed", "again"))).toEqual({
      kind: "found",
      claims: 3,
    });
    expect(finds).toHaveLength(1);
    expect(checks).toHaveLength(0);

    expect(await core.checkMind(request(mindId, "changed"))).toMatchObject({
      kind: "checked",
      sentences: 3,
    });
    expect(finds).toHaveLength(1);
    expect(mindCheckOf(client.doc)?.skipped).toHaveLength(3);
    expect(await core.planMindCheck(mindId, "changed")).toEqual({ kind: "nothing" });
  });

  // protects=on a model billed per token, the estimate for the claims named comes within 1.5 times the run's tokens; fails_when=it prices a check as an Answer, or counts sentences the model leaves out; no_claim=a real model's tokens (tests/core/cost.test.ts has those).
  test("on a model billed per token, the estimate for the claims named comes within 1.5 times the tokens the run then reports", async () => {
    // Each request as an average check of Claude Haiku 5.5 reported it (tests/fixtures/cost/recorded-checks.json).
    const usage: ReportedUsage = {
      inputTokens: { total: 8_260, noCache: 995, cacheRead: 2_903, cacheWrite: 4_362 },
      outputTokens: { total: 129, text: 129, reasoning: undefined },
    };
    const { core, mindId, dataDir } = await setUp({ usage });
    const found = await core.findMindClaims(request(mindId, "changed", "sort"));
    if (found.kind !== "found") throw new Error(`No claims: ${found.kind}.`);

    const estimate = await core.estimateRun({ items: found.claims, each: "check" });
    await core.checkMind(request(mindId, "changed"));

    if (estimate.billing !== "per-token") throw new Error("Not billed per token.");
    const [run] = queryDatabase<{ tokens: number }>(
      dataDir,
      `SELECT sum(input_tokens + output_tokens) AS tokens FROM model_requests
       WHERE deleted_at IS NULL AND status = 'settled' AND flow = 'checks'`,
    );
    expect(estimate.tokens / (run?.tokens ?? 0)).toBeGreaterThan(1 / 1.5);
    expect(estimate.tokens / (run?.tokens ?? 0)).toBeLessThan(1.5);
  });

  test("says why there is nothing to check: no sentences; a Mind in a Folder with no Documents checks every Document (#1359)", async () => {
    const { core } = await setUp();
    const empty = await core.createMind({ title: "Empty", folderId: null });
    expect(await core.planMindCheck(empty.id, "all")).toEqual({ kind: "nothing" });
    const emptyFolder = await core.createLibraryGroup({ name: "Nothing yet", description: "" });
    const lonely = await core.createMind({ title: "Lonely", folderId: emptyFolder.id });
    const client = await connectToMind(core, lonely.id);
    writeMind(client, [{ type: "paragraph", attrs: { id: "moon" }, content: [text(MOON)] }]);
    await client.settled();
    expect(await core.planMindCheck(lonely.id, "all")).toMatchObject({
      kind: "ready",
      sentences: 1,
    });
  });
});
