import { writeFile } from "node:fs/promises";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test, vi } from "vitest";
import type { ClaimCheckAction, ClaimCheckAttributes, Core, Document } from "../../src/core";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { claimCheckShown } from "../../src/shared/claimChecks";
import { checkingModel as checkingModelOf, type Plan, sentenceOf } from "../helpers/checks";
import { setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, manualClock, queryDatabase, startCore } from "../helpers/core";
import { addAndProcess, sha256, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient, typeInto } from "../helpers/mindClient";
import { readMind, writeMind } from "../helpers/minds";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

const TIDES = {
  name: "Tides.md",
  contents: [
    "# Tides",
    "",
    "Spring tides happen at new moon and at full moon, when the Sun and the Moon pull together.",
    "",
    "Harbours publish the times of high water every year.",
  ].join("\n"),
};

const SUPPORTED = "Spring tides come at new and full moon.";
const CONTRADICTED = "Harbours publish the times of high water every week.";
const UNSUPPORTED = "Penguins nest on the cliffs above the harbour.";

const PLANS: Record<string, Plan> = {
  [SUPPORTED]: {
    verdict: "supported",
    quote: "Spring tides happen at new moon and at full moon",
    says: "Tides says spring tides happen at new and full moon.",
  },
  [CONTRADICTED]: {
    verdict: "contradicted",
    quote: "Harbours publish the times of high water every year.",
    says: "Tides says harbours publish them every year, not every week.",
  },
  [UNSUPPORTED]: { verdict: "unsupported" },
};

/** A model that checks as `PLANS` say, unless told otherwise. */
const checkingModel = (
  plans: Record<string, Plan> = PLANS,
  calls: ModelCall[] = [],
  evidence?: string,
) => checkingModelOf(plans, calls, evidence);

let blocks = 0;
const paragraph = (...content: JSONContent[]): JSONContent & { attrs: { id: string } } => ({
  type: "paragraph",
  attrs: { id: `block-${++blocks}` },
  content,
});
const text = (value: string, check?: ClaimCheckAttributes): JSONContent => ({
  type: "text",
  text: value,
  ...(check && { marks: [{ type: "claimCheck", attrs: check }] }),
});

/** A check as a crash leaves it: "checking", with no request running. */
const LEFT_CHECKING: ClaimCheckAttributes = {
  id: "left-checking",
  request: "",
  action: "find-support",
  state: "checking",
  reason: null,
  detail: null,
  text: SUPPORTED,
  evidence: null,
  says: null,
  scope: { folderId: null, widened: false },
  versions: [],
  excluded: [],
  checkedAt: "2026-10-12T09:00:00.000Z",
};

/** Selects the whole of a paragraph's text. */
const whole = (block: { attrs: { id: string } }, length: number) => ({
  from: { blockId: block.attrs.id, offset: 0 },
  to: { blockId: block.attrs.id, offset: length },
});

/** Each check in the Mind, as the editor shows it now. */
async function shownChecks(core: Core, client: MindClient, folderId: string | null = null) {
  await client.settled();
  const versions = new Map((await core.listDocuments()).map((each) => [each.id, each.contentHash]));
  return claimSpans(readMind(client)).map((span) => ({
    check: span.check,
    sentence: span.sentence.text,
    shown: claimCheckShown(span.check, { text: span.sentence.text, versions, folderId }),
  }));
}

/** A Mind with one paragraph of these sentences and the Tides Document, checked by `model`. */
async function mindWith(sentences: readonly string[], model = checkingModel()) {
  const setup = await setUpWithDocuments(model, [TIDES]);
  const body = sentences.join(" ");
  const block = paragraph(text(body));
  writeMind(setup.client, [block]);
  await setup.client.settled();
  const check = (action: ClaimCheckAction = "find-support", range = whole(block, body.length)) =>
    setup.core.checkSentences({
      requestId: crypto.randomUUID(),
      mindId: setup.mind.id,
      action,
      ...range,
    });
  return { ...setup, block, body, check, tides: setup.documents[0] as Document };
}

describe("Find support", () => {
  test("marks each sentence the selection touches with what its check found", async () => {
    const { core, client, check, tides } = await mindWith([SUPPORTED, CONTRADICTED, UNSUPPORTED]);

    expect(await check()).toEqual({ kind: "checked", sentences: 3, checked: 3 });

    const checks = await shownChecks(core, client);
    expect(checks.map((each) => [each.sentence, each.shown])).toEqual([
      [SUPPORTED, { state: "supported" }],
      [CONTRADICTED, { state: "contradicted" }],
      [UNSUPPORTED, { state: "unsupported" }],
    ]);
    const [supported, contradicted, unsupported] = checks.map((each) => each.check);
    expect(supported).toMatchObject({
      action: "find-support",
      text: SUPPORTED,
      evidence: {
        documentId: tides.id,
        contentHash: tides.contentHash,
        quote: "Spring tides happen at new moon and at full moon",
        check: "found",
      },
      scope: { folderId: null, widened: false },
      versions: [{ documentId: tides.id, contentHash: tides.contentHash }],
    });
    expect(contradicted).toMatchObject({
      says: "Tides says harbours publish them every year, not every week.",
      evidence: { quote: "Harbours publish the times of high water every year." },
    });
    expect(unsupported).toMatchObject({ evidence: null, says: null });
  });

  test("a sentence only part selected is checked whole, and the others aren't", async () => {
    const { core, client, block, check } = await mindWith([SUPPORTED, UNSUPPORTED]);
    const inSecond = SUPPORTED.length + 5;

    expect(
      await check("find-support", {
        from: { blockId: block.attrs.id, offset: inSecond },
        to: { blockId: block.attrs.id, offset: inSecond + 3 },
      }),
    ).toEqual({ kind: "checked", sentences: 1, checked: 1 });
    expect((await shownChecks(core, client)).map((each) => each.sentence)).toEqual([UNSUPPORTED]);
  });

  test.each([
    "Verdict: unsupported",
    "**Verdict:** Unsupported",
    "P1 is about spring tides, not penguins.\n\nVerdict: unsupported",
    "Unsupported. No Passage mentions penguins.",
    '{"verdict": "unsupported"}',
    "Verdict: contradicted\nP1 is about another harbour, so it doesn't bear on this.\nVerdict: unsupported",
  ])("reads the verdict however the model words it: %j", async (reply) => {
    const plans = { [UNSUPPORTED]: { reply: { text: reply } } };
    const { core, client, check } = await mindWith([UNSUPPORTED], checkingModel(plans));
    await check();
    expect((await shownChecks(core, client))[0]?.shown).toEqual({ state: "unsupported" });
  });

  test("with nothing to check against, nothing is sent", async () => {
    const calls: ModelCall[] = [];
    const model = checkingModel(PLANS, calls);
    const core = startCore(await createTempDataFolder(), {
      createChatModel: scriptedModels(model).createChatModel,
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const mind = await core.createMind({ title: "Empty" });
    const client = await connectToMind(core, mind.id);
    const block = paragraph(text(SUPPORTED));
    writeMind(client, [block]);
    await client.settled();

    expect(
      await core.checkSentences({
        requestId: "r1",
        mindId: mind.id,
        action: "find-support",
        ...whole(block, SUPPORTED.length),
      }),
    ).toEqual({ kind: "no-documents", folderId: null });
    expect(calls).toEqual([]);
  });
});

describe("One claim check", () => {
  test("Find support and Check against Documents ask the same check Run, offered only search and cite", async () => {
    const calls: ModelCall[] = [];
    const { core, client, check } = await mindWith([SUPPORTED], checkingModel(PLANS, calls));
    await check("find-support");
    const findSupport = calls.splice(0);

    const cited = paragraph(
      text(`${SUPPORTED.slice(0, -1)} `),
      {
        type: "citation",
        attrs: {
          documentName: "Tides",
          quote: "Spring tides happen at new moon and at full moon",
          check: "found",
        },
      },
      text("."),
    );
    writeMind(client, [cited]);
    await client.settled();
    await check("check-against-documents", whole(cited, SUPPORTED.length + 1));
    const againstDocuments = calls.splice(0);

    for (const call of [...findSupport, ...againstDocuments]) {
      expect(call.tools.sort()).toEqual(["cite", "search_documents"]);
    }
    expect(againstDocuments[0]?.system).toBe(findSupport[0]?.system);
    const asked = (call: ModelCall | undefined) =>
      call?.options.prompt.find((message) => message.role === "user")?.content;
    expect(JSON.stringify(asked(againstDocuments[0]))).toContain(
      "Tides: “Spring tides happen at new moon and at full moon”",
    );
    expect(JSON.stringify(asked(findSupport[0]))).not.toContain("The User cites this");
    expect((await shownChecks(core, client)).map((each) => each.check.action)).toEqual([
      "check-against-documents",
    ]);
  });

  test("a contradiction counts only once the sentence and its quote, judged alone, conflict", async () => {
    const near = { ...PLANS[CONTRADICTED], judgement: "neither" as const };
    const { core, client, check } = await mindWith(
      [CONTRADICTED],
      checkingModel({ [CONTRADICTED]: near }),
    );
    await check();
    expect((await shownChecks(core, client))[0]?.shown).toEqual({ state: "unsupported" });

    const same = { ...PLANS[CONTRADICTED], judgement: "same" as const };
    const again = await mindWith([CONTRADICTED], checkingModel({ [CONTRADICTED]: same }));
    await again.check();
    expect((await shownChecks(again.core, again.client))[0]).toMatchObject({
      shown: { state: "supported" },
      check: { evidence: { quote: "Harbours publish the times of high water every year." } },
    });
  });

  test("checkClaim is the same check, for a claim outside a Mind", async () => {
    const { core, documents } = await setUpWithDocuments(checkingModel(), [TIDES]);
    const tides = documents[0] as Document;

    expect(await core.checkClaim({ claim: SUPPORTED })).toMatchObject({
      verdict: "supported",
      evidence: { documentId: tides.id, check: "found" },
      versions: [{ documentId: tides.id, contentHash: tides.contentHash }],
    });
    // The document being checked is never its own evidence (#80).
    expect(
      await core.checkClaim({ claim: SUPPORTED, checked: { documentIds: [tides.id] } }),
    ).toMatchObject({ verdict: "unsupported", versions: [] });
  });
});

describe("When the model can't check a sentence", () => {
  const failing = (reply: ScriptedReply): Plan => ({ reply });
  test.each<{ reason: ClaimCheckAttributes["reason"]; plan: Plan }>([
    { reason: "refused", plan: failing({ text: "I'm sorry, I can't help with that." }) },
    { reason: "malformed", plan: failing({ text: "It reads well to me." }) },
    { reason: "empty", plan: failing({ text: "" }) },
    {
      reason: "rate-limit",
      plan: failing({ error: { status: 429, message: "Rate limit reached." } }),
    },
    // A verdict whose quote isn't in the Passage it names.
    {
      reason: "no-evidence",
      plan: {
        verdict: "supported",
        quote: "Spring tides happen twice a year",
        says: "It says so.",
      },
    },
  ])(
    "$reason: the sentence says “not checked” with why, never supported",
    async ({ reason, plan }) => {
      const { core, client, check } = await mindWith(
        [SUPPORTED],
        checkingModel({ [SUPPORTED]: plan }),
      );

      expect(await check()).toEqual({ kind: "checked", sentences: 1, checked: 0 });

      const [checked] = await shownChecks(core, client);
      expect(checked?.shown).toEqual({ state: "not-checked", reason });
      expect(checked?.check.evidence).toBeNull();
    },
  );

  test("a verdict given without its Passage is asked for once more, and counts once its quote is found", async () => {
    const plans: Record<string, Plan> = {
      [SUPPORTED]: { verdict: "supported", says: "Tides says so." },
    };
    const model = checkingModel(
      plans,
      [],
      JSON.stringify({ passage: "P1", quote: "Spring tides happen at new moon and at full moon" }),
    );
    const { core, client, check } = await mindWith([SUPPORTED], model);

    expect(await check()).toEqual({ kind: "checked", sentences: 1, checked: 1 });

    const [checked] = await shownChecks(core, client);
    expect(checked?.shown).toEqual({ state: "supported" });
    expect(checked?.check.evidence).toMatchObject({
      quote: "Spring tides happen at new moon and at full moon",
      check: "found",
    });
    const asked = model.doGenerateCalls.filter((call) =>
      JSON.stringify(call.prompt).includes("without recording the Passage"),
    );
    expect(asked).toHaveLength(1);
  });

  test("cost cap: nothing is sent, and the sentence says “not checked” with why", async () => {
    const calls: ModelCall[] = [];
    const model = checkingModel(PLANS, calls);
    const folder = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(folder, {
      createChatModel: () => model,
      now: manualClock("2026-10-12T09:00:00.000Z").now,
    });
    core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
    await core.saveChatProvider({
      kind: "anthropic",
      modelId: "claude-haiku-5-5",
      apiKey: "sk-test",
    });
    await core.updateSettings({ user: { monthlyCap: { amount: 0, currency: "USD" } } });
    await addAndProcess(core, [await writeSourceFile(sources, TIDES.name, TIDES.contents)]);
    const mind = await core.createMind({ title: "Capped" });
    const client = await connectToMind(core, mind.id);
    const block = paragraph(text(SUPPORTED));
    writeMind(client, [block]);
    await client.settled();

    await core.checkSentences({
      requestId: "capped",
      mindId: mind.id,
      action: "find-support",
      ...whole(block, SUPPORTED.length),
    });

    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual([
      { state: "not-checked", reason: "cost-cap" },
    ]);
    expect(calls).toEqual([]);
  });
});

describe("Needs re-check", () => {
  test("editing a checked sentence, inside or at its start, makes its check need a re-check", async () => {
    const { core, client, check } = await mindWith([SUPPORTED, UNSUPPORTED]);
    await check();
    const before = await shownChecks(core, client);
    expect(before.map((each) => each.shown.state)).toEqual(["supported", "unsupported"]);

    typeInto(client, 0, SUPPORTED.indexOf("moon"), "blue ");
    typeInto(client, 0, 0, "Not all ");

    const shown = (await shownChecks(core, client)).map((each) => [each.sentence, each.shown]);
    expect(shown).toEqual([
      [
        `Not all ${SUPPORTED.replace("moon", "blue moon")}`,
        { state: "needs-recheck", why: "edited" },
      ],
      [UNSUPPORTED, { state: "unsupported" }],
    ]);
  });

  test("a new version of its evidence Document makes its check need a re-check", async () => {
    const { core, client, check, tides, dataDir } = await mindWith([SUPPORTED]);
    await check();
    expect((await shownChecks(core, client))[0]?.shown).toEqual({ state: "supported" });

    const changed = `${TIDES.contents}\n\nNeap tides come in between.`;
    await writeFile(tides.path, changed);
    await core.reconcileDocuments();
    const [processed] = await waitForProcessing(core, [tides.id]);
    expect(processed?.contentHash).toBe(sha256(changed));

    expect((await shownChecks(core, client))[0]?.shown).toEqual({
      state: "needs-recheck",
      why: "new-version",
    });
    // Its evidence's text is kept for its card and Add as Citation (#79).
    expect(
      queryDatabase(
        dataDir,
        "SELECT content_hash FROM retained_versions WHERE kind = 'check-result' AND deleted_at IS NULL",
      ),
    ).toEqual([{ content_hash: tides.contentHash }]);
  });

  test("a Mind deleted permanently lets go of its checks' evidence", async () => {
    const { core, check, mind, dataDir } = await mindWith([SUPPORTED]);
    await check();
    const retained = () =>
      queryDatabase<{ deleted: number }>(
        dataDir,
        "SELECT deleted_at IS NOT NULL AS deleted FROM retained_versions WHERE kind = 'check-result'",
      );
    expect(retained()).toEqual([{ deleted: 0 }]);

    await core.deleteMind(mind.id);
    await core.deletePermanently([{ kind: "mind", id: mind.id }]);

    expect(retained()).toEqual([{ deleted: 1 }]);
  });

  test("moving the Mind to another Folder makes its checks need a re-check", async () => {
    const { core, client, check, mind } = await mindWith([SUPPORTED]);
    await check();
    const folder = await core.createLibraryGroup({ name: "Coasts", description: "" });
    await core.moveToFolder({ mindIds: [mind.id], folderId: folder.id });

    expect((await shownChecks(core, client, folder.id))[0]?.shown).toEqual({
      state: "needs-recheck",
      why: "scope",
    });
  });
});

describe("Stopping", () => {
  test("a check left “checking” by a crash says “stopped” once its Mind opens", async () => {
    const { core, client, mind } = await mindWith([SUPPORTED]);
    const left = { ...LEFT_CHECKING, request: "a run that no longer exists" };
    writeMind(client, [paragraph(text(SUPPORTED, left))]);
    await client.settled();

    await core.openMind(mind.id);

    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual([
      { state: "not-checked", reason: "stopped" },
    ]);
  });

  test("a check stopped halfway leaves the sentences not checked yet “stopped”", async () => {
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const model = scriptedModel((call) => {
      if (sentenceOf(call) === UNSUPPORTED)
        return { text: "Verdict: unsupported", pause: { after: 0, until: held } };
      return { text: "Verdict: unsupported" };
    });
    const { core, client, mind, block, body } = await mindWith([SUPPORTED, UNSUPPORTED], model);
    const running = core.checkSentences({
      requestId: "stop-me",
      mindId: mind.id,
      action: "find-support",
      ...whole(block, body.length),
    });
    await vi.waitFor(async () =>
      expect((await shownChecks(core, client)).map((each) => each.check.state)).toEqual([
        "unsupported",
        "checking",
      ]),
    );

    await core.cancelCheck("stop-me");
    release();

    expect(await running).toEqual({ kind: "cancelled" });
    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual([
      { state: "unsupported" },
      { state: "not-checked", reason: "stopped" },
    ]);
  });
});
