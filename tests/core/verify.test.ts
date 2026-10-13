/**
 * Verify (#97): a PDF someone else wrote becomes a Mind of its text, its
 * claims are found, and each is checked against the User's Documents, the
 * PDF and its copies never evidence, with marks that go "needs re-check" as
 * Find support's do (#96).
 */
import { copyFile } from "node:fs/promises";
import { join } from "node:path";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test, vi } from "vitest";
import type { Core, VerifyClaimsInput } from "../../src/core";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { claimCheckShown } from "../../src/shared/claimChecks";
import { textOriginOf } from "../../src/shared/textOrigins";
import { verificationOf } from "../../src/shared/verification";
import { checkingModel, type Plan } from "../helpers/checks";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient, typeInto } from "../helpers/mindClient";
import { readMind } from "../helpers/minds";
import { type ModelCall, scriptedModel, scriptedModels } from "../helpers/models";
import { buildPdf } from "../helpers/pdf";

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
const THANKS = "Thank you for reading this report.";

/** The report someone else wrote: a title, three claims under two headings, and thanks. */
const REPORT = buildPdf([
  { lines: ["Harbour report", "1 Tides", SUPPORTED, CONTRADICTED] },
  { lines: ["2 Wildlife", UNSUPPORTED, THANKS] },
]);

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

/** A core with the Tides Document, the report on disk, and a model on this computer. */
async function setUp(model = checkingModel(PLANS)) {
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const sources = await createTempDataFolder();
  const [tides] = await addAndProcess(core, [
    await writeSourceFile(sources, TIDES.name, TIDES.contents),
  ]);
  const report = await writeSourceFile(sources, "Harbour report.pdf", REPORT);
  return { core, sources, report, tides };
}

/** Verifies the report: its Mind, with a client. */
async function verified(core: Core, path: string) {
  const result = await core.verifyDocument({ path });
  if (!result.ok) throw new Error(`The report wasn't verified: ${result.reason}`);
  const client = await connectToMind(core, result.mind.id);
  await client.settled();
  return { ...result, client };
}

const request = (mindId: string, more: Partial<VerifyClaimsInput> = {}): VerifyClaimsInput => ({
  requestId: crypto.randomUUID(),
  mindId,
  ...more,
});

/** Each claim's mark, as the editor shows it now. */
async function shownChecks(core: Core, client: MindClient) {
  await client.settled();
  const versions = new Map((await core.listDocuments()).map((each) => [each.id, each.contentHash]));
  return claimSpans(readMind(client)).map((span) => ({
    sentence: span.sentence.text,
    check: span.check,
    shown: claimCheckShown(span.check, { text: span.sentence.text, versions, folderId: null }),
  }));
}

describe("Verify a document", () => {
  test("its text becomes a Mind, a paragraph or numbered heading to a Block with its page; its claims are found, then checked", async () => {
    const { core, report, tides } = await setUp();
    const { mind, documentId, client } = await verified(core, report);

    expect(mind.title).toBe("Harbour report");
    const paragraphs: JSONContent[] = readMind(client).toJSON().content ?? [];
    expect(paragraphs.map((block) => [block.type, block.content?.[0]?.text])).toEqual([
      ["paragraph", "Harbour report"],
      ["heading", "1 Tides"],
      ["paragraph", SUPPORTED],
      ["paragraph", CONTRADICTED],
      ["heading", "2 Wildlife"],
      ["paragraph", UNSUPPORTED],
      ["paragraph", THANKS],
    ]);
    const ids = paragraphs.map((block) => String(block.attrs?.id));
    expect(verificationOf(client.doc)).toEqual({
      documentId,
      documentName: "Harbour report",
      pages: Object.fromEntries(ids.map((id, index) => [id, index < 4 ? 1 : 2])),
      claims: null,
    });
    // Someone else wrote it: a Run that reads it is third-party (#78).
    expect(ids.map((id) => textOriginOf(client.doc, id))).toEqual(
      ids.map(() => "checked-document"),
    );

    expect(await core.findClaims(request(mind.id))).toEqual({ kind: "found", claims: 3 });
    await client.settled();
    expect(verificationOf(client.doc)?.claims?.map((claim) => claim.text)).toEqual([
      SUPPORTED,
      CONTRADICTED,
      UNSUPPORTED,
    ]);

    expect(await core.verifyClaims(request(mind.id))).toEqual({
      kind: "checked",
      claims: 3,
      checked: 3,
      stop: null,
    });
    const checks = await shownChecks(core, client);
    expect(checks.map((each) => [each.sentence, each.shown])).toEqual([
      [SUPPORTED, { state: "supported" }],
      [CONTRADICTED, { state: "contradicted" }],
      [UNSUPPORTED, { state: "unsupported" }],
    ]);
    expect(checks[1]?.check).toMatchObject({
      action: "check-against-documents",
      says: "Tides says harbours publish them every year, not every week.",
      evidence: { documentId: tides?.id, quote: PLANS[CONTRADICTED]?.quote, check: "found" },
    });
  });

  test("the PDF, and a copy of it at another path, never appear as evidence", async () => {
    const calls: ModelCall[] = [];
    const { core, sources, report } = await setUp(checkingModel(PLANS, calls));
    const copy = join(sources, "Copies", "Harbour report (received).pdf");
    await writeSourceFile(sources, "Copies/.keep", "");
    await copyFile(report, copy);
    await addAndProcess(core, [copy]);
    const { mind } = await verified(core, report);

    await core.findClaims(request(mind.id));
    await core.verifyClaims(request(mind.id));

    const searched = calls.flatMap((call) =>
      call.results
        .filter((result) => result.tool === "search_documents")
        .flatMap((result) => shownPassages(result.text).map((passage) => passage.document)),
    );
    expect(searched.length).toBeGreaterThan(0);
    expect(new Set(searched)).toEqual(new Set(["Tides"]));
  });

  test("a claim the model fails on says “not checked”, and the others are checked", async () => {
    const plans: Record<string, Plan> = {
      ...PLANS,
      [UNSUPPORTED]: { reply: { error: { status: 500, message: "The model fell over." } } },
    };
    const { core, report } = await setUp(checkingModel(plans));
    const { mind, client } = await verified(core, report);
    await core.findClaims(request(mind.id));

    expect(await core.verifyClaims(request(mind.id))).toMatchObject({ claims: 3, checked: 2 });
    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual([
      { state: "supported" },
      { state: "contradicted" },
      { state: "not-checked", reason: "failed" },
    ]);
  });

  test("the monthly cap stops the run: the claims left say “not checked: cost cap”, and nothing is sent", async () => {
    const calls: ModelCall[] = [];
    const model = checkingModel(PLANS, calls);
    const core = startCore(await createTempDataFolder(), {
      createChatModel: () => model,
      now: manualClock("2026-10-13T09:00:00.000Z").now,
    });
    core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
    await core.saveChatProvider({
      kind: "anthropic",
      modelId: "claude-haiku-5-5",
      apiKey: "sk-test",
    });
    const sources = await createTempDataFolder();
    await addAndProcess(core, [await writeSourceFile(sources, TIDES.name, TIDES.contents)]);
    const { mind, client } = await verified(
      core,
      await writeSourceFile(sources, "Harbour report.pdf", REPORT),
    );
    expect(await core.findClaims(request(mind.id))).toEqual({ kind: "found", claims: 3 });

    await core.updateSettings({ user: { monthlyCap: { amount: 0, currency: "USD" } } });
    expect(await core.verifyClaims(request(mind.id))).toEqual({
      kind: "checked",
      claims: 3,
      checked: 0,
      stop: { reason: "cost-cap" },
    });
    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual(
      [SUPPORTED, CONTRADICTED, UNSUPPORTED].map(() => ({
        state: "not-checked",
        reason: "cost-cap",
      })),
    );
    expect(calls).toEqual([]);
  });

  test("results go “needs re-check” as Find support's do: an edited claim", async () => {
    const { core, report } = await setUp();
    const { mind, client } = await verified(core, report);
    await core.findClaims(request(mind.id));
    await core.verifyClaims(request(mind.id));

    typeInto(client, 2, 0, "Not all ");

    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual([
      { state: "needs-recheck", why: "edited" },
      { state: "contradicted" },
      { state: "unsupported" },
    ]);
  });

  test("empty states: no claims found, and nothing to check against", async () => {
    const { core, report } = await setUp(checkingModel({}));
    const { mind } = await verified(core, report);
    expect(await core.findClaims(request(mind.id))).toEqual({ kind: "found", claims: 0 });
    expect(await core.verifyClaims(request(mind.id))).toEqual({ kind: "nothing" });

    const alone = startCore(await createTempDataFolder(), {
      createChatModel: scriptedModels(checkingModel(PLANS)).createChatModel,
    });
    await alone.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const lonely = await verified(alone, report);
    // The PDF is the only Document, and it is never its own evidence.
    expect(await alone.findClaims(request(lonely.mind.id))).toEqual({
      kind: "no-documents",
      folderId: null,
    });
  });

  test("one run at a time: another asked for while one runs says which Mind it is in", async () => {
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const model = scriptedModel(
      () => ({ text: "Verdict: unsupported", pause: { after: 0, until: held } }),
      {
        generate: () => ({ text: JSON.stringify({ claims: [1, 2, 3] }) }),
      },
    );
    const { core, report } = await setUp(model);
    const { mind, client } = await verified(core, report);
    await core.findClaims(request(mind.id));
    const first = request(mind.id);
    const running = core.verifyClaims(first);
    await vi.waitFor(async () => expect(await shownChecks(core, client)).toHaveLength(3));

    expect(await core.verifyClaims(request(mind.id))).toEqual({ kind: "busy", mindId: mind.id });
    expect(await core.findClaims(request(mind.id))).toEqual({ kind: "busy", mindId: mind.id });

    await core.cancelCheck(first.requestId);
    release();
    expect(await running).toEqual({ kind: "cancelled" });
    expect((await shownChecks(core, client)).map((each) => each.shown)).toEqual(
      [1, 2, 3].map(() => ({ state: "not-checked", reason: "stopped" })),
    );
  });
});
