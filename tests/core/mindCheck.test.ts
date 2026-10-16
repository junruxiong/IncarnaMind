/**
 * Check this Mind (#737): every sentence of a Mind that makes a claim,
 * checked in one run; then only what changed. Headings, questions and the
 * User's own opinions are left out, as the model names the claims.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { CheckMindInput, Core } from "../../src/core";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { claimCheckShown } from "../../src/shared/claimChecks";
import { mindCheckOf } from "../../src/shared/mindCheck";
import { checkGenerate, checkReply, isCheckRun, type Plan, sentenceOf } from "../helpers/checks";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, waitForDocuments, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient, typeInto } from "../helpers/mindClient";
import { editMind, readMind, writeMind } from "../helpers/minds";
import {
  type GenerateCall,
  type ModelCall,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";
import { wordFile, paragraph as wordParagraph } from "../helpers/word";
import { importWordBytes } from "../helpers/wordMinds";

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

/** A core with the two Documents in a Folder, a model on this computer, and every call to it kept. */
async function setUp() {
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
    },
  );
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
    now: manualClock("2026-10-02T14:20:00.000Z").now,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
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
  return { core, client, mindId: mind.id, folderId: folder.id, checks, finds };
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

  test("says why there is nothing to check: no sentences, or no Documents to check against, unless every Document is", async () => {
    const { core } = await setUp();
    const empty = await core.createMind({ title: "Empty", folderId: null });
    expect(await core.planMindCheck(empty.id, "all")).toEqual({ kind: "nothing" });
    const emptyFolder = await core.createLibraryGroup({ name: "Nothing yet", description: "" });
    const lonely = await core.createMind({ title: "Lonely", folderId: emptyFolder.id });
    const client = await connectToMind(core, lonely.id);
    writeMind(client, [{ type: "paragraph", attrs: { id: "moon" }, content: [text(MOON)] }]);
    await client.settled();
    expect(await core.planMindCheck(lonely.id, "all")).toEqual({
      kind: "no-documents",
      folderId: emptyFolder.id,
    });
    // Against every Document, the Folder's emptiness doesn't stop it.
    expect(await core.planMindCheck(lonely.id, "all", true)).toMatchObject({
      kind: "ready",
      sentences: 1,
    });
  });
});

describe("A sentence gets the same verdict in Check this Mind as checked alone", () => {
  const LOAN = "The Northgate Bank term loan stood at £214,480.00 at 31 March 2026.";
  const BANK = "Balance outstanding at 31 March 2026: GBP 212,480.00.";
  const MEMO = `Revenue for the year was £1,284,300. ${LOAN}`;

  /**
   * A model that trusts what it reads: the sentence itself supports it, the
   * bank's figure contradicts it. The first `unread` of its check Runs answer
   * before searching, as a model sometimes does.
   */
  function readingModel(runs: ModelCall[], unread = 0) {
    const plans: Record<string, Plan> = { [LOAN]: { judgement: "conflict" } };
    let skipped = 0;
    return scriptedModel(
      (call) => {
        runs.push(call);
        const searches = call.results.filter((result) => result.tool === "search_documents");
        if (searches.length === 0) {
          if (sentenceOf(call) === LOAN && skipped < unread) {
            skipped++;
            return { text: "Verdict: unsupported" };
          }
          return { calls: [{ tool: "search_documents", input: { query: sentenceOf(call) } }] };
        }
        const shown = shownPassages(searches.at(-1)?.text ?? "");
        const itself = shown.find((each) => each.text.includes(sentenceOf(call)));
        const bank = shown.find((each) => each.text.includes(BANK));
        const decided = itself ?? bank;
        if (!decided) return { text: "Verdict: unsupported" };
        const quote = itself ? sentenceOf(call) : BANK;
        if (!call.results.some((result) => result.tool === "cite")) {
          return {
            calls: [
              { tool: "cite", input: { citations: [{ marker: 1, passage: decided.id, quote }] } },
            ],
          };
        }
        return {
          text: itself
            ? "Verdict: supported"
            : "Verdict: contradicted\nThe bank says £212,480.00. [^1]",
        };
      },
      { generate: checkGenerate(plans) },
    );
  }

  /** A bank confirmation, and a memo imported from Word with its original added, in one Folder. */
  async function yearEnd(unread = 0) {
    const runs: ModelCall[] = [];
    const core = startCore(await createTempDataFolder(), {
      createChatModel: scriptedModels(readingModel(runs, unread)).createChatModel,
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const sources = await createTempDataFolder();
    const [bank] = await addAndProcess(core, [
      await writeSourceFile(
        sources,
        "Northgate Bank confirmation.md",
        `# Audit confirmation\n\n${BANK}\n`,
      ),
    ]);
    const folder = await core.createLibraryGroup({ name: "Year end", description: "" });
    await core.moveToFolder({ documentIds: [bank?.id ?? ""], folderId: folder.id });
    const { mindId, client } = await importWordBytes(
      core,
      sources,
      wordFile({ body: wordParagraph(MEMO) }),
      {
        name: "Draft memo.docx",
        origin: "own",
        folderId: folder.id,
        addOriginal: true,
      },
    );
    await waitForDocuments(core, (listed) =>
      expect(listed.filter((each) => each.status === "ready")).toHaveLength(2),
    );
    const loanCheck = () =>
      claimSpans(readMind(client)).find((span) => span.sentence.text.trim() === LOAN)?.check;
    return { core, client, mindId, runs, loanCheck };
  }

  test("an imported draft never counts as its own evidence: the bank's figure contradicts it either way", async () => {
    const { core, client, mindId, runs, loanCheck } = await yearEnd();
    await core.checkMind(request(mindId, "all"));
    expect(loanCheck()).toMatchObject({ state: "contradicted", evidence: { quote: BANK } });

    let blockId = "";
    readMind(client).forEach((node) => {
      if (node.textContent.includes(LOAN)) blockId = String(node.attrs.id);
    });
    const from = MEMO.indexOf(LOAN);
    await core.checkSentences({
      requestId: "alone",
      mindId,
      action: "check-against-documents",
      from: { blockId, offset: from },
      to: { blockId, offset: from + LOAN.length },
    });
    await client.settled();
    expect(loanCheck()).toMatchObject({ state: "contradicted", evidence: { quote: BANK } });

    // No check, whole or alone, was shown a Passage of the draft itself.
    const shownToChecks = runs.flatMap((call) =>
      call.results
        .filter((result) => result.tool === "search_documents")
        .flatMap((result) => shownPassages(result.text)),
    );
    expect(shownToChecks.some((each) => each.document.startsWith("Draft memo"))).toBe(false);
  });

  test("a verdict given before searching isn't kept: the sentence is checked again, and contradicted", async () => {
    const { core, mindId, loanCheck } = await yearEnd(1);
    await core.checkMind(request(mindId, "all"));
    expect(loanCheck()).toMatchObject({ state: "contradicted", evidence: { quote: BANK } });
  });

  test("a model that never searches gets no verdict: the sentence isn't checked, and says why", async () => {
    const { core, mindId, loanCheck } = await yearEnd(2);
    await core.checkMind(request(mindId, "all"));
    expect(loanCheck()).toMatchObject({ state: "not-checked", reason: "unread" });
  });
});
