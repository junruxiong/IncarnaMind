/**
 * Documents added since a Mind's checks (#993): only the checks found
 * unsupported or not checked are offered again, and only for Documents added
 * to the scope they used; Check again checks only those, each in its scope.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { CheckMindInput } from "../../src/core";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { checkGenerate, checkReply, isCheckRun, type Plan } from "../helpers/checks";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { readMind, writeMind } from "../helpers/minds";
import { type ModelCall, scriptedModel, scriptedModels } from "../helpers/models";

const MOON = "Spring tides come at new and full moon.";
const YEARLY = "Harbours publish the times of high water every year.";
const PENGUINS = "Penguins nest on the cliffs above the harbour.";
const PUFFINS = "Puffins dig their burrows in the turf.";
const WEEKLY = "The harbour office publishes the times of high water every week.";

const text = (value: string): JSONContent => ({ type: "text", text: value });

async function setUp() {
  const plans: Record<string, Plan> = {
    [MOON]: { verdict: "supported", quote: MOON },
    [YEARLY]: {
      verdict: "contradicted",
      quote: WEEKLY,
      says: "The harbour office says every week, not every year.",
    },
    [PENGUINS]: { verdict: "unsupported" },
    [PUFFINS]: { verdict: "unsupported" },
  };
  const calls: ModelCall[] = [];
  const model = scriptedModel(
    (call) => {
      calls.push(call);
      return checkReply(call, plans);
    },
    { generate: checkGenerate(plans) },
  );
  const clock = manualClock("2026-10-02T14:20:00.000Z");
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, {
    createChatModel: scriptedModels(model).createChatModel,
    now: clock.now,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const sources = await createTempDataFolder();
  const add = async (name: string, contents: string) => {
    const [document] = await addAndProcess(core, [await writeSourceFile(sources, name, contents)]);
    if (!document) throw new Error(`${name} wasn't added.`);
    return document;
  };
  const tides = await add("Tides.md", `# Tides\n\n${MOON}\n\n${WEEKLY}\n`);
  const folder = await core.createLibraryGroup({ name: "Harbours", description: "" });
  await core.moveToFolder({ documentIds: [tides.id], folderId: folder.id });
  const mind = await core.createMind({ title: "Harbour notes", folderId: folder.id });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [
    { type: "paragraph", attrs: { id: "one" }, content: [text(`${MOON} ${YEARLY} ${PENGUINS}`)] },
    { type: "paragraph", attrs: { id: "two" }, content: [text(PUFFINS)] },
  ]);
  await client.settled();
  const checkRuns = () => calls.filter((call) => isCheckRun(call) && call.results.length === 0);
  return { core, client, clock, plans, add, mindId: mind.id, folderId: folder.id, checkRuns };
}

const request = (mindId: string, mode: CheckMindInput["mode"]): CheckMindInput => ({
  requestId: `run-${mode}`,
  mindId,
  mode,
});

/** Each sentence's check in the Mind now: its id, scope and state as it was made. */
const checksIn = async (client: Awaited<ReturnType<typeof setUp>>["client"]) => {
  await client.settled();
  return claimSpans(readMind(client)).map((span) => ({
    sentence: span.sentence.text.trim(),
    id: span.check.id,
    state: span.check.state,
    widened: span.check.scope.widened,
  }));
};

describe("Documents added since a Mind's checks", () => {
  test("only an unsupported check is offered, for a Document filed into its Folder since, and Check again checks it alone", async () => {
    const { core, client, clock, plans, add, mindId, folderId, checkRuns } = await setUp();
    const colony = await add("Colony.md", `# Colony\n\n${PENGUINS}\n`);
    clock.advance(60_000);
    await core.checkMind(request(mindId, "all"));
    const checked = await checksIn(client);
    expect(checked.map((each) => [each.sentence, each.state])).toEqual([
      [MOON, "supported"],
      [YEARLY, "contradicted"],
      [PENGUINS, "unsupported"],
      [PUFFINS, "unsupported"],
    ]);
    expect(await core.documentsAddedSinceChecks(mindId)).toEqual([]);

    // A Document outside the Mind's Folder isn't in what its checks searched.
    clock.advance(60_000);
    await add("Seabirds.md", `# Seabirds\n\n${PENGUINS}\n`);
    expect(await core.documentsAddedSinceChecks(mindId)).toEqual([]);
    expect(await core.planMindCheck(mindId, "added")).toEqual({ kind: "nothing" });

    // One there before, filed into the Folder since, is: the supported and contradicted checks are never offered.
    clock.advance(60_000);
    await core.moveToFolder({ documentIds: [colony.id], folderId });
    const unsupported = checked.filter((each) => each.state === "unsupported");
    expect(await core.documentsAddedSinceChecks(mindId)).toEqual(
      unsupported.map((each) => ({ checkId: each.id, documentIds: [colony.id] })),
    );
    expect(await core.planMindCheck(mindId, "added")).toEqual({
      kind: "ready",
      sentences: 2,
      findsClaims: false,
      model: null,
    });

    plans[PENGUINS] = { verdict: "supported", quote: PENGUINS };
    const before = checkRuns().length;
    clock.advance(60_000);
    expect(await core.checkMind(request(mindId, "added"))).toMatchObject({
      kind: "checked",
      sentences: 2,
      checked: 2,
    });
    expect(checkRuns().length - before).toBe(2);
    const after = await checksIn(client);
    expect(after.map((each) => [each.sentence, each.state])).toEqual([
      [MOON, "supported"],
      [YEARLY, "contradicted"],
      [PENGUINS, "supported"],
      [PUFFINS, "unsupported"],
    ]);
    expect(after.slice(0, 2).map((each) => each.id)).toEqual(checked.slice(0, 2).map((e) => e.id));
    expect(await core.documentsAddedSinceChecks(mindId)).toEqual([]);
  });

  test("a check made in every Document counts a Document added anywhere, and is checked again in every Document", async () => {
    const { core, client, clock, add, mindId } = await setUp();
    await core.checkMind({ ...request(mindId, "all"), allDocuments: true });
    const checked = await checksIn(client);
    expect(checked.every((each) => each.widened)).toBe(true);

    clock.advance(60_000);
    const elsewhere = await add("Seabirds.md", `# Seabirds\n\n${PENGUINS}\n`);
    const unsupported = checked.filter((each) => each.state === "unsupported");
    expect(await core.documentsAddedSinceChecks(mindId)).toEqual(
      unsupported.map((each) => ({ checkId: each.id, documentIds: [elsewhere.id] })),
    );
    clock.advance(60_000);
    await core.checkMind(request(mindId, "added"));
    const after = await checksIn(client);
    expect(after.filter((each) => each.sentence === PENGUINS || each.sentence === PUFFINS)).toEqual(
      [
        expect.objectContaining({ state: "unsupported", widened: true }),
        expect.objectContaining({ state: "unsupported", widened: true }),
      ],
    );
    expect(await core.documentsAddedSinceChecks(mindId)).toEqual([]);
  });
});
