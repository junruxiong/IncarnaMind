import { describe, expect, test } from "vitest";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { INLINE_NODE } from "../../src/shared/sentences";
import { askAssistant, changesIn, lastResult } from "../helpers/assistant";
import { checkReply, isCheckRun, type Plan } from "../helpers/checks";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { readMind, resolveChanges, shownWithChanges } from "../helpers/minds";
import { type ModelCall, type ScriptedReply, scriptedModel, scriptedModels } from "../helpers/models";

/**
 * What the assistant writes into a Mind is checked against its own Citations
 * before it lands (#1111): a part they don't support is cut, keeping the
 * Citations on the rest, and a sentence still unsupported lands marked, so
 * Check this Mind doesn't find the assistant's overstatements later. The
 * model's verdicts and cuts are scripted; Claude Haiku's were tried by hand.
 */

const MOON = "Spring tides come at new and full moon.";
const OVERSTATED = "Spring tides come at new and full moon, and they flood the harbour every year";
const UNBACKED = "Neap tides are twice as strong";

/** A sentence as a check reads it: the Citation after its words as U+FFFC. */
const asChecked = (words: string) => `${words} ${INLINE_NODE}.`;

const PLANS: Record<string, Plan> = {
  [asChecked(OVERSTATED)]: {
    reply: {
      text: "Verdict: unsupported\nNo Passage says the harbour floods every year.\nCovered: when spring tides come\nNot covered: the yearly floods",
    },
  },
  [asChecked(UNBACKED)]: {
    reply: { text: "Verdict: unsupported\nNo Passage compares the strength of the tides." },
  },
};

const CUT = /<task action="cut-unsupported">/;

/**
 * A model that searches, cites the moon line, writes `content` into a new
 * Mind, and replies; checks a sentence as `PLANS` says; and, asked to cut a
 * sentence, keeps its first clause and its marker.
 */
function writingModel(content: string, calls: ModelCall[]) {
  return scriptedModel((call): ScriptedReply => {
    calls.push(call);
    if (isCheckRun(call)) return checkReply(call, PLANS);
    if (CUT.test(call.prompt)) {
      return { text: "Spring tides come at new and full moon [1].\n<why>Only what the source says.</why>" };
    }
    const { results } = call;
    const search = results.find((result) => result.tool === "search_documents");
    if (!search) return { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] };
    if (!results.some((result) => result.tool === "cite")) {
      const [passage] = shownPassages(search.text);
      const record = { marker: 1, passage: passage?.id, quote: MOON };
      return { calls: [{ tool: "cite", input: { citations: [record] } }] };
    }
    if (!results.some((result) => result.tool === "create_mind")) {
      return { calls: [{ tool: "create_mind", input: { title: "Tide notes", content } }] };
    }
    return { text: "Made “Tide notes” from your Documents." };
  });
}

async function setUp(content: string) {
  const calls: ModelCall[] = [];
  const model = writingModel(content, calls);
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const sources = await createTempDataFolder();
  const documents = await addAndProcess(core, [
    await writeSourceFile(sources, "Tides.md", `# Tides\n\n${MOON}\n`),
  ]);
  const folder = await core.createLibraryGroup({ name: "Coast", description: "" });
  await core.moveToFolder({ documentIds: documents.map((each) => each.id), folderId: folder.id });
  const answer = await askAssistant(core, {
    text: "Summarise these Documents into a new note",
    scope: { folderIds: [folder.id], tagIds: [], documentIds: [] },
  });
  const [change] = changesIn(answer);
  const client = await connectToMind(core, change?.mindId ?? "");
  await client.settled();
  return { core, model, calls, answer, change, client };
}

describe("What the assistant writes is checked before it lands", { timeout: 30_000 }, () => {
  // protects=a written sentence that overstates its Citations lands with that part cut, its Citation kept, and nothing in the Mind marked unsupported (#1111's acceptance); fails_when=the overstated words land, the Citation goes, a mark is left, or the model isn't told; no_claim=how well a real model cuts.
  test("a part its Citations don't support is cut before it lands, the Citation kept", async () => {
    const { model, calls, change, client } = await setUp(
      `## Tides\n\n${OVERSTATED} [^1].`,
    );

    expect(shownWithChanges(client)).toEqual([
      "+ Tides",
      "+ Spring tides come at new and full moon ⟨found⟩.",
    ]);
    expect(claimSpans(readMind(client))).toEqual([]);
    expect(change).toMatchObject({ created: true, proposals: 2 });
    expect(change).not.toHaveProperty("checking");
    // One check, and one edit for the cut.
    expect(calls.filter((call) => isCheckRun(call) && call.results.length === 0)).toHaveLength(1);
    expect(calls.filter((call) => CUT.test(call.prompt))).toHaveLength(1);
    expect(calls.find((call) => CUT.test(call.prompt))?.prompt).toContain("the yearly floods");
    expect(lastResult(model)).toContain(
      "What their Citations don't support was taken out: “the yearly floods”.",
    );
  });

  // protects=a written sentence its Citations don't support at all lands marked unsupported, with the check's reason, so Check this Mind has nothing more to find; fails_when=it lands unmarked, is cut, or Check this Mind sends it again; no_claim=none beyond protects.
  test("a sentence still unsupported lands marked, and Check this Mind leaves it", async () => {
    const { core, model, calls, change, client } = await setUp(
      `## Tides\n\n${UNBACKED} [^1].`,
    );

    expect(shownWithChanges(client)).toEqual([
      "+ Tides",
      "+ Neap tides are twice as strong ⟨found⟩.",
    ]);
    const [mark] = claimSpans(readMind(client));
    expect(mark?.check).toMatchObject({
      state: "unsupported",
      action: "check-against-documents",
      text: asChecked(UNBACKED),
      unsupported: { reason: "No Passage compares the strength of the tides." },
    });
    expect(calls.some((call) => CUT.test(call.prompt))).toBe(false);
    expect(lastResult(model)).toContain(
      "1 sentence is still not supported by its Citations, and marked so in the Mind",
    );
    await resolveChanges(client, "accept");
    expect(claimSpans(readMind(client))).toHaveLength(1);
    expect(await core.planMindCheck(change?.mindId, "changed")).toEqual({ kind: "nothing" });
  });

  // protects=text with no Citation isn't checked as it lands: nothing to check it against; fails_when=a check Run is made; no_claim=none beyond protects.
  test("a sentence with no Citation isn't checked as it lands", async () => {
    const { calls, client } = await setUp("## Tides\n\nThe harbour is quiet in winter.");

    expect(shownWithChanges(client)).toEqual(["+ Tides", "+ The harbour is quiet in winter."]);
    expect(calls.some(isCheckRun)).toBe(false);
  });
});
