import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import type { Core } from "../../src/core";
import { SENTENCES_CHECKED_AT_MOST } from "../../src/core/answers/checks/written";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { claimCheckShown } from "../../src/shared/claimChecks";
import { INLINE_NODE } from "../../src/shared/sentences";
import { askAssistant, changesIn, lastResult } from "../helpers/assistant";
import { checkReply, isCheckRun, type Plan } from "../helpers/checks";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient, typeInto } from "../helpers/mindClient";
import { readMind, resolveChanges, shownWithChanges, writeMind } from "../helpers/minds";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

/**
 * What the assistant writes into a Mind is checked against its own Citations
 * before it lands (#1111): a part they don't support is cut, keeping the
 * Citations and each word's formatting, and the sentence shows it needs a
 * check again; a sentence still unsupported, or one that couldn't be checked,
 * lands marked; at most `SENTENCES_CHECKED_AT_MOST` are checked a write; and
 * words the User types meanwhile stay. The model's verdicts and cuts are
 * scripted; Claude Haiku's were tried by hand.
 */

const MOON = "Spring tides come at new and full moon.";
const OVERSTATED = "Spring tides come at new and full moon, and they flood the harbour every year";
const UNBACKED = "Neap tides are twice as strong";
const FAILING = "The harbour floods at every spring tide";

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
  [asChecked(FAILING)]: { reply: { error: { status: 500, message: "The service broke." } } },
};

const CUT = /<task action="cut-unsupported">/;

interface Writing {
  content: string;
  /** Into the Mind in focus, after its text, rather than a new one. */
  into?: boolean;
  /** Runs while the first check waits for its reply. */
  during?: () => Promise<void>;
}

/**
 * A model that searches, cites the moon line, writes `content`, and replies;
 * checks a sentence as `PLANS` says; and, asked to cut a sentence, keeps its
 * first clause and its marker.
 */
function writingModel({ content, into = false, during }: Writing, calls: ModelCall[]) {
  let waited: Promise<void> | null = null;
  const tool = into ? "add_to_mind" : "create_mind";
  return scriptedModel((call): ScriptedReply => {
    calls.push(call);
    if (isCheckRun(call)) {
      const reply = checkReply(call, PLANS);
      if (!during || waited || "error" in reply) return reply;
      waited = during();
      return { ...reply, pause: { after: 0, until: waited } };
    }
    if (CUT.test(call.prompt)) {
      return {
        text: "Spring tides come at new and full moon [1].\n<why>Only what the source says.</why>",
      };
    }
    const { results } = call;
    const search = results.find((result) => result.tool === "search_documents");
    if (!search) return { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] };
    if (!results.some((result) => result.tool === "cite")) {
      const [passage] = shownPassages(search.text);
      const record = { marker: 1, passage: passage?.id, quote: MOON };
      return { calls: [{ tool: "cite", input: { citations: [record] } }] };
    }
    if (!results.some((result) => result.tool === tool)) {
      const input = into ? { content } : { title: "Tide notes", content };
      return { calls: [{ tool, input }] };
    }
    return { text: "Wrote it from your Documents." };
  });
}

/** A core with the tides line in a Folder, and a Mind there with a heading and a line. */
async function harbour(model: ReturnType<typeof writingModel>) {
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
  const mind = await core.createMind({ title: "Harbour notes", folderId: folder.id });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [
    { type: "heading", attrs: { level: 2 }, content: [{ type: "text", text: "Harbour" }] },
    { type: "paragraph", content: [{ type: "text", text: "Boats moor by the quay." }] },
  ]);
  await client.settled();
  return { core, folderId: folder.id, mindId: mind.id, client };
}

async function write(writing: Writing, onClient?: (client: MindClient) => void) {
  const calls: ModelCall[] = [];
  const model = writingModel(writing, calls);
  const { core, folderId, mindId, client } = await harbour(model);
  onClient?.(client);
  const answer = await askAssistant(core, {
    text: writing.into
      ? "Add a short note on tides to this Mind"
      : "Summarise these Documents into a new note",
    ...(writing.into && { mindId }),
    scope: { folderIds: [folderId], tagIds: [], documentIds: [] },
  });
  const [change] = changesIn(answer);
  const page = writing.into ? client : await connectToMind(core, change?.mindId ?? "");
  await page.settled();
  return { core, model, calls, answer, change, client: page };
}

/** Each check's mark in the Mind and how it shows, by its sentence. */
async function marksIn(core: Core, client: MindClient) {
  await client.settled();
  const versions = new Map((await core.listDocuments()).map((each) => [each.id, each.contentHash]));
  return claimSpans(readMind(client)).map((span) => ({
    sentence: span.sentence.text.replaceAll(INLINE_NODE, "").trim(),
    check: span.check,
    shown: claimCheckShown(span.check, {
      text: span.sentence.text,
      versions,
      folderId: span.check.scope.folderId,
    }),
  }));
}

describe("What the assistant writes is checked before it lands", { timeout: 30_000 }, () => {
  // protects=a written sentence that overstates its Citations lands with that part cut, its Citation kept, and marked to be checked again (#1111's acceptance); fails_when=the overstated words land, the Citation goes, it lands unmarked or unsupported, or the model isn't told; no_claim=how well a real model cuts.
  test("a part its Citations don't support is cut before it lands, the Citation kept, and it shows it needs a check again", async () => {
    const { core, model, calls, change, client } = await write({
      content: `## Tides\n\n${OVERSTATED} [^1].`,
    });

    expect(shownWithChanges(client)).toEqual([
      "+ Tides",
      "+ Spring tides come at new and full moon ⟨found⟩.",
    ]);
    const marks = await marksIn(core, client);
    expect(marks.map(({ shown }) => shown)).toEqual([{ state: "needs-recheck", why: "edited" }]);
    expect(change).toMatchObject({ created: true, proposals: 2 });
    expect(change).not.toHaveProperty("checking");
    expect(change?.selfCheck).toMatchObject({ cited: 1, sent: 1, failed: 0 });
    // One check, and one edit for the cut, told what the Citation quotes.
    expect(calls.filter((call) => isCheckRun(call) && call.results.length === 0)).toHaveLength(1);
    const cuts = calls.filter((call) => CUT.test(call.prompt));
    expect(cuts).toHaveLength(1);
    expect(cuts[0]?.prompt).toContain("the yearly floods");
    expect(cuts[0]?.prompt).toContain(`[1] “${MOON}”`);
    expect(lastResult(model)).toContain(
      "What their Citations don't support was taken out: “the yearly floods”.",
    );
  });

  // protects=a cut keeps each word's own formatting rather than giving all of it the first word's; fails_when=the plain words turn bold; no_claim=none beyond protects.
  test("a cut keeps each word's own formatting", async () => {
    const { client } = await write({
      content: `## Tides\n\n**Spring tides** come at new and full moon, and they flood the harbour every year [^1].`,
    });

    const paragraph = readMind(client).child(1);
    const runs: { text: string; bold: boolean }[] = [];
    paragraph.forEach((node) => {
      if (node.isText) {
        runs.push({
          text: node.text ?? "",
          bold: node.marks.some((mark) => mark.type.name === "bold"),
        });
      }
    });
    expect(runs.filter((run) => run.bold).map((run) => run.text.trim())).toEqual(["Spring tides"]);
    expect(runs.map((run) => run.text).join("")).toBe("Spring tides come at new and full moon .");
  });

  // protects=a written sentence its Citations don't support at all lands marked unsupported, with the check's reason, so Check this Mind has nothing more to find; fails_when=it lands unmarked, is cut, or Check this Mind sends it again; no_claim=none beyond protects.
  test("a sentence still unsupported lands marked, and Check this Mind leaves it", async () => {
    const { core, model, calls, change, client } = await write({
      content: `## Tides\n\n${UNBACKED} [^1].`,
    });

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
    expect(await core.planMindCheck(change?.mindId ?? "", "changed")).toEqual({ kind: "nothing" });
  });

  // protects=a check that fails lands its sentence marked not checked, with why, and the reply and the card's line say so; fails_when=it lands unmarked, which reads as supported; no_claim=the words of the line.
  test("a sentence whose check fails lands marked not checked, and the model is told", async () => {
    const { core, model, change, client } = await write({
      content: `## Tides\n\n${FAILING} [^1].`,
    });

    const [mark] = await marksIn(core, client);
    expect(mark?.check).toMatchObject({ state: "not-checked", reason: "failed" });
    expect(change?.selfCheck).toMatchObject({ cited: 1, sent: 1, failed: 1 });
    expect(lastResult(model)).toContain(
      "1 sentence couldn't be checked, and is marked so in the Mind: say so.",
    );
  });

  // protects=one write checks at most SENTENCES_CHECKED_AT_MOST sentences; the rest land marked not checked, and the reply says so; fails_when=a long draft sends a check for every sentence, or the rest land unmarked; no_claim=how long each check takes.
  test("one write checks at most the first sentences; the rest land marked not checked", async () => {
    const many = SENTENCES_CHECKED_AT_MOST + 1;
    const sentences = Array.from(
      { length: many },
      (_, index) => `Neap tide ${index + 1} is twice as strong [^1].`,
    );
    const { core, model, calls, change, client } = await write({
      content: `## Tides\n\n${sentences.join(" ")}`,
    });

    expect(calls.filter((call) => isCheckRun(call) && call.results.length === 0)).toHaveLength(
      SENTENCES_CHECKED_AT_MOST,
    );
    const marks = await marksIn(core, client);
    expect(marks).toHaveLength(many);
    expect(marks.at(-1)?.check).toMatchObject({ state: "not-checked", reason: "too-many" });
    expect(change?.selfCheck).toMatchObject({ cited: many, sent: SENTENCES_CHECKED_AT_MOST });
    expect(lastResult(model)).toContain(
      `Only the first ${SENTENCES_CHECKED_AT_MOST} of its ${many} sentences with Citations were checked`,
    );
  });

  // protects=words the User types into what the assistant wrote while the check runs stay: only the sentences cut or marked are written, each only while unchanged (the CTO's probe: a Block with no cut lost them); fails_when=the User's words vanish, or a sentence they changed is cut over; no_claim=none beyond protects.
  test("words the User types while the check runs stay, and a sentence they changed is only marked", async () => {
    let client: MindClient | null = null;
    const { core } = await write(
      {
        content: `${OVERSTATED} [^1]. Neap tides come between.\n\nThe harbour office posts the times.\n\n${UNBACKED} [^1].`,
        into: true,
        during: async () => {
          const typing = client;
          if (!typing) return;
          await typing.settled();
          // After the cut sentence, in its paragraph; at the start of a paragraph with no cut; and in a marked sentence.
          const paragraph = typing.blocks.get(2);
          const texts =
            paragraph instanceof Y.XmlElement
              ? paragraph.toArray().filter((each) => each instanceof Y.XmlText)
              : [];
          const last = texts.at(-1);
          if (last instanceof Y.XmlText) last.insert(last.length, " Quiet.");
          typeInto(typing, 3, 0, "Today ");
          typeInto(typing, 4, 0, "All ");
          await typing.settled();
        },
      },
      (connected) => {
        client = connected;
      },
    );
    const page = client as MindClient | null;
    if (!page) throw new Error("No client.");
    await page.settled();

    expect(shownWithChanges(page)).toEqual([
      "Harbour",
      "Boats moor by the quay.",
      "+ Spring tides come at new and full moon ⟨found⟩. Neap tides come between. Quiet.",
      "+ Today The harbour office posts the times.",
      "+ All Neap tides are twice as strong ⟨found⟩.",
    ]);
    const marks = await marksIn(core, page);
    expect(marks.map(({ sentence, shown }) => [sentence, shown.state])).toEqual([
      ["Spring tides come at new and full moon .", "needs-recheck"],
      ["All Neap tides are twice as strong .", "needs-recheck"],
    ]);
  });

  // protects=text with no Citation isn't checked as it lands: nothing to check it against; fails_when=a check Run is made; no_claim=none beyond protects.
  test("a sentence with no Citation isn't checked as it lands", async () => {
    const { calls, client } = await write({
      content: "## Tides\n\nThe harbour is quiet in winter.",
    });

    expect(shownWithChanges(client)).toEqual(["+ Tides", "+ The harbour is quiet in winter."]);
    expect(calls.some(isCheckRun)).toBe(false);
  });
});
