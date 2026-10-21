/**
 * After Stop, a Meeting's notes filled in from its Transcript (#142), through the core's public
 * interface as the meeting bar uses it: under each of the User's lines, or in sections when there
 * are none, every addition AI text citing the Transcript, one Undo, and nothing written when the
 * model fails, a quit cuts it short, or Local mode keeps it off the cloud.
 */
import { describe, expect, test, vi } from "vitest";
import type { Core, Meeting, MeetingFill } from "../../src/core";
import type { WindowedPassage } from "../../src/core/documents/search";
import { FILL_PART_TOKENS, transcriptParts } from "../../src/core/meetings/fill";
import { pendingProposals } from "../../src/shared/proposals";
import { checkReply, isCheckRun, type Plan } from "../helpers/checks";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { heading, note, readMind, shownWithAiText, writeMind } from "../helpers/minds";
import { type ModelCall, scriptedModel } from "../helpers/models";
import { transcribingCore } from "../helpers/transcription";

const SECOND = 1000;
const MY_LINES = [
  "We agreed the enterprise price is 45 dollars a seat",
  "Dana will send the contract by Friday",
  "The rollout starts in the second quarter",
];
const THEIR_LINES = ["The budget is approved for this year", "We still need the security review"];

const block = (text: string, tag: string) =>
  new RegExp(`<${tag}>\\n([\\s\\S]*?)\\n</${tag}>`).exec(text)?.[1] ?? null;

const isFill = (call: ModelCall) =>
  call.prompt.includes("Fill them in from the transcript") || call.prompt.includes("took no notes");

/**
 * A model that fills notes in as `reply` says, citing the transcript lines `quotes` names by
 * marker, and checks claims as `plans` say.
 */
function fillingModel(
  reply: string,
  quotes: Record<number, string>,
  plans: Record<string, Plan> = {},
  { failing = () => false, held }: { failing?: () => boolean; held?: Promise<void> } = {},
) {
  return scriptedModel((call) => {
    if (isCheckRun(call)) return checkReply(call, plans);
    if (!isFill(call)) return { text: "OK" };
    if (failing()) return { error: { status: 500, message: "The model is overloaded." } };
    const shown = shownPassages(block(call.prompt, "live-transcript") ?? "");
    if (!call.results.some((result) => result.tool === "cite")) {
      const citations = Object.entries(quotes).map(([marker, quote]) => ({
        marker: Number(marker),
        passage: shown.find((passage) => passage.text.includes(quote))?.id,
        quote,
      }));
      return { calls: [{ tool: "cite", input: { citations } }] };
    }
    return { text: reply, ...(held && { pause: { after: 0, until: held } }) };
  });
}

/** A Meeting with notes `notes` (none: an empty Mind), recorded for 12 seconds and stopped. */
async function recorded(
  model: ReturnType<typeof scriptedModel>,
  notes: Parameters<typeof writeMind>[1] | null,
  before?: (core: Core) => Promise<void>,
) {
  const dataDir = await createTempDataFolder();
  const { core, record } = await transcribingCore(dataDir, {
    model,
    engine: { lines: MY_LINES, theirLines: THEIR_LINES },
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  await before?.(core);
  const meeting = await core.createMeeting();
  const client = await connectToMind(core, meeting.mindId);
  if (notes) {
    writeMind(client, notes);
    await client.settled();
  }
  await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
  record(12 * SECOND);
  await vi.waitFor(async () => {
    const live = (await core.listMeetings()).find((each) => each.id === meeting.id);
    expect(live?.transcript?.lines ?? 0).toBeGreaterThanOrEqual(5);
  });
  await core.stopMeeting(meeting.id);
  return { core, meeting, client, dataDir };
}

async function fillOf(core: Core, meetingId: string): Promise<MeetingFill | null> {
  return (await core.listMeetings()).find((each) => each.id === meetingId)?.fill ?? null;
}

async function settledFill(core: Core, meeting: Meeting): Promise<MeetingFill> {
  return vi.waitFor(
    async () => {
      const fill = await fillOf(core, meeting.id);
      if (!fill || fill.state === "waiting" || fill.state === "filling") {
        throw new Error(`Not yet: ${JSON.stringify(fill)}`);
      }
      return fill;
    },
    { timeout: 20_000 },
  );
}

/** The Mind's lines as shown, AI text in «», a Citation as \uFFFC. */
const shown = (client: MindClient) => {
  const lines: string[] = [];
  readMind(client).descendants((node) => {
    if (node.type.name === "paragraph" || node.type.name === "heading") {
      lines.push(shownWithAiText(node));
      return false;
    }
    return true;
  });
  return lines;
};

const NOTES = [
  heading(2, "Pricing"),
  note("price → Q2?"),
  note("contract"),
  heading(2, "Rollout"),
  note("start date"),
];

describe("After Stop, the notes are filled in from the Transcript (#142)", {
  timeout: 60_000,
}, () => {
  // protects=each of the User's lines gets what the call said about it right under it, as AI text citing the Transcript, in their order and under their headings, the User's own text unchanged, with no Proposal and a Checkpoint before; Undo leaves the notes as the User left them; fails_when=a line moves or changes, an addition isn't AI text or uncited, a Proposal is made, or Undo leaves anything; no_claim=a real model's wording.
  test("under each of the User's lines, in place, cited, with one Undo", async () => {
    const model = fillingModel(
      "### L1\nThe enterprise price is 45 dollars a seat [^1].\n\n### L2\nDana sends the contract by Friday [^2].\n\n### L3\nThe rollout starts in the second quarter [^3].",
      { 1: MY_LINES[0] as string, 2: MY_LINES[1] as string, 3: MY_LINES[2] as string },
    );
    const { core, meeting, client } = await recorded(model, NOTES);
    const before = shown(client);

    const fill = await settledFill(core, meeting);
    expect(fill).toMatchObject({ state: "filled", mode: "notes", local: true, problem: null });
    await vi.waitFor(() => expect(shown(client)).toHaveLength(before.length + 3));
    expect(shown(client)).toEqual([
      "Pricing",
      "price → Q2?",
      "«The enterprise price is 45 dollars a seat »\uFFFC«.»",
      "contract",
      "«Dana sends the contract by Friday »\uFFFC«.»",
      "Rollout",
      "start date",
      "«The rollout starts in the second quarter »\uFFFC«.»",
    ]);
    // The User's own lines, byte for byte, in their order.
    expect(shown(client).filter((line) => !line.includes("«"))).toEqual(before);
    expect(pendingProposals(client.blocks)).toEqual([]);
    const { checkpoints } = await core.listCheckpoints(meeting.mindId);
    expect(checkpoints.some((each) => each.reason === "enhance")).toBe(true);

    await core.undoMeetingFill(meeting.id);
    await vi.waitFor(() => expect(shown(client)).toEqual(before));
    expect(await fillOf(core, meeting.id)).toMatchObject({ state: "undone" });
    core.close();
  });

  // protects=with no notes, the sections are filled in, each citing the Transcript, and "Said in the call, not in your Documents" keeps only the claims the User's Documents don't support; fails_when=a section is missing, uncited, or a supported claim is listed; no_claim=the template's sections (#688).
  test("with no notes, the sections, and only the claims the Documents don't support", async () => {
    const supported = "The enterprise price is 45 dollars a seat.";
    const model = fillingModel(
      [
        "### SUMMARY",
        "The call settled pricing and the contract [^1].",
        "### DECISIONS",
        "- The price is 45 dollars a seat [^1].",
        "### ACTIONS",
        "- Dana sends the contract by Friday — Me [^2].",
        "### QUESTIONS",
        "- The security review is still open [^3].",
        "### CLAIMS",
        `- ${supported.replace(".", "")} [^1].`,
        "- The budget is approved for this year [^4].",
      ].join("\n"),
      {
        1: MY_LINES[0] as string,
        2: MY_LINES[1] as string,
        3: THEIR_LINES[1] as string,
        4: THEIR_LINES[0] as string,
      },
      {
        "The enterprise price is 45 dollars a seat.": {
          verdict: "supported",
          quote: "Enterprise plan: 45 dollars a seat",
        },
      },
    );
    const { core, meeting, client } = await recorded(model, null, async (core) => {
      const sources = await createSourceFolder();
      await addAndProcess(core, [
        await writeSourceFile(sources, "Price list.md", "Enterprise plan: 45 dollars a seat.\n"),
      ]);
    });

    expect(await settledFill(core, meeting)).toMatchObject({ state: "filled", mode: "sections" });
    await vi.waitFor(() => expect(shown(client).length).toBeGreaterThan(5));
    expect(shown(client)).toEqual([
      "«Summary»",
      "«The call settled pricing and the contract »\uFFFC«.»",
      "«Decisions»",
      "«The price is 45 dollars a seat »\uFFFC«.»",
      "«Action items»",
      "«Dana sends the contract by Friday — Me »\uFFFC«.»",
      "«Open questions»",
      "«The security review is still open »\uFFFC«.»",
      "«Said in the call, not in your Documents»",
      "«The budget is approved for this year »\uFFFC«.»",
    ]);
    core.close();
  });

  // protects=a model that fails leaves the Mind as it was and says why, and Try again fills it in; fails_when=anything is written on a failure, or Try again does nothing; no_claim=which provider error it was.
  test("a failing model writes nothing and says why; Try again fills the notes in", async () => {
    let failing = true;
    const model = fillingModel(
      "### L1\nThe enterprise price is 45 dollars a seat [^1].",
      { 1: MY_LINES[0] as string },
      {},
      { failing: () => failing },
    );
    const { core, meeting, client } = await recorded(model, NOTES);
    const before = shown(client);

    const failed = await settledFill(core, meeting);
    expect(failed).toMatchObject({ state: "failed", problem: { kind: "model" } });
    expect(failed.problem?.message).not.toBe("");
    expect(shown(client)).toEqual(before);

    failing = false;
    await core.fillMeetingNotes(meeting.id);
    await vi.waitFor(async () =>
      expect(await fillOf(core, meeting.id)).toMatchObject({ state: "filled" }),
    );
    await vi.waitFor(() => expect(shown(client)).toHaveLength(before.length + 1));
    core.close();
  });

  // protects=lines the User deletes or moves while the notes are filled in keep their additions, at the end of their heading's part, and nothing the User typed changes; fails_when=an addition is lost, overwrites, or lands under the wrong heading; no_claim=a line moved under another heading, whose additions follow it there.
  test("a line deleted while it fills in gets its addition at the end of its heading's part", async () => {
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const model = fillingModel(
      "### L1\nThe enterprise price is 45 dollars a seat [^1].\n\n### L2\nDana sends the contract by Friday [^2].",
      { 1: MY_LINES[0] as string, 2: MY_LINES[1] as string },
      {},
      { held },
    );
    const { core, meeting, client } = await recorded(model, NOTES);
    await vi.waitFor(async () =>
      expect(await fillOf(core, meeting.id)).toMatchObject({ state: "filling" }),
    );
    // The User deletes "price → Q2?" and types a new line at the end meanwhile.
    writeMind(client, [NOTES[0], NOTES[2], NOTES[3], NOTES[4], note("Who signs?")] as Parameters<
      typeof writeMind
    >[1]);
    await client.settled();
    const typed = shown(client);
    release();

    expect(await settledFill(core, meeting)).toMatchObject({ state: "filled" });
    await vi.waitFor(() => expect(shown(client)).toHaveLength(typed.length + 2));
    expect(shown(client)).toEqual([
      "Pricing",
      "contract",
      "«Dana sends the contract by Friday »\uFFFC«.»",
      "«The enterprise price is 45 dollars a seat »\uFFFC«.»",
      "Rollout",
      "start date",
      "Who signs?",
    ]);
    core.close();
  });

  // protects=quitting while the notes are filled in leaves the Mind as it was, and the next start says so with Try again; fails_when=anything half-written stays, or the Meeting says it is still filling; no_claim=none beyond protects.
  test("a quit while it fills in leaves nothing written, and the next start says so", async () => {
    const held = new Promise<void>(() => {});
    const model = fillingModel(
      "### L1\nThe enterprise price is 45 dollars a seat [^1].",
      { 1: MY_LINES[0] as string },
      {},
      { held },
    );
    const { core, meeting, client, dataDir } = await recorded(model, NOTES);
    const before = shown(client);
    await vi.waitFor(async () =>
      expect(await fillOf(core, meeting.id)).toMatchObject({ state: "filling" }),
    );
    core.close();

    const { core: again } = await transcribingCore(dataDir, {
      model: fillingModel("", {}),
    });
    const reopened = await connectToMind(again, meeting.mindId);
    expect(await fillOf(again, meeting.id)).toMatchObject({
      state: "failed",
      problem: { kind: "stopped" },
    });
    expect(shown(reopened)).toEqual(before);
    again.close();
  });

  // protects=with "Keep everything on this computer" on and a cloud Answers model, nothing is filled in or sent, and the Meeting says why; fails_when=it runs on the cloud model, or says nothing; no_claim=a model on this computer, which the other tests use.
  test("in Local mode with a cloud Answers model, it doesn't run, and says why", async () => {
    const calls: ModelCall[] = [];
    const model = scriptedModel((call) => {
      calls.push(call);
      return { text: "OK" };
    });
    const { core, meeting, client } = await recorded(model, NOTES, async (core) => {
      const cloud = await core.saveChatProvider({
        kind: "openai",
        apiKey: "sk-test",
        modelId: "gpt-test",
      });
      await core.updateSettings({
        user: { chatModel: { providerId: cloud.id, modelId: "gpt-test" } },
      });
      await core.setLocalOnly(true);
    });
    const before = shown(client);
    expect(await settledFill(core, meeting)).toMatchObject({
      state: "off",
      problem: { kind: "local-mode" },
    });
    expect(calls.filter(isFill)).toEqual([]);
    expect(shown(client)).toEqual(before);
    core.close();
  });
});

describe("A long Transcript is read in parts", () => {
  const passage = (index: number, words: number): WindowedPassage =>
    ({ text: "word ".repeat(words), seq: index }) as unknown as WindowedPassage;

  // protects=a two-hour Transcript fits one part on a model with a large window, and a longer one or a small window is split, every Passage kept once in order; fails_when=a Passage is dropped or repeated, or a short Transcript is split; no_claim=the token estimate's accuracy.
  test("parts fit the budget and keep every Passage once, in order", () => {
    const twoHours = Array.from({ length: 120 }, (_, index) => passage(index, 120));
    expect(transcriptParts(twoHours, FILL_PART_TOKENS)).toHaveLength(1);
    const parts = transcriptParts(twoHours, 4_000);
    expect(parts.length).toBeGreaterThan(1);
    expect(parts.flat().map((each) => each.seq)).toEqual(twoHours.map((each) => each.seq));
  });
});
