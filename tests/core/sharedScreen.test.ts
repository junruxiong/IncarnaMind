/**
 * The shared screen (ADR-0024, #834), through the core's public interface as
 * the meeting bar and the assistant use it: asked at every meeting, one
 * window read as text once it settles, a Question during the call going with
 * that text and its Answer citing it, and nothing of it kept once the
 * Meeting stops: not in memory, not in the data folder, not in the Citation.
 */
import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test, vi } from "vitest";
import type { Core, Meeting } from "../../src/core";
import { citationState } from "../../src/shared/citations";
import { answerEnded } from "../helpers/citations";
import { createTempDataFolder } from "../helpers/core";
import { addAndProcess } from "../helpers/documents";
import { type ModelCall, scriptedModel } from "../helpers/models";
import {
  type FakeScreenOptions,
  fakeScreen,
  type Slides,
  screenAdapters,
  slideReader,
} from "../helpers/screens";
import { transcribingCore } from "../helpers/transcription";

const SLIDES: Slides = {
  1: [
    "Launch budget",
    "Budget £1.2m",
    "Spent to date £640k",
    "Contingency left £90k",
    "Confidential draft for the steering committee",
  ],
  2: ["华东区第三季度收入", "收入 3.2 亿元", "同比增长 18%"],
  // Read twice, the first reading says 640 and the second 648.
  3: [{ first: "Spent to date £640k", second: "Spent to date £648k" }],
};

/** The text of the User's messages of a request. */
const userText = (call: ModelCall) =>
  call.options.prompt
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => ("text" in part ? part.text : "")).join(""),
    )
    .join("\n");

/** Every part of every message a request sent, by its type: text and Tools' only, never a picture. */
const partTypes = (call: ModelCall) =>
  new Set(
    call.options.prompt.flatMap((message) =>
      typeof message.content === "string"
        ? ["text"]
        : message.content.map((part: { type: string }) => part.type),
    ),
  );

/** A model that cites the line of the newest screen holding `quote`, and, given one, the Library Passage it matches. */
function screenReadingModel(calls: ModelCall[], quote: string, answer: string) {
  return scriptedModel((call) => {
    calls.push(call);
    const text = userText(call);
    const screen = [
      ...text.matchAll(/<shared-screen id="(S\d+)"[^>]*?(?: same-as="([^"]+)")?>/g),
    ].at(-1);
    if (!screen) return { text: "Nothing was shared." };
    if (!call.results.some((result) => result.tool === "cite")) {
      const library = screen[2]?.split(" ")[0];
      return {
        calls: [
          {
            tool: "cite",
            input: {
              citations: [
                { marker: 1, passage: screen[1], quote },
                ...(library ? [{ marker: 2, passage: library, quote }] : []),
              ],
            },
          },
        ],
      };
    }
    return { text: answer };
  });
}

async function meetingOf(core: Core, meetingId: string): Promise<Meeting> {
  const meeting = (await core.listMeetings()).find((each) => each.id === meetingId);
  if (!meeting) throw new Error(`No Meeting ${meetingId}.`);
  return meeting;
}

async function setUp(model?: MockLanguageModelV4, options: FakeScreenOptions = {}) {
  const dataDir = await createTempDataFolder();
  const screen = fakeScreen(options);
  const reader = slideReader(SLIDES);
  const { core, clock, record } = await transcribingCore(dataDir, {
    ...(model && { model }),
    adapters: screenAdapters(screen.capture, reader),
  });
  if (model) await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const meeting = await core.createMeeting();
  /** A second passes, and the window's capture sends what it shows. */
  const tick = () => {
    clock.advance(1000);
    screen.sample();
  };
  /** Seconds pass until the Meeting has read `screens` screens. */
  const readUntil = (screens: number) =>
    vi.waitFor(
      async () => {
        tick();
        const now = await meetingOf(core, meeting.id);
        if (now.screen?.screens !== screens)
          throw new Error(`Not yet: ${JSON.stringify(now.screen)}`);
        return now;
      },
      { timeout: 15_000, interval: 20 },
    );
  return { dataDir, core, clock, record, meeting, screen, reader, tick, readUntil };
}

async function ask(core: Core, mindId: string, text: string) {
  const result = await core.askInConversation({ mindId, text });
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  expect((await answerEnded(core, result.answerId)).event).toBe("finished");
  return answerIn(core, result.conversation.id, result.answerId);
}

/** An Answer as its conversation keeps it. */
async function answerIn(core: Core, conversationId: string, answerId: string) {
  const { messages } = await core.getConversation(conversationId);
  const answer = messages.find((message) => message.id === answerId);
  if (answer?.kind !== "answer") throw new Error("No Answer.");
  return { ...answer, conversationId };
}

/** Every file under a folder, read whole. */
async function filesUnder(folder: string): Promise<{ path: string; bytes: Buffer }[]> {
  const found: { path: string; bytes: Buffer }[] = [];
  for (const entry of await readdir(folder, { withFileTypes: true, recursive: true })) {
    if (!entry.isFile()) continue;
    const path = join(entry.parentPath, entry.name);
    found.push({ path, bytes: await readFile(path) });
  }
  return found;
}

describe("The shared screen", { timeout: 60_000 }, () => {
  test("the bar asks when recording starts, until answered; Not this time stops it; the setting turns the question off", async () => {
    const { core, meeting, screen } = await setUp();
    expect((await meetingOf(core, meeting.id)).screen).toBeNull();
    await core.startMeeting({ meetingId: meeting.id });
    expect((await meetingOf(core, meeting.id)).screen).toMatchObject({
      state: "ask",
      window: null,
      problem: null,
      screens: 0,
      permission: "granted",
    });

    await core.declineSharedScreen(meeting.id);
    expect((await meetingOf(core, meeting.id)).screen?.state).toBe("off");
    await core.pauseMeeting(meeting.id);
    await core.resumeMeeting(meeting.id);
    expect((await meetingOf(core, meeting.id)).screen?.state).toBe("off");
    await core.stopMeeting(meeting.id);
    expect((await meetingOf(core, meeting.id)).screen).toBeNull();

    await core.updateSettings({ device: { meetingScreenAsk: false } });
    const next = await core.createMeeting();
    await core.startMeeting({ meetingId: next.id });
    expect((await meetingOf(core, next.id)).screen?.state).toBe("off");
    expect(screen.started).toBe(0);
  });

  test("a chosen window is read once it settles; a Question during the call goes with its text, never a picture, and its Answer cites the screen without keeping the quote", async () => {
    const calls: ModelCall[] = [];
    const model = screenReadingModel(
      calls,
      "Spent to date £640k",
      "£640k of the £1.2m launch budget has been spent so far [^1].",
    );
    const { core, meeting, screen, reader, tick, readUntil } = await setUp(model);
    await core.startMeeting({ meetingId: meeting.id });
    await core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" });
    expect((await meetingOf(core, meeting.id)).screen).toMatchObject({
      state: "reading",
      window: "Zoom Meeting",
    });

    const read = await readUntil(1);
    // Read once: the second frame was the same picture, so it wasn't read again.
    expect(reader.reads).toBe(1);
    expect(screen.grabs).toBe(2);
    // The same slide, a cursor over it: nothing new is read.
    screen.moveCursor();
    for (let second = 0; second < 5; second++) tick();
    expect(screen.grabs).toBe(2);

    const answer = await ask(core, meeting.mindId, "How much of the budget has been spent so far?");

    const sent = userText(calls[0] as ModelCall);
    expect(sent).toMatch(/<shared-screen id="S1" window="Zoom Meeting" read="\d\d:\d\d">/);
    expect(sent).toContain("Spent to date £640k");
    expect(calls[0]?.system).toContain(
      "<shared-screen> holds the text read from what someone shares",
    );
    for (const call of calls)
      expect([...partTypes(call)].every((type) => type === "text" || type.startsWith("tool"))).toBe(
        true,
      );
    const [citation] = answer.citations;
    expect(answer.citations).toHaveLength(1);
    expect(citation).toMatchObject({
      documentId: "",
      passageId: "",
      contentHash: "",
      quote: "",
      check: "found",
      location: {
        kind: "screen",
        screen: expect.stringMatching(new RegExp(`^${read.screen?.memory}/`)),
      },
    });
    if (citation?.location?.kind !== "screen") throw new Error("Not a screen's Citation.");
    const frame = await core.getSharedScreen(citation.location);
    expect(frame).toMatchObject({
      window: "Zoom Meeting",
      quote: "Spent to date £640k",
      image: expect.stringMatching(/^data:image\/jpeg;base64,/),
    });
    expect(frame?.boxes).toHaveLength(1);
    // What someone else showed is third-party text: a later write or send asks first (#78).
    expect(answer.textOrigin).toBe("shared-screen");
  });

  test("at Stop, nothing of the screen is kept: not in memory, not in the data folder, not in the Answer's Citation", async () => {
    const calls: ModelCall[] = [];
    const model = screenReadingModel(
      calls,
      "Spent to date £640k",
      "£640k of the £1.2m launch budget has been spent so far [^1].",
    );
    const { dataDir, core, meeting, readUntil } = await setUp(model);
    await core.startMeeting({ meetingId: meeting.id });
    await core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" });
    await readUntil(1);
    const answer = await ask(core, meeting.mindId, "How much of the budget has been spent so far?");
    const location = answer.citations[0]?.location;
    if (location?.kind !== "screen") throw new Error("Not a screen's Citation.");

    await core.stopMeeting(meeting.id);

    expect((await meetingOf(core, meeting.id)).screen).toBeNull();
    expect(await core.getSharedScreen(location)).toBeNull();
    const kept = await answerIn(core, answer.conversationId, answer.id);
    expect(kept.citations[0]).toMatchObject({ quote: "", location: { kind: "screen" } });
    expect(citationState(kept.citations[0] ?? {}, [], [], [])).toMatchObject({
      check: "cant-check",
      reason: "screen-deleted",
    });
    for (const { path, bytes } of await filesUnder(dataDir)) {
      for (const words of ["Confidential draft", "Spent to date", "Contingency left"]) {
        expect(bytes.includes(Buffer.from(words, "utf8")), `${words} in ${path}`).toBe(false);
        expect(bytes.includes(Buffer.from(words, "utf16le")), `${words} in ${path}`).toBe(false);
      }
    }
  });

  test("a screen that matches a page in the Library goes with that page, and the Answer's Citation of it stays after the meeting", async () => {
    const calls: ModelCall[] = [];
    const model = screenReadingModel(
      calls,
      "Spent to date £640k",
      "£640k of the £1.2m launch budget has been spent so far [^1][^2].",
    );
    const { dataDir, core, meeting, readUntil } = await setUp(model);
    const deck = join(dataDir, "Launch budget.md");
    await writeFile(
      deck,
      [
        "# Steering committee pack",
        "",
        "Launch budget",
        "",
        "Budget £1.2m",
        "",
        "Spent to date £640k",
        "",
        "Contingency left £90k",
        "",
        "Confidential draft for the steering committee",
      ].join("\n"),
    );
    const [document] = await addAndProcess(core, [deck]);
    await core.startMeeting({ meetingId: meeting.id });
    await core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" });
    await readUntil(1);

    const answer = await ask(core, meeting.mindId, "How much of the budget has been spent so far?");

    expect(userText(calls[0] as ModelCall)).toMatch(/<shared-screen id="S1"[^>]* same-as="P1">/);
    expect(answer.citations).toEqual([
      expect.objectContaining({ location: expect.objectContaining({ kind: "screen" }) }),
      expect.objectContaining({
        documentId: document?.id,
        quote: "Spent to date £640k",
        check: "found",
      }),
    ]);
    await core.stopMeeting(meeting.id);
    const after = await answerIn(core, answer.conversationId, answer.id);
    const documents = await core.listDocuments();
    expect(after.citations.map((each) => citationState(each, documents).check)).toEqual([
      "cant-check",
      "found",
    ]);
  });

  test("a figure the two readings disagree on goes marked as hard to read, and a quote of it is still found", async () => {
    const calls: ModelCall[] = [];
    const model = screenReadingModel(
      calls,
      "Spent to date £⟦640⟧k",
      "The slide shows £640k spent, but that figure was hard to read on the screen [^1].",
    );
    const { core, meeting, readUntil, screen } = await setUp(model, { noisy: true });
    screen.show(3);
    await core.startMeeting({ meetingId: meeting.id });
    await core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" });
    await readUntil(1);

    const answer = await ask(core, meeting.mindId, "How much has been spent?");

    expect(userText(calls[0] as ModelCall)).toContain("Spent to date £⟦640⟧k");
    expect(calls[0]?.system).toContain("was hard to read on the screen");
    expect(answer.citations[0]).toMatchObject({ check: "found", location: { kind: "screen" } });
  });

  test("Chinese on the shared screen is read and cited as English is", async () => {
    const calls: ModelCall[] = [];
    const model = screenReadingModel(
      calls,
      "收入 3.2 亿元",
      "华东区第三季度收入为 3.2 亿元 [^1]。",
    );
    const { core, meeting, readUntil, screen } = await setUp(model);
    screen.show(2);
    await core.startMeeting({ meetingId: meeting.id });
    await core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" });
    await readUntil(1);

    const answer = await ask(core, meeting.mindId, "屏幕上的收入是多少？");

    expect(userText(calls[0] as ModelCall)).toContain(
      "华东区第三季度收入\n收入 3.2 亿元\n同比增长 18%",
    );
    expect(answer.citations[0]).toMatchObject({ check: "found", quote: "" });
  });

  test("macOS's Screen Recording is asked for when the windows are first listed; refused, the meeting goes on without the screen", async () => {
    const asked = await setUp(undefined, { permission: "not-asked" });
    await asked.core.startMeeting({ meetingId: asked.meeting.id });
    expect((await meetingOf(asked.core, asked.meeting.id)).screen?.permission).toBe("not-asked");
    expect(await asked.core.listShareableWindows(asked.meeting.id)).toEqual([
      { id: "zoom", name: "Zoom Meeting", thumbnail: "" },
    ]);
    expect((await meetingOf(asked.core, asked.meeting.id)).screen?.permission).toBe("granted");

    const refused = await setUp(undefined, { permission: "refused" });
    await refused.core.startMeeting({ meetingId: refused.meeting.id });
    expect(await refused.core.listShareableWindows(refused.meeting.id)).toBeNull();
    const meeting = await meetingOf(refused.core, refused.meeting.id);
    expect(meeting.screen).toMatchObject({ state: "off", problem: "denied", permission: "denied" });
    expect(meeting.status).toBe("recording");
    expect(refused.screen.started).toBe(0);
  });

  test("a whole screen is refused, a closed window says so, and Pause stops the capture until Resume", async () => {
    const refused = await setUp(undefined, { refuse: "whole-screen" });
    await refused.core.startMeeting({ meetingId: refused.meeting.id });
    await refused.core.readSharedScreen({ meetingId: refused.meeting.id, windowId: "zoom" });
    expect((await meetingOf(refused.core, refused.meeting.id)).screen).toMatchObject({
      state: "off",
      problem: "whole-screen",
    });

    const { core, meeting, screen, readUntil } = await setUp();
    await core.startMeeting({ meetingId: meeting.id });
    await core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" });
    await readUntil(1);
    await core.pauseMeeting(meeting.id);
    expect(screen.capturing).toBe(false);
    expect((await meetingOf(core, meeting.id)).screen).toMatchObject({ state: "off", screens: 1 });
    await expect(
      core.readSharedScreen({ meetingId: meeting.id, windowId: "zoom" }),
    ).rejects.toThrow();
    await core.resumeMeeting(meeting.id);
    await vi.waitFor(async () => {
      expect((await meetingOf(core, meeting.id)).screen?.state).toBe("reading");
    });
    expect(screen.capturing).toBe(true);
    screen.close();
    expect((await meetingOf(core, meeting.id)).screen).toMatchObject({
      state: "off",
      problem: "closed",
      screens: 1,
    });
  });
});
