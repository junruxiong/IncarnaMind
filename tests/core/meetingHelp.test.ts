/**
 * One-click help during a call (#839), through the core's public interface as the floating
 * panel and the assistant ask: a button's Question goes with the live transcript, the shape its
 * Answer takes and what the User's Documents say about the newest lines, found ahead; the
 * Meeting names its latest help, the line it answers and the conversation it went to; the
 * template adds its buttons.
 */
import { describe, expect, test } from "vitest";
import type { Core, Meeting, MeetingHelpKind } from "../../src/core";
import { answerEnded, shownPassages } from "../helpers/citations";
import { createTempDataFolder } from "../helpers/core";
import { createSourceFolder, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { type ModelCall, scriptedModel } from "../helpers/models";
import { transcribingCore, until } from "../helpers/transcription";

const SECOND = 1000;
const PRICE = "Enterprise plan: $45 per seat a month, single sign-on included.";

const block = (text: string, tag: string) =>
  new RegExp(`<${tag}>\\n([\\s\\S]*?)\\n</${tag}>`).exec(text)?.[1] ?? null;

/** A model that answers in the help shape, citing the first Passage found ahead if it has one. */
function helpingModel(calls: ModelCall[]) {
  return scriptedModel((call) => {
    calls.push(call);
    const [found] = shownPassages(block(call.prompt, "found-in-documents") ?? "");
    if (found && !call.results.some((result) => result.tool === "cite")) {
      return {
        calls: [
          { tool: "cite", input: { citations: [{ marker: 1, passage: found.id, quote: PRICE }] } },
        ],
      };
    }
    return {
      text: found
        ? "★ Yes, Enterprise is $45 a seat [^1]\n→ answers at once\n\nI'll send the price sheet\n→ buys time"
        : "★ Let me confirm by Friday\n→ keeps it moving\n\nWhat matters most to you?\n→ learns their priority",
    };
  });
}

/** A meeting recording with lines from both sides; `before`, what to do first (Documents wait while it records). */
async function setUp(before?: (core: Core) => Promise<void>) {
  const calls: ModelCall[] = [];
  const dataDir = await createTempDataFolder();
  const { core, record } = await transcribingCore(dataDir, { model: helpingModel(calls) });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  await before?.(core);
  const meeting = await core.createMeeting();
  await core.startMeeting({ meetingId: meeting.id, systemAudio: true });
  record(9 * SECOND);
  await until(core, meeting.id, (transcript) => transcript.lines >= 4);
  return { core, meeting, calls };
}

async function press(core: Core, meeting: Meeting, kind: MeetingHelpKind, conversationId?: string) {
  const result = await core.askMeetingHelp({
    meetingId: meeting.id,
    kind,
    ...(conversationId && { conversationId }),
  });
  if (!result.asked) throw new Error(`Not asked: ${JSON.stringify(result)}`);
  expect((await answerEnded(core, result.answerId)).event).toBe("finished");
  const { messages } = await core.getConversation(result.conversation.id);
  const answer = messages.find((message) => message.id === result.answerId);
  if (answer?.kind !== "answer") throw new Error("No Answer.");
  const latest = (await core.listMeetings()).find((each) => each.id === meeting.id);
  return { result, answer, messages, latest };
}

describe("Help during the call", { timeout: 60_000 }, () => {
  test("a button asks its name with the live transcript and the options' shape, offered the Documents' Tools only, and the Meeting names its help", async () => {
    const { core, meeting, calls } = await setUp();

    const { result, answer, messages, latest } = await press(core, meeting, "no");

    const [first] = calls;
    expect(first?.system).toContain("pressed a button for help");
    expect(first?.system).toContain('starting with "★ "');
    expect(first?.prompt).toContain("<live-transcript>");
    expect(first?.tools).toEqual(expect.arrayContaining(["search_documents", "cite"]));
    expect(first?.tools).not.toContain("add_to_mind");
    expect(messages[0]).toMatchObject({ kind: "question", text: "Say no politely" });
    expect(answer.text).toMatch(/^★ /);
    expect(latest?.help).toEqual({
      kind: "no",
      conversationId: result.asked && result.conversation.id,
      answerId: result.asked && result.answerId,
      heard: { text: expect.any(String), speaker: "Them" },
    });
  });

  test("Answer their question finds the User's Documents ahead for the newest lines, not the transcript, and cites them", async () => {
    const { core, meeting, calls } = await setUp(async (before) => {
      const sources = await createSourceFolder();
      const added = await before.addDocuments([
        await writeSourceFile(
          sources,
          "Price sheet.txt",
          `Price sheet 2026\n\nTeam plan: $25 per seat a month.\n${PRICE}\nThe launch moves to the eighteenth of November for every plan.`,
        ),
      ]);
      await waitForProcessing(
        before,
        added.documents.map((document) => document.id),
      );
    });

    const { answer } = await press(core, meeting, "answer");

    const found = block(calls[0]?.prompt ?? "", "found-in-documents");
    expect(found).toContain("Price sheet");
    expect(found).not.toContain("Transcript");
    expect(calls[0]?.tools).toContain("search_documents");
    expect(answer.citations).toEqual([
      expect.objectContaining({ documentName: "Price sheet", quote: PRICE, check: "found" }),
    ]);
  });

  test("a second press goes on in the same conversation, the assistant's when it says so, and a deleted one starts anew", async () => {
    const { core, meeting } = await setUp();
    const first = await press(core, meeting, "say");
    const second = await press(core, meeting, "ask");
    expect(second.result.asked && second.result.conversation.id).toBe(
      first.result.asked && first.result.conversation.id,
    );
    expect(second.latest?.help?.kind).toBe("ask");

    if (first.result.asked) await core.deleteConversation(first.result.conversation.id);
    const third = await press(core, meeting, "sum");
    expect(third.result.asked && third.result.conversation.id).not.toBe(
      first.result.asked && first.result.conversation.id,
    );
    expect(third.latest?.help?.heard).toBeNull();
  });

  test("the template adds its buttons; another template's, or help after Stop, is refused", async () => {
    const { core, meeting } = await setUp();
    await expect(core.askMeetingHelp({ meetingId: meeting.id, kind: "leading" })).rejects.toThrow();

    const set = await core.setMeetingTemplate({ meetingId: meeting.id, template: "interview" });
    expect(set.helpTemplate).toBe("interview");
    const { messages, latest } = await press(core, meeting, "leading");
    expect(messages.at(-2)).toMatchObject({ kind: "question", text: "Am I leading them?" });
    expect(latest?.help?.heard?.speaker).toBe("Me");
    await expect(
      core.askMeetingHelp({ meetingId: meeting.id, kind: "objection" }),
    ).rejects.toThrow();

    await core.stopMeeting(meeting.id);
    const stopped = (await core.listMeetings()).find((each) => each.id === meeting.id);
    expect(stopped?.help).toBeNull();
    expect(stopped?.helpTemplate).toBeNull();
    await expect(core.askMeetingHelp({ meetingId: meeting.id, kind: "say" })).rejects.toThrow();
  });

  test("the buttons' shortcuts are kept per device, and only keys with ⌘, ⌃ or ⌥ are taken", async () => {
    const { core } = await setUp();
    const saved = await core.updateSettings({ device: { meetingHelpKeys: { say: "Ctrl+Alt+9" } } });
    expect(saved.device.meetingHelpKeys).toEqual({ say: "Ctrl+Alt+9" });
    expect(saved.device.meetingPanel).toBe(true);
    expect(saved.device.meetingPanelHidden).toBe(true);
    await expect(
      core.updateSettings({ device: { meetingHelpKeys: { say: "Shift+A" } } }),
    ).rejects.toThrow();
    await expect(
      core.updateSettings({ device: { meetingHelpKeys: { leading: "Ctrl+Alt+1" } } } as never),
    ).rejects.toThrow();
  });
});
