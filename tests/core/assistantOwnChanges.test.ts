import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import { EDIT_INSTRUCTIONS } from "../../src/core/edits/prompt";
import { waitingProblemOf } from "../../src/renderer/src/changeProblems";
import { pendingProposals } from "../../src/shared/proposals";
import { askAssistant, changesIn, lastResult, mindCalls } from "../helpers/assistant";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { heading, note, shownWithChanges, writeMind } from "../helpers/minds";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

/**
 * The assistant and its own changes still waiting (#893), through the core's
 * public interface as the assistant asks: a later Question knows what an
 * earlier Answer's Tools really did, the Mind is read with its waiting
 * changes marked, "make it longer" revises the same waiting text, changes it
 * can't revise make it say so and change nothing, a reply that claims a
 * change no call made is asked for once more, and a failed change says why.
 * The model's choices are scripted; real models' are checked by hand and in
 * the everyday-requests evaluation.
 */

/** The text of the last user message of a request: the Question it answers. */
const asked = ({ options }: ModelCall): string => {
  const last = options.prompt.filter((message) => message.role === "user").at(-1);
  if (!last) return "";
  return typeof last.content === "string"
    ? last.content
    : last.content.map((part) => ("text" in part ? part.text : "")).join("");
};

/** Edit with AI's request, and the passage it asks to rewrite; null for any other request. */
const editPassage = (call: ModelCall): string | null =>
  call.system === EDIT_INSTRUCTIONS
    ? (/<passage>\n([\s\S]*?)\n<\/passage>/.exec(call.prompt)?.[1] ?? null)
    : null;

const called = (call: ModelCall, tool: string) => call.results.some((each) => each.tool === tool);

/** A Mind "Q3 review" with the User's own summary, on a scripted model. */
async function setUp(model: MockLanguageModelV4) {
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const mind = await core.createMind({ title: "Q3 review" });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [
    heading(2, "Summary"),
    note("The quarter was very good, and sales were really strong."),
  ]);
  await client.settled();
  return { core, mind, client };
}

/** The Mind an Answer made, connected. */
async function madeBy(core: Core, answer: Awaited<ReturnType<typeof askAssistant>>) {
  const made = changesIn(answer).find((change) => change.created);
  if (!made) throw new Error("The Answer made no Mind.");
  return connectToMind(core, made.mindId);
}

/** How many changes wait in a Mind from a request. */
const waitingFrom = (client: MindClient, request: string) =>
  pendingProposals(client.blocks).filter((each) => each.request === request).length;

const SUMMARY_PAGE = "## Summary\n\nThe paper is about spring tides.";

describe("The assistant revises its own changes still waiting", { timeout: 30_000 }, () => {
  // protects=after "write a summary page", "make it longer" lengthens that same waiting summary: the next Question knows what the first Answer made, reads its text marked as waiting, and the Mind holds one summary whose count is the Answer's; fails_when=a second summary is added, the waiting text reads as empty, or the bar and the card disagree; no_claim=how well a real model lengthens it.
  test("“make it longer” lengthens the same waiting summary, and the Mind holds one", async () => {
    const seen: { record: string; read: string } = { record: "", read: "" };
    const model = scriptedModel((call): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) {
        return { text: `${passage} It also covers neap tides.\n<why>Longer.</why>` };
      }
      if (asked(call).endsWith("write a summary page")) {
        return called(call, "create_mind")
          ? { text: "Made “Tide notes” with a short summary." }
          : {
              calls: [
                { tool: "create_mind", input: { title: "Tide notes", content: SUMMARY_PAGE } },
              ],
            };
      }
      if (!called(call, "read_mind")) {
        seen.record = asked(call);
        return { calls: [{ tool: "read_mind", input: { mind: "Tide notes" } }] };
      }
      if (!called(call, "edit_mind")) {
        seen.read = call.results.find((each) => each.tool === "read_mind")?.text ?? "";
        return {
          calls: [
            {
              tool: "edit_mind",
              input: { mind: "Tide notes", part: "Summary", instruction: "Make it longer" },
            },
          ],
        };
      }
      return { text: "Lengthened the summary in “Tide notes”." };
    });
    const { core, mind } = await setUp(model);

    const first = await askAssistant(core, { text: "write a summary page", mindId: mind.id });
    const page = await madeBy(core, first);
    const [made] = changesIn(first);
    const second = await askAssistant(core, {
      text: "make the summary longer",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    expect(seen.record).toContain(
      "- create_mind: made the Mind “Tide notes”, its text as 2 changes waiting for the User",
    );
    expect(seen.read).toContain("## [waiting change, adds: Summary]");
    expect(seen.read).toContain("[waiting change, adds: The paper is about spring tides.]");
    await page.settled();
    expect(shownWithChanges(page)).toEqual([
      "+ Summary",
      "+ The paper is about spring tides. It also covers neap tides.",
    ]);
    const [revised] = changesIn(second);
    expect(revised).toMatchObject({ mindId: made?.mindId, revised: true, proposals: 2 });
    expect(pendingProposals(page.blocks)).toHaveLength(revised?.proposals ?? -1);
    expect(waitingFrom(page, made?.request ?? "")).toBe(0);
    expect(lastResult(model)).toMatch(/^Revised your own waiting text in “Tide notes”/);
    expect(second.steps?.unbackedClaims).toBeUndefined();
  });

  // protects=removing part of the assistant's waiting text removes it there and leaves one version, the rest moving to the new Answer's change so the counts agree; fails_when=the part stays, the rest is lost, or the counts differ; no_claim=none beyond protects.
  test("“remove the part about…” removes it from the waiting text, and the counts agree", async () => {
    const model = scriptedModel((call): ScriptedReply => {
      if (editPassage(call) !== null) return { text: "<why>Removed it.</why>" };
      if (asked(call).endsWith("write a page on tides")) {
        const content =
          "## Spring tides\n\nThey come at new moon.\n\n## Neap tides\n\nThey come at half moon.";
        return called(call, "create_mind")
          ? { text: "Made “Tide notes”." }
          : { calls: [{ tool: "create_mind", input: { title: "Tide notes", content } }] };
      }
      return called(call, "edit_mind")
        ? { text: "Removed the neap tides section from “Tide notes”." }
        : {
            calls: [
              {
                tool: "edit_mind",
                input: { mind: "Tide notes", part: "Neap tides", instruction: "Remove it" },
              },
            ],
          };
    });
    const { core, mind } = await setUp(model);
    const first = await askAssistant(core, { text: "write a page on tides", mindId: mind.id });
    const page = await madeBy(core, first);

    const second = await askAssistant(core, {
      text: "remove the part about neap tides",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    await page.settled();
    expect(shownWithChanges(page)).toEqual(["+ Spring tides", "+ They come at new moon."]);
    const [revised] = changesIn(second);
    expect(revised).toMatchObject({ revised: true, proposals: 2 });
    expect(pendingProposals(page.blocks)).toHaveLength(2);
    expect(waitingFrom(page, changesIn(first)[0]?.request ?? "")).toBe(0);
  });

  // protects=edit_mind with new text replaces the assistant's own waiting section with it, as one version; fails_when=both versions stay, or it is refused; no_claim=none beyond protects.
  test("new text for its own waiting section takes that section's place", async () => {
    const longer = "## Summary\n\nThe paper is about spring tides, and how the moon drives them.";
    const model = scriptedModel((call): ScriptedReply => {
      if (asked(call).endsWith("write a summary page")) {
        return called(call, "create_mind")
          ? { text: "Made “Tide notes”." }
          : {
              calls: [
                { tool: "create_mind", input: { title: "Tide notes", content: SUMMARY_PAGE } },
              ],
            };
      }
      return called(call, "edit_mind")
        ? { text: "Rewrote the summary in “Tide notes”." }
        : {
            calls: [
              {
                tool: "edit_mind",
                input: { mind: "Tide notes", part: "Summary", content: longer },
              },
            ],
          };
    });
    const { core, mind } = await setUp(model);
    const first = await askAssistant(core, { text: "write a summary page", mindId: mind.id });
    const page = await madeBy(core, first);

    const second = await askAssistant(core, {
      text: "make it longer",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    await page.settled();
    expect(shownWithChanges(page)).toEqual([
      "+ Summary",
      "+ The paper is about spring tides, and how the moon drives them.",
    ]);
    expect(changesIn(second)).toEqual([expect.objectContaining({ revised: true, proposals: 2 })]);
  });

  // protects=asked to change all of a Mind where its own waiting text sits beside the User's, the assistant is told to name the section, and revises only its own; fails_when=the User's text is changed, or nothing is done; no_claim=none beyond protects.
  test("its waiting section beside the User's text is revised once the section is named", async () => {
    const model = scriptedModel((call): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) return { text: `${passage} The courier is late too.` };
      if (asked(call).endsWith("add a risks section")) {
        return called(call, "add_to_mind")
          ? { text: "Added a risks section to “Q3 review”." }
          : {
              calls: [
                {
                  tool: "add_to_mind",
                  input: { content: "## Risks\n\nThe supplier is late." },
                },
              ],
            };
      }
      const edits = call.results.filter((each) => each.tool === "edit_mind").length;
      if (edits === 0) {
        return { calls: [{ tool: "edit_mind", input: { instruction: "Make the risks longer" } }] };
      }
      if (edits === 1) {
        return {
          calls: [
            { tool: "edit_mind", input: { part: "Risks", instruction: "Make the risks longer" } },
          ],
        };
      }
      return { text: "Lengthened the risks in “Q3 review”." };
    });
    const { core, mind, client } = await setUp(model);
    const first = await askAssistant(core, { text: "add a risks section", mindId: mind.id });

    const second = await askAssistant(core, {
      text: "make the risks longer",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    const [whole, named] = mindCalls(second);
    expect(whole).toMatchObject({
      status: "failed",
      mindProblem: { kind: "waiting", mindId: mind.id, mixed: true },
    });
    expect(named).toMatchObject({ status: "done", mindChange: { revised: true } });
    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "Summary",
      "The quarter was very good, and sales were really strong.",
      "+ Risks",
      "+ The supplier is late. The courier is late too.",
    ]);
    expect(waitingProblemOf(second)).toBeNull();
  });
});

describe("Changes it can't revise", { timeout: 30_000 }, () => {
  // protects=over changes the User asked for (Edit with AI's, here from an earlier edit_mind), the assistant changes nothing, is told to say so first, the card says why, and the next Question knows it failed; fails_when=it changes the Mind another way, the problem isn't recorded, or the record says it worked; no_claim=how the model words it.
  test("over changes it can't revise, nothing changes, and it says why", async () => {
    let record = "";
    const model = scriptedModel((call): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) return { text: passage.replace(/\b(?:very|really) /g, "") };
      if (asked(call).endsWith("anything else?")) {
        record = call.prompt;
        return { text: "No." };
      }
      return called(call, "edit_mind")
        ? { text: "Done." }
        : { calls: [{ tool: "edit_mind", input: { instruction: "Shorten it" } }] };
    });
    const { core, mind, client } = await setUp(model);
    const first = await askAssistant(core, { text: "shorten the summary", mindId: mind.id });
    await client.settled();
    const before = shownWithChanges(client);
    const checkpoints = (await core.listCheckpoints(mind.id)).checkpoints.length;

    const second = await askAssistant(core, {
      text: "shorten it more",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    const [refused] = mindCalls(second);
    expect(refused).toMatchObject({
      status: "failed",
      mindProblem: { kind: "waiting", mindId: mind.id, count: 2, mixed: false },
    });
    expect(lastResult(model)).toContain("Say so first, in one sentence");
    await client.settled();
    expect(shownWithChanges(client)).toEqual(before);
    expect((await core.listCheckpoints(mind.id)).checkpoints).toHaveLength(checkpoints);
    expect(waitingProblemOf(second)).toEqual({ mindId: mind.id, count: 2 });

    await askAssistant(core, {
      text: "anything else?",
      mindId: mind.id,
      conversationId: first.conversationId,
    });
    expect(record).toContain("- edit_mind “Q3 review”: proposed 2 changes waiting for the User");
    expect(record).toContain(
      "- edit_mind “Q3 review”: nothing changed: 2 changes the assistant can't revise waited there for the User",
    );
    expect(record).toContain("[waiting change, removes: very ]");
  });

  // protects=a failed change keeps why on its card: a heading that isn't there, a Mind that isn't; fails_when=the card has no reason; no_claim=none beyond protects.
  test("a heading or a Mind that isn't there is the reason on the card", async () => {
    const model = scriptedModel(
      ({ results }): ScriptedReply =>
        results.length === 0
          ? {
              calls: [
                { tool: "edit_mind", input: { part: "Risks", instruction: "Shorten it" } },
                { tool: "read_mind", input: { mind: "Chapter 9" } },
              ],
            }
          : { text: "There is no risks section, and no Chapter 9." },
    );
    const { core, mind } = await setUp(model);

    const answer = await askAssistant(core, { text: "shorten the risks", mindId: mind.id });

    expect(mindCalls(answer).map((call) => call.mindProblem)).toEqual([
      { kind: "no-heading", mindId: mind.id, heading: "Risks" },
      { kind: "not-found", name: "Chapter 9" },
    ]);
  });
});

describe("A reply that claims a change no call made", { timeout: 30_000 }, () => {
  const CLAIM = "I've added a risks section to “Q3 review”.";
  const ASKED_AGAIN = "but no Tool did that";

  // protects=a reply claiming a change no call made is asked for once more with the record, and the honest second reply is what the User reads, unflagged; fails_when=the claim stands, or the model isn't asked again; no_claim=that a real model corrects itself.
  test("is asked for once more, and the honest reply stands", async () => {
    let note = "";
    const model = scriptedModel((call): ScriptedReply => {
      if (!call.prompt.includes(ASKED_AGAIN)) return { text: CLAIM };
      note = asked(call);
      return { text: "I haven't changed “Q3 review” yet: say if you'd like a risks section." };
    });
    const { core, mind, client } = await setUp(model);

    const answer = await askAssistant(core, { text: "add risks", mindId: mind.id });

    expect(model.doStreamCalls).toHaveLength(2);
    expect(note).toContain("<tool-record>");
    expect(note).toContain("- No Tool was called.");
    expect(note).toContain("Your reply says you added text to a Mind, but no Tool did that.");
    expect(answer.text).toBe(
      "I haven't changed “Q3 review” yet: say if you'd like a risks section.",
    );
    expect(answer.steps?.unbackedClaims).toBeUndefined();
    expect(pendingProposals(client.blocks)).toEqual([]);
  });

  // protects=asked again, the model may make the change it claimed, and then nothing is flagged; fails_when=the change isn't made, or the reply is still flagged; no_claim=none beyond protects.
  test("asked once more, it may make the change, and nothing is flagged", async () => {
    const model = scriptedModel((call): ScriptedReply => {
      if (!call.prompt.includes(ASKED_AGAIN)) return { text: CLAIM };
      return called(call, "add_to_mind")
        ? { text: "Added a risks section to “Q3 review”." }
        : {
            calls: [
              { tool: "add_to_mind", input: { content: "## Risks\n\nThe supplier is late." } },
            ],
          };
    });
    const { core, mind, client } = await setUp(model);

    const answer = await askAssistant(core, { text: "add risks", mindId: mind.id });

    expect(answer.text).toBe("Added a risks section to “Q3 review”.");
    expect(answer.steps?.unbackedClaims).toBeUndefined();
    await client.settled();
    expect(shownWithChanges(client).slice(-2)).toEqual(["+ Risks", "+ The supplier is late."]);
  });

  // protects=a reply that still claims a change no call made is shown flagged; fails_when=it is shown as if the change were made; no_claim=none beyond protects.
  test("still claiming it, the reply is flagged", async () => {
    const model = scriptedModel((): ScriptedReply => ({ text: CLAIM }));
    const { core, mind } = await setUp(model);

    const answer = await askAssistant(core, { text: "add risks", mindId: mind.id });

    expect(model.doStreamCalls).toHaveLength(2);
    expect(answer.text).toBe(CLAIM);
    expect(answer.steps?.unbackedClaims).toEqual(["added"]);
  });

  // protects=a claim a call backs is never asked about again; fails_when=the model is asked once more after a real change; no_claim=none beyond protects.
  test("a change a call made isn't asked about", async () => {
    const model = scriptedModel(
      ({ results }): ScriptedReply =>
        results.length === 0
          ? {
              calls: [
                { tool: "add_to_mind", input: { content: "## Risks\n\nThe supplier is late." } },
              ],
            }
          : { text: "Added a risks section to “Q3 review”." },
    );
    const { core, mind } = await setUp(model);

    const answer = await askAssistant(core, { text: "add risks", mindId: mind.id });

    expect(model.doStreamCalls).toHaveLength(2);
    expect(answer.steps?.unbackedClaims).toBeUndefined();
  });
});
