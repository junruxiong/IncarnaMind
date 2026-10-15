import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type {
  AnswerToolCall,
  AskInConversationInput,
  ConversationAnswer,
  Core,
  MindChange,
} from "../../src/core";
import { pendingProposals } from "../../src/shared/proposals";
import { answerEnded, shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { createSourceFolder, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { heading, note, shownWithChanges, writeMind } from "../helpers/minds";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";
import { buildPdf } from "../helpers/pdf";

/**
 * The assistant changes Minds and makes them (#202), through the core's
 * public interface as the assistant asks: what the User types decides whether
 * the Answer answers, changes a Mind or makes one, and every change arrives
 * in its Mind as Proposals, after a Checkpoint of it. The model's choices are
 * scripted here; real models' are checked by hand and in
 * tests/live/assistantChangesMinds.live.test.ts.
 */

/** A model that calls one Minds' Tool first, then replies with `reply`; edits drop "very" and "really". */
function toolModel(call: { tool: string; input: Record<string, unknown> } | null, reply: string) {
  return scriptedModel(
    (model: ModelCall): ScriptedReply =>
      shortened(model) ??
      (call && model.results.length === 0 ? { calls: [call] } : { text: reply }),
  );
}

/** Edit with AI's reply, to the request edit_mind makes: the passage with its "very"s and "really"s taken out. */
function shortened({ prompt }: ModelCall): ScriptedReply | null {
  const passage = /<passage>\n([\s\S]*?)\n<\/passage>/.exec(prompt)?.[1];
  return passage === undefined ? null : { text: passage.replace(/\b(?:very|really) /g, "") };
}

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

/** Asks in the assistant and resolves with the Answer as stored once it has ended. */
async function ask(core: Core, input: AskInConversationInput): Promise<ConversationAnswer> {
  const result = await core.askInConversation(input);
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  expect((await answerEnded(core, result.answerId)).event).toBe("finished");
  const { messages } = await core.getConversation(result.conversation.id);
  const answer = messages.find((message) => message.id === result.answerId);
  if (answer?.kind !== "answer") throw new Error("No Answer.");
  return answer;
}

const mindCalls = (answer: ConversationAnswer): AnswerToolCall[] =>
  answer.toolCalls.filter((call) => call.source === "minds");

const changesIn = (answer: ConversationAnswer): MindChange[] =>
  mindCalls(answer).flatMap((call) => (call.mindChange ? [call.mindChange] : []));

async function checkpointsOf(core: Core, mindId: string) {
  return (await core.listCheckpoints(mindId)).checkpoints;
}

/** What the model was told a Tool call came to: the last Tool result it read. */
const lastResult = (model: MockLanguageModelV4) => {
  const prompt = model.doStreamCalls.at(-1)?.prompt ?? [];
  const results = prompt.flatMap((message) =>
    message.role === "tool"
      ? message.content.flatMap((part) =>
          part.type === "tool-result" &&
          (part.output.type === "text" || part.output.type === "error-text")
            ? [part.output.value]
            : [],
        )
      : [],
  );
  return results.at(-1) ?? "";
};

describe("The assistant changes Minds as Proposals", { timeout: 30_000 }, () => {
  // protects=a change request edits the Mind in focus as Proposals, after a Checkpoint, and its card says so; fails_when=the Mind changes outright, no Checkpoint is saved first, or the card doesn't name the Mind; no_claim=how well a real model rewrites.
  test("a change request proposes changes in the Mind in focus, after a Checkpoint", async () => {
    const model = toolModel(
      { tool: "edit_mind", input: { instruction: "Shorten it" } },
      "Shortened the summary in “Q3 review”: dropped two fillers.",
    );
    const { core, mind, client } = await setUp(model);
    expect(await checkpointsOf(core, mind.id)).toHaveLength(0);

    const answer = await ask(core, { text: "Shorten the summary", mindId: mind.id });

    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "Summary",
      "The quarter was [-very -]good, and sales were [-really -]strong.",
    ]);
    const [checkpoint] = await checkpointsOf(core, mind.id);
    expect(checkpoint?.reason).toBe("accept");
    expect(changesIn(answer)).toEqual([
      { mindId: mind.id, created: false, request: expect.any(String), proposals: 2 },
    ]);
    expect(mindCalls(answer)[0]?.status).toBe("done");
    expect(answer.text).toBe("Shortened the summary in “Q3 review”: dropped two fillers.");
    expect(lastResult(model)).toMatch(/^Proposed 2 changes in “Q3 review” \(from 11 to 9 words\)/);
  });

  // protects=a Question is answered and changes nothing, though the Minds' Tools are offered; fails_when=the Mind or its Checkpoints change; no_claim=that a real model never edits on a Question.
  test("a Question is answered and changes no Mind", async () => {
    const model = toolModel(null, "The quarter was good.");
    const { core, mind, client } = await setUp(model);
    const before = JSON.stringify(client.blocks.toJSON());

    const answer = await ask(core, { text: "How was the quarter?", mindId: mind.id });

    expect(model.doStreamCalls[0]?.tools?.map((each) => each.name)).toEqual(
      expect.arrayContaining(["read_mind", "edit_mind", "add_to_mind", "create_mind"]),
    );
    expect(answer.text).toBe("The quarter was good.");
    expect(mindCalls(answer)).toEqual([]);
    await client.settled();
    expect(JSON.stringify(client.blocks.toJSON())).toBe(before);
    expect(await checkpointsOf(core, mind.id)).toHaveLength(0);
  });

  // protects=a write request naming another Mind adds to that Mind, not the one in focus; fails_when=the focused Mind changes, or the named one gets no Proposals or Checkpoint; no_claim=none beyond protects.
  test("a write request adds to the Mind it names, under the section it names", async () => {
    const model = toolModel(
      {
        tool: "add_to_mind",
        input: {
          mind: "Project plan",
          // As <mind> shows it, with its Markdown mark.
          after: "## Decisions",
          content: "Ship the beta in May.",
        },
      },
      "Added today's decision to “Project plan”.",
    );
    const { core, mind, client } = await setUp(model);
    const plan = await core.createMind({ title: "Project plan" });
    const planClient = await connectToMind(core, plan.id);
    writeMind(planClient, [
      heading(2, "Decisions"),
      note("Hire a designer."),
      heading(2, "Risks"),
      note("The supplier is late."),
    ]);
    await planClient.settled();

    const answer = await ask(core, {
      text: "Add today's decision to Project plan",
      mindId: mind.id,
    });

    await planClient.settled();
    expect(shownWithChanges(planClient)).toEqual([
      "Decisions",
      "Hire a designer.",
      "+ Ship the beta in May.",
      "Risks",
      "The supplier is late.",
    ]);
    expect(pendingProposals(client.blocks)).toEqual([]);
    expect(await checkpointsOf(core, plan.id)).toHaveLength(1);
    expect(await checkpointsOf(core, mind.id)).toHaveLength(0);
    expect(changesIn(answer)).toEqual([
      { mindId: plan.id, created: false, request: expect.any(String), proposals: 1 },
    ]);
  });

  // protects=a request for a new note makes a Mind in the Search scope's Folder, its draft as Proposals with checked Citations; fails_when=the Mind is made elsewhere, its text is accepted outright, or a Citation isn't checked; no_claim=how a real model drafts.
  test("a request for a new note makes a Mind in the Folder, its draft as changes with checked Citations", async () => {
    const SPRING = "Spring tides happen just after a new moon or a full moon.";
    const model = scriptedModel(({ results }): ScriptedReply => {
      const search = results.find((result) => result.tool === "search_documents");
      if (!search) {
        return { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] };
      }
      if (!results.some((result) => result.tool === "cite")) {
        const [passage] = shownPassages(search.text);
        const record = { marker: 1, passage: passage?.id, pageFrom: 1, pageTo: 1, quote: SPRING };
        return { calls: [{ tool: "cite", input: { citations: [record] } }] };
      }
      if (!results.some((result) => result.tool === "create_mind")) {
        const content = "## Tides\n\nSpring tides come at new and full moon [^1].";
        return { calls: [{ tool: "create_mind", input: { title: "Tide notes", content } }] };
      }
      return { text: "Made “Tide notes” from your Documents." };
    });
    const { core, mind } = await setUp(model);
    const folder = await core.createLibraryGroup({ name: "Coast", description: "" });
    const sources = await createSourceFolder();
    const pdf = buildPdf([{ lines: ["Spring and neap tides", SPRING] }]);
    const added = await core.addDocuments([await writeSourceFile(sources, "Tides.pdf", pdf)], {
      groupId: folder.id,
    });
    await waitForProcessing(
      core,
      added.documents.map((document) => document.id),
    );

    const answer = await ask(core, {
      text: "Summarise these Documents into a new note",
      mindId: mind.id,
      scope: { folderIds: [folder.id], tagIds: [], documentIds: [] },
    });

    const [change] = changesIn(answer);
    expect(change).toMatchObject({ created: true, proposals: 2 });
    const made = (await core.listMinds()).find((each) => each.id === change?.mindId);
    expect(made).toMatchObject({ title: "Tide notes", folderId: folder.id });
    const madeClient = await connectToMind(core, change?.mindId ?? "");
    expect(shownWithChanges(madeClient)).toEqual([
      "+ Tides",
      "+ Spring tides come at new and full moon ⟨found⟩.",
    ]);
  });

  // protects=a Mind the User names that doesn't exist, or no Mind at all, changes nothing and the model is told why; fails_when=another Mind is changed, or the call says it worked; no_claim=how the model words it.
  test("a Mind that isn't there, or none in focus, changes nothing and says why", async () => {
    const named = toolModel(
      { tool: "edit_mind", input: { mind: "Chapter 9", instruction: "Shorten it" } },
      "There is no Mind called Chapter 9.",
    );
    const { core, mind, client } = await setUp(named);
    const answer = await ask(core, { text: "Shorten Chapter 9", mindId: mind.id });
    expect(mindCalls(answer)[0]?.status).toBe("failed");
    expect(lastResult(named)).toContain("The User has no Mind called “Chapter 9”.");
    expect(lastResult(named)).toContain("“Q3 review”");
    expect(pendingProposals(client.blocks)).toEqual([]);

    const none = toolModel(
      { tool: "edit_mind", input: { instruction: "Shorten it" } },
      "No Mind is open.",
    );
    const other = await setUp(none);
    const unfocused = await ask(other.core, { text: "Shorten the summary" });
    expect(mindCalls(unfocused)[0]?.status).toBe("failed");
    expect(lastResult(none)).toContain("No Mind is open beside the assistant");
    expect(pendingProposals(other.client.blocks)).toEqual([]);
    expect(await checkpointsOf(other.core, other.mind.id)).toHaveLength(0);
  });

  // protects=what the User selected is what edit_mind changes when no section is named; fails_when=the rest of the Mind is rewritten too; no_claim=none beyond protects.
  test("with text selected, the change is to the selected paragraphs only", async () => {
    const model = toolModel(
      { tool: "edit_mind", input: { instruction: "Shorten it" } },
      "Shortened the selected paragraph.",
    );
    const { core, mind, client } = await setUp(model);
    writeMind(client, [
      note("Very first, really."),
      note("The quarter was very good, and sales were really strong."),
    ]);
    await client.settled();
    const selected = blockIdAt(client, 1);

    await ask(core, {
      text: "Shorten this",
      mindId: mind.id,
      focus: { selection: "The quarter was very good", blockIds: [selected] },
    });

    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "Very first, really.",
      "The quarter was [-very -]good, and sales were [-really -]strong.",
    ]);
  });
});

describe("Written again", { timeout: 30_000 }, () => {
  // protects=Regenerate on an Answer that changed Minds takes back what still waits from it first, so nothing is proposed twice and a Mind it made, still empty of the User's words, goes; fails_when=the changes double, or the first new Mind stays beside the second; no_claim=what the User has accepted, which stays.
  test("regenerating takes back its changes still waiting, and the Mind it made", async () => {
    const model = scriptedModel(
      (call): ScriptedReply =>
        shortened(call) ??
        (call.results.length === 0
          ? {
              calls: [
                { tool: "edit_mind", input: { instruction: "Shorten it" } },
                { tool: "create_mind", input: { title: "Notes", content: "One line." } },
              ],
            }
          : { text: "Done." }),
    );
    const { core, mind, client } = await setUp(model);
    const first = await ask(core, { text: "Shorten it and make notes", mindId: mind.id });
    const made = changesIn(first).find((change) => change.created)?.mindId;
    await client.settled();
    expect(pendingProposals(client.blocks)).toHaveLength(2);

    const result = await core.regenerateConversationAnswer({
      conversationId: first.conversationId,
      answerId: first.id,
    });
    if (!result.asked) throw new Error("Not asked again.");
    await answerEnded(core, result.answerId);

    await client.settled();
    expect(pendingProposals(client.blocks)).toHaveLength(2);
    expect(shownWithChanges(client)).toEqual([
      "Summary",
      "The quarter was [-very -]good, and sales were [-really -]strong.",
    ]);
    const notes = (await core.listMinds()).filter((each) => each.title === "Notes");
    expect(notes).toHaveLength(1);
    expect(notes[0]?.id).not.toBe(made);
  });
});

describe("A Search scope with no Documents", { timeout: 30_000 }, () => {
  // protects=a request to change a Mind in a Folder with no Documents still changes it, and the model is told the scope is empty; fails_when=the canned empty-scope Answer comes instead, or search is offered; no_claim=what a real model says to a Question there.
  test("a Mind in a Folder with no Documents can still be changed, and the model knows the scope is empty", async () => {
    const model = toolModel(
      { tool: "edit_mind", input: { instruction: "Shorten it" } },
      "Shortened the summary.",
    );
    const { core, mind, client } = await setUp(model);
    const folder = await core.createLibraryGroup({ name: "Empty", description: "" });

    const answer = await ask(core, {
      text: "Shorten the summary",
      mindId: mind.id,
      scope: { folderIds: [folder.id], tagIds: [], documentIds: [] },
    });

    expect(answer.text).toBe("Shortened the summary.");
    expect(answer.model).not.toBeNull();
    const [first] = model.doStreamCalls;
    expect(first?.tools?.map((each) => each.name)).not.toContain("search_documents");
    expect(JSON.stringify(first?.prompt)).toContain(
      "holds no Documents, so there is nothing to search",
    );
    await client.settled();
    expect(pendingProposals(client.blocks).length).toBe(2);
  });
});

describe("The Minds' Tools find what the model names", { timeout: 30_000 }, () => {
  // protects=edit_mind with a section's heading changes that section only, to the next heading as high; fails_when=the rest of the Mind is rewritten, or a lower heading ends the section; no_claim=none beyond protects.
  test("a section named by its heading is what changes", async () => {
    const model = toolModel(
      { tool: "edit_mind", input: { part: "risks", instruction: "Shorten it" } },
      "Shortened the risks.",
    );
    const { core, mind, client } = await setUp(model);
    writeMind(client, [
      heading(2, "Summary"),
      note("Sales were really strong."),
      heading(2, "Risks"),
      note("The supplier is very late."),
      heading(3, "Mitigation"),
      note("We are really looking."),
      heading(2, "Next"),
      note("Really soon."),
    ]);
    await client.settled();

    await ask(core, { text: "Shorten the risks", mindId: mind.id });

    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "Summary",
      "Sales were really strong.",
      "Risks",
      "The supplier is [-very -]late.",
      "Mitigation",
      "We are [-really -]looking.",
      "Next",
      "Really soon.",
    ]);
  });

  // protects=read_mind gives another Mind's text, as the Mind in focus is read; fails_when=the model gets nothing, or the wrong Mind; no_claim=none beyond protects.
  test("read_mind gives the Mind the User names", async () => {
    const model = toolModel({ tool: "read_mind", input: { mind: "“project PLAN”" } }, "Read it.");
    const { core, mind } = await setUp(model);
    const plan = await core.createMind({ title: "Project plan" });
    const planClient = await connectToMind(core, plan.id);
    writeMind(planClient, [heading(2, "Decisions"), note("Hire a designer.")]);
    await planClient.settled();

    await ask(core, { text: "What does the project plan decide?", mindId: mind.id });

    expect(lastResult(model)).toBe(
      ['<mind name="Project plan">', "## Decisions", "", "Hire a designer.", "</mind>"].join("\n"),
    );
  });
});

const blockIdAt = (client: MindClient, index: number): string => {
  const block = client.blocks.get(index);
  const id = "getAttribute" in block ? block.getAttribute("id") : undefined;
  if (typeof id !== "string") throw new Error("That Block has no id.");
  return id;
};

describe("New Blocks come in as the model writes them", { timeout: 30_000 }, () => {
  const CONTENT = "## Risks\n\nThe supplier is late.\n\nThe courier contract renews in February.";

  /** A model that adds CONTENT, its input streamed, held back from "courier" on until `until`. */
  function streamingAdd(until: Promise<void>) {
    const after = JSON.stringify({ content: CONTENT }).indexOf("courier");
    return scriptedModel(
      ({ results }): ScriptedReply =>
        results.length === 0
          ? {
              calls: [{ tool: "add_to_mind", input: { content: CONTENT }, streamInput: true }],
              pause: { after, until },
            }
          : { text: "Added a risks section." },
    );
  }

  /** Each "mind.writing" event, in order, as they come: whether it is writing. */
  const writingEvents = (core: Core) => {
    const seen: { writing: boolean }[] = [];
    core.on("mind.writing", ({ place }) => seen.push({ writing: place !== null }));
    return seen;
  };

  // protects=each Block the model finishes goes into the Mind as a change while the rest is still being written, with "Writing…" where the next goes, and all of them once it ends, none twice; fails_when=nothing shows until the call ends, a Block is written twice, or Writing… stays; no_claim=how a real provider chunks its input.
  test("finished Blocks are in the Mind while the rest is written, then all of them, once each", async () => {
    let release: () => void = () => undefined;
    const until = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { core, mind, client } = await setUp(streamingAdd(until));
    const seen = writingEvents(core);
    const asked = await core.askInConversation({ text: "Add a risks section", mindId: mind.id });
    if (!asked.asked) throw new Error("Not asked.");

    await expect
      .poll(() => shownWithChanges(client), { timeout: 10_000 })
      .toEqual([
        "Summary",
        "The quarter was very good, and sales were really strong.",
        "+ Risks",
        "+ The supplier is late.",
      ]);
    expect(seen.at(-1)?.writing).toBe(true);
    // The card says so while the call is still being written.
    const stored = async () => {
      const { messages } = await core.getConversation(asked.conversation.id);
      const answer = messages.find((message) => message.id === asked.answerId);
      return answer?.kind === "answer" ? changesIn(answer) : [];
    };
    await expect
      .poll(stored)
      .toEqual([{ mindId: mind.id, created: false, request: expect.any(String), proposals: 2 }]);

    release();
    expect((await answerEnded(core, asked.answerId)).event).toBe("finished");
    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "Summary",
      "The quarter was very good, and sales were really strong.",
      "+ Risks",
      "+ The supplier is late.",
      "+ The courier contract renews in February.",
    ]);
    expect(seen.at(-1)?.writing).toBe(false);
    expect(await checkpointsOf(core, mind.id)).toHaveLength(1);
  });

  // protects=a call that names another Mind writes its Blocks there as they come, never first into the Mind in focus; fails_when=Blocks show in the focused Mind, even for a moment; no_claim=a model that names the Mind after its content.
  test("a call naming another Mind writes there as it comes, never into the Mind in focus", async () => {
    const input = { mind: "Project plan", content: CONTENT };
    const after = JSON.stringify(input).indexOf("courier");
    const { core, mind, client } = await setUp(
      scriptedModel(
        ({ results }): ScriptedReply =>
          results.length === 0
            ? {
                calls: [{ tool: "add_to_mind", input, streamInput: true }],
                pause: { after, until: new Promise(() => undefined) },
              }
            : { text: "Added." },
        { chunkSize: 4 },
      ),
    );
    const plan = await core.createMind({ title: "Project plan" });
    const planClient = await connectToMind(core, plan.id);
    let focusedChanged = false;
    client.blocks.observeDeep(() => {
      if (pendingProposals(client.blocks).length > 0) focusedChanged = true;
    });

    const asked = await core.askInConversation({
      text: "Add risks to Project plan",
      mindId: mind.id,
    });
    if (!asked.asked) throw new Error("Not asked.");
    await expect
      .poll(() => shownWithChanges(planClient), { timeout: 10_000 })
      .toEqual(["+ Risks", "+ The supplier is late."]);
    expect(focusedChanged).toBe(false);
    await core.stopConversationAnswer({
      conversationId: asked.conversation.id,
      answerId: asked.answerId,
    });
  });

  // protects=a model that writes its content before naming the Mind gets its Blocks taken back from the Mind in focus and written into the one it named, once each, with nothing left behind; fails_when=a streamed Block stays in the focused Mind, or one is written twice; no_claim=how long the take-back shows.
  test("content written before the Mind is named is taken back from the Mind in focus and goes to the one named", async () => {
    // Content first, then the Mind: against the schema's order, as a model may write it.
    const input = { content: CONTENT, mind: "Project plan" };
    const { core, mind, client } = await setUp(
      scriptedModel(
        ({ results }): ScriptedReply =>
          results.length === 0
            ? { calls: [{ tool: "add_to_mind", input, streamInput: true }] }
            : { text: "Added." },
        { chunkSize: 4 },
      ),
    );
    const plan = await core.createMind({ title: "Project plan" });
    const planClient = await connectToMind(core, plan.id);
    let streamedHere = false;
    client.blocks.observeDeep(() => {
      if (pendingProposals(client.blocks).length > 0) streamedHere = true;
    });

    const asked = await core.askInConversation({ text: "Add risks", mindId: mind.id });
    if (!asked.asked) throw new Error("Not asked.");
    expect((await answerEnded(core, asked.answerId)).event).toBe("finished");
    await client.settled();
    await planClient.settled();

    expect(streamedHere).toBe(true);
    expect(shownWithChanges(client)).toEqual([
      "Summary",
      "The quarter was very good, and sales were really strong.",
    ]);
    expect(pendingProposals(client.blocks)).toEqual([]);
    expect(shownWithChanges(planClient)).toEqual([
      "+ Risks",
      "+ The supplier is late.",
      "+ The courier contract renews in February.",
    ]);
  });

  // protects=Stop while the model writes keeps the Blocks already written, as changes, on the Answer's card, and Writing… goes; fails_when=they vanish, or the card has no row for them; no_claim=none beyond protects.
  test("Stop keeps the Blocks already written, as changes on the card", async () => {
    const { core, mind, client } = await setUp(streamingAdd(new Promise(() => undefined)));
    const seen = writingEvents(core);
    const asked = await core.askInConversation({ text: "Add a risks section", mindId: mind.id });
    if (!asked.asked) throw new Error("Not asked.");
    await expect.poll(() => pendingProposals(client.blocks).length, { timeout: 10_000 }).toBe(2);

    await core.stopConversationAnswer({
      conversationId: asked.conversation.id,
      answerId: asked.answerId,
    });
    await client.settled();
    expect(pendingProposals(client.blocks)).toHaveLength(2);
    expect(seen.at(-1)?.writing).toBe(false);
    const { messages } = await core.getConversation(asked.conversation.id);
    const stopped = messages.find((message) => message.id === asked.answerId);
    if (stopped?.kind !== "answer") throw new Error("No Answer.");
    expect(changesIn(stopped)).toEqual([
      { mindId: mind.id, created: false, request: expect.any(String), proposals: 2 },
    ]);
  });
});
