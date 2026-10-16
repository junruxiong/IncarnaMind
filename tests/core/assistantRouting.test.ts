import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import { EDIT_INSTRUCTIONS } from "../../src/core/edits/prompt";
import { pendingProposals, proposalRequestOf, resolveProposals } from "../../src/shared/proposals";
import { askAssistant, changesIn, lastResult } from "../helpers/assistant";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { heading, note, shownWithChanges, writeMind } from "../helpers/minds";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

/**
 * How the assistant reads what the User asked for (`routeRequest`), through
 * the core's public interface: the model is told, a request for text in a
 * Mind that got a reply instead is asked for once more, and the requests the
 * calls make record what was asked, for the changes that apply at once
 * (#880). The model's choices are scripted.
 */

const asked = ({ options }: ModelCall): string => {
  const last = options.prompt.filter((message) => message.role === "user").at(-1);
  if (!last) return "";
  return typeof last.content === "string"
    ? last.content
    : last.content.map((part) => ("text" in part ? part.text : "")).join("");
};

const called = (call: ModelCall, tool: string) => call.results.some((each) => each.tool === tool);

const editPassage = (call: ModelCall): string | null =>
  call.system === EDIT_INSTRUCTIONS
    ? (/<passage>\n([\s\S]*?)\n<\/passage>/.exec(call.prompt)?.[1] ?? null)
    : null;

const ASKED_TO_WRITE = "The User asked for this in a Mind";
const SUMMARY_PAGE = "## Summary\n\nThe paper is about spring tides.";

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
  return { core, mind };
}

async function madeBy(core: Core, answer: Awaited<ReturnType<typeof askAssistant>>) {
  const made = changesIn(answer).find((change) => change.created);
  if (!made) throw new Error("The Answer made no Mind.");
  return connectToMind(core, made.mindId);
}

describe("A request for text in a Mind", { timeout: 30_000 }, () => {
  // protects=asked for a page, a reply that writes it in the panel is held back, the model is asked once more, and it writes the page, tagged as asked; fails_when=the panel text shows, the model isn't asked again, or the page's request isn't tagged "write"; no_claim=that a real model complies.
  test("written in the reply instead, it is asked for once more and goes into a page", async () => {
    let system = "";
    const model = scriptedModel((call): ScriptedReply => {
      system = call.system;
      if (!asked(call).includes(ASKED_TO_WRITE))
        return { text: "Summary: the paper is about tides." };
      return called(call, "create_mind")
        ? { text: "It covers the paper." }
        : {
            calls: [{ tool: "create_mind", input: { title: "Tide notes", content: SUMMARY_PAGE } }],
          };
    });
    const { core, mind } = await setUp(model);
    const shown: string[] = [];
    core.on("answer.delta", (event) => shown.push(event.text));

    const answer = await askAssistant(core, { text: "write a summary page", mindId: mind.id });

    expect(system).toContain("This request asks for a new page or note");
    expect(shown.join("")).not.toContain("Summary: the paper");
    expect(answer.text).toBe("It covers the paper.");
    const page = await madeBy(core, answer);
    await page.settled();
    const request = changesIn(answer)[0]?.request ?? "";
    expect(proposalRequestOf(page.doc, request)).toMatchObject({
      source: "answer-added",
      requested: "write",
    });
  });

  // protects=a Question, or a request that can't be told, is answered in the reply with no instruction to write and nothing asked again; fails_when=the model is pushed to change a Mind; no_claim=none beyond protects.
  test.each(["how was the quarter?", "Could you shorten my summary?"])(
    "“%s” is answered in the reply, and nothing is asked again",
    async (text) => {
      let system = "";
      const model = scriptedModel((call): ScriptedReply => {
        system = call.system;
        return { text: "The quarter was good." };
      });
      const { core, mind } = await setUp(model);

      const answer = await askAssistant(core, { text, mindId: mind.id });

      expect(answer.text).toBe("The quarter was good.");
      expect(model.doStreamCalls).toHaveLength(1);
      expect(system).not.toContain("This request asks");
    },
  );

  // protects=a follow-up with no object is read as a change to what the conversation's last Answer wrote into a Mind, and as unclear with nothing written before; fails_when=the model is pushed to change a Mind with nothing written, or isn't told to after a write; no_claim=that a real model then edits.
  test("“make it longer” asks for an edit after an Answer that changed a Mind, and nothing in a new conversation", async () => {
    const systems: string[] = [];
    const model = scriptedModel((call): ScriptedReply => {
      if (asked(call).endsWith("make it longer")) systems.push(call.system);
      if (!asked(call).endsWith("add a risks section")) return { text: "Which text?" };
      return called(call, "add_to_mind")
        ? { text: "Done." }
        : {
            calls: [
              { tool: "add_to_mind", input: { content: "## Risks\n\nThe supplier is late." } },
            ],
          };
    });
    const { core, mind } = await setUp(model);

    await askAssistant(core, { text: "make it longer", mindId: mind.id });
    const first = await askAssistant(core, { text: "add a risks section", mindId: mind.id });
    await askAssistant(core, {
      text: "make it longer",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    expect(systems).toHaveLength(2);
    expect(systems[0]).not.toContain("This request asks");
    expect(systems[1]).toContain("This request asks to change text in a Mind");
  });

  // protects=a request a Tool refused (no Mind open) isn't asked for again: the refusal is the answer; fails_when=the model is asked to write after it was told it can't; no_claim=none beyond protects.
  test("one a Tool refused isn't asked for again", async () => {
    const model = scriptedModel(
      ({ results }): ScriptedReply =>
        results.length === 0
          ? { calls: [{ tool: "add_to_mind", input: { content: "## Risks\n\nLate." } }] }
          : { text: "No Mind is open: open one, or ask for a new Mind." },
    );
    const { core } = await setUp(model);

    const answer = await askAssistant(core, { text: "add a risks section" });

    expect(answer.text).toBe("No Mind is open: open one, or ask for a new Mind.");
    expect(model.doStreamCalls).toHaveLength(2);
  });
});

describe("What each change's request records", { timeout: 30_000 }, () => {
  // protects=the owner's two requests tag their changes as asked: the new page "write", the edit of it "edit", and a revision of waiting text nothing; fails_when=a tag is missing or wrong, or a revision is tagged; no_claim=how #880 applies them.
  test("“write a summary page” is tagged write, “make it longer” edit, a revision nothing", async () => {
    const model = scriptedModel((call): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) return { text: `${passage} It also covers neap tides.` };
      if (asked(call).endsWith("write a summary page")) {
        return called(call, "create_mind")
          ? { text: "It covers the paper." }
          : {
              calls: [
                { tool: "create_mind", input: { title: "Tide notes", content: SUMMARY_PAGE } },
              ],
            };
      }
      return called(call, "edit_mind")
        ? { text: "It says more now." }
        : {
            calls: [
              {
                tool: "edit_mind",
                input: { mind: "Tide notes", part: "Summary", instruction: "Make it longer" },
              },
            ],
          };
    });
    const { core, mind } = await setUp(model);
    const first = await askAssistant(core, { text: "write a summary page", mindId: mind.id });
    const page = await madeBy(core, first);
    await page.settled();
    const tagOf = (answer: Awaited<ReturnType<typeof askAssistant>>) =>
      proposalRequestOf(page.doc, changesIn(answer)[0]?.request ?? "");
    expect(tagOf(first)).toMatchObject({ requested: "write" });

    const revising = await askAssistant(core, {
      text: "make the summary section longer",
      mindId: mind.id,
      conversationId: first.conversationId,
    });
    await page.settled();
    expect(changesIn(revising)[0]).toMatchObject({ revised: true });
    expect(tagOf(revising)).not.toHaveProperty("requested");

    page.doc.transact(() => resolveProposals(page.blocks, () => true, "accept"));
    await page.settled();
    expect(pendingProposals(page.blocks)).toEqual([]);
    const editing = await askAssistant(core, {
      text: "make the summary section longer",
      mindId: mind.id,
      conversationId: first.conversationId,
    });
    await page.settled();

    expect(changesIn(editing)[0]).not.toHaveProperty("revised");
    expect(tagOf(editing)).toMatchObject({ source: "edit-with-ai", requested: "edit" });
  });
});

describe("A second copy", { timeout: 30_000 }, () => {
  const RISKS = [
    "## Risks",
    "The supplier of the main part is three weeks late, and nobody has told us why.",
    "The courier who takes the parts to the plant has raised its prices twice this year.",
  ].join("\n\n");

  // protects=adding again a section the assistant wrote and that still waits is refused, the Mind keeps one, and the model is told to revise it with edit_mind, as it streams in and as it ends; fails_when=a second copy is written, or the model isn't told how to revise; no_claim=that a real model then revises.
  test.each([false, true])(
    "of its own waiting section is refused, streamed: %s",
    async (streamed) => {
      const model = scriptedModel((call): ScriptedReply => {
        if (call.results.some((each) => each.tool === "add_to_mind")) return { text: "Done." };
        const longer = asked(call).includes("write more about the risks");
        return {
          calls: [
            {
              tool: "add_to_mind",
              input: { content: longer ? `${RISKS}\n\nThe courier is late too.` : RISKS },
              streamInput: streamed,
            },
          ],
        };
      });
      const { core, mind } = await setUp(model);
      const client = await connectToMind(core, mind.id);
      const first = await askAssistant(core, { text: "add a risks section", mindId: mind.id });

      const second = await askAssistant(core, {
        text: "write more about the risks",
        mindId: mind.id,
        conversationId: first.conversationId,
      });

      await client.settled();
      expect(shownWithChanges(client).filter((line) => line.includes("Risks"))).toEqual([
        "+ Risks",
      ]);
      const [refused] = second.toolCalls.filter((call) => call.tool === "add_to_mind");
      expect(refused).toMatchObject({
        status: "failed",
        mindProblem: { kind: "copy", mindId: mind.id, text: "Risks" },
      });
      expect(lastResult(model)).toContain("Revise it with edit_mind");
    },
  );

  // protects=a second summary goes in beside the first though they share headings, short fixed lines, and "not covered" lines under different headings, as the summarise Skill writes them; fails_when=it is refused as a second copy; no_claim=none beyond protects.
  test("isn't a second summary that shares headings, short lines, and long lines under other headings with the first", async () => {
    const summary = (name: string, claim: string) =>
      [
        `## ${name}`,
        "### Key claims",
        `${name} claims that ${claim}, in every harbour it measured over two years.`,
        "### Methods",
        "The Passages found don't describe the methods.",
        `### ${name === "Paper A" ? "Limitations" : "Open questions"}`,
        "The Passages found don't say how the tide was measured or over how many harbours.",
        "Nor do they say which years the measurements cover, or who took them.",
        "Not reported.",
        "Not reported.",
      ].join("\n\n");
    const model = scriptedModel((call): ScriptedReply => {
      if (called(call, "add_to_mind")) return { text: "Done." };
      const content = asked(call).includes("Paper B")
        ? summary("Paper B", "the tide falls faster in winter")
        : summary("Paper A", "the tide rises faster at new moon");
      return { calls: [{ tool: "add_to_mind", input: { content } }] };
    });
    const { core, mind } = await setUp(model);
    const client = await connectToMind(core, mind.id);
    const first = await askAssistant(core, { text: "add a summary of Paper A", mindId: mind.id });

    const second = await askAssistant(core, {
      text: "add a summary of Paper B",
      mindId: mind.id,
      conversationId: first.conversationId,
    });

    await client.settled();
    expect(second.toolCalls.find((call) => call.tool === "add_to_mind")).toMatchObject({
      status: "done",
    });
    const shown = shownWithChanges(client);
    expect(shown.filter((line) => line === "+ Key claims")).toHaveLength(2);
    expect(shown.filter((line) => line === "+ Not reported.")).toHaveLength(4);
  });
});
