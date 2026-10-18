import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import { askAssistant, changesIn, mindCalls } from "../helpers/assistant";
import { answerEnded, setUpWithDocuments, shownPassages } from "../helpers/citations";
import { connectToMind } from "../helpers/mindClient";
import { note, readMind, writeMind } from "../helpers/minds";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

/**
 * A request to write something goes into a Mind, and "make it a document"
 * after an Answer puts that Answer in a new one as it is (#1109), through the
 * core's public interface as the assistant asks: the model is told where the
 * text goes, and the Answer's text, Citations and scope come over with no new
 * search. The model's choices are scripted; real models' are checked by hand.
 */

const LEASES = Array.from({ length: 7 }, (_, index) => ({
  name: `Lease ${index + 1}.md`,
  contents: `# Lease ${index + 1}\n\nThe rent for unit ${index + 1} is ${1000 + index * 100} euros a month.\n`,
}));
const QUOTES = LEASES.map(
  (_, index) => `The rent for unit ${index + 1} is ${1000 + index * 100} euros a month`,
);
const CITED = QUOTES.map((quote, index) => `${quote} [^${index + 1}].`).join("\n\n");

const NEW_MIND_LINE = "write it into a new Mind with create_mind";
const AS_A_DOCUMENT = "This request asks for your last Answer as a document";

/** The instructions of the request that offers the Minds' Tools. */
const withMindTools = (systems: readonly string[]) =>
  systems.find((system) => system.includes("read_mind")) ?? "";

const called = (call: ModelCall, tool: string) => call.results.some((each) => each.tool === tool);

/** Searches for every lease, then cites each one's rent, then `next`. */
function citingAll(call: ModelCall, next: () => ScriptedReply): ScriptedReply {
  if (!called(call, "search_documents")) {
    return {
      calls: LEASES.map((_, index) => ({
        tool: "search_documents",
        input: { query: `rent for unit ${index + 1}` },
      })),
    };
  }
  if (!called(call, "cite")) {
    const shown = call.results
      .filter((each) => each.tool === "search_documents")
      .flatMap((each) => shownPassages(each.text));
    const citations = QUOTES.map((quote, index) => ({
      marker: index + 1,
      passage: shown.find((passage) => passage.text.includes(quote))?.id ?? "",
      quote,
    }));
    return { calls: [{ tool: "cite", input: { citations } }] };
  }
  return next();
}

function checks(doc: ProseMirrorNode): string[] {
  const found: string[] = [];
  doc.descendants((node) => {
    if (node.type.name === "citation") found.push(String(node.attrs.check));
    return true;
  });
  return found;
}

async function madeBy(core: Core, answer: Awaited<ReturnType<typeof askAssistant>>) {
  const made = changesIn(answer).find((change) => change.created);
  if (!made) throw new Error("The Answer made no Mind.");
  const client = await connectToMind(core, made.mindId);
  await client.settled();
  return { mindId: made.mindId, doc: readMind(client) };
}

describe("A request to write something (#1109)", { timeout: 60_000 }, () => {
  // protects=the owner's "根据我现有的文件帮我写一份summary", and its English, tell the model to write into a new Mind, and the Mind made holds every Document's Citation, checked; fails_when=the model is told nothing, or the Mind made leaves Documents or Citations out; no_claim=that a real model writes it there.
  test.each(["根据我现有的文件帮我写一份summary", "Based on my files, write a summary"])(
    "“%s” goes into a new Mind, every Document cited",
    async (request) => {
      const systems: string[] = [];
      const model = scriptedModel((call) => {
        systems.push(call.system);
        return citingAll(call, () =>
          called(call, "create_mind")
            ? { text: "Done." }
            : { calls: [{ tool: "create_mind", input: { title: "Summary", content: CITED } }] },
        );
      });
      const { core, mind, client } = await setUpWithDocuments(model, LEASES);
      writeMind(client, [note("Notes on the leases.")]);
      await client.settled();

      const answer = await askAssistant(core, { text: request, mindId: mind.id });

      expect(withMindTools(systems)).toContain(NEW_MIND_LINE);
      const made = await madeBy(core, answer);
      expect(checks(made.doc)).toEqual(QUOTES.map(() => "found"));
    },
  );

  // protects=asked for a piece of its own, text added to the User's Mind is refused once with how to make the new Mind, the User's Mind is left as it was, and the new Mind is made (#1109); fails_when=the piece lands in the User's Mind, or the model isn't told to use create_mind; no_claim=that a real model then does.
  test("added to the User's Mind instead, it is refused once and made a new Mind", async () => {
    const told: string[] = [];
    const model = scriptedModel((call) =>
      citingAll(call, () => {
        if (!called(call, "add_to_mind")) {
          return { calls: [{ tool: "add_to_mind", input: { content: CITED } }] };
        }
        if (!called(call, "create_mind")) {
          told.push(call.results.find((each) => each.tool === "add_to_mind")?.text ?? "");
          return { calls: [{ tool: "create_mind", input: { title: "Summary", content: CITED } }] };
        }
        return { text: "Done." };
      }),
    );
    const { core, mind, client } = await setUpWithDocuments(model, LEASES);
    writeMind(client, [note("Notes on the leases.")]);
    await client.settled();

    const answer = await askAssistant(core, {
      text: "根据我现有的文件帮我写一份summary",
      mindId: mind.id,
    });

    expect(told[0]).toContain("the User asked for a new Mind for this");
    expect(told[0]).toContain("Call create_mind");
    expect(mindCalls(answer)[0]?.mindProblem).toEqual({ kind: "new-mind", mindId: mind.id });
    await client.settled();
    expect(readMind(client).textContent).toBe("Notes on the leases.");
    const made = await madeBy(core, answer);
    expect(checks(made.doc)).toEqual(QUOTES.map(() => "found"));
  });

  // protects=in an empty Mind, a request to write fills it rather than making another; fails_when=the model is told to make a new Mind; no_claim=none beyond protects.
  test("in an empty Mind, it fills that Mind", async () => {
    const systems: string[] = [];
    const model = scriptedModel((call) => {
      systems.push(call.system);
      return { text: "Done." };
    });
    const { core } = await setUpWithDocuments(model, LEASES.slice(0, 1));
    const empty = await core.createMind({ title: "" });

    await askAssistant(core, { text: "根据我现有的文件帮我写一份summary", mindId: empty.id });

    expect(withMindTools(systems)).toContain("write it there with add_to_mind");
    expect(withMindTools(systems)).not.toContain(NEW_MIND_LINE);
  });
});

describe("“Make it a document” after an Answer (#1109)", { timeout: 60_000 }, () => {
  // protects=after a panel Answer citing all seven Documents, asking for it as a document, in the owner's words or in English, makes a Mind of that Answer as it is: the same Citations, checked, with no new search, in the Folder the Answer searched; fails_when=the model searches again, the Mind covers fewer Documents, its Citations are lost, or it lands in another Folder; no_claim=that a real model gives answer: true.
  test.each(["把这个做成文档", "Make it a document"])(
    "“%s” makes a Mind of the Answer, every Citation kept, nothing searched again",
    async (request) => {
      const systems: string[] = [];
      let second: ModelCall[] = [];
      const model = scriptedModel((call) => {
        if (call.prompt.includes(request)) {
          second = [...second, call];
          systems.push(call.system);
          return called(call, "create_mind")
            ? { text: "Done." }
            : { calls: [{ tool: "create_mind", input: { title: "Leases", answer: true } }] };
        }
        return citingAll(call, () => ({ text: CITED }));
      });
      const { core, mind, client, documents } = await setUpWithDocuments(model, LEASES);
      writeMind(client, [note("Notes on the leases.")]);
      await client.settled();
      const folder = await core.createLibraryGroup({ name: "Leases", description: "" });
      await core.moveToFolder({
        documentIds: documents.map((document) => document.id),
        folderId: folder.id,
      });
      const scope = { folderIds: [folder.id], tagIds: [], documentIds: [] };

      const first = await askAssistant(core, {
        text: "What are the rents?",
        mindId: mind.id,
        scope,
      });
      expect(first.citations).toHaveLength(7);
      const answer = await askAssistant(core, {
        text: request,
        mindId: mind.id,
        conversationId: first.conversationId,
      });

      expect(withMindTools(systems)).toContain(AS_A_DOCUMENT);
      expect(withMindTools(systems)).toContain(
        "call create_mind with a short title and answer: true",
      );
      expect(second.flatMap((call) => call.results.map((each) => each.tool))).not.toContain(
        "search_documents",
      );
      expect(mindCalls(answer)[0]?.input).toEqual({ title: "Leases", answer: true });
      const made = await madeBy(core, answer);
      expect(checks(made.doc)).toEqual(QUOTES.map(() => "found"));
      const minds = await core.listMinds();
      expect(minds.find((each) => each.id === made.mindId)?.folderId).toBe(folder.id);
    },
  );
});

describe("“Make it a document” after an Answer with no words (#1133 review)", {
  timeout: 60_000,
}, () => {
  // protects=after an Answer that failed, or came back empty, "make it a document" isn't read as putting that Answer in a Mind, and no Mind holding only its title is made; fails_when=the model is told to call create_mind with answer: true, or a title-only Mind appears; no_claim=what a real model then says.
  test.each([
    ["after a failed Answer", { error: { status: 500, message: "The server failed." } }],
    ["after an empty Answer", { text: "" }],
  ] as const)("%s, nothing is put in a Mind", async (_, first) => {
    const systems: string[] = [];
    const model = scriptedModel((call) => {
      if (!call.prompt.includes("Make it a document")) return first;
      systems.push(call.system);
      if (call.system.includes(AS_A_DOCUMENT) && !called(call, "create_mind")) {
        return { calls: [{ tool: "create_mind", input: { title: "Leases", answer: true } }] };
      }
      return { text: "There's no Answer to put in a document yet." };
    });
    const { core, mind } = await setUpWithDocuments(model, LEASES.slice(0, 1));
    const before = (await core.listMinds()).length;

    const result = await core.askInConversation({ text: "What are the rents?", mindId: mind.id });
    if (!result.asked) throw new Error("Not asked.");
    await answerEnded(core, result.answerId);
    await askAssistant(core, {
      text: "Make it a document",
      mindId: mind.id,
      conversationId: result.conversation.id,
    });

    expect(withMindTools(systems)).not.toContain(AS_A_DOCUMENT);
    expect(await core.listMinds()).toHaveLength(before);
  });
});
