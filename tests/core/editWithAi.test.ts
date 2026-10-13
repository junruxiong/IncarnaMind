import type { JSONContent } from "@tiptap/core";
import { APICallError } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  type ConsentRequest,
  type Core,
  type Document,
  type EditWithAiInput,
  MIND_PROPOSAL_REQUESTS_FIELD,
  PROPOSAL_ATTRIBUTE,
} from "../../src/core";
import {
  aiUseOfAccepted,
  outOfDateProposals,
  pendingProposals,
  proposalRequestOf,
  resolveProposals,
} from "../../src/shared/proposals";
import { setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { readMind, resolveChanges, shownWithChanges, writeMind } from "../helpers/minds";
import { scriptedModels } from "../helpers/models";

const TIDES = {
  name: "Tides.md",
  contents: [
    "# Tides",
    "",
    "Spring tides happen at new moon and at full moon, when the Sun and the Moon pull together.",
    "",
    "Harbours publish the times of high water every year.",
  ].join("\n"),
};

const USAGE = {
  inputTokens: { total: 5, noCache: 5, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 5, text: 5, reasoning: undefined },
};

/** What an Edit with AI request asked: the passage, the action, the whole prompt and the instructions. */
interface EditCall {
  passage: string;
  action: string;
  prompt: string;
  system: string;
}

type Prompt = MockLanguageModelV4["doGenerateCalls"][number]["prompt"];

const textOf = (prompt: Prompt, role: "user" | "system") =>
  prompt
    .filter((message) => message.role === role)
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");

/** A model that rewrites each passage with `edit`; automatic tagging gets no Tags. */
function editingModel(
  edit: (call: EditCall) => string | Promise<string>,
  calls: EditCall[] = [],
): MockLanguageModelV4 {
  return new MockLanguageModelV4({
    doGenerate: async ({ prompt, responseFormat }) => {
      if (responseFormat?.type === "json") {
        return {
          content: [{ type: "text", text: JSON.stringify({ tags: [] }) }],
          finishReason: { unified: "stop", raw: undefined },
          usage: USAGE,
          warnings: [],
        };
      }
      const text = textOf(prompt, "user");
      const call = {
        passage: /<passage>\n([\s\S]*?)\n<\/passage>/.exec(text)?.[1] ?? "",
        action: /<task action="([^"]+)">/.exec(text)?.[1] ?? "",
        prompt: text,
        system: textOf(prompt, "system"),
      };
      calls.push(call);
      return {
        content: [{ type: "text", text: await edit(call) }],
        finishReason: { unified: "stop", raw: undefined },
        usage: USAGE,
        warnings: [],
      };
    },
  });
}

const text = (value: string, ...marks: string[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks: marks.map((type) => ({ type })) }),
});

let ids = 0;
const paragraph = (...content: JSONContent[]): JSONContent & { attrs: { id: string } } => ({
  type: "paragraph",
  attrs: { id: `block-${++ids}` },
  content,
});

/** A Citation of the Tides Document's spring-tides line, as a Note holds one copied from an Answer. */
function citation(tides: Document, check: "found" | "not-found" = "found"): JSONContent {
  return {
    type: "citation",
    attrs: {
      passageId: "passage-1",
      documentId: tides.id,
      documentName: "Tides",
      contentHash: tides.contentHash,
      pageFrom: null,
      pageTo: null,
      quote: "Spring tides happen at new moon and at full moon",
      check,
      checkReason: check === "found" ? null : "quote-not-on-pages",
    },
  };
}

/** The editor's reading of a paragraph's text (an inline node as U+FFFC), for `EditWithAiInput.text`. */
function plainOf(client: MindClient, blockId: string): string {
  let found = "";
  readMind(client).descendants((node) => {
    if (node.attrs.id !== blockId) return true;
    found = node.textBetween(0, node.content.size, "", "￼");
    return false;
  });
  return found;
}

/** Edit with AI over the whole of one paragraph. */
function wholeParagraph(
  client: MindClient,
  mindId: string,
  blockId: string,
  action: EditWithAiInput["action"],
  instruction?: string,
): EditWithAiInput {
  const plain = plainOf(client, blockId);
  return {
    requestId: `edit-${++ids}`,
    mindId,
    from: { blockId, offset: 0 },
    to: { blockId, offset: plain.length },
    text: plain,
    action,
    ...(instruction !== undefined && { instruction }),
  };
}

async function setUp(model: MockLanguageModelV4) {
  const set = await setUpWithDocuments(model, [TIDES]);
  return { ...set, tides: set.documents[0] as Document };
}

async function exported(core: Core, mindId: string): Promise<string> {
  const file = await core.exportMind(mindId, { format: "markdown" });
  return new TextDecoder().decode(file.data);
}

describe("Edit with AI", () => {
  test("a request comes back as Proposals, one per place that changed, and nothing changes until they are accepted", async () => {
    const calls: EditCall[] = [];
    const { core, mind, client } = await setUp(
      editingModel(() => "The tides here are strong and regular.", calls),
    );
    const note = paragraph(text("The tides here are very strong and really regular."));
    writeMind(client, [note]);
    await client.settled();

    const result = await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "shorten"));

    expect(result).toMatchObject({ kind: "proposed", proposals: 2, droppedCitations: 0 });
    expect(shownWithChanges(client)).toEqual([
      "The tides here are [-very -]strong and [-really -]regular.",
    ]);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.action).toBe("shorten");
    expect(calls[0]?.passage).toBe("The tides here are very strong and really regular.");
    // Pending Proposals aren't the Mind's text yet: exports read it as it was.
    expect(await exported(core, mind.id)).toContain(
      "The tides here are very strong and really regular.",
    );
    // The Mind says which model made them.
    const request = result.kind === "proposed" ? result.request : "";
    expect(proposalRequestOf(client.doc, request)).toMatchObject({
      source: "edit-with-ai",
      modelId: "local-model",
    });

    // Accept one, reject the other.
    await resolveChanges(client, "accept", (_id, index) => index === 0);
    expect(shownWithChanges(client)).toEqual(["The tides here are strong and [-really -]regular."]);
    await resolveChanges(client, "reject");
    expect(shownWithChanges(client)).toEqual(["The tides here are strong and really regular."]);
    expect(await exported(core, mind.id)).toContain(
      "The tides here are strong and really regular.",
    );
  });

  test("words added come after those they replace, with the marks of the words they replace", async () => {
    const { core, mind, client } = await setUp(
      editingModel(() => "Spring tides come twice each month."),
    );
    const note = paragraph(text("Spring tides "), text("happen twice a month", "bold"), text("."));
    writeMind(client, [note]);
    await client.settled();

    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));

    expect(shownWithChanges(client)).toEqual([
      "Spring tides [-happen-]{+come+} twice [-a-]{+each+} month.",
    ]);
    const added = readMind(client).firstChild?.content.content.filter((node) =>
      node.marks.some((mark) => mark.type.name === "proposalInsert"),
    );
    expect(added?.every((node) => node.marks.some((mark) => mark.type.name === "bold"))).toBe(true);
    // Until accepted, what is read of the Mind (Question context, exports) is the text as it was.
    const before = await exported(core, mind.id);
    expect(before).toContain("Spring tides **happen twice a month**.");
    expect(before).not.toContain("come");
    await resolveChanges(client, "accept");
    expect(await exported(core, mind.id)).toContain("Spring tides **come twice each month**.");
    expect(shownWithChanges(client)).toEqual(["Spring tides come twice each month."]);
    // The words the changes added are AI text; those they kept are as they were.
    expect(readMind(client).firstChild?.toJSON().content).toEqual([
      text("Spring tides "),
      text("come", "generated", "bold"),
      text(" twice ", "bold"),
      text("each", "generated", "bold"),
      text(" month", "bold"),
      text("."),
    ]);
  });

  test("a selection inside a paragraph changes only what was selected, and the model sees the words around it", async () => {
    const calls: EditCall[] = [];
    const { core, mind, client } = await setUp(editingModel(() => "Moon", calls));
    const long =
      "Long before anyone kept tide tables, sailors watched how the water rose and fell. ";
    const note = paragraph(text(`${long}The moon pulls the sea, and the moon sets the tides.`));
    writeMind(client, [note]);
    await client.settled();
    const plain = plainOf(client, note.attrs.id);
    const from = plain.lastIndexOf("moon");

    await core.editWithAi({
      requestId: "edit-capital",
      mindId: mind.id,
      from: { blockId: note.attrs.id, offset: from },
      to: { blockId: note.attrs.id, offset: from + 4 },
      text: "moon",
      action: "instruction",
      instruction: "Capitalise it",
    });

    expect(shownWithChanges(client)).toEqual([
      `${long}The moon pulls the sea, and the [-moon-]{+Moon+} sets the tides.`,
    ]);
    // About 80 characters before it, no word cut in two, and what follows it.
    expect(calls[0]?.prompt).toContain(
      "<before>\nsailors watched how the water rose and fell. The moon pulls the sea, and the\n</before>",
    );
    expect(calls[0]?.prompt).toContain("<after>\nsets the tides.\n</after>");
  });

  test("a Citation the new text keeps stays, and is checked again; one it drops is shown for review", async () => {
    const { core, mind, client, tides } = await setUp(
      editingModel(({ passage }) =>
        passage.includes("[2]") ? "Spring tides come at new and full moon [1]." : passage,
      ),
    );
    // Two Citations, the first said not found when it was made: it is found now.
    const note = paragraph(
      text("Spring tides happen at new moon and at full moon "),
      citation(tides, "not-found"),
      text(", and they are the highest "),
      citation(tides),
      text("."),
    );
    writeMind(client, [note]);
    await client.settled();

    const result = await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "shorten"));

    expect(result).toMatchObject({ kind: "proposed", droppedCitations: 1 });
    expect(shownWithChanges(client)).toEqual([
      "Spring tides [-happen-]{+come+} at new [-moon and at -]{+and +}full moon [-⟨not-found⟩, and they are the highest ⟨found⟩-]{+⟨found⟩+}.",
    ]);

    await resolveChanges(client, "accept");
    expect(shownWithChanges(client)).toEqual(["Spring tides come at new and full moon ⟨found⟩."]);
    const kept: Record<string, unknown>[] = [];
    readMind(client).descendants((node) => {
      if (node.type.name === "citation") kept.push(node.attrs);
    });
    expect(kept).toEqual([
      expect.objectContaining({
        documentId: tides.id,
        contentHash: tides.contentHash,
        quote: "Spring tides happen at new moon and at full moon",
        check: "found",
        [PROPOSAL_ATTRIBUTE]: null,
      }),
    ]);
  });

  test("the model sees Citations as numbered markers, formulas as written, and the paragraphs around the selection", async () => {
    const calls: EditCall[] = [];
    const { core, mind, client, tides } = await setUp(
      editingModel(({ passage }) => passage, calls),
    );
    const note = paragraph(
      text("The force is "),
      { type: "inlineMath", attrs: { latex: "F = ma" } },
      text(" for spring tides "),
      citation(tides),
      text("."),
    );
    writeMind(client, [note]);
    await client.settled();

    const result = await core.editWithAi(
      wholeParagraph(client, mind.id, note.attrs.id, "translate", "简体中文"),
    );

    expect(result).toEqual({ kind: "unchanged" });
    expect(calls[0]?.passage).toBe("The force is $F = ma$ for spring tides [1].");
    expect(calls[0]?.prompt).toContain("<context>\nThe force is $F = ma$ for spring tides [·].");
    expect(calls[0]?.prompt).toContain("Translate it into 简体中文");
    expect(calls[0]?.system).toContain("Keep every Citation marker");
    expect(shownWithChanges(client)).toEqual(["The force is ￼ for spring tides ⟨found⟩."]);
  });

  test("a footnote the User wrote stays where the model keeps its marker, moves with its text, or is struck when left out", async () => {
    const calls: EditCall[] = [];
    let reply = "";
    const { core, mind, client } = await setUp(editingModel(() => reply, calls));
    const footnote = (id: string, words: string): JSONContent => ({
      type: "footnote",
      attrs: { id },
      content: [text(words)],
    });
    const note = paragraph(
      text("Neap tides"),
      footnote("fn-before", "At the quarter moons."),
      text(" are weak. Spring tides are very strong"),
      footnote("fn-in", "Twice a month."),
      text(" and regular."),
    );
    writeMind(client, [note]);
    await client.settled();
    // From "Spring" to the end: the footnote before it counts as one character, its text none.
    const from = "Neap tides￼ are weak. ".length;
    const selected = "Spring tides are very strong￼ and regular.";
    const edit = async (requestId: string, rewrite: string) => {
      reply = rewrite;
      return core.editWithAi({
        requestId,
        mindId: mind.id,
        from: { blockId: note.attrs.id, offset: from },
        to: { blockId: note.attrs.id, offset: from + selected.length },
        text: selected,
        action: "shorten",
      });
    };
    const footnotes = () => {
      const found: { id: unknown; text: string; proposal: unknown }[] = [];
      readMind(client).descendants((node) => {
        if (node.type.name !== "footnote") return true;
        const kind = (node.attrs[PROPOSAL_ATTRIBUTE] as { kind?: string } | null)?.kind ?? null;
        found.push({ id: node.attrs.id, text: node.textContent, proposal: kind });
        return false;
      });
      return found;
    };

    // Kept: only the words change around it.
    expect(await edit("edit-kept", "Spring tides are strong[^1] and regular.")).toMatchObject({
      kind: "proposed",
      proposals: 1,
    });
    expect(calls[0]?.passage).toBe("Spring tides are very strong[^1] and regular.");
    expect(calls[0]?.system).toContain("footnote marker, such as [^1]");
    expect(shownWithChanges(client)).toEqual([
      "Neap tides￼ are weak. Spring tides are [-very -]strong￼ and regular.",
    ]);
    await resolveChanges(client, "reject");

    // Moved: struck where it was, and copied with its text where the model put it.
    await edit("edit-moved", "Spring tides are very strong and regular[^1].");
    expect(shownWithChanges(client)).toEqual([
      "Neap tides￼ are weak. Spring tides are very strong[-￼-] and regular{+￼+}.",
    ]);
    const [, moved, copy] = footnotes();
    expect(moved).toEqual({ id: "fn-in", text: "Twice a month.", proposal: "delete" });
    expect(copy).toMatchObject({ text: "Twice a month.", proposal: "insert" });
    expect(copy?.id).not.toBe("fn-in");
    // A footnote's text was the User's: accepting its move writes nothing into the record of AI use.
    const accepted = await resolveChanges(client, "accept");
    expect(accepted.map((each) => each.characters)).toEqual([0, 0]);
    expect(shownWithChanges(client)).toEqual([
      "Neap tides￼ are weak. Spring tides are very strong and regular￼.",
    ]);
    expect(footnotes().map((each) => each.text)).toEqual([
      "At the quarter moons.",
      "Twice a month.",
    ]);
    expect(await exported(core, mind.id)).toContain("regular[^2].");

    // Left out: struck, for the User to review.
    const after = "Spring tides are very strong and regular￼.";
    reply = "Spring tides are strong and regular.";
    await core.editWithAi({
      requestId: "edit-dropped",
      mindId: mind.id,
      from: { blockId: note.attrs.id, offset: from },
      to: { blockId: note.attrs.id, offset: from + after.length },
      text: after,
      action: "shorten",
    });
    expect(shownWithChanges(client)).toEqual([
      "Neap tides￼ are weak. Spring tides are [-very -]strong and regular[-￼-].",
    ]);
  });

  test("Chinese is compared character by character", async () => {
    const { core, mind, client } = await setUp(editingModel(() => "潮水每天很规律地涨落两次。"));
    const note = paragraph(text("潮水每天非常规律地涨落两次。"));
    writeMind(client, [note]);
    await client.settled();

    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));

    expect(shownWithChanges(client)).toEqual(["潮水每天[-非常-]{+很+}规律地涨落两次。"]);
  });

  test("Explain adds a new paragraph under the selection, as a Proposal of its own", async () => {
    const { core, mind, client } = await setUp(
      editingModel(({ action }) =>
        action === "explain" ? "It says the Moon lifts the sea twice a day [1]." : "",
      ),
    );
    const first = paragraph(text("The Moon raises two bulges of water."));
    const second = paragraph(text("A closing line."));
    writeMind(client, [first, second]);
    await client.settled();

    const result = await core.editWithAi(
      wholeParagraph(client, mind.id, first.attrs.id, "explain"),
    );

    expect(result).toMatchObject({ kind: "proposed", proposals: 1 });
    expect(shownWithChanges(client)).toEqual([
      "The Moon raises two bulges of water.",
      "+ It says the Moon lifts the sea twice a day.",
      "A closing line.",
    ]);
    expect(await exported(core, mind.id)).not.toContain("It says");
    await resolveChanges(client, "reject");
    expect(shownWithChanges(client)).toEqual([
      "The Moon raises two bulges of water.",
      "A closing line.",
    ]);
  });

  test("paragraphs the new text drops are Proposals on their whole Blocks, and those it adds are new Blocks", async () => {
    const { core, mind, client } = await setUp(
      editingModel(({ passage }) =>
        passage.includes("Second") ? "First and second, together." : "One line.\n\nAnother line.",
      ),
    );
    const first = paragraph(text("First paragraph."));
    const second = paragraph(text("Second paragraph."));
    const third = paragraph(text("Single line."));
    writeMind(client, [first, second, third]);
    await client.settled();

    await core.editWithAi({
      requestId: "edit-join",
      mindId: mind.id,
      from: { blockId: first.attrs.id, offset: 0 },
      to: { blockId: second.attrs.id, offset: "Second paragraph.".length },
      text: "First paragraph.\nSecond paragraph.",
      action: "shorten",
    });
    await core.editWithAi(wholeParagraph(client, mind.id, third.attrs.id, "instruction", "Split"));

    expect(shownWithChanges(client)).toEqual([
      "First [-paragraph-]{+and second, together+}.",
      "- Second paragraph.",
      "[-Single-]{+One+} line.",
      "+ Another line.",
    ]);
    await resolveChanges(client, "accept");
    expect(shownWithChanges(client)).toEqual([
      "First and second, together.",
      "One line.",
      "Another line.",
    ]);
  });

  test("an empty paragraph in the selection stays out of the pairing: each paragraph gets its own new text", async () => {
    const calls: EditCall[] = [];
    const { core, mind, client } = await setUp(
      editingModel(() => "First line.\n\nThird line.", calls),
    );
    const first = paragraph(text("First paragraph."));
    const empty = paragraph();
    const third = paragraph(text("Third paragraph."));
    writeMind(client, [first, empty, third]);
    await client.settled();

    const result = await core.editWithAi({
      requestId: "edit-across-a-blank-line",
      mindId: mind.id,
      from: { blockId: first.attrs.id, offset: 0 },
      to: { blockId: third.attrs.id, offset: "Third paragraph.".length },
      text: "First paragraph.\n\nThird paragraph.",
      action: "shorten",
    });

    expect(calls[0]?.passage).toBe("First paragraph.\n\nThird paragraph.");
    expect(result).toMatchObject({ kind: "proposed", proposals: 2 });
    expect(shownWithChanges(client)).toEqual([
      "First [-paragraph-]{+line+}.",
      "",
      "Third [-paragraph-]{+line+}.",
    ]);
  });

  test("nothing is proposed over text that is gone or holds Proposals, nor when the model gives it back as it was", async () => {
    let release = () => {};
    const waiting = new Promise<void>((resolve) => {
      release = resolve;
    });
    let calls = 0;
    const { core, mind, client } = await setUp(
      editingModel(async ({ passage }) => {
        calls++;
        if (passage.startsWith("Slow")) await waiting;
        return passage.startsWith("Same") ? passage : `${passage} Indeed.`;
      }),
    );
    const slow = paragraph(text("Slow paragraph."));
    const same = paragraph(text("Same paragraph."));
    writeMind(client, [slow, same]);
    await client.settled();

    // What the editor read isn't what the Mind holds.
    expect(
      await core.editWithAi({
        ...wholeParagraph(client, mind.id, slow.attrs.id, "improve"),
        text: "Other",
      }),
    ).toEqual({ kind: "changed" });
    expect(calls).toBe(0);

    // The User deletes the paragraph while the model works: there is nothing to change.
    const gone = paragraph(text("Slow and gone."));
    writeMind(client, [slow, gone, same]);
    await client.settled();
    const editing = core.editWithAi(wholeParagraph(client, mind.id, gone.attrs.id, "improve"));
    await expect.poll(() => calls).toBe(1);
    client.blocks.delete(1, 1);
    await client.settled();
    release();
    expect(await editing).toEqual({ kind: "changed" });
    expect(shownWithChanges(client)).toEqual(["Slow paragraph.", "Same paragraph."]);
    expect(pendingProposals(client.blocks)).toEqual([]);

    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, same.attrs.id, "improve")),
    ).toEqual({ kind: "unchanged" });

    // Proposals waiting in the selection: accept or reject them first.
    await core.editWithAi(wholeParagraph(client, mind.id, slow.attrs.id, "improve"));
    expect(shownWithChanges(client)[0]).toBe("Slow paragraph.{+ Indeed.+}");
    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, slow.attrs.id, "improve")),
    ).toEqual({ kind: "has-proposals" });
  });

  test("a request can be stopped, and one still waiting when the app quits writes nothing", async () => {
    let asked = () => {};
    const hanging = new MockLanguageModelV4({
      doGenerate: async ({ prompt, abortSignal }) => {
        if (textOf(prompt, "user").includes("<passage>")) asked();
        return new Promise((_resolve, reject) => {
          abortSignal?.addEventListener("abort", () =>
            reject(new DOMException("The request was aborted.", "AbortError")),
          );
        });
      },
    });
    const { core, mind, client } = await setUp(hanging);
    const note = paragraph(text("A paragraph to improve."));
    writeMind(client, [note]);
    await client.settled();

    const input = wholeParagraph(client, mind.id, note.attrs.id, "improve");
    const waiting = new Promise<void>((resolve) => {
      asked = resolve;
    });
    const editing = core.editWithAi(input);
    await waiting;
    await core.cancelEdit(input.requestId);
    expect(await editing).toEqual({ kind: "cancelled" });
    expect(shownWithChanges(client)).toEqual(["A paragraph to improve."]);

    // Quitting halfway: the request stops, and nothing is left half written.
    const again = new Promise<void>((resolve) => {
      asked = resolve;
    });
    const quitting = core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    await again;
    core.close();
    expect(await quitting).toEqual({ kind: "cancelled" });
    expect(pendingProposals(client.blocks)).toEqual([]);
  });

  test("a model that fails says why, and nothing is written", async () => {
    const { core, mind, client } = await setUp(
      new MockLanguageModelV4({
        doGenerate: async () => {
          throw new APICallError({
            message: "Too many requests",
            url: "http://127.0.0.1:11434/v1/chat/completions",
            requestBodyValues: {},
            statusCode: 429,
            isRetryable: false,
          });
        },
      }),
    );
    const note = paragraph(text("A paragraph to improve."));
    writeMind(client, [note]);
    await client.settled();

    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve")),
    ).toEqual({ kind: "failed", error: expect.objectContaining({ kind: "rate-limit" }) });
    expect(shownWithChanges(client)).toEqual(["A paragraph to improve."]);
  });

  test("Proposals survive sync to another client and a restart, and resolve the same there", async () => {
    const dataDir = await createTempDataFolder();
    const model = editingModel(() => "The tides are strong and regular.");
    const models = scriptedModels(model);
    let core = startCore(dataDir, { createChatModel: models.createChatModel });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const mind = await core.createMind({ title: "Tides" });
    const first = await connectToMind(core, mind.id);
    const note = paragraph(text("The tides are very strong and really regular."));
    const extra = paragraph(text("Moon."));
    writeMind(first, [note, extra]);
    await first.settled();
    await core.editWithAi(wholeParagraph(first, mind.id, note.attrs.id, "shorten"));
    // A whole new Block too.
    model.doGenerate = async () => ({
      content: [{ type: "text", text: "The Moon pulls the sea." }],
      finishReason: { unified: "stop", raw: undefined },
      usage: USAGE,
      warnings: [],
    });
    await core.editWithAi(wholeParagraph(first, mind.id, extra.attrs.id, "explain"));
    const pending = shownWithChanges(first);
    expect(pending).toEqual([
      "The tides are [-very -]strong and [-really -]regular.",
      "Moon.",
      "+ The Moon pulls the sea.",
    ]);

    // Another client has them as they are.
    const second = await connectToMind(core, mind.id);
    expect(shownWithChanges(second)).toEqual(pending);

    // So does the Mind after a restart, from what was stored.
    core.close();
    core = startCore(dataDir, { createChatModel: models.createChatModel });
    const reloaded = await connectToMind(core, mind.id);
    expect(shownWithChanges(reloaded)).toEqual(pending);
    expect(pendingProposals(reloaded.blocks)).toHaveLength(3);
    expect(reloaded.doc.getMap(MIND_PROPOSAL_REQUESTS_FIELD).size).toBe(2);

    // Accepting one, rejecting one and accepting the Block gives the same text everywhere.
    const ids = pendingProposals(reloaded.blocks).map((proposal) => proposal.id);
    const apply = (doc: Y.Doc, blocks: Y.XmlFragment) =>
      doc.transact(() => {
        resolveProposals(blocks, (proposal) => proposal.id === ids[0], "accept");
        resolveProposals(blocks, (proposal) => proposal.id === ids[1], "reject");
        resolveProposals(blocks, (proposal) => proposal.id === ids[2], "accept");
      });
    const copy = new Y.Doc();
    Y.applyUpdate(copy, Y.encodeStateAsUpdate(reloaded.doc));
    apply(reloaded.doc, reloaded.blocks);
    await reloaded.settled();
    apply(copy, copy.getXmlFragment("blocks"));
    const expected = [
      "The tides are strong and really regular.",
      "Moon.",
      "The Moon pulls the sea.",
    ];
    expect(shownWithChanges(reloaded)).toEqual(expected);
    const again = await connectToMind(core, mind.id);
    expect(shownWithChanges(again)).toEqual(expected);
    expect(
      copy
        .getXmlFragment("blocks")
        .toArray()
        .map((block) =>
          (block as Y.XmlElement)
            .toArray()
            .map((child) => child.toString())
            .join(""),
        ),
    ).toEqual([
      "The tides are strong and really regular.",
      "Moon.",
      // Stored as AI text, as the editor's binding keeps a mark.
      "<generated>The Moon pulls the sea.</generated>",
    ]);
  });

  test("an accepted change goes into the record of AI use with the model that wrote it; a rejected one, or one that only removes words, doesn't", async () => {
    const { core, mind, client } = await setUp(
      editingModel(() => "The tides are strong and quite regular."),
    );
    const note = paragraph(text("The tides are very strong and regular."));
    writeMind(client, [note]);
    await client.settled();
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    expect(shownWithChanges(client)).toEqual([
      "The tides are [-very -]strong and {+quite +}regular.",
    ]);

    // As the editor does: accept, send the change, then record what was accepted.
    const [removing, adding] = pendingProposals(client.blocks).map((proposal) => proposal.id);
    const accepted = await resolveChanges(client, "accept", (id) => id === adding);
    await resolveChanges(client, "reject", (id) => id === removing);
    const entries = aiUseOfAccepted(client.doc, mind.id, accepted, "2026-10-11T09:00:00.000Z");
    await core.recordAiUse(entries);

    expect(await core.listAiUse(mind.id)).toEqual([
      expect.objectContaining({
        kind: "edit-with-ai-accepted",
        outcome: "kept",
        providerId: expect.any(String),
        modelId: "local-model",
        characters: "quite ".length,
        blockIds: [note.attrs.id],
      }),
    ]);
    // Removing words writes nothing of the AI's.
    const removed = { id: removing ?? "", request: adding ?? "", characters: 0, blockIds: [] };
    expect(aiUseOfAccepted(client.doc, mind.id, [removed], "")).toEqual([]);
  });

  test("an accept is one step of Undo, which brings the Proposal back", async () => {
    const { core, mind, client } = await setUp(
      editingModel(() => "The tides are strong and regular."),
    );
    const note = paragraph(text("The tides are very strong and really regular."));
    writeMind(client, [note]);
    await client.settled();
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "shorten"));
    const undo = new Y.UndoManager(client.blocks, { trackedOrigins: new Set(["editor"]) });

    await resolveChanges(client, "accept");
    expect(shownWithChanges(client)).toEqual(["The tides are strong and regular."]);
    undo.undo();
    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "The tides are [-very -]strong and [-really -]regular.",
    ]);
    expect(pendingProposals(client.blocks)).toHaveLength(2);
    undo.redo();
    expect(shownWithChanges(client)).toEqual(["The tides are strong and regular."]);
  });
});

/** The Mind's paragraphs and headings with what kind of Block each is: "h2", "•" a bullet, "1." a numbered item, "p". */
function structure(client: MindClient): string[] {
  const lines: string[] = [];
  readMind(client).descendants((node, _position, parent) => {
    if (!node.isTextblock) return true;
    const list = parent?.type.name === "listItem" ? readMindParentOf(client, node) : null;
    const kind =
      node.type.name === "heading"
        ? `h${node.attrs.level}`
        : list === "bulletList"
          ? "•"
          : list === "orderedList"
            ? "1."
            : "p";
    let line = "";
    node.forEach((child) => {
      line += child.isText ? (child.text ?? "") : `⟨${child.attrs.check}⟩`;
    });
    lines.push(`${kind} ${line}`);
    return false;
  });
  return lines;
}

/** The kind of list a list item's paragraph is in. */
function readMindParentOf(
  client: MindClient,
  paragraph: ReturnType<typeof readMind>,
): string | null {
  let found: string | null = null;
  readMind(client).descendants((node) => {
    if (node.type.name !== "bulletList" && node.type.name !== "orderedList") return true;
    node.descendants((inner) => {
      if (inner === paragraph || inner.eq(paragraph)) found = node.type.name;
    });
    return false;
  });
  return found;
}

describe("Edit with AI over Blocks of several kinds", () => {
  const heading = (value: string): JSONContent => ({
    type: "heading",
    attrs: { id: `block-${++ids}`, level: 2 },
    content: [text(value)],
  });
  const list = (type: "bulletList" | "orderedList", ...items: JSONContent[][]): JSONContent => ({
    type,
    content: items.map((content) => ({ type: "listItem", content: [paragraph(...content)] })),
  });
  /** Findings with a cited bullet, decisions as numbered items, and a closing paragraph. */
  const mixed = (tides: Document): JSONContent[] => [
    heading("Findings"),
    list(
      "bulletList",
      [text("Spring tides are strongest at new moon. "), citation(tides)],
      [text("Neap tides are weakest.")],
    ),
    heading("Decisions"),
    list(
      "orderedList",
      [text("Publish the tide tables in March.")],
      [text("Check the harbour gauges weekly.")],
    ),
    paragraph(text("Harbours publish the times of high water every year.")),
  ];
  const whole = (client: MindClient, mindId: string): EditWithAiInput => {
    const blocks: { id: string; size: number }[] = [];
    readMind(client).descendants((node) => {
      if (!node.isTextblock) return true;
      blocks.push({ id: String(node.attrs.id), size: node.content.size });
      return false;
    });
    const first = blocks[0] as { id: string };
    const last = blocks.at(-1) as { id: string; size: number };
    return {
      requestId: `edit-${++ids}`,
      mindId,
      from: { blockId: first.id, offset: 0 },
      to: { blockId: last.id, offset: last.size },
      text: blocks.map((block) => plainOf(client, block.id)).join("\n"),
      action: "instruction",
      instruction: "Restructure so the decisions come first. Cut the last paragraph.",
    };
  };
  const BEFORE = [
    "h2 Findings",
    "• Spring tides are strongest at new moon. ⟨found⟩",
    "• Neap tides are weakest.",
    "h2 Decisions",
    "1. Publish the tide tables in March.",
    "1. Check the harbour gauges weekly.",
    "p Harbours publish the times of high water every year.",
  ];
  const RESTRUCTURED = [
    "h2 Decisions",
    "1. Publish the tide tables in March.",
    "1. Check the harbour gauges weekly.",
    "h2 Findings",
    "• Spring tides are strongest at new moon. ⟨found⟩",
    "• Neap tides are weakest.",
  ];

  test("the model reads each Block's kind by its mark, and a restructure keeps headings, lists and Citations; Accept all is one step of Undo (#703)", async () => {
    const calls: EditCall[] = [];
    const { core, mind, client, tides } = await setUp(
      editingModel(
        () =>
          [
            "## Decisions",
            "1. Publish the tide tables in March.",
            "1. Check the harbour gauges weekly.",
            "## Findings",
            "- Spring tides are strongest at new moon. [1]",
            "- Neap tides are weakest.",
          ].join("\n\n"),
        calls,
      ),
    );
    writeMind(client, mixed(tides));
    await client.settled();
    expect(structure(client)).toEqual(BEFORE);

    await core.editWithAi(whole(client, mind.id));

    expect(calls[0]?.passage).toBe(
      [
        "## Findings",
        "- Spring tides are strongest at new moon. [1]",
        "- Neap tides are weakest.",
        "## Decisions",
        "1. Publish the tide tables in March.",
        "1. Check the harbour gauges weekly.",
        "Harbours publish the times of high water every year.",
      ].join("\n\n"),
    );
    const undo = new Y.UndoManager(client.blocks, { trackedOrigins: new Set(["editor"]) });
    await resolveChanges(client, "accept");
    expect(structure(client)).toEqual(RESTRUCTURED);
    const markdown = await exported(core, mind.id);
    expect(markdown).toContain("## Decisions\n\n1. Publish the tide tables in March.");
    expect(markdown).toContain("## Findings\n\n- Spring tides are strongest at new moon.");

    undo.undo();
    await client.settled();
    expect(pendingProposals(client.blocks).length).toBeGreaterThan(0);
    await resolveChanges(client, "reject");
    expect(structure(client)).toEqual(BEFORE);
  });

  test("a reply without marks pairs paragraphs by their words, not their place: none is written into a heading, and moved ones keep their kind", async () => {
    const { core, mind, client, tides } = await setUp(
      editingModel(() =>
        [
          "Decisions",
          "Publish the tide tables in March.",
          "Check the harbour gauges weekly.",
          "Findings",
          "Spring tides are strongest at new moon. [1]",
          "Neap tides are weakest.",
        ].join("\n\n"),
      ),
    );
    writeMind(client, mixed(tides));
    await client.settled();

    await core.editWithAi(whole(client, mind.id));
    await resolveChanges(client, "accept");

    expect(structure(client)).toEqual(RESTRUCTURED);
  });

  test("a line that starts with a mark is a paragraph of its own, even after a single line break", async () => {
    const { core, mind, client, tides } = await setUp(
      editingModel(() =>
        [
          "## Decisions",
          "1. Publish the tide tables in March.",
          "1. Check the harbour gauges weekly.",
          "## Findings",
          "- Spring tides are strongest at new moon. [1]",
          "- Neap tides are weakest.",
        ].join("\n"),
      ),
    );
    writeMind(client, mixed(tides));
    await client.settled();

    await core.editWithAi(whole(client, mind.id));
    await resolveChanges(client, "accept");

    expect(structure(client)).toEqual(RESTRUCTURED);
  });

  test("a Block the model marks as another kind becomes that kind", async () => {
    const { core, mind, client } = await setUp(
      editingModel(() =>
        ["## Tides", "- Harbours publish the times of high water every year."].join("\n\n"),
      ),
    );
    writeMind(client, [
      heading("Tides"),
      paragraph(text("Harbours publish the times of high water every year.")),
    ]);
    await client.settled();

    await core.editWithAi(whole(client, mind.id));
    await resolveChanges(client, "accept");

    expect(structure(client)).toEqual([
      "h2 Tides",
      "• Harbours publish the times of high water every year.",
    ]);
  });
});

describe("Changes out of date", () => {
  test("a paragraph edited while the model works keeps the User's edits, and the change to words they edited is out of date until it is made again", async () => {
    let release = () => {};
    const waiting = new Promise<void>((resolve) => {
      release = resolve;
    });
    const calls: EditCall[] = [];
    const { core, mind, client } = await setUp(
      editingModel(async ({ passage }) => {
        if (passage === "really") return "truly";
        await waiting;
        return "The tides are strong and truly regular.";
      }, calls),
    );
    const note = paragraph(text("The tides are very strong and realy regular."));
    writeMind(client, [note]);
    await client.settled();

    // The model is paused; meanwhile the User fixes a typo in words it is about to replace.
    const editing = core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    await expect.poll(() => calls.length).toBe(1);
    const words = (client.blocks.get(0) as Y.XmlElement).get(0) as Y.XmlText;
    words.insert("The tides are very strong and rea".length, "l");
    await client.settled();
    release();

    expect(await editing).toMatchObject({ kind: "proposed", proposals: 2, outOfDate: 1 });
    // The fix stays; the change to the words fixed is out of date, the other isn't.
    expect(shownWithChanges(client)).toEqual([
      "The tides are [-very -]strong and [-really-]{+truly+} regular.",
    ]);
    const [fresh, stale] = pendingProposals(client.blocks).map((proposal) => proposal.id);
    expect([...outOfDateProposals(client.blocks)]).toEqual([stale]);
    // Neither counts until it is accepted: the Mind reads as the User has it.
    expect(await exported(core, mind.id)).toContain(
      "The tides are very strong and really regular.",
    );

    // Accepting all leaves the one out of date as it is: it can't be accepted.
    const accepted = await resolveChanges(client, "accept");
    expect(accepted.map((proposal) => proposal.id)).toEqual([fresh]);
    expect(shownWithChanges(client)).toEqual([
      "The tides are strong and [-really-]{+truly+} regular.",
    ]);

    // Regenerate makes it again, as it was asked, from the words as they are now.
    expect(
      await core.regenerateProposal({
        requestId: "again",
        mindId: mind.id,
        proposalId: stale ?? "",
      }),
    ).toMatchObject({ kind: "proposed", proposals: 1, outOfDate: 0 });
    expect(calls.at(-1)).toMatchObject({ passage: "really", action: "improve" });
    // A few words in a sentence: the model sees the words around them, to fit between and not repeat.
    expect(calls.at(-1)?.prompt).toContain(
      "<before>\nThe tides are strong and\n</before>\n\n<passage>\nreally\n</passage>\n\n<after>\nregular.\n</after>",
    );
    expect(calls.at(-1)?.system).toContain(
      "The new text must fit between the words before and after it. Never repeat those words in it.",
    );
    // A whole paragraph has no words around it.
    expect(calls[0]?.prompt).not.toContain("<before>");
    expect(calls[0]?.prompt).not.toContain("<after>");
    expect(shownWithChanges(client)).toEqual([
      "The tides are strong and [-really-]{+truly+} regular.",
    ]);
    expect(pendingProposals(client.blocks).map((proposal) => proposal.id)).not.toContain(stale);
    expect(outOfDateProposals(client.blocks).size).toBe(0);
    await resolveChanges(client, "accept");
    expect(shownWithChanges(client)).toEqual(["The tides are strong and truly regular."]);
  });

  test("a change out of date can be rejected; Regenerate keeps it when the model fails, and takes it away when there is nothing to change", async () => {
    let reply: (passage: string) => string = () => "The tides are strong.";
    const { core, mind, client } = await setUp(editingModel(({ passage }) => reply(passage)));
    const note = paragraph(text("The tides are very strong."));
    writeMind(client, [note]);
    await client.settled();
    const again = (proposalId: string) =>
      core.regenerateProposal({ requestId: `again-${++ids}`, mindId: mind.id, proposalId });

    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "shorten"));
    expect(shownWithChanges(client)).toEqual(["The tides are [-very -]strong."]);
    // The User changes the words the change removes.
    const words = (client.blocks.get(0) as Y.XmlElement).get(0) as Y.XmlText;
    words.insert("The tides are v".length, "e");
    await client.settled();
    const [stale] = pendingProposals(client.blocks).map((proposal) => proposal.id);
    expect([...outOfDateProposals(client.blocks)]).toEqual([stale]);

    // The model fails: the change stays as it was.
    reply = () => {
      throw new Error("The model is down.");
    };
    expect(await again(stale ?? "")).toMatchObject({ kind: "failed" });
    expect(shownWithChanges(client)).toEqual(["The tides are [-veery -]strong."]);

    // The model gives the words back as they are: there is no change to make, so it goes.
    reply = (passage) => passage;
    expect(await again(stale ?? "")).toEqual({ kind: "unchanged" });
    expect(shownWithChanges(client)).toEqual(["The tides are veery strong."]);
    expect(await again(stale ?? "")).toEqual({ kind: "changed" });

    // Rejecting one out of date works as for any change.
    reply = () => "The tides are strong.";
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "shorten"));
    words.insert("The tides are v".length, "e");
    await client.settled();
    expect(outOfDateProposals(client.blocks).size).toBe(1);
    expect(await resolveChanges(client, "accept")).toEqual([]);
    expect(await resolveChanges(client, "reject")).toHaveLength(1);
    expect(shownWithChanges(client)).toEqual(["The tides are veeery strong."]);
  });

  test("a change goes out of date when the words it removes are edited, not when the User writes in what it adds or elsewhere", async () => {
    const { core, mind, client } = await setUp(
      editingModel(() => "The tides are strong and truly regular. Moon."),
    );
    const note = paragraph(text("The tides are very strong and really regular. Moon."));
    writeMind(client, [note]);
    await client.settled();
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    expect(shownWithChanges(client)).toEqual([
      "The tides are [-very -]strong and [-really-]{+truly+} regular. Moon.",
    ]);
    const words = (client.blocks.get(0) as Y.XmlElement).get(0) as Y.XmlText;

    // Words written elsewhere in the paragraph, or in what a change adds, leave both as they are.
    words.insert(words.length, " Sun.");
    words.insert("The tides are very strong and reallytru".length, "u");
    await client.settled();
    expect(shownWithChanges(client)).toEqual([
      "The tides are [-very -]strong and [-really-]{+truuly+} regular. Moon. Sun.",
    ]);
    expect(outOfDateProposals(client.blocks).size).toBe(0);

    // Deleting some of the words a change removes puts it out of date.
    words.delete("The tides are ".length, "ve".length);
    await client.settled();
    const [first] = pendingProposals(client.blocks).map((proposal) => proposal.id);
    expect([...outOfDateProposals(client.blocks)]).toEqual([first]);
  });

  test("Proposals made before changes recorded their text are never out of date", async () => {
    const { core, mind, client } = await setUp(editingModel(() => "The tides are strong."));
    const note = paragraph(text("The tides are very strong."));
    writeMind(client, [note]);
    await client.settled();
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "shorten"));
    // As version 4 wrote it: the change names its id and request only.
    const words = (client.blocks.get(0) as Y.XmlElement).get(0) as Y.XmlText;
    const [change] = pendingProposals(client.blocks);
    const removed = "The tides are ".length;
    client.doc.transact(() => {
      words.format(removed, "very ".length, {
        proposalDelete: { id: change?.id, request: change?.request },
      });
    });
    words.insert(removed + 1, "e");
    await client.settled();
    expect(shownWithChanges(client)).toEqual(["The tides are [-veery -]strong."]);
    expect(outOfDateProposals(client.blocks).size).toBe(0);
    expect(await resolveChanges(client, "accept")).toHaveLength(1);
    expect(shownWithChanges(client)).toEqual(["The tides are strong."]);
  });

  test("deleting a Document a waiting change cites leaves the change, with its Citation, for the User", async () => {
    const { core, mind, client, tides } = await setUp(
      editingModel(() => "Spring tides come at new and full moon [1]."),
    );
    const note = paragraph(
      text("Spring tides happen at new moon and at full moon"),
      citation(tides),
      text("."),
    );
    writeMind(client, [note]);
    await client.settled();
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    const waiting = pendingProposals(client.blocks);
    expect(waiting.length).toBeGreaterThan(0);

    await core.deleteDocument(tides.id);
    await client.settled();

    // Nothing is dropped: the change and the Citation it keeps are still there, to show "Source removed".
    expect(pendingProposals(client.blocks)).toEqual(waiting);
    expect(shownWithChanges(client).join("")).toContain("⟨");
  });
});

describe("Edit with AI and the model's service", () => {
  test("a cloud model is asked only once the User allows the edits flow, never in local mode", async () => {
    const calls: EditCall[] = [];
    const model = editingModel(() => "A better paragraph.", calls);
    const dataDir = await createTempDataFolder();
    const models = scriptedModels(model);
    const core = startCore(dataDir, { createChatModel: models.createChatModel });
    await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-test" });
    const mind = await core.createMind({ title: "Cloud" });
    const client = await connectToMind(core, mind.id);
    const note = paragraph(text("A paragraph."));
    writeMind(client, [note]);
    await client.settled();

    // Declined: nothing is sent.
    const asked = nextEvent(core, "consent.requested");
    const declined = core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    const request = (await asked) as ConsentRequest;
    expect(request.flow).toMatchObject({ id: "edits", sends: ["selection"] });
    await core.respondToConsent(request.requestId, false);
    expect(await declined).toMatchObject({
      kind: "not-ready",
      readiness: { reason: "consent-declined" },
    });
    expect(calls).toHaveLength(0);

    // Allowed: it is sent, with thinking off.
    await core.allowDataFlow("edits", request.flow.service.id);
    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve")),
    ).toMatchObject({ kind: "proposed" });
    expect(calls).toHaveLength(1);
    expect(models.specs.at(-1)).toMatchObject({ kind: "openai", thinking: "off" });
    await resolveChanges(client, "reject");

    // Local mode: a cloud model isn't used.
    await core.setLocalOnly(true);
    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve")),
    ).toEqual({ kind: "local-mode" });
    expect(calls).toHaveLength(1);
  });

  test("with no chat model it says so, and nothing is written", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const mind = await core.createMind({ title: "No model" });
    const client = await connectToMind(core, mind.id);
    const note = paragraph(text("A paragraph."));
    writeMind(client, [note]);
    await client.settled();

    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve")),
    ).toEqual({ kind: "not-ready", readiness: { ready: false, reason: "no-provider" } });
    expect(shownWithChanges(client)).toEqual(["A paragraph."]);
  });
});
