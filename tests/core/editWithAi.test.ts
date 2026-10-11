import type { JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
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
  pendingProposals,
  proposalRequestOf,
  resolveProposals,
} from "../../src/shared/proposals";
import { setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { readMind, writeMind } from "../helpers/minds";
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

/**
 * The Mind's text, Proposals shown: removed words as [-…-], added words as
 * {+…+}, a Citation as ⟨its check⟩, Blocks a Proposal adds or removes after
 * "+ " or "- ", one line per paragraph.
 */
function shown(client: MindClient): string[] {
  const lines: string[] = [];
  readMind(client).descendants((node: ProseMirrorNode) => {
    if (!node.isTextblock) return true;
    let line = "";
    node.forEach((child) => {
      const kind = (child.attrs[PROPOSAL_ATTRIBUTE] as { kind?: string } | null)?.kind;
      const inline = child.isText
        ? (child.text ?? "")
        : child.type.name === "citation"
          ? `⟨${child.attrs.check}⟩`
          : "￼";
      const marks = child.marks.map((mark) => mark.type.name);
      if (marks.includes("proposalDelete") || kind === "delete") line += `[-${inline}-]`;
      else if (marks.includes("proposalInsert") || kind === "insert") line += `{+${inline}+}`;
      else line += inline;
    });
    const block = (node.attrs[PROPOSAL_ATTRIBUTE] as { kind?: string } | null)?.kind;
    lines.push(`${block === "insert" ? "+ " : block === "delete" ? "- " : ""}${line}`);
    return false;
  });
  return lines.map((line) => line.replace(/-\]\[-|\+\}\{\+/g, ""));
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

/** Accepts or rejects Proposals in the client's copy, as the editor does, and sends it to the core. */
async function resolve(
  client: MindClient,
  outcome: "accept" | "reject",
  which: (id: string, index: number) => boolean = () => true,
) {
  const order = pendingProposals(client.blocks).map((proposal) => proposal.id);
  let resolved: ReturnType<typeof resolveProposals> = [];
  client.doc.transact(() => {
    resolved = resolveProposals(
      client.blocks,
      (proposal) => which(proposal.id, order.indexOf(proposal.id)),
      outcome,
    );
  }, "editor");
  await client.settled();
  return resolved;
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
    expect(shown(client)).toEqual(["The tides here are [-very -]strong and [-really -]regular."]);
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
    await resolve(client, "accept", (_id, index) => index === 0);
    expect(shown(client)).toEqual(["The tides here are strong and [-really -]regular."]);
    await resolve(client, "reject");
    expect(shown(client)).toEqual(["The tides here are strong and really regular."]);
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

    expect(shown(client)).toEqual(["Spring tides [-happen-]{+come+} twice [-a-]{+each+} month."]);
    const added = readMind(client).firstChild?.content.content.filter((node) =>
      node.marks.some((mark) => mark.type.name === "proposalInsert"),
    );
    expect(added?.every((node) => node.marks.some((mark) => mark.type.name === "bold"))).toBe(true);
    // Until accepted, what is read of the Mind (Question context, exports) is the text as it was.
    const before = await exported(core, mind.id);
    expect(before).toContain("Spring tides **happen twice a month**.");
    expect(before).not.toContain("come");
    await resolve(client, "accept");
    expect(await exported(core, mind.id)).toContain("Spring tides **come twice each month**.");
    expect(shown(client)).toEqual(["Spring tides come twice each month."]);
    expect(readMind(client).firstChild?.toJSON().content).toEqual([
      text("Spring tides "),
      text("come twice each month", "bold"),
      text("."),
    ]);
  });

  test("a selection inside a paragraph changes only what was selected", async () => {
    const { core, mind, client } = await setUp(editingModel(() => "Moon"));
    const note = paragraph(text("The moon pulls the sea, and the moon sets the tides."));
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

    expect(shown(client)).toEqual([
      "The moon pulls the sea, and the [-moon-]{+Moon+} sets the tides.",
    ]);
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
    expect(shown(client)).toEqual([
      "Spring tides [-happen-]{+come+} at new [-moon and at -]{+and +}full moon [-⟨not-found⟩, and they are the highest ⟨found⟩-]{+⟨found⟩+}.",
    ]);

    await resolve(client, "accept");
    expect(shown(client)).toEqual(["Spring tides come at new and full moon ⟨found⟩."]);
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
    expect(shown(client)).toEqual(["The force is ￼ for spring tides ⟨found⟩."]);
  });

  test("Chinese is compared character by character", async () => {
    const { core, mind, client } = await setUp(editingModel(() => "潮水每天很规律地涨落两次。"));
    const note = paragraph(text("潮水每天非常规律地涨落两次。"));
    writeMind(client, [note]);
    await client.settled();

    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));

    expect(shown(client)).toEqual(["潮水每天[-非常-]{+很+}规律地涨落两次。"]);
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
    expect(shown(client)).toEqual([
      "The Moon raises two bulges of water.",
      "+ It says the Moon lifts the sea twice a day.",
      "A closing line.",
    ]);
    expect(await exported(core, mind.id)).not.toContain("It says");
    await resolve(client, "reject");
    expect(shown(client)).toEqual(["The Moon raises two bulges of water.", "A closing line."]);
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

    expect(shown(client)).toEqual([
      "First [-paragraph-]{+and second, together+}.",
      "- Second paragraph.",
      "[-Single-]{+One+} line.",
      "+ Another line.",
    ]);
    await resolve(client, "accept");
    expect(shown(client)).toEqual(["First and second, together.", "One line.", "Another line."]);
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
    expect(shown(client)).toEqual([
      "First [-paragraph-]{+line+}.",
      "",
      "Third [-paragraph-]{+line+}.",
    ]);
  });

  test("nothing is proposed over text that changed or holds Proposals, nor when the model gives it back as it was", async () => {
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

    // The User edits the paragraph while the model works.
    const editing = core.editWithAi(wholeParagraph(client, mind.id, slow.attrs.id, "improve"));
    await expect.poll(() => calls).toBe(1);
    const content = (client.blocks.get(0) as Y.XmlElement).get(0) as Y.XmlText;
    content.insert(0, "Very ");
    await client.settled();
    release();
    expect(await editing).toEqual({ kind: "changed" });
    expect(shown(client)).toEqual(["Very Slow paragraph.", "Same paragraph."]);

    expect(
      await core.editWithAi(wholeParagraph(client, mind.id, same.attrs.id, "improve")),
    ).toEqual({ kind: "unchanged" });

    // Proposals waiting in the selection: accept or reject them first.
    await core.editWithAi(wholeParagraph(client, mind.id, slow.attrs.id, "improve"));
    expect(shown(client)[0]).toBe("Very Slow paragraph.{+ Indeed.+}");
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
    expect(shown(client)).toEqual(["A paragraph to improve."]);

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
    expect(shown(client)).toEqual(["A paragraph to improve."]);
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
    const pending = shown(first);
    expect(pending).toEqual([
      "The tides are [-very -]strong and [-really -]regular.",
      "Moon.",
      "+ The Moon pulls the sea.",
    ]);

    // Another client has them as they are.
    const second = await connectToMind(core, mind.id);
    expect(shown(second)).toEqual(pending);

    // So does the Mind after a restart, from what was stored.
    core.close();
    core = startCore(dataDir, { createChatModel: models.createChatModel });
    const reloaded = await connectToMind(core, mind.id);
    expect(shown(reloaded)).toEqual(pending);
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
    expect(shown(reloaded)).toEqual(expected);
    const again = await connectToMind(core, mind.id);
    expect(shown(again)).toEqual(expected);
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
    ).toEqual(expected);
  });

  test("an accepted change goes into the record of AI use with the model that wrote it; a rejected one, or one that only removes words, doesn't", async () => {
    const { core, mind, client } = await setUp(
      editingModel(() => "The tides are strong and quite regular."),
    );
    const note = paragraph(text("The tides are very strong and regular."));
    writeMind(client, [note]);
    await client.settled();
    await core.editWithAi(wholeParagraph(client, mind.id, note.attrs.id, "improve"));
    expect(shown(client)).toEqual(["The tides are [-very -]strong and {+quite +}regular."]);

    // As the editor does: accept, send the change, then record what was accepted.
    const [removing, adding] = pendingProposals(client.blocks).map((proposal) => proposal.id);
    const accepted = await resolve(client, "accept", (id) => id === adding);
    await resolve(client, "reject", (id) => id === removing);
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

    await resolve(client, "accept");
    expect(shown(client)).toEqual(["The tides are strong and regular."]);
    undo.undo();
    await client.settled();
    expect(shown(client)).toEqual(["The tides are [-very -]strong and [-really -]regular."]);
    expect(pendingProposals(client.blocks)).toHaveLength(2);
    undo.redo();
    expect(shown(client)).toEqual(["The tides are strong and regular."]);
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
    await resolve(client, "reject");

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
    expect(shown(client)).toEqual(["A paragraph."]);
  });
});
