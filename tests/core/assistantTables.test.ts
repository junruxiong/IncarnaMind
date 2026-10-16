import type { JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { Document } from "../../src/core";
import { EDIT_INSTRUCTIONS } from "../../src/core/edits/prompt";
import { pendingProposals } from "../../src/shared/proposals";
import { askAssistant, changesIn, lastResult, mindCalls } from "../helpers/assistant";
import { setUpWithDocuments, shownPassages } from "../helpers/citations";
import { part, unzip } from "../helpers/docx";
import { readMind, resolveChanges, writeMind } from "../helpers/minds";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

/**
 * A table the assistant writes into a Mind (#961), through the core's public
 * interface as the assistant asks: by edit_mind's rewrite of the User's text,
 * by add_to_mind, and by revising its own waiting text, the page holds a
 * table with rows and columns, each cell's source a Citation, and Word gets a
 * table with footnotes. Lines of pipes that make no table go in nowhere, and
 * the model is told to say so and give the text in its reply. The model's
 * choices are scripted; real models' are checked by hand.
 */

const SPRING = "Spring tides happen at new moon and at full moon";

const TIDES = {
  name: "Tides.md",
  contents: [
    "# Tides",
    "",
    `${SPRING}, when the Sun and the Moon pull together.`,
    "",
    "Neap tides happen at the first and last quarter moon.",
  ].join("\n"),
};

const TABLE = [
  "| Term | What it means |",
  "|---|---|",
  "| Spring tide | The biggest tide, at new and full moon [^1] |",
  "| Neap tide | The smallest tide |",
].join("\n");

/** The table as the issue found it: the right rows, but with no row of --- under the first. */
const BROKEN_TABLE = [
  "| Term | What it means |",
  "| Spring tide | The biggest tide [Tides, p. 1] |",
].join("\n");

const EXPECTED_ROWS = [
  ["Term", "What it means"],
  ["Spring tide", "The biggest tide, at new and full moon ⟨found⟩"],
  ["Neap tide", "The smallest tide"],
];

/** A Citation of the spring-tides line, as a Note holds one copied from an Answer. */
function citation(tides: Document): JSONContent {
  return {
    type: "citation",
    attrs: {
      passageId: "passage-1",
      documentId: tides.id,
      documentName: "Tides",
      contentHash: tides.contentHash,
      quote: SPRING,
      check: "found",
    },
  };
}

/** Edit with AI's request, and the passage it asks to rewrite; null for any other request. */
const editPassage = (call: ModelCall): string | null =>
  call.system === EDIT_INSTRUCTIONS
    ? (/<passage>\n([\s\S]*?)\n<\/passage>/.exec(call.prompt)?.[1] ?? null)
    : null;

const called = (call: ModelCall, tool: string) => call.results.some((each) => each.tool === tool);

/** Searches the Documents and records Citation 1 of the spring-tides line, then `next`. */
function citingFirst(call: ModelCall, next: () => ScriptedReply): ScriptedReply {
  const search = call.results.find((result) => result.tool === "search_documents");
  if (!search) {
    return { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] };
  }
  if (!called(call, "cite")) {
    const [passage] = shownPassages(search.text);
    const record = { marker: 1, passage: passage?.id, quote: SPRING };
    return { calls: [{ tool: "cite", input: { citations: [record] } }] };
  }
  return next();
}

/** The Mind's tables: each row's cells' text, a Citation as ⟨its check⟩. */
function tablesIn(doc: ProseMirrorNode): string[][][] {
  const tables: string[][][] = [];
  doc.descendants((node) => {
    if (node.type.name !== "table") return true;
    const rows: string[][] = [];
    node.forEach((row) => {
      const cells: string[] = [];
      row.forEach((cell) => {
        let shown = "";
        cell.descendants((inner) => {
          if (inner.isText) shown += inner.text ?? "";
          else if (inner.type.name === "citation") shown += `⟨${String(inner.attrs.check)}⟩`;
          return true;
        });
        cells.push(shown);
      });
      rows.push(cells);
    });
    tables.push(rows);
    return false;
  });
  return tables;
}

/** Whether a paragraph of the Mind reads as a table's rows: what a table that wasn't made leaves. */
function pipeText(doc: ProseMirrorNode): boolean {
  let found = false;
  doc.descendants((node) => {
    if (node.isTextblock && /\|/.test(node.textBetween(0, node.content.size, "\n", "\n"))) {
      found = true;
    }
    return !found;
  });
  return found;
}

async function setUp(model: MockLanguageModelV4) {
  const set = await setUpWithDocuments(model, [TIDES]);
  const [tides] = set.documents;
  if (!tides) throw new Error("No Document.");
  writeMind(set.client, [
    {
      type: "heading",
      attrs: { id: "summary", level: 2 },
      content: [{ type: "text", text: "Summary" }],
    },
    {
      type: "paragraph",
      attrs: { id: "spring" },
      content: [
        { type: "text", text: "Spring tides come at new and full moon. " },
        citation(tides),
      ],
    },
  ]);
  await set.client.settled();
  return { ...set, tides };
}

describe("A table the assistant writes into a Mind is a table", { timeout: 30_000 }, () => {
  // protects=edit_mind's rewrite of the User's text that adds a Markdown table proposes a table Block, its cells' markers the passage's Citations, and the model reads that a table was added; fails_when=the table lands as a paragraph of pipe lines (#961), or a cell's Citation is lost; no_claim=how a real model writes a table.
  test("a rewrite that adds a table proposes a table, its cells cited", async () => {
    const model = scriptedModel((call): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) {
        return {
          text: `${passage}\n\n${TABLE.replace("[^1]", "[1]")}\n<why>Longer, with a table.</why>`,
        };
      }
      if (!called(call, "edit_mind")) {
        return {
          calls: [{ tool: "edit_mind", input: { instruction: "Make it longer, with a table" } }],
        };
      }
      return { text: "" };
    });
    const { core, mind, client } = await setUp(model);

    const answer = await askAssistant(core, { text: "Make the summary longer", mindId: mind.id });
    await client.settled();

    expect(pipeText(readMind(client))).toBe(false);
    expect(tablesIn(readMind(client))).toEqual([EXPECTED_ROWS]);
    expect(changesIn(answer)).toEqual([
      { mindId: mind.id, created: false, request: expect.any(String), proposals: 1, tables: 1 },
    ]);
    expect(lastResult(model)).toContain("Proposed 1 change in “Tides”");
    expect(lastResult(model)).toContain("One of them adds a table.");
    await resolveChanges(client, "accept");
    expect(tablesIn(readMind(client))).toEqual([EXPECTED_ROWS]);
  });

  // protects=add_to_mind with a Markdown table cited with cite adds a table whose cell holds a checked Citation, and once accepted, Word gets a table with that Citation as a footnote; fails_when=the cell's marker stays text or is lost, or the export writes pipes; no_claim=the export's look.
  test("an added table's cells hold checked Citations, and Word writes a table with footnotes", async () => {
    const model = scriptedModel(
      (call): ScriptedReply =>
        citingFirst(call, () =>
          called(call, "add_to_mind")
            ? { text: "" }
            : {
                calls: [
                  {
                    tool: "add_to_mind",
                    input: { after: "Summary", content: `## Key terms\n\n${TABLE}` },
                  },
                ],
              },
        ),
    );
    const { core, mind, client } = await setUp(model);

    const answer = await askAssistant(core, { text: "Add a table of key terms", mindId: mind.id });
    await client.settled();

    expect(tablesIn(readMind(client))).toEqual([EXPECTED_ROWS]);
    expect(changesIn(answer)).toEqual([
      {
        mindId: mind.id,
        created: false,
        request: expect.any(String),
        proposals: 2,
        tables: 1,
        cited: 1,
      },
    ]);
    expect(lastResult(model)).toContain("One of them adds a table.");

    await resolveChanges(client, "accept");
    const exported = await core.exportMind(mind.id, { format: "docx" });
    const document = part(unzip(exported.data), "word/document.xml");
    const table = /<w:tbl>[\s\S]*<\/w:tbl>/.exec(document)?.[0] ?? "";
    expect([...table.matchAll(/<w:tr>/g)]).toHaveLength(3);
    expect(table).toContain("Spring tide");
    expect(table).toContain("<w:footnoteReference");
    expect(document).not.toContain("|---|");
    expect(part(unzip(exported.data), "word/footnotes.xml")).toContain("Tides");
  });

  // protects=revising the assistant's own waiting text, by instruction or with new content, makes its table a table; fails_when=either path leaves pipe lines in the page; no_claim=none beyond protects.
  test("a revision of its own waiting text, by instruction or by content, makes a table", async () => {
    const byInstruction = (call: ModelCall): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) {
        return { text: `${passage}\n\n${TABLE.replace(" [^1]", "")}\n<why>A table.</why>` };
      }
      return citingFirst(call, () => {
        if (!called(call, "add_to_mind")) {
          const content = "## Terms\n\nSpring tides come at new and full moon [^1].";
          return { calls: [{ tool: "add_to_mind", input: { content } }] };
        }
        if (!called(call, "edit_mind")) {
          return {
            calls: [{ tool: "edit_mind", input: { part: "Terms", instruction: "Add a table" } }],
          };
        }
        return { text: "" };
      });
    };
    const byContent = (call: ModelCall): ScriptedReply =>
      citingFirst(call, () =>
        called(call, "edit_mind")
          ? { text: "" }
          : {
              calls: [
                { tool: "edit_mind", input: { part: "Terms", content: `## Terms\n\n${TABLE}` } },
              ],
            },
      );
    let script = byInstruction;
    const model = scriptedModel((call) => script(call));
    const { core, mind, client } = await setUp(model);

    const first = await askAssistant(core, { text: "Add the terms", mindId: mind.id });
    await client.settled();

    expect(changesIn(first).at(-1)).toMatchObject({ revised: true, tables: 1 });
    expect(pipeText(readMind(client))).toBe(false);
    expect(tablesIn(readMind(client))).toEqual([
      [
        ["Term", "What it means"],
        ["Spring tide", "The biggest tide, at new and full moon"],
        ["Neap tide", "The smallest tide"],
      ],
    ]);

    script = byContent;
    const second = await askAssistant(core, {
      text: "Cite the table",
      mindId: mind.id,
      conversationId: first.conversationId,
    });
    await client.settled();

    expect(changesIn(second)).toEqual([expect.objectContaining({ revised: true, tables: 1 })]);
    expect(tablesIn(readMind(client))).toEqual([EXPECTED_ROWS]);
  });
});

describe("A table that can't be made goes in nowhere", { timeout: 30_000 }, () => {
  // protects=add_to_mind whose table has no row of --- adds nothing, its card says why, and the model is told to say so in the User's line and give the text in its reply; fails_when=the pipe lines land in the page, or the model isn't told; no_claim=that a real model words it so.
  test("added: nothing goes in, and the model is told to say so and give the text", async () => {
    const model = scriptedModel(
      (call): ScriptedReply =>
        called(call, "add_to_mind")
          ? { text: "I couldn't make this into a table." }
          : {
              calls: [
                {
                  tool: "add_to_mind",
                  input: { content: `## Key terms\n\n${BROKEN_TABLE}\n\nThat is all.` },
                },
              ],
            },
    );
    const { core, mind, client } = await setUp(model);
    const before = JSON.stringify(client.blocks.toJSON());

    const answer = await askAssistant(core, { text: "Add a table of key terms", mindId: mind.id });
    await client.settled();

    expect(JSON.stringify(client.blocks.toJSON())).toBe(before);
    expect(pendingProposals(client.blocks)).toEqual([]);
    expect(mindCalls(answer)[0]).toMatchObject({
      status: "failed",
      mindProblem: { kind: "no-table", mindId: mind.id },
    });
    expect(lastResult(model)).toContain(
      "“I couldn't make this into a table, so it's below as text. Nothing was added to “Tides”.”",
    );
    expect(lastResult(model)).toContain("这些内容没法做成表格，所以写在下面，没有添加到“Tides”。");
    expect(lastResult(model)).toContain("so the User still has it");
  });

  // protects=edit_mind whose rewrite writes pipe lines that make no table proposes nothing, takes back what it had written, and gives the model the text without its markers; fails_when=the pipe lines or part of the rewrite stay in the page; no_claim=none beyond protects.
  test("rewritten: nothing is proposed, and the model gets the text to give", async () => {
    const model = scriptedModel((call): ScriptedReply => {
      const passage = editPassage(call);
      if (passage !== null) {
        return { text: `${passage} More here.\n\n${BROKEN_TABLE}\n\nThe end.\n<why>Longer.</why>` };
      }
      return called(call, "edit_mind")
        ? { text: "" }
        : {
            calls: [{ tool: "edit_mind", input: { instruction: "Make it longer, with a table" } }],
          };
    });
    const { core, mind, client } = await setUp(model);
    const before = JSON.stringify(client.blocks.toJSON());

    const answer = await askAssistant(core, { text: "Make the summary longer", mindId: mind.id });
    await client.settled();

    expect(JSON.stringify(client.blocks.toJSON())).toBe(before);
    expect(mindCalls(answer)[0]).toMatchObject({
      status: "failed",
      mindProblem: { kind: "no-table", mindId: mind.id },
    });
    const told = lastResult(model);
    expect(told).toContain("Nothing was added to “Tides”");
    expect(told).toContain(
      "<text>## Summary\n\nSpring tides come at new and full moon. More here.",
    );
    expect(told).toContain("| Spring tide | The biggest tide [Tides, p. 1] |");
    expect(told).not.toContain("[1]");
  });
});
