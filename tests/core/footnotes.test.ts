import { randomUUID } from "node:crypto";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  type Document,
  FOOTNOTE_NODE,
  MIND_SETTINGS_FIELD,
  type MindExport,
} from "../../src/core";
import { createElement, type NodeJSON, toMarkdown } from "../../src/core/answers/blocks";
import { firstLineOf } from "../../src/core/mindText";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { checkWellFormed, footnotesOf, paragraphsOf, part, unzip } from "../helpers/docx";
import { connectToMind } from "../helpers/mindClient";
import { question, readMind, writeMind } from "../helpers/minds";

/*
 * Footnotes the User writes (#284), in the core: read with the Mind's text,
 * numbered together with the Citations' footnotes in every export, kept in
 * the Mind's Yjs content across a restart and between clients.
 */

const text = (value: string, ...marks: string[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks: marks.map((type) => ({ type })) }),
});

const link = (value: string, href: string): JSONContent => ({
  type: "text",
  text: value,
  marks: [{ type: "link", attrs: { href } }],
});

const paragraph = (...content: JSONContent[]): JSONContent => ({
  type: "paragraph",
  attrs: { id: randomUUID() },
  content,
});

const footnote = (...content: JSONContent[]): JSONContent => ({
  type: FOOTNOTE_NODE,
  attrs: { id: randomUUID() },
  content,
});

/** A Citation of the Document's page 2, found there. */
const citation = (document: Document): JSONContent => ({
  type: "citation",
  attrs: {
    passageId: "passage-1",
    documentId: document.id,
    documentName: "Tides",
    contentHash: document.contentHash,
    pageFrom: 2,
    pageTo: 2,
    quote: "Spring tides happen at new moon.",
    check: "found",
    checkReason: null,
  },
});

async function setUp() {
  const dataDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { now: manualClock().now });
  const path = await writeSourceFile(sources, "Tides.txt", "Spring tides happen at new moon.");
  const [tides] = await addAndProcess(core, [path]);
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  return { dataDir, core, mind, client, tides: tides as Document };
}

/**
 * A footnote in a Note, an Answer's Citation, a footnote in Chinese, one
 * holding a found Citation, and one with a link and formatting.
 */
function sample(tides: Document): JSONContent[] {
  const asked = question("When do spring tides happen?");
  return [
    paragraph(text("Neap tides "), footnote(text("At the quarter moons.")), text(" are weak.")),
    asked,
    {
      type: "answer",
      attrs: { questionId: asked.attrs.id, status: "done" },
      content: [paragraph(text("Spring tides come at new moon "), citation(tides), text("."))],
    },
    paragraph(text("潮汐"), footnote(text("每月两次。"))),
    paragraph(text("See also"), footnote(text("Compare"), citation(tides), text("."))),
    paragraph(
      text("Tables"),
      footnote(
        text("Published "),
        text("yearly", "italic"),
        text(" by "),
        link("the port", "https://port.example"),
      ),
    ),
  ];
}

const docxParts = (exported: MindExport) => {
  const files = unzip(exported.data);
  return {
    files,
    document: part(files, "word/document.xml"),
    footnotes: part(files, "word/footnotes.xml"),
  };
};

describe("exporting the User's footnotes", () => {
  test("to Word: real footnotes, numbered with the Citations' in the order they appear", async () => {
    const { core, mind, client, tides } = await setUp();
    writeMind(client, sample(tides));
    await client.settled();

    const exported = await core.exportMind(mind.id, { format: "docx" });
    // A Citation inside a footnote is a Citation all the same.
    expect(exported).toMatchObject({ citations: 2, unverifiedCitations: 0 });
    const { files, document, footnotes } = docxParts(exported);
    checkWellFormed(footnotes);

    expect(paragraphsOf(document).map((each) => each.text)).toEqual([
      "Tides",
      "Neap tides[1] are weak.",
      "Spring tides come at new moon[2].",
      "潮汐[3]",
      "See also[4]",
      "Tables[5]",
    ]);
    expect(footnotesOf(footnotes)).toEqual({
      "1": "At the quarter moons.",
      "2": "Tides, p. 2",
      "3": "每月两次。",
      // The Citation in it is written into its text, as its source.
      "4": "Compare Tides, p. 2.",
      "5": "Published yearly by the port",
    });
    // Its formatting, and its link, which is the footnotes part's own relationship.
    expect(footnotes).toContain(
      '<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">yearly</w:t></w:r>',
    );
    expect(footnotes).toContain('<w:hyperlink r:id="rIdNoteLink1" w:history="1">');
    const relationships = part(files, "word/_rels/footnotes.xml.rels");
    checkWellFormed(relationships);
    expect(relationships).toContain(
      'Id="rIdNoteLink1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://port.example" TargetMode="External"',
    );
    expect(part(files, "word/_rels/document.xml.rels")).not.toContain("port.example");
  });

  test("to Markdown: [^n] for both kinds, the User's text in its definition", async () => {
    const { core, mind, client, tides } = await setUp();
    writeMind(client, sample(tides));
    await client.settled();

    const markdown = new TextDecoder().decode(
      (await core.exportMind(mind.id, { format: "markdown" })).data,
    );

    expect(markdown).toContain("Neap tides[^1] are weak.");
    expect(markdown).toContain("Spring tides come at new moon[^2].");
    expect(markdown).toContain("潮汐[^3]");
    expect(markdown).toContain(
      [
        "[^1]: At the quarter moons.",
        "[^2]: Tides, p. 2",
        "[^3]: 每月两次。",
        "[^4]: Compare Tides, p. 2.",
        "[^5]: Published *yearly* by [the port](https://port.example)",
      ].join("\n"),
    );
  });

  test("a Mind without footnotes of the User's exports as before", async () => {
    const { core, mind, client, tides } = await setUp();
    writeMind(client, [paragraph(text("Spring tides "), citation(tides))]);
    await client.settled();

    const { files } = docxParts(await core.exportMind(mind.id, { format: "docx" }));
    expect(files.has("word/_rels/footnotes.xml.rels")).toBe(false);
  });
});

describe("a footnote in the Mind's text", () => {
  const fragmentOf = (...blocks: NodeJSON[]) => {
    const fragment = new Y.Doc().getXmlFragment("blocks");
    fragment.insert(0, blocks.map(createElement));
    return fragment;
  };

  test("Question context shows it where its marker is, as an inline note", () => {
    const fragment = fragmentOf(
      paragraph(
        text("Neap tides"),
        footnote(text("At the quarter "), text("moons", "bold")),
        text("."),
      ) as NodeJSON,
    );
    expect(toMarkdown(fragment.get(0) as Y.XmlElement)).toBe(
      "Neap tides^[At the quarter **moons**].",
    );
  });

  test("its text isn't part of the line, so a Mind isn't named by it", () => {
    const fragment = fragmentOf(
      paragraph(footnote(text("A footnote first"))) as NodeJSON,
      paragraph(
        text("Spring tides"),
        footnote(text("Twice a month.")),
        text(" are strong"),
      ) as NodeJSON,
    );
    expect(firstLineOf(fragment)).toBe("Spring tides are strong");
  });
});

describe("footnotes in the Mind's Yjs content", () => {
  test("reach another client, and are there after a restart, with the new content schema version", async () => {
    const { dataDir, core, mind, client, tides } = await setUp();
    const other = await connectToMind(core, mind.id);
    writeMind(client, sample(tides));
    await client.settled();

    const footnotesIn = (doc: ReturnType<typeof readMind>) => {
      const found: { id: unknown; text: string }[] = [];
      doc.descendants((node) => {
        if (node.type.name !== FOOTNOTE_NODE) return true;
        found.push({ id: node.attrs.id, text: node.textContent });
        return false;
      });
      return found;
    };
    const written = footnotesIn(readMind(client));
    expect(written.map((each) => each.text)).toEqual([
      "At the quarter moons.",
      "每月两次。",
      "Compare.",
      "Published yearly by the port",
    ]);
    await expect.poll(() => footnotesIn(readMind(other))).toEqual(written);
    expect(client.doc.getMap(MIND_SETTINGS_FIELD).get(CONTENT_SCHEMA_VERSION_KEY)).toBe(
      CONTENT_SCHEMA_VERSION,
    );

    core.close();
    const restarted = startCore(dataDir);
    const reader = await connectToMind(restarted, mind.id);
    expect(footnotesIn(readMind(reader))).toEqual(written);
    // The one holding a Citation still holds it, checked as it was.
    const citations: string[] = [];
    readMind(reader).descendants((node, _pos, parent) => {
      if (node.type.name === "citation") citations.push(`${parent?.type.name}:${node.attrs.check}`);
    });
    expect(citations).toEqual(["paragraph:found", `${FOOTNOTE_NODE}:found`]);
    restarted.close();
  });
});
