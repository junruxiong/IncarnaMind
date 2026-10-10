import { createHash } from "node:crypto";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  type CitationCheck,
  type CitationCheckReason,
  type Document,
  InvalidInputError,
  type MindExport,
  NotFoundError,
} from "../../src/core";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { checkWellFormed, footnotesOf, paragraphsOf, part, textOf, unzip } from "../helpers/docx";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { heading, note, question, writeMind } from "../helpers/minds";
import { zipFingerprint } from "../helpers/zip";

/** A core with one Document ("Tides") and a Mind titled "Tides", with a client. */
async function setUp(title = "Tides") {
  const dataDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { now: manualClock().now });
  const path = await writeSourceFile(sources, "Tides.txt", "Spring tides happen at new moon.");
  const [tides] = await addAndProcess(core, [path]);
  const mind = await core.createMind({ title });
  const client = await connectToMind(core, mind.id);
  return { core, mind, client, tides: tides as Document };
}

const text = (value: string, ...marks: string[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks: marks.map((type) => ({ type })) }),
});

const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });

const item = (...content: JSONContent[]): JSONContent => ({ type: "listItem", content });

/** A Citation of the Document, as the core writes one into an Answer. */
function citation(
  document: Document,
  pages: [number, number] | null,
  check: CitationCheck,
  checkReason: CitationCheckReason | null = null,
): JSONContent {
  return {
    type: "citation",
    attrs: {
      passageId: "passage-1",
      documentId: document.id,
      documentName: "Tides",
      contentHash: document.contentHash,
      pageFrom: pages?.[0] ?? null,
      pageTo: pages?.[1] ?? null,
      quote: "Spring tides happen at new moon.",
      check,
      checkReason,
    },
  };
}

/**
 * Notes of every kind (one switched out of Question context), a Question, and
 * its Answer, whose three Citations were found, not found, and can't be checked.
 */
function writeSample(client: MindClient, tides: Document): void {
  const asked = question("When do spring tides happen?");
  writeMind(client, [
    heading(1, "Spring tides"),
    paragraph(
      text("Neap", "bold"),
      text(" tides are "),
      text("weaker", "italic"),
      text(", at $0."),
    ),
    note("Kept out of Questions, still exported.", { off: true }),
    {
      type: "bulletList",
      content: [item(paragraph(text("High water"))), item(paragraph(text("Low water")))],
    },
    {
      type: "orderedList",
      attrs: { start: 1 },
      content: [
        item(paragraph(text("New moon"))),
        item(paragraph(text("Full moon")), {
          type: "bulletList",
          content: [item(paragraph(text("Twice a month")))],
        }),
      ],
    },
    { type: "codeBlock", attrs: { language: "js" }, content: [text('console.log("tide");')] },
    { type: "blockMath", attrs: { latex: "F = G\\frac{m_1 m_2}{r^2}" } },
    asked,
    {
      type: "answer",
      attrs: { questionId: asked.attrs.id, status: "done" },
      content: [
        paragraph(
          text("At new moon "),
          citation(tides, [2, 2], "found"),
          text(" and at full moon "),
          citation(tides, [2, 2], "not-found", "quote-not-on-pages"),
          text("."),
        ),
        paragraph(
          text("The pull falls off as "),
          { type: "inlineMath", attrs: { latex: "1/r^3" } },
          text(" "),
          citation(tides, [3, 4], "cant-check", "no-text"),
          text("."),
        ),
      ],
    },
  ]);
}

/**
 * What the sample leaves out that the .docx writer knows: a quote, a rule, web
 * links (one on code), struck and underlined text, a line break, Citations of
 * slides, rows and sections, maths Word's equations can't hold, a table with a
 * cell over two columns, and an image.
 */
function writeEverythingElse(client: MindClient, tides: Document): void {
  const at = (location: object, name: string): JSONContent => {
    const node = citation(tides, [1, 1], "found");
    return { ...node, attrs: { ...node.attrs, documentName: name, location } };
  };
  const linked = (value: string, href: string, ...marks: string[]): JSONContent => ({
    type: "text",
    text: value,
    marks: [...marks.map((type) => ({ type })), { type: "link", attrs: { href } }],
  });
  writeMind(client, [
    { type: "blockquote", content: [paragraph(text("Time and tide wait for no one."))] },
    { type: "horizontalRule" },
    paragraph(
      linked("Tide tables", "https://tides.example/tables"),
      text(" and "),
      linked("tide.js", "https://tides.example/code", "code"),
      text(": "),
      text("old", "strike"),
      text(" "),
      text("new", "underline"),
      { type: "hardBreak" },
      text("Growth"),
      at({ kind: "slide", from: 3, to: 3 }, "Deck"),
      text(", revenue"),
      at({ kind: "rows", sheet: "Revenue", from: 12, to: 14 }, "Model"),
      text(", the crest"),
      at({ kind: "section", heading: "2.1 Sensitivity" }, "Review"),
      text("."),
    ),
    { type: "blockMath", attrs: { latex: "\\undefinedmacro{x}" } },
  ]);
  // The editor has no tables or images yet; they are written as Tiptap's would be.
  client.doc.transact(() => {
    client.blocks.push([
      element("table", {}, [
        element("tableRow", {}, [
          element("tableHeader", { colspan: "2" }, [element("paragraph", {}, ["Tides"])]),
        ]),
        element("tableRow", {}, [
          element("tableCell", {}, [element("paragraph", {}, ["Spring"])]),
          element("tableCell", {}, [element("paragraph", {}, ["4 m"])]),
        ]),
      ]),
      element("image", { src: PIXEL, alt: "A chart" }, []),
    ]);
  });
}

const decode = (exported: MindExport) => new TextDecoder().decode(exported.data);

/** The text of an equation's runs, joined. */
const mathText = (omml: string) =>
  [...omml.matchAll(/<m:t xml:space="preserve">([^<]*)<\/m:t>/g)].map((match) => match[1]).join("");

/** The parts of an exported .docx, each checked to be well-formed XML. */
function docxParts(exported: MindExport) {
  const files = unzip(exported.data);
  for (const [name, content] of files) {
    if (name.endsWith(".xml") || name.endsWith(".rels")) checkWellFormed(content.toString("utf8"));
  }
  return {
    files,
    document: part(files, "word/document.xml"),
    footnotes: part(files, "word/footnotes.xml"),
    numbering: part(files, "word/numbering.xml"),
    relationships: part(files, "word/_rels/document.xml.rels"),
  };
}

describe("exporting a Mind", () => {
  test("the bytes of an export don't change: a fingerprint of the sample's Markdown and .docx", async () => {
    const { core, mind, client, tides } = await setUp();
    writeSample(client, tides);
    await client.settled();

    const exported = async (format: "markdown" | "docx") =>
      (await core.exportMind(mind.id, { format })).data;

    expect(
      createHash("sha256")
        .update(await exported("markdown"))
        .digest("hex"),
    ).toBe("a4c7d2f0c6856d0ed5a4df3984ab53e8b1bd89057a4cef980a14629bb86685fa");
    // The .docx's compressed bytes depend on zlib's build, so they are checked against this
    // Node's deflate, and the rest is pinned (see `zipFingerprint`).
    expect(zipFingerprint(await exported("docx"))).toBe(
      "657f2cf0604e8470b9bcc513f41111ce89230884050fc75f4c900bb92ecb3b1b",
    );
  });

  test.each([
    [
      "the sample, with its Question",
      "sample",
      { includeQuestions: true },
      "4d550b58862572e9cfacb754c17da2186f0962f2d4b26579c442e48e8dcbf0f5",
    ],
    [
      "the sample, in Chinese, on A4",
      "sample",
      { language: "zh-CN" },
      "64ab43cb53ab415dfe722a7983f1416a75c7b2fc0a32e8886c2b10c648498624",
    ],
    [
      "quotes, rules, links, marks, breaks, tables, images and Locations",
      "rest",
      {},
      "96c56a2c299134fa9209614c25a085974a7306317ebc41a09b5302bacef54ac8",
    ],
  ] as const)(
    "the bytes of a .docx export don't change either: %s",
    async (_, content, options, expected) => {
      const { core, mind, client, tides } = await setUp();
      if (content === "sample") writeSample(client, tides);
      else writeEverythingElse(client, tides);
      await client.settled();
      if ("language" in options)
        await core.updateSettings({ user: { language: options.language } });

      const exported = await core.exportMind(mind.id, {
        format: "docx",
        includeQuestions: "includeQuestions" in options,
      });

      expect(zipFingerprint(exported.data)).toBe(expected);
    },
  );

  test("to Markdown: Notes and Answers as Markdown, the Question marked, math kept, and a footnote per Citation", async () => {
    const { core, mind, client, tides } = await setUp();
    writeSample(client, tides);
    await client.settled();

    const exported = await core.exportMind(mind.id, { format: "markdown" });

    expect(decode(exported)).toBe(
      [
        "# Tides",
        "",
        "# Spring tides",
        "",
        "**Neap** tides are *weaker*, at \\$0.",
        "",
        "Kept out of Questions, still exported.",
        "",
        "- High water",
        "- Low water",
        "",
        "1. New moon",
        "2. Full moon",
        "   - Twice a month",
        "",
        "```js",
        'console.log("tide");',
        "```",
        "",
        "$$",
        "F = G\\frac{m_1 m_2}{r^2}",
        "$$",
        "",
        "**Question:** When do spring tides happen?",
        "",
        "At new moon[^1] and at full moon[^2].",
        "",
        "The pull falls off as $1/r^3$[^3].",
        "",
        "[^1]: Tides, p. 2",
        "[^2]: Tides, p. 2 [unverified]",
        "[^3]: Tides, p. 3–4 [unverified]",
        "",
      ].join("\n"),
    );
    expect(exported).toMatchObject({
      fileName: "Tides.md",
      citations: 3,
      unverifiedCitations: 2,
      questions: 1,
    });
  });

  test("the preview counts the unverified Citations before anything is written", async () => {
    const { core, mind, client, tides } = await setUp("Tides: spring/neap?");
    writeSample(client, tides);
    await client.settled();

    expect(await core.previewMindExport(mind.id, { format: "docx" })).toEqual({
      fileName: "Tides spring neap.docx",
      citations: 3,
      unverifiedCitations: 2,
      questions: 1,
    });
  });

  test("to .docx: headings, lists, code and math, the Question left out, and each Citation a Word footnote", async () => {
    const { core, mind, client, tides } = await setUp();
    writeSample(client, tides);
    await client.settled();

    const exported = await core.exportMind(mind.id, { format: "docx" });
    expect(exported).toMatchObject({
      fileName: "Tides.docx",
      citations: 3,
      unverifiedCitations: 2,
    });
    const { files, document, footnotes, numbering } = docxParts(exported);
    expect([...files.keys()]).toEqual(
      expect.arrayContaining([
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "word/styles.xml",
        "word/numbering.xml",
        "word/footnotes.xml",
      ]),
    );
    expect(part(files, "[Content_Types].xml")).toContain(
      'PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"',
    );

    // Questions are left out by default; the Note switched out of Question context is in.
    expect(paragraphsOf(document)).toEqual([
      { style: "Title", text: "Tides" },
      { style: "Heading1", text: "Spring tides" },
      { text: "Neap tides are weaker, at $0." },
      { text: "Kept out of Questions, still exported." },
      { style: "ListParagraph", list: "1/0", text: "High water" },
      { style: "ListParagraph", list: "1/0", text: "Low water" },
      { style: "ListParagraph", list: "2/0", text: "New moon" },
      { style: "ListParagraph", list: "2/0", text: "Full moon" },
      { style: "ListParagraph", list: "3/1", text: "Twice a month" },
      { style: "Code", text: 'console.log("tide");' },
      // The maths is in Word's equations, not text (below).
      { style: "Math", text: "" },
      { text: "At new moon[1] and at full moon[2]." },
      { text: "The pull falls off as [3]." },
    ]);
    // Maths is Word's own equations: a display equation in its paragraph, and inline.
    expect(document).toContain(
      'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"',
    );
    const display =
      /<w:p><w:pPr><w:pStyle w:val="Math"\/><\/w:pPr><m:oMathPara>(.*?)<\/m:oMathPara><\/w:p>/.exec(
        document,
      )?.[1];
    expect(display).toMatch(/^<m:oMath>.*<\/m:oMath>$/);
    expect(display).toContain("<m:f><m:num>");
    expect(mathText(display ?? "")).toBe("F=Gm1m2r2");
    const inline = /falls off as <\/w:t><\/w:r>(<m:oMath>.*?<\/m:oMath>)/.exec(document)?.[1];
    expect(mathText(inline ?? "")).toBe("1/r3");
    expect(inline).toContain("<m:sSup>");
    expect(document).not.toContain("$$");
    expect(document).not.toContain("$1/r^3$");
    expect(document).not.toContain("When do spring tides happen?");
    expect(document).toContain(
      '<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">Neap</w:t></w:r>',
    );
    expect(document).toContain(
      '<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">weaker</w:t></w:r>',
    );

    // Real Word footnotes, one per Citation even when two cite the same page.
    expect(footnotesOf(footnotes)).toEqual({
      "1": "Tides, p. 2",
      "2": "Tides, p. 2 [unverified]",
      "3": "Tides, p. 3–4 [unverified]",
    });
    expect(footnotes).toContain('<w:footnote w:type="separator" w:id="-1">');
    expect(part(files, "word/settings.xml")).toContain(
      '<w:footnotePr><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr>',
    );

    // Each list numbers itself: bullets, numbers from 1, and bullets again a level down.
    expect(
      [...numbering.matchAll(/<w:num w:numId="\d+">[\s\S]*?<\/w:num>/g)].map((m) => m[0]),
    ).toEqual([
      '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>',
      '<w:num w:numId="2"><w:abstractNumId w:val="1"/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>',
      '<w:num w:numId="3"><w:abstractNumId w:val="0"/></w:num>',
    ]);
    // Headings are Word's own heading styles, so Word's navigation finds them.
    expect(part(files, "word/styles.xml")).toContain(
      '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>',
    );
  });

  test("highlighted text is ==marked== in Markdown and highlighted in Word", async () => {
    const { core, mind, client } = await setUp();
    writeMind(client, [
      paragraph(
        text("Spring tides are "),
        text("strongest", "highlight"),
        text(" at "),
        text("new moon", "bold", "highlight"),
        text("."),
      ),
      // A "==" that isn't a highlight stays text; so does a highlight that starts with "=".
      paragraph(text("In code, a == b; "), text("= b", "highlight")),
    ]);
    await client.settled();

    const markdown = decode(await core.exportMind(mind.id, { format: "markdown" }));
    expect(markdown).toBe(
      [
        "# Tides",
        "",
        "Spring tides are ==strongest== at **==new moon==**.",
        "",
        "In code, a \\== b; ==\\= b==",
        "",
      ].join("\n"),
    );

    const { document } = docxParts(await core.exportMind(mind.id, { format: "docx" }));
    expect(paragraphsOf(document)).toEqual([
      { style: "Title", text: "Tides" },
      { text: "Spring tides are strongest at new moon." },
      { text: "In code, a == b; = b" },
    ]);
    // Word's own highlighting, which its Text Highlight Color button shows and removes.
    expect(document).toContain(
      '<w:r><w:rPr><w:highlight w:val="yellow"/></w:rPr><w:t xml:space="preserve">strongest</w:t></w:r>',
    );
    expect(document).toContain(
      '<w:r><w:rPr><w:b/><w:highlight w:val="yellow"/></w:rPr><w:t xml:space="preserve">new moon</w:t></w:r>',
    );
    expect(document.match(/<w:highlight /g)).toHaveLength(3);
  });

  test("to .docx with Questions: they are put back, marked as Questions", async () => {
    const { core, mind, client, tides } = await setUp();
    writeSample(client, tides);
    await client.settled();

    const exported = await core.exportMind(mind.id, { format: "docx", includeQuestions: true });

    const paragraphs = paragraphsOf(docxParts(exported).document);
    expect(paragraphs.filter((each) => each.style === "Question")).toEqual([
      { style: "Question", text: "Question: When do spring tides happen?" },
    ]);
    const at = paragraphs.findIndex((each) => each.style === "Question");
    expect(paragraphs[at + 1]?.text).toBe("At new moon[1] and at full moon[2].");
  });

  test("Markdown can leave Questions out too", async () => {
    const { core, mind, client } = await setUp("");
    writeMind(client, [note("A Note."), question("A Question?")]);
    await client.settled();

    const exported = await core.exportMind(mind.id, {
      format: "markdown",
      includeQuestions: false,
    });
    expect(decode(exported)).toBe("A Note.\n");
    expect(exported).toMatchObject({ fileName: "Untitled.md", citations: 0, questions: 1 });
  });

  test("a Citation whose Document has been deleted is unverified, whatever its check said", async () => {
    const { core, mind, client, tides } = await setUp();
    writeMind(client, [paragraph(text("Spring tides"), citation(tides, [2, 2], "found"))]);
    await client.settled();
    expect(await core.previewMindExport(mind.id, { format: "docx" })).toMatchObject({
      citations: 1,
      unverifiedCitations: 0,
    });

    await core.deleteDocument(tides.id);

    expect(await core.previewMindExport(mind.id, { format: "docx" })).toMatchObject({
      citations: 1,
      unverifiedCitations: 1,
    });
    const exported = await core.exportMind(mind.id, { format: "markdown" });
    expect(decode(exported)).toContain("[^1]: Tides, p. 2 [unverified]");
  });

  test("a Citation of a Document without pages names only the Document", async () => {
    const { core, mind, client, tides } = await setUp("");
    writeMind(client, [paragraph(text("Spring tides."), citation(tides, null, "found"))]);
    await client.settled();

    const exported = await core.exportMind(mind.id, { format: "docx" });
    expect(footnotesOf(docxParts(exported).footnotes)).toEqual({ "1": "Tides" });
  });

  test("tables become Word tables and GitHub tables, and images are embedded", async () => {
    const { core, mind, client } = await setUp("");
    // The editor has no tables or images yet; they are written as Tiptap's would be.
    client.doc.transact(() => {
      client.blocks.push([
        element("table", {}, [
          element("tableRow", {}, [
            element("tableHeader", {}, [element("paragraph", {}, ["Tide"])]),
            element("tableHeader", {}, [element("paragraph", {}, ["Height"])]),
          ]),
          element("tableRow", {}, [
            element("tableCell", {}, [element("paragraph", {}, ["Spring"])]),
            element("tableCell", {}, [element("paragraph", {}, ["4 m | high"])]),
          ]),
        ]),
        element("image", { src: PIXEL, alt: "A chart" }, []),
      ]);
    });
    await client.settled();

    const markdown = decode(await core.exportMind(mind.id, { format: "markdown" }));
    expect(markdown).toBe(
      [
        "| Tide | Height |",
        "| --- | --- |",
        "| Spring | 4 m \\| high |",
        "",
        `![A chart](${PIXEL})`,
        "",
      ].join("\n"),
    );

    const { files, document, relationships } = docxParts(
      await core.exportMind(mind.id, { format: "docx" }),
    );
    const table = /<w:tbl>[\s\S]*<\/w:tbl>/.exec(document)?.[0] ?? "";
    expect(table).toContain("<w:trPr><w:tblHeader/></w:trPr>");
    expect(
      [...table.matchAll(/<w:tc>([\s\S]*?)<\/w:tc>/g)].map((cell) => textOf(cell[1] as string)),
    ).toEqual(["Tide", "Height", "Spring", "4 m | high"]);
    expect(files.get("word/media/image1.png")).toEqual(
      Buffer.from(PIXEL.split(",")[1] as string, "base64"),
    );
    expect(relationships).toContain(
      'Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"',
    );
    expect(document).toContain('<a:blip r:embed="rIdImage1"/>');
    // One pixel at 96 dpi.
    expect(document).toContain('<wp:extent cx="9525" cy="9525"/>');
  });

  test("its own words are in the interface language", async () => {
    const { core, mind, client, tides } = await setUp();
    await core.updateSettings({ user: { language: "zh-CN" } });
    writeMind(client, [
      question("大潮什么时候出现？"),
      paragraph(
        text("新月时"),
        citation(tides, [2, 3], "not-found", "quote-not-on-pages"),
        text("。"),
      ),
    ]);
    await client.settled();

    expect(decode(await core.exportMind(mind.id, { format: "markdown" }))).toBe(
      [
        "# Tides",
        "",
        "**问题：** 大潮什么时候出现？",
        "",
        "新月时[^1]。",
        "",
        "[^1]: Tides，第 2–3 页 [未核实]",
        "",
      ].join("\n"),
    );
  });

  test("a footnote names its Citation's Location: a slide, rows of a sheet, a section, lines", async () => {
    const { core, mind, client, tides } = await setUp();
    const at = (location: object, name: string): JSONContent => {
      const node = citation(tides, [1, 1], "found");
      return { ...node, attrs: { ...node.attrs, documentName: name, location } };
    };
    writeMind(client, [
      paragraph(
        text("Growth"),
        at({ kind: "slide", from: 3, to: 3 }, "Deck"),
        text(", revenue"),
        at({ kind: "rows", sheet: "Revenue", from: 12, to: 14 }, "Model"),
        text(", the crest"),
        at({ kind: "section", heading: "2.1 Sensitivity" }, "Review"),
        text(", the log"),
        at({ kind: "lines", from: 120, to: 134 }, "Log"),
        text("."),
      ),
    ]);
    await client.settled();

    expect(decode(await core.exportMind(mind.id, { format: "markdown" }))).toContain(
      [
        "[^1]: Deck, slide 3",
        "[^2]: Model, Revenue, rows 12–14",
        "[^3]: Review, § 2.1 Sensitivity",
        "[^4]: Log, lines 120–134",
      ].join("\n"),
    );
    const docx = unzip((await core.exportMind(mind.id, { format: "docx" })).data);
    expect(footnotesOf(part(docx, "word/footnotes.xml"))).toEqual({
      "1": "Deck, slide 3",
      "2": "Model, Revenue, rows 12–14",
      "3": "Review, § 2.1 Sensitivity",
      "4": "Log, lines 120–134",
    });
    await core.updateSettings({ user: { language: "zh-CN" } });
    expect(decode(await core.exportMind(mind.id, { format: "markdown" }))).toContain(
      [
        "[^1]: Deck，第 3 张幻灯片",
        "[^2]: Model，Revenue，第 12–14 行",
        "[^3]: Review，§ 2.1 Sensitivity",
        "[^4]: Log，第 120–134 行",
      ].join("\n"),
    );
  });

  test("maths Word's equations can't hold stays LaTeX text in a .docx that still opens; Markdown keeps $…$", async () => {
    const { core, mind, client } = await setUp();
    writeMind(client, [
      { type: "blockMath", attrs: { latex: "\\undefinedmacro{x}" } },
      paragraph(
        text("Linked "),
        { type: "inlineMath", attrs: { latex: "\\href{https://example.com}{x}" } },
        text(", and fine "),
        { type: "inlineMath", attrs: { latex: "E = mc^2" } },
        text("."),
      ),
    ]);
    await client.settled();

    const { document } = docxParts(await core.exportMind(mind.id, { format: "docx" }));
    expect(paragraphsOf(document)).toEqual([
      { style: "Title", text: "Tides" },
      { style: "Math", text: "$$\\undefinedmacro{x}$$" },
      { text: "Linked $\\href{https://example.com}{x}$, and fine ." },
    ]);
    expect(document.match(/<m:oMath>/g)).toHaveLength(1);
    expect(document).not.toContain("<m:oMathPara>");

    expect(decode(await core.exportMind(mind.id, { format: "markdown" }))).toContain(
      "Linked $\\href{https://example.com}{x}$, and fine $E = mc^2$.",
    );
  });

  test("refuses unknown formats and Minds", async () => {
    const { core, mind } = await setUp();
    await expect(
      core.exportMind(mind.id, { format: "pdf" } as unknown as { format: "docx" }),
    ).rejects.toThrow(InvalidInputError);
    await expect(
      core.exportMind(mind.id, { format: "docx", includeQuestions: "yes" } as unknown as {
        format: "docx";
      }),
    ).rejects.toThrow(InvalidInputError);
    await core.deleteMind(mind.id);
    await expect(core.previewMindExport(mind.id, { format: "docx" })).rejects.toThrow(
      NotFoundError,
    );
  });
});

/** A 1×1 PNG. */
const PIXEL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/** A node as Tiptap's Yjs binding stores it; strings are its text. */
function element(
  type: string,
  attributes: Record<string, string>,
  children: (Y.XmlElement | string)[],
): Y.XmlElement {
  const node = new Y.XmlElement(type);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  node.insert(
    0,
    children.map((child) => (typeof child === "string" ? new Y.XmlText(child) : child)),
  );
  return node;
}
