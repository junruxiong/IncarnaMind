/**
 * Exporting a Mind imported from a Word file back into that file (#105),
 * through the core's public interface: what the Mind didn't change is byte
 * for byte what the file had, edits are written into the paragraphs they were
 * made in with their formatting kept, new Blocks take the document's own
 * styles, lists and footnotes, and what can't be kept is said before anything
 * is saved.
 */
import { readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { Core, Document } from "../../src/core";
import { wordFileOf, wordImportProgressOf } from "../../src/shared/wordFiles";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { textOf, unzip } from "../helpers/docx";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { editMind } from "../helpers/minds";
import {
  commented,
  field,
  NUMBERING,
  PNG,
  paragraph,
  run,
  table,
  tracked,
  type WordFileInput,
  wordFile,
} from "../helpers/word";
import { buildZip } from "../helpers/zip";

async function importFile(core: Core, bytes: Buffer, origin: "own" | "someone-else" = "own") {
  const folder = await createTempDataFolder();
  const path = join(folder, "Memo.docx");
  await writeFile(path, bytes);
  const result = await core.importWordFile({ path, origin });
  if (!result.ok) throw new Error(`Refused: ${result.refusal}`);
  const client = await connectToMind(core, result.mind.id);
  await new Promise<void>((resolve) => {
    const check = () => {
      if (wordFileOf(client.doc) && wordImportProgressOf(client.doc) === null) {
        client.doc.off("update", check);
        resolve();
      }
    };
    client.doc.on("update", check);
    check();
  });
  return { mindId: result.mind.id, client };
}

async function setUp(input: WordFileInput, origin: "own" | "someone-else" = "own") {
  const core = startCore(await createTempDataFolder());
  const bytes = wordFile(input);
  return { core, bytes, ...(await importFile(core, bytes, origin)) };
}

async function exported(core: Core, mindId: string) {
  const result = await core.exportMind(mindId, { format: "docx" });
  const files = unzip(result.data);
  const part = (name: string) => files.get(name)?.toString("utf8") ?? "";
  return { files, word: result.word, document: part("word/document.xml"), part };
}

/** The entries an export changed, added or left out. */
function changed(before: Map<string, Buffer>, after: Map<string, Buffer>): string[] {
  const names = new Set([...before.keys(), ...after.keys()]);
  return [...names]
    .filter((name) => !before.get(name)?.equals(after.get(name) ?? Buffer.alloc(0)))
    .sort();
}

/** Changes the text of every text node in the Mind. */
function editText(client: MindClient, change: (text: string) => string): void {
  const visit = (node: JSONContent): JSONContent => ({
    ...node,
    ...(node.text !== undefined && { text: change(node.text) }),
    ...(node.content && { content: node.content.map(visit) }),
  });
  editMind(client, (blocks) => blocks.map(visit));
}

const text = (value: string, ...marks: string[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks: marks.map((type) => ({ type })) }),
});

const item = (value: string): JSONContent => ({
  type: "listItem",
  content: [{ type: "paragraph", content: [text(value)] }],
});

/** A Citation of the Document, its quote found. */
const citation = (document: Document): JSONContent => ({
  type: "citation",
  attrs: {
    passageId: "p1",
    documentId: document.id,
    documentName: "Tides",
    contentHash: document.contentHash,
    quote: "Spring tides happen at new moon.",
    check: "found",
  },
});

async function tidesDocument(core: Core): Promise<Document> {
  const sources = await createTempDataFolder();
  const path = await writeSourceFile(sources, "Tides.txt", "Spring tides happen at new moon.");
  const [tides] = await addAndProcess(core, [path]);
  return tides as Document;
}

/** The body's paragraphs as written, in order. */
const paragraphsIn = (xml: string) =>
  [...xml.matchAll(/<w:p[ >][\s\S]*?<\/w:p>|<w:p\/>/g)].map((match) => match[0]);

describe("what the Mind didn't change", () => {
  test("is byte for byte the file's, every entry, when nothing changed", async () => {
    const { core, bytes, mindId } = await setUp({
      body:
        paragraph("Heading", { style: "Heading1" }) +
        paragraph(`${run("Plain and ")}${run("bold", { b: true })}`) +
        paragraph("First item", { numId: 1 }) +
        table([
          ["A", "B"],
          ["C", "D"],
        ]),
      numbering: NUMBERING,
    });
    const { files, word } = await exported(core, mindId);
    expect(changed(unzip(bytes), files)).toEqual([]);
    expect(word).toEqual({ notKept: [], unknownParts: [], keptFurniture: [] });
  });

  test("a part IncarnaMind doesn't know is copied as it is, and listed", async () => {
    const plain = unzip(wordFile({ body: paragraph("Text.") }));
    const types = plain.get("[Content_Types].xml")?.toString("utf8") ?? "";
    const { core, mindId, client } = await setUp({
      body: paragraph("Text."),
      parts: [
        { name: "vendor/settings.vendor", data: "<vendor>kept</vendor>" },
        {
          name: "[Content_Types].xml",
          data: types.replace(
            "</Types>",
            '<Default Extension="vendor" ContentType="application/vnd.example.settings+xml"/></Types>',
          ),
        },
      ],
    });
    editText(client, (value) => value.replace("Text.", "Text, edited."));
    await client.settled();
    const { files, word } = await exported(core, mindId);
    expect(word?.unknownParts).toEqual(["vendor/settings.vendor"]);
    expect(files.get("vendor/settings.vendor")?.toString("utf8")).toBe("<vendor>kept</vendor>");
  });
});

describe("an edited paragraph", () => {
  test("keeps its own XML: a changed word takes its run's formatting, the other runs and paragraphs stay", async () => {
    const first = paragraph(
      `${run("The rent is ", { font: "Georgia" })}${run("due", { b: true, font: "Georgia" })}${run(" monthly.", { font: "Georgia" })}`,
      { paraId: "1A2B3C4D" },
    );
    const second = paragraph("Unchanged paragraph.", { paraId: "5E6F7A8B" });
    const { core, bytes, mindId, client } = await setUp({ body: first + second });
    editText(client, (value) => (value === " monthly." ? " every month." : value));
    await client.settled();

    const { files, document } = await exported(core, mindId);
    expect(changed(unzip(bytes), files)).toEqual(["word/document.xml"]);
    expect(document).toContain(second);
    expect(document).toContain(
      `<w:p w14:paraId="1A2B3C4D">${run("The rent is ", { font: "Georgia" })}${run("due", { b: true, font: "Georgia" })}`,
    );
    expect(document).toContain(
      '<w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/></w:rPr><w:t xml:space="preserve"> every month.</w:t>',
    );
  });

  test("a word made bold, or no longer bold, changes only that toggle: the font stays", async () => {
    const { core, mindId, client } = await setUp({
      body: paragraph(
        `${run("Keep ", { font: "Georgia" })}${run("this", { b: true, font: "Georgia" })}`,
      ),
    });
    editMind(client, ([first]) => [{ ...first, content: [text("Keep", "bold"), text(" this")] }]);
    await client.settled();
    const { document } = await exported(core, mindId);
    expect(document).toContain(
      '<w:r><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">Keep</w:t></w:r>',
    );
    expect(document).toContain(
      '<w:r><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/></w:rPr><w:t xml:space="preserve"> this</w:t></w:r>',
    );
  });

  test("what the import kept as is goes back as it was: a field, a footnote mark, someone's tracked change", async () => {
    const date = field('DATE \\@ "d MMMM yyyy"', "12 October 2026");
    const change = tracked("ins", 7, "Lars", " (draft)");
    const footnoteMark =
      '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="1"/></w:r>';
    const { core, mindId, client } = await setUp(
      {
        body: paragraph(
          `${run("Dated ")}${date}${run(" and noted")}${footnoteMark}${change}${run(".")}`,
        ),
        footnotes: `<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r>${run(" A note.")}</w:p></w:footnote>`,
      },
      "someone-else",
    );
    editText(client, (value) => value.replace("noted", "signed"));
    await client.settled();
    const { document } = await exported(core, mindId);
    expect(document).toContain(date);
    expect(document).toContain(change);
    expect(document).toContain(footnoteMark);
    expect(document).toContain("signed");
    expect(document).not.toContain("noted");
  });

  test("a heading's new level takes the document's own heading style", async () => {
    const { core, mindId, client } = await setUp({
      body: paragraph("Scope", { style: "Heading1" }) + paragraph("Body."),
    });
    editMind(client, ([heading, ...rest]) => [
      { ...heading, attrs: { ...heading?.attrs, level: 2 } },
      ...rest,
    ]);
    await client.settled();
    const { document } = await exported(core, mindId);
    expect(paragraphsIn(document)[0]).toBe(
      `<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>${run("Scope")}</w:p>`,
    );
  });

  test("a link added to text is a hyperlink with a relationship of its own", async () => {
    const { core, mindId, client, bytes } = await setUp({ body: paragraph("See the site.") });
    editMind(client, ([first]) => [
      {
        ...first,
        content: [
          text("See "),
          {
            type: "text",
            text: "the site",
            marks: [{ type: "link", attrs: { href: "https://example.com/" } }],
          },
          text("."),
        ],
      },
    ]);
    await client.settled();
    const { document, part, files } = await exported(core, mindId);
    const id = /<w:hyperlink r:id="([^"]+)" w:history="1">/.exec(document)?.[1];
    expect(id).toBeDefined();
    expect(part("word/_rels/document.xml.rels")).toContain(
      `<Relationship Id="${id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/" TargetMode="External"/>`,
    );
    // The link's character style is added to the document's styles.
    expect(changed(unzip(bytes), files)).toEqual([
      "word/_rels/document.xml.rels",
      "word/document.xml",
      "word/styles.xml",
    ]);
  });

  test("a change still waiting to be accepted isn't written: the paragraph stays as it was", async () => {
    const { core, mindId, client, bytes } = await setUp({
      body: paragraph("Rent is due monthly."),
    });
    editMind(client, ([first]) => [
      {
        ...first,
        content: [
          text("Rent is due "),
          {
            type: "text",
            text: "weekly ",
            marks: [{ type: "proposalInsert", attrs: { proposalId: "p", requestId: "r" } }],
          },
          text("monthly."),
        ],
      },
    ]);
    await client.settled();
    const { files } = await exported(core, mindId);
    expect(changed(unzip(bytes), files)).toEqual([]);
  });
});

describe("Blocks added, deleted and moved", () => {
  test("a new paragraph with a Citation goes after the one before it, in its look; the Citation is a footnote", async () => {
    const { core, mindId, client } = await setUp({
      body: paragraph(run("Intro.", { font: "Georgia" })) + paragraph("Outro."),
    });
    const tides = await tidesDocument(core);
    editMind(client, ([first, last]) => [
      first as JSONContent,
      { type: "paragraph", content: [text("Tides are strongest at new moon. "), citation(tides)] },
      last as JSONContent,
    ]);
    await client.settled();
    const { document, part } = await exported(core, mindId);
    // As if Enter were pressed at the end of the paragraph before: its text's look.
    expect(paragraphsIn(document)[1]).toBe(
      '<w:p><w:r><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/></w:rPr><w:t xml:space="preserve">Tides are strongest at new moon.</w:t></w:r><w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="1"/></w:r></w:p>',
    );
    expect(part("word/footnotes.xml")).toContain("Tides");
    const styles = part("word/styles.xml");
    expect(styles).toContain('w:styleId="FootnoteText"');
    expect(styles).toContain('w:styleId="FootnoteReference"');
    expect(part("[Content_Types].xml")).toContain(
      '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>',
    );
  });

  test("in a citation style, a new Citation is written in the text and the Bibliography goes at the end", async () => {
    const { core, mindId, client } = await setUp({ body: paragraph("Intro.") });
    const tides = await tidesDocument(core);
    await core.setMindCitationStyle(mindId, "apa");
    editMind(client, (blocks) => [
      ...blocks,
      {
        type: "paragraph",
        content: [text("Tides are strongest at new moon "), citation(tides), text(".")],
      },
    ]);
    await client.settled();
    const { document, files } = await exported(core, mindId);
    const paragraphs = paragraphsIn(document);
    expect(textOf(paragraphs[1] ?? "")).toBe("Tides are strongest at new moon (Tides, n.d.).");
    expect(textOf(paragraphs.at(-2) ?? "")).toBe("References");
    expect(textOf(paragraphs.at(-1) ?? "")).toContain("Tides");
    expect(document).toMatch(/<\/w:p><w:sectPr>/);
    expect(files.has("word/footnotes.xml")).toBe(false);
  });

  test("a deleted paragraph goes, and a moved one moves as it was written", async () => {
    const one = paragraph("One.", { paraId: "00000001" });
    const two = paragraph("Two.", { paraId: "00000002" });
    const three = paragraph("Three.", { paraId: "00000003" });
    const { core, mindId, client } = await setUp({ body: one + two + three });
    editMind(client, ([first, , third]) => [third as JSONContent, first as JSONContent]);
    await client.settled();
    const { document } = await exported(core, mindId);
    expect(document).toContain(`<w:body>${three}${one}<w:sectPr>`);
  });

  test("a new item in a list goes on with the list's own numbering", async () => {
    const { core, mindId, client } = await setUp({
      body: paragraph("First", { numId: 2 }) + paragraph("Second", { numId: 2 }),
      numbering: NUMBERING,
    });
    editMind(client, ([list]) => [{ ...list, content: [...(list?.content ?? []), item("Third")] }]);
    await client.settled();
    const { document } = await exported(core, mindId);
    expect(paragraphsIn(document).at(-1)).toBe(
      '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr><w:r><w:t xml:space="preserve">Third</w:t></w:r></w:p>',
    );
  });

  test("a new list is a list of its own in the document's numbering, after those it has", async () => {
    const { core, mindId, client, bytes } = await setUp({
      body: paragraph("Text."),
      numbering: NUMBERING,
    });
    editMind(client, (blocks) => [
      ...blocks,
      { type: "orderedList", attrs: { start: 1 }, content: [item("New")] },
    ]);
    await client.settled();
    const { document, part } = await exported(core, mindId);
    expect(document).toContain('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="3"/></w:numPr>');
    const numbering = part("word/numbering.xml");
    const original = unzip(bytes).get("word/numbering.xml")?.toString("utf8") ?? "";
    // The document's own lists are as they were: the new definition follows its definitions, the new list its lists.
    expect(numbering).toContain(
      original.slice(original.indexOf("<w:abstractNum "), original.indexOf("<w:num ")),
    );
    expect(numbering.indexOf('w:abstractNumId="2"')).toBeLessThan(
      numbering.indexOf('<w:num w:numId="1">'),
    );
    expect(numbering).toContain('<w:num w:numId="3"><w:abstractNumId w:val="2"/>');
  });

  test("a table is written into cell by cell; a new row writes it again, its properties kept", async () => {
    const { core, mindId, client } = await setUp({
      body: table([
        ["A", "B"],
        ["C", "D"],
      ]),
    });
    editText(client, (value) => (value === "D" ? "D, edited" : value));
    await client.settled();
    let { document } = await exported(core, mindId);
    expect(document).toContain(`${paragraph("A")}</w:tc><w:tc>${paragraph("B")}`);
    expect(document).toContain("D, edited");

    editMind(client, ([grid, ...rest]) => [
      {
        ...grid,
        content: [
          ...(grid?.content ?? []),
          {
            type: "tableRow",
            content: ["E", "F"].map((cell) => ({
              type: "tableCell",
              content: [{ type: "paragraph", content: [text(cell)] }],
            })),
          },
        ],
      },
      ...rest,
    ]);
    await client.settled();
    ({ document } = await exported(core, mindId));
    expect(document).toContain('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/></w:tblPr>');
    expect([...document.matchAll(/<w:tr>/g)]).toHaveLength(3);
    expect(document).toMatch(/E<\/w:t>[\s\S]*F<\/w:t>/);
  });

  test("a new picture is a media part of its own, with its relationship, content type and a drawing id the document hasn't used", async () => {
    const { core, mindId, client } = await setUp({
      body: paragraph(
        '<w:r><w:drawing><wp:anchor><wp:extent cx="1" cy="1"/><wp:docPr id="7" name="Existing"/></wp:anchor></w:drawing></w:r>',
      ),
    });
    editMind(client, (blocks) => [
      ...blocks,
      {
        type: "paragraph",
        content: [
          {
            type: "image",
            attrs: { src: `data:image/png;base64,${PNG.toString("base64")}`, alt: "A dot" },
          },
        ],
      },
    ]);
    await client.settled();
    const { files, part, document } = await exported(core, mindId);
    expect(files.get("word/media/image1.png")?.equals(PNG)).toBe(true);
    expect(part("[Content_Types].xml")).toContain(
      '<Default Extension="png" ContentType="image/png"/>',
    );
    expect(document).toContain('<wp:docPr id="1" name="Picture 1" descr="A dot"/>');
    expect([...document.matchAll(/<wp:docPr id="7"/g)]).toHaveLength(1);
  });
});

describe("what can't be kept", () => {
  test("someone's comment on a paragraph the User deleted is said before saving, and goes only once they say so", async () => {
    const { inline, comment } = commented(
      5,
      "Lars Eklund",
      "Check duty rates first",
      run("Duty adds 12%."),
    );
    const { core, mindId, client } = await setUp(
      { body: paragraph("Keep.") + paragraph(inline), comments: comment },
      "someone-else",
    );
    const output = await createTempDataFolder();
    await core.setOutputLocation(output);
    editMind(client, (blocks) => blocks.slice(0, 1));
    await client.settled();

    const review = await core.saveMindExport(mindId, { format: "docx" });
    expect(review).toEqual({
      status: "review",
      word: {
        notKept: [
          {
            key: expect.stringMatching(/^comment:/),
            kind: "comment",
            author: "Lars Eklund",
            text: "Check duty rates first",
          },
        ],
        unknownParts: [],
        keptFurniture: [],
      },
    });
    expect(await readdir(output)).toEqual([]);

    if (review.status !== "review") throw new Error("Not a review.");
    const saved = await core.saveMindExport(mindId, {
      format: "docx",
      accept: review.word.notKept.map((loss) => loss.key),
    });
    expect(saved).toMatchObject({ status: "saved", file: { name: "Memo.docx" } });
    const { document } = await exported(core, mindId);
    expect(document).not.toContain("comment");
  });

  test("a deleted header stays in the file, and is listed", async () => {
    const { core, mindId, client, bytes } = await setUp({
      body: paragraph("Body."),
      headers: { rIdH1: paragraph("Confidential") },
      sectPr: '<w:headerReference w:type="default" r:id="rIdH1"/><w:pgSz w:w="11906" w:h="16838"/>',
    });
    editMind(client, (blocks) => blocks.filter((block) => block.type !== "wordBlock"));
    await client.settled();
    const { files, word } = await exported(core, mindId);
    expect(word?.keptFurniture).toEqual(["header"]);
    expect(changed(unzip(bytes), files)).toEqual([]);
  });

  test("a part that can't be written stops the export, naming it, and nothing is saved", async () => {
    const files = unzip(wordFile({ body: paragraph("Cafe.") }));
    const original = files.get("word/document.xml") ?? Buffer.alloc(0);
    // A byte that isn't UTF-8 in the main part: its text can't be written back without changing it.
    const at = original.indexOf("Cafe") + 3;
    const broken = Buffer.concat([
      original.subarray(0, at),
      Buffer.from([0xe9]),
      original.subarray(at + 1),
    ]);
    const bytes = buildZip(
      [...files].map(([name, data]) => ({
        name,
        data: name === "word/document.xml" ? broken : data,
      })),
    );
    const core = startCore(await createTempDataFolder());
    const { mindId, client } = await importFile(core, bytes);
    const output = await createTempDataFolder();
    await core.setOutputLocation(output);
    editText(client, (value) => `${value} Edited.`);
    await client.settled();
    await expect(core.saveMindExport(mindId, { format: "docx" })).rejects.toThrow(
      "word/document.xml",
    );
    expect(await readdir(output)).toEqual([]);
  });
});

describe("Word's rules", () => {
  test("a new footnote's id is one the document hasn't used, below 0x80000000; the document's own start tag stays", async () => {
    const { core, mindId, client, bytes } = await setUp({
      body: paragraph(
        'Text.<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="1"/></w:r>',
      ),
      footnotes: `<w:footnote w:id="1"><w:p>${run("Old.")}</w:p></w:footnote>`,
    });
    const tides = await tidesDocument(core);
    editMind(client, (blocks) => [
      ...blocks,
      { type: "paragraph", content: [text("New."), citation(tides)] },
    ]);
    await client.settled();
    const { document, part } = await exported(core, mindId);
    const ids = [...part("word/footnotes.xml").matchAll(/<w:footnote\b[^>]*w:id="(-?\d+)"/g)].map(
      (match) => Number(match[1]),
    );
    expect(ids).toEqual([-1, 0, 1, 2]);
    const original = unzip(bytes).get("word/document.xml")?.toString("utf8") ?? "";
    const head = (xml: string) => xml.slice(0, xml.indexOf("<w:body>"));
    expect(head(document)).toBe(head(original));
  });
});
