/**
 * Words the User selects in a Document's viewer are cited where they were
 * selected (#101): checked in those pages only, never moved to the same words
 * on another page, and without the running header and footer a selection
 * across a page break takes in.
 */
import { describe, expect, test } from "vitest";
import type { Core, Document } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { buildPdf, type PdfPage } from "../helpers/pdf";

const HEADER = "Ferrand and Moss, Rate cuts and retail margins";
const REPEATED = "Margins recover within two quarters of each cut.";

/** Pages with a running header and a page number, as a paper prints them. */
const paper = (bodies: readonly (readonly string[])[]) =>
  buildPdf(bodies.map((lines, index): PdfPage => ({ lines: [HEADER, ...lines, `${index + 1}`] })));

const BODIES = [
  ["Grocers studied across the eleven chains.", REPEATED],
  ["Pooling the cuts, the recovery peaks in the second quarter.", REPEATED],
  [
    "The effect is larger for grocers that fund their stock",
    "at floating rates, as Table 3 shows.",
  ],
  ["Smaller chains recovered less, about 0.2 points.", "Only three of them are in the panel."],
];

async function addPaper(core: Core, bytes = paper(BODIES), name = "Rate cuts.pdf") {
  const [document] = await addAndProcess(core, [
    await writeSourceFile(await createSourceFolder(), name, bytes),
  ]);
  return document as Document;
}

describe("citeSelection", () => {
  test("cites the page the words were selected on, though the same words come first on another", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: 2, text: REPEATED, linesBefore: 2, linesAfter: 1 }],
    });

    expect(cited).toMatchObject({ check: "found", pageFrom: 2, pageTo: 2, quote: REPEATED });
    expect(cited.location).toEqual({ kind: "page", from: 2, to: 2 });
    // Asked to find the same words, the general check takes the first page that has them.
    const anywhere = await core.recheckCitation({
      documentId: document.id,
      quote: REPEATED,
      pageFrom: null,
      pageTo: null,
    });
    expect(anywhere.pageFrom).toBe(1);
  });

  // The page number "3" is in the page's text too, but not at its edge: it isn't taken for the words.
  test("across a page break, leaves out the footer and header selected with the words", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [
        {
          unit: 3,
          text: "at floating rates, as Table 3 shows.\n3",
          linesBefore: 2,
          linesAfter: 0,
        },
        {
          unit: 4,
          text: `${HEADER}\nSmaller chains recovered less,`,
          linesBefore: 0,
          linesAfter: 2,
        },
      ],
    });

    expect(cited).toMatchObject({
      check: "found",
      pageFrom: 3,
      pageTo: 4,
      quote: "at floating rates, as Table 3 shows. Smaller chains recovered less,",
    });
    expect(cited.location).toEqual({ kind: "page", from: 3, to: 4 });
  });

  test("a selection that reaches only the next page's header cites one page", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [
        {
          unit: 3,
          text: "The effect is larger for grocers that fund their stock\n",
          linesBefore: 1,
          linesAfter: 2,
        },
        { unit: 4, text: HEADER, linesBefore: 0, linesAfter: 3 },
      ],
    });

    expect(cited).toMatchObject({ check: "found", pageFrom: 3, pageTo: 3 });
    expect(cited.quote).toBe("The effect is larger for grocers that fund their stock");
  });

  test("leaves nothing out away from the page's edges", async () => {
    const core = startCore(await createTempDataFolder());
    const pages = [...BODIES.slice(0, 2), ["One", "Two", "Three", "Four", "Five", "Six", "Seven"]];
    const document = await addPaper(core, paper([...pages, ...BODIES.slice(2)]));

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: 3, text: "Five\nNot on the page", linesBefore: 5, linesAfter: 5 }],
    });

    expect(cited).toMatchObject({ check: "not-found", pageFrom: 3, pageTo: 3 });
    expect(cited.quote).toBe("Five Not on the page");
  });

  test("words not on the page they were selected on are not found there, and not looked for elsewhere", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: 4, text: "Grocers studied across the eleven chains." }],
    });

    expect(cited).toMatchObject({
      check: "not-found",
      checkReason: "quote-not-on-pages",
      pageFrom: 4,
      pageTo: 4,
    });
  });

  test("more than two pages can't be cited", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [
        { unit: 1, text: REPEATED, linesBefore: 2, linesAfter: 1 },
        { unit: 2, text: BODIES[1]?.join("\n") ?? "", linesBefore: 1, linesAfter: 1 },
        { unit: 3, text: "The effect is larger", linesBefore: 1, linesAfter: 2 },
      ],
    });

    expect(cited).toMatchObject({ check: "not-found", checkReason: "too-many-pages" });
  });

  test("Chinese selected across a line break reads as one line, and is found", async () => {
    const core = startCore(await createTempDataFolder());
    const bytes = buildPdf([
      { chineseLines: ["降息之后，零售商的毛利率", "在两个季度内回升。"] },
      { chineseLines: ["小型连锁店回升较少。"] },
    ]);
    const document = await addPaper(core, bytes, "降息.pdf");

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: 1, text: "零售商的毛利率\n在两个季度内回升", linesBefore: 0, linesAfter: 0 }],
    });

    expect(cited).toMatchObject({ check: "found", pageFrom: 1, pageTo: 1 });
    expect(cited.quote).toBe("零售商的毛利率在两个季度内回升");
  });

  test("a word split at a line's end reads whole", async () => {
    const core = startCore(await createTempDataFolder());
    const bytes = buildPdf([{ lines: ["The panel covers asset man-", "agement firms only."] }]);
    const document = await addPaper(core, bytes, "Panel.pdf");

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: 1, text: "covers asset man-\nagement firms", linesBefore: 0, linesAfter: 0 }],
    });

    expect(cited).toMatchObject({ check: "found", quote: "covers asset management firms" });
  });

  test("words on no known Unit are looked for in the whole Document", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: null, text: "Smaller chains recovered less" }],
    });

    expect(cited).toMatchObject({ check: "found", pageFrom: 4, pageTo: 4 });
  });

  test("a selection that names its Unit for some parts and not others isn't looked for elsewhere", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [
        { unit: 4, text: "Pooling the cuts," },
        { unit: null, text: "the recovery peaks in the second quarter." },
      ],
    });

    expect(cited).toMatchObject({ check: "not-found", pageFrom: null });
  });

  test("a Unit that isn't one, or nothing but spaces, is refused", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);
    const cite = (parts: unknown[]) =>
      core.citeSelection({ documentId: document.id, parts } as Parameters<
        typeof core.citeSelection
      >[0]);

    await expect(cite([{ unit: 0, text: REPEATED }])).rejects.toThrow("No text is selected.");
    await expect(cite([{ unit: 1.5, text: REPEATED }])).rejects.toThrow("No text is selected.");
    await expect(cite([{ unit: 2, text: " \n " }])).rejects.toThrow("No text is selected.");
    await expect(cite([])).rejects.toThrow("No text is selected.");
  });

  test("a deleted Document can't be checked", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await addPaper(core);
    await core.deleteDocuments([document.id]);

    const cited = await core.citeSelection({
      documentId: document.id,
      parts: [{ unit: 1, text: REPEATED }],
    });

    expect(cited).toMatchObject({ check: "cant-check", checkReason: "document-removed" });
  });
});
