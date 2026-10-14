import { describe, expect, test } from "vitest";
import { confirmDrawn, drawnTargetOf } from "../../src/renderer/src/viewer/wordPlaces";
import { paragraph, run, table, textBox, wordFile } from "../helpers/word";

/** A title, a text box Word saved twice (its newer copy, in mc:Choice, isn't drawn), a table, then text. */
const FILE = wordFile({
  body: [
    paragraph("Title"),
    paragraph(textBox(["Box one", "Box two"])),
    table([["Cell"]]),
    paragraph("After"),
  ].join(""),
});

/** A paragraph a page break crosses, as Word marks it: drawn as two, one on each page. */
const crossing = (before: string, after: string) =>
  paragraph(
    `${run(before)}<w:r><w:lastRenderedPageBreak/><w:t xml:space="preserve">${after}</w:t></w:r>`,
  );

describe("a place in a Word file, among what the viewer draws", () => {
  test("counts past a text box's undrawn copy, with the paragraph's text", async () => {
    // w:p in document order: Title 0, the box's holder 1, its newer copy 2–3, its older copy
    // 4–5, Cell 6, After 7. Drawn: Title, holder, the older copy, Cell, After.
    expect(await drawnTargetOf(FILE, "w:p#0")).toEqual({ index: 0, text: "Title" });
    expect(await drawnTargetOf(FILE, "w:p#4")).toEqual({ index: 2, text: "Box one" });
    expect(await drawnTargetOf(FILE, "w:p#7")).toEqual({ index: 5, text: "After" });
    expect(await drawnTargetOf(FILE, "w:tbl#0")).toEqual({ index: 0, text: "" });
  });

  test("counts a paragraph a page break crosses as the two the viewer draws", async () => {
    const file = wordFile({
      body: [
        crossing("First, ", "on the next page."),
        paragraph("Second."),
        // A break at its very end splits nothing.
        paragraph(`${run("Third.")}<w:r><w:lastRenderedPageBreak/></w:r>`),
        crossing("Fourth, ", "on the page after."),
        paragraph("Fifth."),
      ].join(""),
    });
    expect(await drawnTargetOf(file, "w:p#1")).toEqual({ index: 2, text: "Second." });
    expect(await drawnTargetOf(file, "w:p#4")).toEqual({ index: 6, text: "Fifth." });
  });

  test("gives the paragraph that holds an undrawn one, and nothing for a place the file hasn't", async () => {
    expect((await drawnTargetOf(FILE, "w:p#2"))?.index).toBe(1);
    expect(await drawnTargetOf(FILE, "w:p#40")).toBeNull();
    expect(await drawnTargetOf(FILE, "w:r#1")).toBeNull();
  });
});

describe("the drawn paragraph to go to", () => {
  const drawn = [
    "Title",
    "Recital 1: one.",
    "Recital 2: two, on",
    "the next page.",
    "Clause 9: the cap.",
  ];

  test("is the one at its place when its text begins as the paragraph's", () => {
    expect(confirmDrawn(drawn, { index: 4, text: "Clause 9: the cap." })).toBe(4);
    // The first half of a paragraph split across pages.
    expect(confirmDrawn(drawn, { index: 2, text: "Recital 2: two, on the next page." })).toBe(2);
  });

  test("else the nearest that does, and its place when none does or it has no text", () => {
    expect(confirmDrawn(drawn, { index: 3, text: "Clause 9: the cap." })).toBe(4);
    expect(
      confirmDrawn([...drawn, "Clause 9: the cap."], { index: 1, text: "Clause 9: the cap." }),
    ).toBe(4);
    expect(confirmDrawn(drawn, { index: 1, text: "Gone from the pages." })).toBe(1);
    expect(confirmDrawn(drawn, { index: 3, text: "" })).toBe(3);
    expect(confirmDrawn(drawn, { index: 9, text: "" })).toBeNull();
  });
});
