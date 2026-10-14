import { describe, expect, test } from "vitest";
import { drawnIndexOf } from "../../src/renderer/src/viewer/wordPlaces";
import { paragraph, table, textBox, wordFile } from "../helpers/word";

/** A title, a text box Word saved twice (its newer copy, in mc:Choice, isn't drawn), a table, then text. */
const FILE = wordFile({
  body: [
    paragraph("Title"),
    paragraph(textBox(["Box one", "Box two"])),
    table([["Cell"]]),
    paragraph("After"),
  ].join(""),
});

describe("a place in a Word file, among what the viewer draws", () => {
  test("counts past a text box's undrawn copy", async () => {
    // w:p in document order: Title 0, the box's holder 1, its newer copy 2–3, its older copy
    // 4–5, Cell 6, After 7. Drawn: Title, holder, the older copy, Cell, After.
    expect(await drawnIndexOf(FILE, "w:p#0")).toBe(0);
    expect(await drawnIndexOf(FILE, "w:p#4")).toBe(2);
    expect(await drawnIndexOf(FILE, "w:p#7")).toBe(5);
    expect(await drawnIndexOf(FILE, "w:tbl#0")).toBe(0);
  });

  test("gives the paragraph that holds an undrawn one, and nothing for a place the file hasn't", async () => {
    expect(await drawnIndexOf(FILE, "w:p#2")).toBe(1);
    expect(await drawnIndexOf(FILE, "w:p#40")).toBeNull();
    expect(await drawnIndexOf(FILE, "w:r#1")).toBeNull();
  });
});
