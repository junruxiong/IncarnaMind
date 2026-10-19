import { getSchema } from "@tiptap/core";
import { DOMParser as ProseMirrorDOMParser } from "@tiptap/pm/model";
import { parseHTML } from "linkedom";
import { describe, expect, test } from "vitest";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { listsFromWord } from "../../src/renderer/src/editor/wordListPaste";

const { document } = parseHTML("<!doctype html><html><body></body></html>");
const schema = getSchema(noteExtensions());

/** What Word puts on the clipboard for a list paragraph: its level in the style, its marker in a span it ignores. */
const bullet = (level: number, words: string) =>
  `<p class=MsoListParagraph style='margin-left:${level * 0.5}in;text-indent:-.25in;mso-list:l0 level${level} lfo1'><![if !supportLists]><span style='font-family:Symbol'><span style='mso-list:Ignore'>·<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></span><![endif]>${words}<o:p></o:p></p>`;
const number = (level: number, marker: string, words: string) =>
  `<p class=MsoListParagraph style='text-indent:-.25in;mso-list:l1 level${level} lfo2'><![if !supportLists]><span style='mso-list:Ignore'>${marker}<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp; </span></span><![endif]>${words}<o:p></o:p></p>`;

describe("pasting a list from Word", () => {
  test("a bulleted list of two levels keeps both levels", () => {
    const html = `<!--StartFragment-->${bullet(1, "Spring tides")}${bullet(2, "At new moon")}${bullet(2, "At full moon")}${bullet(1, "Neap tides")}<!--EndFragment-->`;

    const root = document.createElement("div");
    root.innerHTML = listsFromWord(html, document);
    const shape = (list: Element): unknown =>
      [...list.children].map((item) => [
        item.querySelector(":scope > p")?.textContent,
        ...[...item.children]
          .filter((each) => /^(UL|OL)$/.test(each.tagName))
          .map((inner) => [inner.tagName, shape(inner)]),
      ]);
    const [list] = [...root.children];
    expect(list?.tagName).toBe("UL");
    expect(shape(list as Element)).toEqual([
      ["Spring tides", ["UL", [["At new moon"], ["At full moon"]]]],
      ["Neap tides"],
    ]);
    expect(root.textContent).toBe("Spring tidesAt new moonAt full moonNeap tides");
  });

  test("a numbered list with a bulleted one inside it, and a paragraph that ends it", () => {
    const html = `${number(1, "1.", "First")}${bullet(2, "Detail")}${number(1, "2.", "Second")}<p>After the list.</p>${number(1, "1.", "A new list")}`;

    const root = document.createElement("div");
    root.innerHTML = listsFromWord(html, document);
    expect([...root.children].map((each) => each.tagName)).toEqual(["OL", "P", "OL"]);
    const first = root.children[0] as Element;
    expect(
      [...first.children].map((item) => item.querySelector(":scope > p")?.textContent),
    ).toEqual(["First", "Second"]);
    expect(first.querySelector("li > ul > li > p")?.textContent).toBe("Detail");
  });

  test("the editor's own schema reads it as lists, nested", () => {
    const html = `${bullet(1, "One")}${bullet(2, "Two")}${bullet(1, "Three")}`;
    const root = document.createElement("div");
    root.innerHTML = listsFromWord(html, document);
    const doc = ProseMirrorDOMParser.fromSchema(schema).parse(root as unknown as Node);
    const list = doc.firstChild;
    expect(list?.type.name).toBe("bulletList");
    expect(list?.childCount).toBe(2);
    expect(list?.firstChild?.textContent).toBe("OneTwo");
    expect(list?.firstChild?.lastChild?.type.name).toBe("bulletList");
  });

  test.each([
    ["1、", "Chinese numbers"],
    ["（1）", "Chinese numbers in brackets"],
    ["一、", "Chinese numerals"],
    ["1.1.", "legal numbering"],
    ["a)", "letters"],
  ])("a list numbered %s (%s) stays paragraphs with its numbers typed", (marker) => {
    const html = `${number(1, marker, "First")}${number(1, marker, "Second")}`;
    const root = document.createElement("div");
    root.innerHTML = listsFromWord(html, document);
    expect(root.querySelectorAll("ul, ol")).toHaveLength(0);
    expect(
      [...root.querySelectorAll("p")].map((each) => each.textContent?.replace(/\s+/g, "")),
    ).toEqual([`${marker}First`, `${marker}Second`]);
  });

  test("a numbered list that starts after 1 keeps its first number, and any bullet glyph is a bullet", () => {
    const root = document.createElement("div");
    root.innerHTML = listsFromWord(
      `${number(1, "3.", "Third")}${number(1, "4.", "Fourth")}`,
      document,
    );
    expect(root.querySelector("ol")?.getAttribute("start")).toBe("3");
    for (const glyph of ["·", "o", "§", "\uf0b7"]) {
      const bulleted = document.createElement("div");
      bulleted.innerHTML = listsFromWord(
        `<p style='mso-list:l0 level1 lfo1'><span style='mso-list:Ignore'>${glyph}</span>Item</p>`,
        document,
      );
      expect(bulleted.querySelectorAll("ul > li > p")).toHaveLength(1);
    }
  });

  test("a run with a style the editor can't draw among its levels stays paragraphs, the next run is a list", () => {
    const html = `${number(1, "1.", "One")}${number(2, "a.", "Sub")}<p>Between.</p>${bullet(1, "Bullet")}`;
    const root = document.createElement("div");
    root.innerHTML = listsFromWord(html, document);
    expect([...root.children].map((each) => each.tagName)).toEqual(["P", "P", "P", "UL"]);
  });

  test("HTML with no Word list in it, a Google Docs list among it, comes back as it was", () => {
    const docs = `<ul><li><p>One</p></li><ul><li><p>Two</p></li></ul></ul>`;
    expect(listsFromWord(docs, document)).toBe(docs);
    const plain = "<p>Some words</p>";
    expect(listsFromWord(plain, document)).toBe(plain);
  });
});
