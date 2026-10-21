import { readFileSync } from "node:fs";
import { join } from "node:path";
import { getSchema } from "@tiptap/core";
import { DOMParser as ProseMirrorDOMParser, type Node as ProseMirrorNode } from "@tiptap/pm/model";
import { DOMParser, parseHTML } from "linkedom";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { fromWord } from "../../src/renderer/src/editor/wordPaste";

const fieldReport = readFileSync(
  join(__dirname, "../fixtures/formats/word-clipboard-field-report.html"),
  "utf8",
);

beforeEach(() => {
  vi.stubGlobal("DOMParser", DOMParser);
});

function pasted(html: string): ProseMirrorNode {
  const { document } = parseHTML(`<!doctype html><html><body>${fromWord(html)}</body></html>`);
  return ProseMirrorDOMParser.fromSchema(getSchema(noteExtensions())).parse(
    document.body as unknown as Node,
  );
}

const blocks = (doc: ProseMirrorNode) =>
  doc.children.map((node) => ({ type: node.type.name, text: node.textContent }));

const items = (list: ProseMirrorNode | null | undefined) =>
  list?.children.map((item) => item.firstChild?.textContent);

const listParagraph = (list: string, level: number, marker: string, words: string) =>
  `<p class=MsoListParagraphCxSpMiddle style='margin-left:36.0pt;text-indent:-18.0pt;mso-list:${list} level${level} lfo1'><![if !supportLists]><span><span style='mso-list:Ignore'>${marker}<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp; </span></span></span><![endif]><span lang=EN-US>${words}<o:p></o:p></span></p>`;

const word = (body: string) =>
  `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><body><!--StartFragment-->${body}<!--EndFragment--></body></html>`;

describe("pasting what Word copies", () => {
  test("the Field Report keeps its Title, headings, lists and table, and leaves its comment out", () => {
    const doc = pasted(fieldReport);
    const shown = blocks(doc);
    expect(shown[0]).toEqual({ type: "heading", text: "Field Report: Northern Estuary" });
    expect(doc.firstChild?.attrs.level).toBe(1);
    expect(shown.map((block) => block.type)).toEqual(
      expect.arrayContaining(["orderedList", "bulletList", "table"]),
    );
    const steps = doc.children.find((node) => node.type.name === "orderedList");
    expect(items(steps)).toEqual([
      "Calibrate each gauge against the datum.",
      "Read it every fifteen minutes.",
      "Reject gaps longer than a month.",
    ]);
    expect(steps?.attrs.start).toBe(1);
    const equipment = doc.children.find((node) => node.type.name === "bulletList");
    expect(items(equipment)).toEqual(["Pressure gauges", "Radar level sensors", "A survey boat"]);
    expect(shown).toContainEqual({ type: "heading", text: "1 Methods" });
    expect(shown).toContainEqual({ type: "heading", text: "1.1 Sites" });
    expect(doc.textContent).toContain("Highlighted text marks open questions.");
    expect(doc.textContent).not.toContain("[JM1]");
    expect(doc.textContent).not.toContain("Confirm with the hydrology team");
    expect(shown.at(-1)).toEqual({
      type: "paragraph",
      text: "The central and southern estuaries gain less, because their defences are already above the 2050 line.",
    });
  });

  // The rule for Word's numbering: a run becomes a list only when the editor can draw every marker in it.
  test.each(["1、", "（1）", "一、", "1.1.", "a)"])(
    "a run numbered %s stays paragraphs, each with its number typed",
    (marker) => {
      const doc = pasted(
        word(listParagraph("l0", 1, marker, "First") + listParagraph("l0", 1, marker, "Second")),
      );
      expect(blocks(doc)).toEqual([
        { type: "paragraph", text: `${marker} First` },
        { type: "paragraph", text: `${marker} Second` },
      ]);
    },
  );

  test("a run numbered from 3. is a list that starts at 3", () => {
    const doc = pasted(
      word(listParagraph("l0", 1, "3.", "Third") + listParagraph("l0", 1, "4.", "Fourth")),
    );
    expect(blocks(doc).map((block) => block.type)).toEqual(["orderedList"]);
    expect(doc.firstChild?.attrs.start).toBe(3);
    expect(items(doc.firstChild)).toEqual(["Third", "Fourth"]);
  });

  test.each(["·", "o", "§", "\uf0b7"])("a run of %s bullets is a bullet list", (bullet) => {
    const doc = pasted(word(listParagraph("l1", 1, bullet, "Tide tables")));
    expect(doc.firstChild?.type.name).toBe("bulletList");
    expect(doc.firstChild?.firstChild?.type.name).toBe("listItem");
    expect(doc.firstChild?.firstChild?.firstChild?.type.name).toBe("paragraph");
    expect(doc.textContent).toBe("Tide tables");
  });

  test("a run with one marker the editor can't draw stays plain; the next run is a list", () => {
    const doc = pasted(
      word(
        listParagraph("l0", 1, "1.", "Collect") +
          listParagraph("l0", 2, "a.", "Label") +
          "<p class=MsoNormal>Then</p>" +
          listParagraph("l1", 1, "·", "Boats"),
      ),
    );
    expect(blocks(doc)).toEqual([
      { type: "paragraph", text: "1. Collect" },
      { type: "paragraph", text: "a. Label" },
      { type: "paragraph", text: "Then" },
      { type: "bulletList", text: "Boats" },
    ]);
  });

  test("bullets nest by level", () => {
    const doc = pasted(
      word(
        listParagraph("l1", 1, "·", "Spring tides") +
          listParagraph("l1", 2, "o", "At new moon") +
          listParagraph("l1", 2, "o", "At full moon") +
          listParagraph("l1", 1, "·", "Neap tides"),
      ),
    );
    expect(blocks(doc).map((block) => block.type)).toEqual(["bulletList"]);
    const list = doc.firstChild;
    expect(items(list)).toEqual(["Spring tides", "Neap tides"]);
    const inner = list?.firstChild?.lastChild;
    expect(inner?.type.name).toBe("bulletList");
    expect(items(inner)).toEqual(["At new moon", "At full moon"]);
  });

  test("a bullet under a numbered step nests in it, and the numbering goes on after", () => {
    const doc = pasted(
      word(
        listParagraph("l0", 1, "1.", "First") +
          listParagraph("l0", 2, "·", "Detail") +
          listParagraph("l0", 1, "2.", "Second"),
      ),
    );
    const list = doc.firstChild;
    expect(list?.type.name).toBe("orderedList");
    expect(items(list)).toEqual(["First", "Second"]);
    const inner = list?.firstChild?.lastChild;
    expect(inner?.type.name).toBe("bulletList");
    expect(items(inner)).toEqual(["Detail"]);
  });

  test("words beside a list, outside any paragraph, are kept", () => {
    const doc = pasted(word(`Loose words before${listParagraph("l0", 1, "1.", "First")}and after`));
    expect(blocks(doc)).toEqual([
      { type: "paragraph", text: "Loose words before" },
      { type: "orderedList", text: "First" },
      { type: "paragraph", text: "and after" },
    ]);
  });

  test("a paragraph between two runs ends the first", () => {
    const doc = pasted(
      word(
        listParagraph("l0", 1, "1.", "One") +
          "<p class=MsoNormal>Between</p>" +
          listParagraph("l0", 1, "2.", "Two"),
      ),
    );
    expect(blocks(doc)).toEqual([
      { type: "orderedList", text: "One" },
      { type: "paragraph", text: "Between" },
      { type: "orderedList", text: "Two" },
    ]);
    expect(doc.child(2).attrs.start).toBe(2);
  });
  test("a list in a table cell stays in its cell, and Windows Word's comment-wrapped markers go", () => {
    const windows = (marker: string, words: string) =>
      `<p class=MsoListParagraph style='mso-list:l0 level1 lfo1'><!--[if !supportLists]--><span><span style='mso-list:Ignore'>${marker}<span>&nbsp; </span></span></span><!--[endif]-->${words}</p>`;
    const doc = pasted(
      word(
        `<table><tr><td>${windows("1.", "First")}${windows("2.", "Second")}</td><td><p class=MsoNormal>Notes</p></td></tr></table>`,
      ),
    );
    const cell = doc.firstChild?.firstChild?.firstChild;
    expect(cell?.firstChild?.type.name).toBe("orderedList");
    expect(items(cell?.firstChild)).toEqual(["First", "Second"]);
    expect(doc.textContent).not.toContain("1.");
  });

  test("a Subtitle is a plain line, and HTML from anywhere else is left as it is", () => {
    const doc = pasted(word(`<p class=MsoSubtitle>Spring 2026</p>`));
    expect(blocks(doc)).toEqual([{ type: "paragraph", text: "Spring 2026" }]);
    const web = `<ol><li>One</li></ol><p class="Title">Not Word</p>`;
    expect(fromWord(web)).toBe(web);
  });
});
