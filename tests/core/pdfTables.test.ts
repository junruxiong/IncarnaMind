/**
 * A PDF page's tables (#1368), from where pdf.js places its text: words in the same columns over
 * three lines or more, with a column of figures or three columns, are a table; prose isn't.
 */
import { describe, expect, test } from "vitest";
import { documentBlocks } from "../../src/core/answers/checks/verifyTables";
import { type PagePart, tablesIn } from "../../src/core/documents/pdfTables";

const SIZE = 10;
const CHARACTER = 5;

/** A text item as pdf.js gives it: `str` at `x` on the baseline `y`, its width from its length. */
const item = (str: string, x: number, y: number, eol = false) => ({
  str,
  dir: "ltr",
  transform: [SIZE, 0, 0, SIZE, x, y],
  width: str.length * CHARACTER,
  height: SIZE,
  hasEOL: eol,
});

/** A line of cells, each at its column's left edge, or its right edge when `right` says so. */
function line(y: number, cells: readonly (readonly [string, number, "right"?])[]) {
  return cells.flatMap(([text, at, align], index) => {
    const x = align === "right" ? at - text.length * CHARACTER : at;
    const gap = index < cells.length - 1 ? [item(" ", x + text.length * CHARACTER, y)] : [];
    return [item(text, x, y, index === cells.length - 1), ...gap];
  });
}

const textOf = (part: PagePart | undefined) =>
  part?.kind === "text"
    ? part.items.map((each) => ("str" in each ? String(each.str) : "")).join("")
    : null;

describe("tables on a PDF page", () => {
  test("rows of a label and figures are a table, between the text before and after it", () => {
    const parts = tablesIn([
      item("Balance sheet at 31 December", 72, 700, true),
      ...line(680, [
        ["Item", 72],
        ["Closing", 300, "right"],
        ["Opening", 400, "right"],
      ]),
      ...line(666, [
        ["Cash and equivalents", 72],
        ["3,820", 300, "right"],
        ["3,145", 400, "right"],
      ]),
      ...line(652, [
        ["Receivables", 72],
        ["2,610", 300, "right"],
        ["2,290", 400, "right"],
      ]),
      ...line(638, [
        ["Short-term loans", 72],
        ["(1,500)", 300, "right"],
        ["", 400, "right"],
      ]),
      item("Figures in thousands.", 72, 610, true),
    ]);
    expect(parts?.map((part) => part.kind)).toEqual(["text", "table", "text"]);
    expect(textOf(parts?.[0])).toBe("Balance sheet at 31 December");
    expect(parts?.[1]).toEqual({
      kind: "table",
      rows: [
        ["Item", "Closing", "Opening"],
        ["Cash and equivalents", "3,820", "3,145"],
        ["Receivables", "2,610", "2,290"],
        ["Short-term loans", "(1,500)", ""],
      ],
      numeric: [false, true, true],
    });
    expect(textOf(parts?.[2])).toBe("Figures in thousands.");
  });

  test("a heading row of years with no label over it heads the table", () => {
    const parts = tablesIn([
      ...line(700, [
        ["2025", 300, "right"],
        ["2024", 400, "right"],
      ]),
      ...line(686, [
        ["Revenue", 72],
        ["21,640", 300, "right"],
        ["19,080", 400, "right"],
      ]),
      ...line(672, [
        ["Cost", 72],
        ["16,930", 300, "right"],
        ["15,140", 400, "right"],
      ]),
    ]);
    expect(parts).toEqual([
      {
        kind: "table",
        rows: [
          ["", "2025", "2024"],
          ["Revenue", "21,640", "19,080"],
          ["Cost", "16,930", "15,140"],
        ],
        numeric: [false, true, true],
      },
    ]);
  });

  test("two columns of prose side by side are no table, and a page without a table gives none", () => {
    const prose = [0, 1, 2, 3].flatMap((row) =>
      line(700 - row * 14, [
        [`The left column goes on with words ${row}`, 72],
        [`and the right column does too ${row}`, 320],
      ]),
    );
    expect(tablesIn(prose)).toBeNull();
    expect(tablesIn([item("Just a sentence.", 72, 700, true)])).toBeNull();
  });

  test("two rows aren't enough to be a table", () => {
    const parts = tablesIn([
      ...line(700, [
        ["Revenue", 72],
        ["21,640", 300, "right"],
      ]),
      ...line(686, [
        ["Cost", 72],
        ["16,930", 300, "right"],
      ]),
    ]);
    expect(parts).toBeNull();
  });

  test("three columns of words are a table without figures", () => {
    const parts = tablesIn(
      [
        ["Site", "Gauge", "Read by"],
        ["Harbour mouth", "Radar", "Port office"],
        ["Mid estuary", "Pressure", "Survey team"],
      ].flatMap((row, index) =>
        line(
          700 - index * 14,
          row.map((text, column): readonly [string, number] => [text, 72 + column * 150]),
        ),
      ),
    );
    expect(parts).toEqual([
      {
        kind: "table",
        rows: [
          ["Site", "Gauge", "Read by"],
          ["Harbour mouth", "Radar", "Port office"],
          ["Mid estuary", "Pressure", "Survey team"],
        ],
        numeric: [false, false, false],
      },
    ]);
  });

  test("in the Mind, a first row of years or words over the figures is the table's heading", () => {
    const unit = { page: 1, text: "", kind: "page" as const, label: null };
    const headingsOf = (first: string[]) =>
      documentBlocks(
        [unit],
        new Map([
          [
            1,
            [
              {
                kind: "table" as const,
                rows: [first, ["Revenue", "21,640", "19,080"], ["Cost", "16,930", "15,140"]],
                numeric: [false, true, true],
              },
            ],
          ],
        ]),
      )[0]?.node.content?.[0]?.content?.map((cell) => cell.type);
    expect(headingsOf(["", "2025", "2024"])).toEqual(["tableHeader", "tableHeader", "tableHeader"]);
    expect(headingsOf(["Item", "This year", "Last year"])).toEqual([
      "tableHeader",
      "tableHeader",
      "tableHeader",
    ]);
    expect(headingsOf(["Equity", "10,930", "9,310"])).toEqual([
      "tableCell",
      "tableCell",
      "tableCell",
    ]);
  });
});
