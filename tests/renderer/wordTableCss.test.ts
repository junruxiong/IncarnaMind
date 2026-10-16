import { describe, expect, onTestFinished, test } from "vitest";
import * as Y from "yjs";
import { MIND_WORD_LOOK_FIELD, type WordLook } from "../../src/core";
import { wordLookCss } from "../../src/renderer/src/editor/wordLookCss";
import { wordTableCss } from "../../src/renderer/src/editor/wordTableCss";
import { wordLookOf } from "../../src/shared/wordLook";
import { createTempDataFolder, startCore } from "../helpers/core";
import { paragraph, run, wordFile } from "../helpers/word";
import { contractFile, resumeFile } from "../helpers/wordLookFiles";
import { importWordBytes } from "../helpers/wordMinds";

async function importedLook(bytes: Buffer): Promise<WordLook> {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir);
  onTestFinished(() => core.close());
  const { client } = await importWordBytes(core, dataDir, bytes, { origin: "own" });
  const look = wordLookOf(client.doc);
  if (!look) throw new Error("No look.");
  return look;
}

const SCOPE = ".word-look-mind";

function keyOf<T>(styles: Record<string, T>, matches: (style: T) => boolean): string {
  const found = Object.entries(styles).find(([, style]) => matches(style));
  if (!found) throw new Error("No such style.");
  return found[0];
}

const rulesFor = (css: readonly string[], key: string) =>
  css.filter((rule) => rule.includes(`"${key}"`));

describe("a Word table's stylesheet", () => {
  test("a table with no lines draws none; the skills table draws its own, with insideH and insideV on the cells between", async () => {
    const look = await importedLook(resumeFile());
    const css = wordTableCss(SCOPE, look);
    const layout = keyOf(look.tables, (table) => !table.borders);
    const lined = keyOf(look.tables, (table) => !!table.borders);

    expect(rulesFor(css, layout).join("\n")).not.toMatch(/border-(top|bottom|left|right):/);
    const own = rulesFor(css, lined).join("\n");
    expect(own).toMatch(
      /table\[data-word-look="t\w+"\] \{[^}]*border-top: calc\(var\(--word-pt\) \* 0\.5\) solid #4472C4/,
    );
    expect(own).toContain("tr + tr > *) { border-top: calc(var(--word-pt) * 0.5) solid #4472C4; }");
    expect(own).toContain("tr > * + *) { border-left: calc(var(--word-pt) * 0.5) solid #4472C4; }");
    expect(css.join("\n")).toContain("border-collapse: collapse");
  });

  test("columns take the file's widths in proportion, one rule a column of each table look; a percent table is that share of the text width", async () => {
    const resume = await importedLook(resumeFile());
    const lined = keyOf(resume.tables, (table) => !!table.borders);
    const widths = rulesFor(wordTableCss(SCOPE, resume), lined).flatMap((rule) => {
      const width = /col:nth-child\(\d\) \{ width: calc\(var\(--word-pt\) \* ([\d.]+)\)/.exec(rule);
      return width ? [Number(width[1])] : [];
    });
    expect(widths).toEqual([100, 381.9]);

    const contract = await importedLook(contractFile());
    const definitions = keyOf(contract.tables, (table) => !!table.borders);
    expect(rulesFor(wordTableCss(SCOPE, contract), definitions).join("\n")).toContain(
      `width: calc(var(--word-pt) * ${contract.width})`,
    );
  });

  test("cells draw their shading, alignment, padding and own sides; a side set to none hides the table's line", () => {
    const look: WordLook = {
      version: 3,
      body: 12,
      width: 450,
      paragraphs: {},
      runs: {},
      tables: {
        t1: { borders: { insideH: { style: "single", width: 0.5, color: null, space: 0 } } },
      },
      cells: {
        c1: {
          shade: "F2F2F2",
          vertical: "center",
          noWrap: true,
          margins: { left: 2 },
          borders: {
            top: { style: "nil", width: 0, color: null, space: 0 },
            bottom: { style: "double", width: 1.5, color: "000000", space: 0 },
          },
        },
      },
    };
    const cell = rulesFor(wordTableCss(SCOPE, look), "c1").join("\n");
    expect(cell).toBe(
      `${SCOPE} :is(td, th)[data-word-look="c1"] { background-color: #F2F2F2; vertical-align: middle; text-wrap-mode: nowrap; padding-left: calc(var(--word-pt) * 2); border-top: hidden; border-bottom: calc(var(--word-pt) * 1.5) double #000000; }`,
    );
    // The table's inside lines are :where() rules, so any cell's own rule wins over them.
    for (const rule of rulesFor(wordTableCss(SCOPE, look), "t1").filter((each) =>
      each.includes("> tr"),
    )) {
      expect(rule.startsWith(`${SCOPE} :where(`)).toBe(true);
    }
  });

  test("a heading's own bottom line stays above the next row's 'none' tops; inside lines sit only on the cell below or to the right, never on a bottom", () => {
    const single = { style: "single", width: 0.5, color: null, space: 0 };
    const look: WordLook = {
      version: 3,
      body: 12,
      width: 450,
      paragraphs: {},
      runs: {},
      tables: { t1: { borders: { insideH: single, insideV: single } } },
      cells: {
        c1: { borders: { bottom: { ...single, width: 1.5 } } },
        c2: { borders: { top: { style: "none", width: 0, color: null, space: 0 } } },
      },
    };
    const css = wordTableCss(SCOPE, look);
    expect(rulesFor(css, "c2").join("\n")).toContain("border-top: none;");
    expect(css.join("\n")).not.toContain("hidden");
    const inside = rulesFor(css, "t1").filter((each) => each.includes("> tr"));
    expect(inside.join("\n")).not.toContain("border-bottom");
    expect(inside.some((each) => each.includes("[rowspan]:not(:last-child)) { border-right"))).toBe(
      true,
    );
  });

  test("rules grow with the file's table styles, never with its tables", async () => {
    const table = `<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/></w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="2000"/><w:gridCol w:w="2000"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:shd w:val="clear" w:fill="EEEEEE"/></w:tcPr>${paragraph(run("a"))}</w:tc><w:tc>${paragraph(run("b"))}</w:tc></w:tr></w:tbl>`;
    const one = await importedLook(wordFile({ body: table }));
    const many = await importedLook(wordFile({ body: table.repeat(80) }));
    expect(wordTableCss(SCOPE, many)).toHaveLength(wordTableCss(SCOPE, one).length);
  });

  test("a table look another device wrote can't reach outside its rule", () => {
    const doc = new Y.Doc();
    const map = doc.getMap(MIND_WORD_LOOK_FIELD);
    map.set("tables", {
      t1: {
        width: { value: "1px; } body { display: none", unit: "pt" },
        columns: [100, "2px } body { display: none"],
        borders: { top: { style: "single; } body {", width: 1, color: null, space: 0 } },
        align: "center; display: none",
      },
      "t2] , body [x": { fixed: true },
    });
    map.set("cells", {
      c1: {
        shade: "red; } body { display: none",
        vertical: "middle; display: none",
        margins: { left: "1px" },
      },
    });
    map.set("body", 12);
    map.set("width", 400);
    map.set("version", 3);
    const look = wordLookOf(doc);
    if (!look) throw new Error("No look.");
    expect(look.tables).toEqual({ t1: { columns: [100] } });
    expect(look.cells).toEqual({ c1: {} });
    const css = wordLookCss("mind", look);
    expect(css).not.toContain("display");
    expect(css.split("{").length).toBe(css.split("}").length);
  });
});
