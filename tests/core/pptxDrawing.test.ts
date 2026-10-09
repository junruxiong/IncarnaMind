/**
 * PowerPoint decks read as drawings for the viewer's slide renderer
 * (ADR-0011): what each slide draws, with what it inherits from its layout,
 * master and theme resolved. The committed fixture is the pptxgenjs deck of
 * tests/core/formats.test.ts; the rest are built here (tests/helpers/pptx.ts).
 */
import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import {
  type ColourContext,
  css,
  DEFAULT_COLOUR_MAP,
  readTheme,
  resolveColour,
} from "../../src/core/documents/formats/drawingml";
import { ExtractionError } from "../../src/core/documents/formats/errors";
import {
  autoNumber,
  type DeckDrawing,
  drawPptx,
  type FailedSlide,
  fontFamily,
  type GroupItem,
  type Item,
  type PictureItem,
  type ShapeItem,
  type SlideDrawing,
  type TableItem,
} from "../../src/core/documents/formats/pptxDrawing";
import { parseXml } from "../../src/core/documents/formats/xml";
import { para, pptxOf, rel, sp, themeXml, txBody, xfrm } from "../helpers/pptx";

const PPTX = new Uint8Array(
  readFileSync(new URL("../fixtures/formats/Quarterly Research Update.pptx", import.meta.url)),
);

const drawn = (deck: DeckDrawing, number: number): SlideDrawing => {
  const slide = deck.slides[number - 1];
  if (!slide || "failed" in slide) throw new Error(`Slide ${number} wasn't drawn.`);
  return slide;
};
const shapes = (slide: SlideDrawing) =>
  slide.items.filter((item): item is ShapeItem => item.kind === "shape");
const textOf = (shape: Item | undefined) =>
  shape?.kind === "shape"
    ? (shape.text?.paragraphs.map((p) => p.runs.map((run) => run.text).join("")) ?? [])
    : [];
const firstRun = (shape: ShapeItem | undefined) => shape?.text?.paragraphs[0]?.runs[0];

describe("the fixture deck", () => {
  test("is drawn slide by slide at its size, 16:9 at 10 inches, with nothing failed", async () => {
    const deck = await drawPptx(PPTX);
    expect(deck.width).toBe(960);
    expect(deck.height).toBe(540);
    expect(deck.slides.map((slide) => slide.number)).toEqual([1, 2, 3, 4, 5]);
    expect(deck.slides.some((slide) => "failed" in slide)).toBe(false);
  });

  test("draws the title slide's background and its title as set", async () => {
    const slide = drawn(await drawPptx(PPTX), 1);
    expect(slide.background).toEqual({ kind: "solid", colour: "#1e3a5f" });
    const [title, subtitle] = shapes(slide);
    expect(title?.box).toMatchObject({ x: 57.6, y: 172.8, w: 844.8, h: 115.2 });
    const run = firstRun(title);
    expect(run).toMatchObject({ text: "Quarterly Research Update", bold: true, colour: "#ffffff" });
    expect(run?.size).toBeCloseTo((40 * 4) / 3);
    // Calibri, from the theme, with its metric twin Carlito behind it.
    expect(run?.fontFamily).toMatch(/^'Calibri', 'Carlito', /);
    expect(subtitle?.text?.anchor).toBe("middle");
    expect(firstRun(subtitle)?.colour).toBe("#cbd5e1");
  });

  test("draws bullets hanging at their indent", async () => {
    const slide = drawn(await drawPptx(PPTX), 2);
    const body = shapes(slide)[1]?.text;
    expect(body?.paragraphs).toHaveLength(3);
    const first = body?.paragraphs[0];
    expect(first?.bullet?.text).toBe("•");
    expect(first?.marginLeft).toBe(36);
    expect(first?.indent).toBe(-36);
    expect(first?.spaceAfter).toEqual({ px: 16 });
  });

  test("places the picture, by its part in the package", async () => {
    const slide = drawn(await drawPptx(PPTX), 3);
    const picture = slide.items.find((item): item is PictureItem => item.kind === "picture");
    expect(picture).toMatchObject({
      part: "ppt/media/image-3-1.png",
      box: { x: 57.6, y: 115.2, w: 460.8, h: 288 },
      crop: { left: 0, top: 0, right: 0, bottom: 0 },
    });
  });

  test("draws the table with its cells' fills and borders", async () => {
    const slide = drawn(await drawPptx(PPTX), 4);
    const table = slide.items.find((item): item is TableItem => item.kind === "table");
    expect(table?.columns).toEqual([268.8, 268.8, 268.8]);
    expect(table?.rows).toHaveLength(4);
    const header = table?.rows[0]?.cells[0];
    expect(header?.fill).toEqual({ kind: "solid", colour: "#e2e8f0" });
    expect(header?.borders.left).toMatchObject({ colour: "#94a3b8", width: 12700 / 9525 });
    expect(header?.text.paragraphs[0]?.runs[0]).toMatchObject({ text: "Stage", bold: true });
    expect(table?.rows[1]?.cells.map((cell) => cell.text.paragraphs[0]?.runs[0]?.text)).toEqual([
      "Qualified",
      "42",
      "1,260",
    ]);
  });

  test("draws the chart from its cached values", async () => {
    const slide = drawn(await drawPptx(PPTX), 5);
    const chart = slide.items.find((item) => item.kind === "chart");
    expect(chart?.kind === "chart" && chart.chart).toMatchObject({
      kind: "column",
      grouping: "clustered",
      categories: ["Q4 25", "Q1 26", "Q2 26", "Q3 26"],
      series: [{ name: "Revenue (£m)", values: [3.1, 3.4, 3.9, 4.4], colour: "#2563eb" }],
      dataLabels: true,
      gridlines: true,
    });
  });
});

/** A master with Office's title and body styles, and title and body placeholders. */
const MASTER = {
  txStyles: `<p:titleStyle><a:lvl1pPr algn="ctr"><a:defRPr sz="4400"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mj-lt"/></a:defRPr></a:lvl1pPr></p:titleStyle><p:bodyStyle><a:lvl1pPr marL="342900" indent="-342900"><a:buFont typeface="Arial"/><a:buChar char="•"/><a:defRPr sz="3200"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/></a:defRPr></a:lvl1pPr><a:lvl2pPr marL="742950" indent="-285750"><a:buFont typeface="Arial"/><a:buChar char="–"/><a:defRPr sz="2800"/></a:lvl2pPr></p:bodyStyle><p:otherStyle><a:lvl1pPr><a:defRPr sz="1800"/></a:lvl1pPr></p:otherStyle>`,
  shapes: [
    sp({
      id: 2,
      name: "Title",
      ph: `<p:ph type="title"/>`,
      spPr: xfrm(457200, 274638, 8229600, 1143000),
      body: txBody(para("Click to edit"), `<a:bodyPr anchor="ctr"/>`),
    }),
    sp({
      id: 3,
      name: "Body",
      ph: `<p:ph type="body" idx="1"/>`,
      spPr: xfrm(457200, 1600200, 8229600, 3000000),
      body: txBody(para("Click to edit")),
    }),
    // A logo bar the master draws on every slide.
    sp({
      id: 4,
      name: "Bar",
      spPr: `${xfrm(0, 5000000, 9144000, 143500)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:schemeClr val="accent1"/></a:solidFill>`,
    }),
  ].join(""),
};

describe("placeholders and the master", () => {
  test("a title placeholder takes its place from the master and its style from the master's title style", async () => {
    const deck = await drawPptx(
      pptxOf({
        master: MASTER,
        slides: [{ shapes: sp({ ph: `<p:ph type="title"/>`, body: txBody(para("Results")) }) }],
      }),
    );
    const slide = drawn(deck, 1);
    const title = shapes(slide).find((shape) => shape.title);
    expect(title?.box).toMatchObject({ x: 48, y: 274638 / 9525, w: 864, h: 120 });
    expect(title?.text?.anchor).toBe("middle");
    expect(title?.text?.paragraphs[0]?.align).toBe("center");
    expect(firstRun(title)).toMatchObject({ text: "Results", colour: "#000000" });
    expect(firstRun(title)?.size).toBeCloseTo((44 * 4) / 3);
    expect(firstRun(title)?.fontFamily).toMatch(/^'Calibri Light', 'Carlito'/);
  });

  test("a body placeholder found by its index in the layout takes the layout's place and the master's bullets by level", async () => {
    const deck = await drawPptx(
      pptxOf({
        master: MASTER,
        layout: {
          shapes: sp({
            ph: `<p:ph idx="1"/>`,
            spPr: xfrm(100000, 200000, 3000000, 2000000),
            body: txBody(para("")),
          }),
        },
        slides: [
          {
            shapes: sp({
              ph: `<p:ph idx="1"/>`,
              body: txBody([para("First point"), para("Detail", `<a:pPr lvl="1"/>`)].join("")),
            }),
          },
        ],
      }),
    );
    const body = shapes(drawn(deck, 1)).find((shape) => shape.origin === "slide");
    expect(body?.box).toMatchObject({ x: 100000 / 9525, y: 200000 / 9525 });
    const [first, second] = body?.text?.paragraphs ?? [];
    expect(first?.bullet).toMatchObject({ text: "•" });
    expect(first?.marginLeft).toBe(36);
    expect(first?.runs[0]?.size).toBeCloseTo((32 * 4) / 3);
    expect(second?.bullet).toMatchObject({ text: "–" });
    expect(second?.runs[0]?.size).toBeCloseTo((28 * 4) / 3);
  });

  test("the master's own shapes are drawn behind the slide's; its placeholders aren't", async () => {
    const deck = await drawPptx(pptxOf({ master: MASTER, slides: [{ shapes: "" }] }));
    const items = drawn(deck, 1).items;
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      kind: "shape",
      origin: "master",
      fill: { kind: "solid", colour: "#4472c4" },
    });
  });

  test("a slide that hides the master's shapes doesn't draw them", async () => {
    const deck = await drawPptx(
      pptxOf({ master: MASTER, slides: [{ shapes: "", attrs: `showMasterSp="0"` }] }),
    );
    expect(drawn(deck, 1).items).toEqual([]);
  });

  test("a hidden shape isn't drawn", async () => {
    const deck = await drawPptx(
      pptxOf({
        slides: [
          {
            shapes: sp({ hidden: true, spPr: xfrm(0, 0, 100, 100), body: txBody(para("Hidden")) }),
          },
        ],
      }),
    );
    expect(drawn(deck, 1).items).toEqual([]);
  });
});

describe("colours", () => {
  test("a dark master maps the background and the text to the theme's dark and light colours", async () => {
    const deck = await drawPptx(
      pptxOf({
        theme: themeXml({ dk1: "111827", lt1: "F9FAFB" }),
        master: {
          ...MASTER,
          clrMap: `bg1="dk1" tx1="lt1" bg2="dk2" tx2="lt2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"`,
        },
        slides: [{ shapes: sp({ ph: `<p:ph type="title"/>`, body: txBody(para("Night shift")) }) }],
      }),
    );
    const slide = drawn(deck, 1);
    expect(slide.background).toEqual({ kind: "solid", colour: "#111827" });
    expect(firstRun(shapes(slide).find((shape) => shape.title))?.colour).toBe("#f9fafb");
  });

  test("theme colours are darkened and lightened as Office does", () => {
    const context: ColourContext = { theme: readTheme(themeXml()), map: DEFAULT_COLOUR_MAP };
    const colour = (xml: string) =>
      css(resolveColour(parseXml(xml), context) ?? { r: 0, g: 0, b: 0, a: 0 });
    /** Each channel within one step of PowerPoint's own value. */
    const near = (xml: string, hex: string) => {
      const got = resolveColour(parseXml(xml), context);
      const want = [1, 3, 5].map((at) => Number.parseInt(hex.slice(at, at + 2), 16));
      expect(
        [got?.r, got?.g, got?.b].map((value, at) => Math.abs((value ?? -9) - (want[at] ?? 0)) <= 1),
      ).toEqual([true, true, true]);
    };
    // "Blue, Accent 1, Darker 25%" and "Lighter 60%", as PowerPoint's palette names them.
    near(`<a:schemeClr val="accent1"><a:lumMod val="75000"/></a:schemeClr>`, "#2F5597");
    near(
      `<a:schemeClr val="accent1"><a:lumMod val="40000"/><a:lumOff val="60000"/></a:schemeClr>`,
      "#B4C7E7",
    );
    expect(colour(`<a:srgbClr val="FF0000"><a:alpha val="50000"/></a:srgbClr>`)).toBe(
      "rgb(255 0 0 / 0.5)",
    );
    expect(colour(`<a:schemeClr val="tx1"/>`)).toBe("#000000");
    expect(colour(`<a:sysClr val="window" lastClr="FFFFFF"/>`)).toBe("#ffffff");
  });

  test("a shape's style references take the theme's fill, line and font colour", async () => {
    const style = `<p:style><a:lnRef idx="2"><a:schemeClr val="accent1"><a:shade val="50000"/></a:schemeClr></a:lnRef><a:fillRef idx="1"><a:schemeClr val="accent2"/></a:fillRef><a:effectRef idx="0"><a:schemeClr val="accent1"/></a:effectRef><a:fontRef idx="minor"><a:schemeClr val="lt1"/></a:fontRef></p:style>`;
    const deck = await drawPptx(
      pptxOf({
        slides: [
          {
            shapes: sp({
              spPr: `${xfrm(0, 0, 952500, 952500)}<a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom>`,
              style,
              body: txBody(para("Go"), `<a:bodyPr anchor="ctr"/>`),
            }),
          },
        ],
      }),
    );
    const shape = shapes(drawn(deck, 1))[0];
    expect(shape?.geometry).toEqual({ kind: "preset", name: "roundRect", adjust: {} });
    expect(shape?.fill).toEqual({ kind: "solid", colour: "#ed7d31" });
    expect(shape?.line?.width).toBeCloseTo(12700 / 9525);
    expect(shape?.line?.colour).not.toBe("#4472c4");
    expect(firstRun(shape)?.colour).toBe("#ffffff");
  });
});

describe("groups, pictures and tables", () => {
  test("a group's children are placed in its box, scaled from its child space", async () => {
    const group = `<p:grpSp><p:nvGrpSpPr><p:cNvPr id="5" name="Group"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="952500" y="952500"/><a:ext cx="1905000" cy="952500"/><a:chOff x="0" y="0"/><a:chExt cx="952500" cy="952500"/></a:xfrm></p:grpSpPr>${sp({ id: 6, spPr: xfrm(476250, 0, 476250, 476250), body: txBody(para("Inside")) })}</p:grpSp>`;
    const deck = await drawPptx(pptxOf({ slides: [{ shapes: group }] }));
    const item = drawn(deck, 1).items[0] as GroupItem;
    expect(item.kind).toBe("group");
    expect(item.box).toMatchObject({ x: 100, y: 100, w: 200, h: 100 });
    expect(item.children[0]?.box).toMatchObject({ x: 100, y: 0, w: 100, h: 50 });
    expect(textOf(item.children[0])).toEqual(["Inside"]);
  });

  test("a picture linked from outside the file is never fetched: it has no part", async () => {
    const pic = `<p:pic><p:nvPicPr><p:cNvPr id="7" name="Picture" descr="A chart"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rImg"/><a:srcRect l="10000" r="20000"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>${xfrm(0, 0, 952500, 952500)}<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom></p:spPr></p:pic>`;
    const deck = await drawPptx(
      pptxOf({
        slides: [
          { shapes: pic, rels: rel("rImg", "image", "https://example.com/pixel.png", true) },
        ],
      }),
    );
    const picture = drawn(deck, 1).items[0] as PictureItem;
    expect(picture).toMatchObject({ kind: "picture", part: null, alt: "A chart" });
    expect(picture.crop).toEqual({ left: 0.1, top: 0, right: 0.2, bottom: 0 });
    expect(picture.geometry).toMatchObject({ name: "ellipse" });
  });

  test("a table in PowerPoint's default style, which files leave out, has its header row and bands", async () => {
    const cell = (text: string) =>
      `<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>${para(text)}</a:txBody><a:tcPr/></a:tc>`;
    const table = `<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="8" name="Table"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="0" y="0"/><a:ext cx="1905000" cy="952500"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl><a:tblPr firstRow="1" bandRow="1"><a:tableStyleId>{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}</a:tableStyleId></a:tblPr><a:tblGrid><a:gridCol w="952500"/><a:gridCol w="952500"/></a:tblGrid><a:tr h="317500">${cell("Site")}${cell("Mean")}</a:tr><a:tr h="317500">${cell("A")}${cell("112")}</a:tr><a:tr h="317500">${cell("B")}${cell("98")}</a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>`;
    const deck = await drawPptx(pptxOf({ slides: [{ shapes: table }] }));
    const item = drawn(deck, 1).items[0] as TableItem;
    expect(item.columns).toEqual([100, 100]);
    const [header, band1, band2] = item.rows;
    expect(header?.cells[0]?.fill).toEqual({ kind: "solid", colour: "#4472c4" });
    expect(header?.cells[0]?.text.paragraphs[0]?.runs[0]).toMatchObject({
      bold: true,
      colour: "#ffffff",
    });
    expect(header?.cells[0]?.borders.bottom?.width).toBeCloseTo(4);
    // The first band is accent 1 at a 40% tint, the second the whole table's 20%.
    expect(band1?.cells[0]?.fill).not.toEqual(band2?.cells[0]?.fill);
    expect(band1?.cells[0]?.text.paragraphs[0]?.runs[0]).toMatchObject({
      bold: false,
      colour: "#000000",
    });
  });
});

describe("text", () => {
  test("automatic numbers count on at their level and restart under a shallower paragraph", async () => {
    const numbered = (text: string, level = 0) =>
      para(
        text,
        `<a:pPr lvl="${level}" marL="342900" indent="-342900"><a:buAutoNum type="arabicPeriod"/></a:pPr>`,
      );
    const deck = await drawPptx(
      pptxOf({
        slides: [
          {
            shapes: sp({
              spPr: xfrm(0, 0, 952500, 952500),
              body: txBody(
                [
                  numbered("One"),
                  numbered("Two"),
                  numbered("Two a", 1),
                  numbered("Two b", 1),
                  numbered("Three"),
                  numbered("Three a", 1),
                ].join(""),
              ),
            }),
          },
        ],
      }),
    );
    const bullets = shapes(drawn(deck, 1))[0]?.text?.paragraphs.map((p) => p.bullet?.text);
    expect(bullets).toEqual(["1.", "2.", "1.", "2.", "3.", "1."]);
  });

  test("numbering schemes", () => {
    expect(autoNumber("arabicPeriod", 3)).toBe("3.");
    expect(autoNumber("romanUcParenR", 4)).toBe("IV)");
    expect(autoNumber("alphaLcParenBoth", 28)).toBe("(ab)");
    expect(autoNumber("arabicPlain", 12)).toBe("12");
  });

  test("runs keep their own formatting, and a shrunk text body its scale", async () => {
    const runs = [
      `<a:r><a:rPr lang="en-GB" b="1"/><a:t>Bold, </a:t></a:r>`,
      `<a:r><a:rPr lang="en-GB" i="1" u="sng"><a:solidFill><a:srgbClr val="C00000"/></a:solidFill></a:rPr><a:t>red</a:t></a:r>`,
      `<a:br><a:rPr lang="en-GB"/></a:br>`,
      `<a:r><a:rPr lang="en-GB" baseline="30000" strike="sngStrike"/><a:t>2</a:t></a:r>`,
    ];
    const deck = await drawPptx(
      pptxOf({
        slides: [
          {
            shapes: sp({
              spPr: xfrm(0, 0, 952500, 952500),
              body: txBody(
                para(runs),
                `<a:bodyPr wrap="none"><a:normAutofit fontScale="50000" lnSpcReduction="20000"/></a:bodyPr>`,
              ),
            }),
          },
        ],
      }),
    );
    const body = shapes(drawn(deck, 1))[0]?.text;
    expect(body?.wrap).toBe(false);
    const [bold, red, lineBreak, raised] = body?.paragraphs[0]?.runs ?? [];
    expect(bold).toMatchObject({ text: "Bold, ", bold: true });
    expect(bold?.size).toBeCloseTo(12);
    expect(red).toMatchObject({ italic: true, underline: "sng", colour: "#c00000" });
    expect(lineBreak?.lineBreak).toBe(true);
    expect(raised).toMatchObject({ baseline: 0.3, strike: "single" });
    expect(body?.paragraphs[0]?.lineSpacing).toEqual({ percent: 0.8 });
  });

  test("a font family falls back by its kind, with Office's fonts' metric twins", () => {
    expect(fontFamily("Calibri", undefined)).toMatch(/^'Calibri', 'Carlito', 'Helvetica Neue'/);
    expect(fontFamily("Georgia", undefined)).toMatch(/^'Georgia', 'Times New Roman'/);
    expect(fontFamily("Arial", "等线")).toMatch(/^'Arial', '等线', 'Helvetica Neue'/);
    expect(fontFamily("It's", undefined)).toMatch(/^'Its'/);
  });
});

describe("failures", () => {
  test("a slide that can't be read fails on its own; the others are drawn", async () => {
    const deck = await drawPptx(
      pptxOf({
        slides: [
          { shapes: sp({ spPr: xfrm(0, 0, 952500, 952500), body: txBody(para("Fine")) }) },
          { shapes: "", raw: "<p:sld><p:cSld><p:spTree></p:sld>" },
        ],
      }),
    );
    expect(textOf(drawn(deck, 1).items[0])).toEqual(["Fine"]);
    expect((deck.slides[1] as FailedSlide).failed).toMatch(/Broken XML/);
  });

  test("a file that isn't a deck is refused", async () => {
    await expect(drawPptx(new Uint8Array(Buffer.from("not a zip")))).rejects.toBeInstanceOf(
      ExtractionError,
    );
  });
});
