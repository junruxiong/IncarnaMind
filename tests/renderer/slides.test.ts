/**
 * The slide renderer's pure parts (src/renderer/src/viewer/slides): where a
 * Citation's quote is in a drawn slide's text, shapes' geometry as SVG
 * paths, and a chart's value axis.
 */
import { describe, expect, test } from "vitest";
import { drawPptx, type SlideDrawing } from "../../src/core/documents/formats/pptxDrawing";
import { niceScale } from "../../src/renderer/src/viewer/slides/chartScale";
import {
  customPaths,
  presetPaths,
  textInsets,
} from "../../src/renderer/src/viewer/slides/geometry";
import { findInDrawing, runKey, slidePieces } from "../../src/renderer/src/viewer/slides/pieces";
import { para, pptxOf, sp, txBody, xfrm } from "../helpers/pptx";

async function slideOf(shapes: string, masterShapes = ""): Promise<SlideDrawing> {
  const deck = await drawPptx(pptxOf({ master: { shapes: masterShapes }, slides: [{ shapes }] }));
  const slide = deck.slides[0];
  if (!slide || "failed" in slide) throw new Error("Not drawn.");
  return slide;
}

describe("a quote in a drawn slide", () => {
  test("is found across runs of one paragraph, each run's part keyed where it is drawn", async () => {
    const slide = await slideOf(
      sp({
        spPr: xfrm(0, 0, 952500, 952500),
        body: txBody(
          para([
            `<a:r><a:rPr lang="en-GB"/><a:t>Churn fell to </a:t></a:r>`,
            `<a:r><a:rPr lang="en-GB" b="1"/><a:t>4.2 per cent</a:t></a:r>`,
            `<a:r><a:rPr lang="en-GB"/><a:t> after the redesign.</a:t></a:r>`,
          ]),
        ),
      }),
    );
    const found = findInDrawing(slide, "fell to 4.2 per cent after");
    expect(found && [...found.entries()]).toEqual([
      [runKey("0", 0, 0), [{ start: 6, end: 14 }]],
      [runKey("0", 0, 1), [{ start: 0, end: 12 }]],
      [runKey("0", 0, 2), [{ start: 0, end: 6 }]],
    ]);
    expect(findInDrawing(slide, "fell to 5 per cent")).toBeNull();
  });

  test("reads the title first, then the shapes, then the tables, as the slide's Unit does", async () => {
    const cell = (text: string) =>
      `<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>${para(text)}</a:txBody><a:tcPr/></a:tc>`;
    const table = `<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="9" name="Table"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="0" y="0"/><a:ext cx="952500" cy="952500"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl><a:tblPr/><a:tblGrid><a:gridCol w="476250"/><a:gridCol w="476250"/></a:tblGrid><a:tr h="0">${cell("Qualified")}${cell("42")}</a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>`;
    const slide = await slideOf(
      [
        table,
        sp({ id: 3, spPr: xfrm(0, 0, 952500, 952500), body: txBody(para("Body text")) }),
        sp({
          id: 4,
          ph: `<p:ph type="title"/>`,
          spPr: xfrm(0, 0, 952500, 952500),
          body: txBody(para("The title")),
        }),
      ].join(""),
      // The master's own text isn't the slide's: a quote is never found in it.
      sp({ id: 5, spPr: xfrm(0, 0, 952500, 952500), body: txBody(para("Confidential")) }),
    );
    expect(slidePieces(slide).map((piece) => piece.text)).toEqual([
      "The title",
      "Body text",
      "Qualified",
      "42",
    ]);
    expect(findInDrawing(slide, "The title Body text")?.size).toBe(2);
    expect(findInDrawing(slide, "Qualified 42")?.size).toBe(2);
    expect(findInDrawing(slide, "Confidential")).toBeNull();
  });
});

describe("geometry", () => {
  test("a preset it doesn't know is its rectangle", () => {
    expect(presetPaths("someNewShape", 100, 50, {})).toEqual([
      { d: "M0 0 L100 0 L100 50 L0 50 Z", fill: true, stroke: true },
    ]);
  });

  test("a rounded rectangle's corners follow its adjust value", () => {
    const [path] = presetPaths("roundRect", 200, 100, { adj: 50000 });
    // A radius of half the shorter side: 50.
    expect(path?.d).toContain("A50 50 0 0 1 200 50");
  });

  test("a chevron's point and notch, and a line with no inside", () => {
    expect(presetPaths("chevron", 100, 40, {})[0]?.d).toBe(
      "M0 0 L80 0 L100 20 L80 40 L0 40 L20 20 Z",
    );
    expect(presetPaths("straightConnector1", 100, 0, {})).toEqual([
      { d: "M0 0 L100 0", fill: false, stroke: true },
    ]);
  });

  test("text sits in a preset's text rectangle: after a chevron's notch, inside an ellipse", () => {
    const chevron = { kind: "preset" as const, name: "chevron", adjust: {} };
    expect(textInsets(chevron, 200, 100)).toEqual({ left: 50, top: 0, right: 50, bottom: 0 });
    const ellipse = textInsets({ kind: "preset", name: "ellipse", adjust: {} }, 100, 100);
    expect(ellipse.left).toBeCloseTo(14.64);
    expect(textInsets({ kind: "preset", name: "rect", adjust: {} }, 100, 100)).toEqual({
      left: 0,
      top: 0,
      right: 0,
      bottom: 0,
    });
  });

  test("a custom path is scaled from its own space, its arcs from the current point", () => {
    const [path] = customPaths(
      [
        {
          w: 10,
          h: 10,
          fill: true,
          stroke: true,
          commands: [
            { op: "M", x: 0, y: 5 },
            { op: "A", wR: 5, hR: 5, start: 180, swing: 180 },
            { op: "Z" },
          ],
        },
      ],
      100,
      50,
    );
    expect(path?.d).toBe("M0 25 A50 25 0 0 1 100 25 Z");
  });
});

describe("a chart's value axis", () => {
  test("runs from a round number to a round number in round steps", () => {
    expect(niceScale(0, 4.4)).toEqual({ min: 0, max: 5, step: 1 });
    expect(niceScale(0, 262)).toEqual({ min: 0, max: 300, step: 100 });
    expect(niceScale(-3, 14)).toEqual({ min: -5, max: 15, step: 5 });
    expect(niceScale(0, 0)).toEqual({ min: 0, max: 1, step: 0.2 });
  });
});
