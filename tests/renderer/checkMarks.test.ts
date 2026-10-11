import { describe, expect, test } from "vitest";
import { contrast, token } from "../helpers/contrast";

/*
 * The margin marks of a Citation's check (DESIGN.md, Citation check): the
 * amber "!" of a quote not found or a claim unsupported, and the outlined red
 * "≠" of a contradicted one. Each is a small glyph, so each passes 4.5:1 on
 * what it sits on, and its outline passes 3:1 against the sheet.
 */
describe("Citation check marks", () => {
  test('the amber "!" reads on its wash, and in the text marker on the sheet', () => {
    expect(contrast(token("warning"), token("warning-wash"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(token("warning"), token("sheet"))).toBeGreaterThanOrEqual(4.5);
  });

  test('the red "≠" reads on the sheet, and its outline stands out', () => {
    expect(contrast(token("danger"), token("sheet"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(token("danger"), token("danger-wash"))).toBeGreaterThanOrEqual(4.5);
  });

  test("the quiet tick and its number read on the sheet", () => {
    expect(contrast(token("success"), token("sheet"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(token("ink-meta"), token("sheet"))).toBeGreaterThanOrEqual(4.5);
  });
});
