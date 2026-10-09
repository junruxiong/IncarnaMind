import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { TAG_COLOURS } from "../../src/shared/tagColours";

/*
 * A Tag's dot (DESIGN.md, Tags) is all that says which Tag it is on a
 * sidebar row, so it must stand out from where it sits: 3:1 or better,
 * WCAG's contrast for graphics, on the sidebar (frame), a row pointed at
 * (hover) and a selected row (sheet).
 */

const styles = readFileSync(join(__dirname, "../../src/renderer/src/styles.css"), "utf8");

/** A colour token's value, from styles.css. */
function token(name: string): string {
  const match = styles.match(new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6});`));
  if (!match?.[1]) throw new Error(`No --color-${name} in styles.css.`);
  return match[1];
}

const luminance = (hex: string) => {
  const [r, g, b] = [1, 3, 5].map((at) => {
    const channel = Number.parseInt(hex.slice(at, at + 2), 16) / 255;
    return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

const contrast = (a: string, b: string) => {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (light + 0.05) / (dark + 0.05);
};

describe("Tag dots", () => {
  test("every colour has one, 3:1 or better on the sidebar, a row pointed at and a selected row", () => {
    const surfaces = ["frame", "hover", "sheet"].map(token);
    for (const colour of TAG_COLOURS) {
      const dot = token(`tag-${colour}-dot`);
      for (const surface of surfaces) {
        expect(contrast(dot, surface), `${colour} (${dot}) on ${surface}`).toBeGreaterThanOrEqual(
          3,
        );
      }
    }
  });
});
