import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { TAG_COLOURS } from "../../src/shared/tagColours";

/*
 * A Tag's dot (DESIGN.md, Tags) is all that says which Tag it is on a
 * sidebar row, so it must stand out from where it sits: 3:1 or better,
 * WCAG's contrast for graphics, on the sidebar (frame), a row pointed at
 * (hover) and a selected row (sheet). A light hue (yellow, say) may be under
 * that if its hairline edge, a darker shade, is not. A named chip's text
 * must read on its fill.
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

const surfaces = ["frame", "hover", "sheet"];

describe("Tag dots", () => {
  test("each stands out 3:1 on the sidebar, a row pointed at and a selected row, by itself or by its edge", () => {
    for (const colour of TAG_COLOURS) {
      const dot = token(`tag-${colour}-dot`);
      const edge = token(`tag-${colour}-edge`);
      for (const surface of surfaces) {
        const on = token(surface);
        expect(
          Math.max(contrast(dot, on), contrast(edge, on)),
          `${colour} (${dot}, edge ${edge}) on ${surface}`,
        ).toBeGreaterThanOrEqual(3);
      }
      // The edge is the dot's own colour or a darker shade of it, never a lighter one.
      expect(luminance(edge)).toBeLessThanOrEqual(luminance(dot));
    }
  });

  test("yellow, the lightest, needs its edge; purple stands out by itself", () => {
    const frame = token("frame");
    expect(contrast(token("tag-yellow-dot"), frame)).toBeLessThan(3);
    expect(contrast(token("tag-yellow-edge"), frame)).toBeGreaterThanOrEqual(3);
    expect(token("tag-purple-edge")).toBe(token("tag-purple-dot"));
  });

  test("the review ring stands out 3:1 too, and is darker than the orange dot", () => {
    const ring = token("review");
    for (const surface of surfaces) {
      expect(contrast(ring, token(surface)), surface).toBeGreaterThanOrEqual(3);
    }
    expect(luminance(ring)).toBeLessThan(luminance(token("tag-orange-dot")));
  });
});

describe("Tag chips", () => {
  test("each chip's text reads on its fill at 7:1 or better", () => {
    for (const colour of TAG_COLOURS) {
      const fill = token(`tag-${colour}`);
      const ink = token(`tag-${colour}-ink`);
      expect(contrast(ink, fill), `${colour} (${ink} on ${fill})`).toBeGreaterThanOrEqual(7);
    }
  });
});
