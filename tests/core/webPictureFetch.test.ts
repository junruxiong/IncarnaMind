/**
 * Fetching a page's pictures (#756): what a picture is comes from its bytes,
 * and one whose header declares more pixels than can be drawn is left out,
 * however small its file.
 */
import { createCanvas } from "@napi-rs/canvas";
import { describe, expect, test } from "vitest";
import type { PageFetch } from "../../src/core/documents/web/fetch";
import {
  fetchPictures,
  PICTURE_LIMITS,
  pictureSize,
  pictureType,
} from "../../src/core/documents/web/pictures";

/** A PNG's first bytes declaring `width` × `height`, with no pixels after them. */
function pngHeader(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(33);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(8, 13);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12);
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes;
}

/** A GIF whose screen is `screen` px square, and whose first frame is `frame` px square. */
function gif(screen: number, frame: number): Uint8Array {
  const bytes = new Uint8Array(13 + 10 + 2);
  bytes.set(new TextEncoder().encode("GIF89a"), 0);
  const view = new DataView(bytes.buffer);
  view.setUint16(6, screen, true);
  view.setUint16(8, screen, true);
  bytes[13] = 0x2c;
  view.setUint16(18, frame, true);
  view.setUint16(20, frame, true);
  return bytes;
}

/**
 * An AVIF's first bytes: its `ftyp`, a `colr` box of `padding` bytes, then an
 * `ispe` box for each size given, in order.
 */
function avif(sizes: number[], padding = 0): Uint8Array {
  const bytes = new Uint8Array(16 + padding + 20 * sizes.length + 16);
  const text = new TextEncoder();
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 16);
  bytes.set(text.encode("ftypavif"), 4);
  if (padding > 0) {
    view.setUint32(16, padding);
    bytes.set(text.encode("colr"), 20);
  }
  sizes.forEach((size, index) => {
    const at = 16 + padding + index * 20;
    view.setUint32(at, 20);
    bytes.set(text.encode("ispe"), at + 4);
    view.setUint32(at + 12, size);
    view.setUint32(at + 16, size);
  });
  return bytes;
}

const served =
  (pictures: Record<string, Uint8Array>): PageFetch =>
  async (address) => {
    const bytes = pictures[address];
    return bytes
      ? new Response(Buffer.from(bytes), { headers: { "Content-Type": "image/png" } })
      : new Response("Not found", { status: 404 });
  };

describe("fetching a page's pictures", () => {
  test("a picture declaring more pixels than can be drawn is left out, however small its file; one that isn't a picture isn't kept", async () => {
    const canvas = createCanvas(200, 120);
    const chart = canvas.toBuffer("image/png");
    const fetched = await fetchPictures(
      [
        { index: 0, address: "http://images.example/chart.png" },
        { index: 1, address: "http://images.example/bomb.png" },
        { index: 2, address: "http://images.example/not-a-picture.png" },
      ],
      served({
        "http://images.example/chart.png": chart,
        "http://images.example/bomb.png": pngHeader(20_000, 20_000),
        "http://images.example/not-a-picture.png": new TextEncoder().encode("<html></html>"),
      }),
    );
    expect(fetched.kept.map((each) => [each.index, each.width, each.height])).toEqual([
      [0, 200, 120],
    ]);
    expect(fetched.leftOut).toBe(1);
    expect(20_000 * 20_000).toBeGreaterThan(PICTURE_LIMITS.maxPixels);
  });

  test("a GIF whose first frame reaches past its screen, or an AVIF with a larger image after a small one or a large box, is as large as what is drawn", () => {
    const sizeOf = (bytes: Uint8Array) => {
      const type = pictureType(bytes);
      return type && pictureSize(bytes, type);
    };
    expect(sizeOf(gif(100, 100))).toEqual({ width: 100, height: 100 });
    expect(sizeOf(gif(100, 20_000))).toEqual({ width: 20_000, height: 20_000 });
    expect(sizeOf(avif([100]))).toEqual({ width: 100, height: 100 });
    expect(sizeOf(avif([64, 20_000]))).toEqual({ width: 20_000, height: 20_000 });
    // Its main image past a large colour profile is found too.
    expect(sizeOf(avif([20_000], 200_000))).toEqual({ width: 20_000, height: 20_000 });
    expect(20_000 * 20_000).toBeGreaterThan(PICTURE_LIMITS.maxPixels);
  });
});
