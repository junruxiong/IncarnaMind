/**
 * Fetching a page's pictures (#756): what a picture is comes from its bytes,
 * and one whose header declares more pixels than can be drawn is left out,
 * however small its file.
 */
import { createCanvas } from "@napi-rs/canvas";
import { describe, expect, test } from "vitest";
import type { PageFetch } from "../../src/core/documents/web/fetch";
import { fetchPictures, PICTURE_LIMITS } from "../../src/core/documents/web/pictures";

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
});
