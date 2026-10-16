import { describe, expect, test } from "vitest";
import type { SharePagePreview } from "../../src/core/api";
import { LARGE_BYTES, sharedSize, withFewerPictures } from "../../src/renderer/src/sharePageSize";

const MB = 1_000_000;

const preview: SharePagePreview = {
  textBytes: 200_000,
  documents: [
    {
      documentId: "results",
      name: "Results",
      quotes: 5,
      pictures: 4,
      pictureBytes: 9 * MB,
      textOnly: false,
    },
    {
      documentId: "facility",
      name: "Facility",
      quotes: 3,
      pictures: 3,
      pictureBytes: 12 * MB,
      textOnly: false,
    },
    {
      documentId: "waiver",
      name: "Waiver",
      quotes: 2,
      pictures: 2,
      pictureBytes: 3 * MB,
      textOnly: false,
    },
    { documentId: "memo", name: "Memo", quotes: 2, pictures: 0, pictureBytes: 0, textOnly: false },
  ],
};

describe("the shared page's size in the sheet", () => {
  test("it adds the pictures of the Documents that keep theirs, at once", () => {
    expect(sharedSize(preview, new Set())).toEqual({ bytes: 24_200_000, pictures: 9 });
    expect(sharedSize(preview, new Set(["facility"]))).toEqual({ bytes: 12_200_000, pictures: 6 });
  });

  test("Use fewer pictures keeps the heaviest Documents as text until the file is under the warning", () => {
    const fewer = withFewerPictures(preview, new Set());

    expect([...fewer]).toEqual(["facility"]);
    expect(sharedSize(preview, fewer).bytes).toBeLessThan(LARGE_BYTES);
    expect([...withFewerPictures(preview, new Set(["waiver"]))]).toEqual(["waiver", "facility"]);
  });
});
