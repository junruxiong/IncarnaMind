import { afterEach, describe, expect, test } from "vitest";
import {
  pinCitation,
  staysPinned,
  takePinnedCitation,
  unpinCitation,
} from "../../src/renderer/src/pinnedCitation";

afterEach(unpinCitation);

describe("the Citation whose card waits for its Mind to come back", () => {
  test("is given back once, to the Citation at the place it was pinned, and to no other", () => {
    pinCitation("mind-1", "text:12");

    expect(takePinnedCitation("mind-1", "text:13")).toBe(false);
    expect(takePinnedCitation("mind-2", "text:12")).toBe(false);
    expect(takePinnedCitation("mind-1", "text:12")).toBe(true);
    expect(takePinnedCitation("mind-1", "text:12")).toBe(false);
  });

  test("a Citation in a footnote isn't taken for one in the text at the same position", () => {
    pinCitation("mind-1", "footnote-a:3");

    expect(takePinnedCitation("mind-1", "text:3")).toBe(false);
    expect(takePinnedCitation("mind-1", "footnote-b:3")).toBe(false);
    expect(takePinnedCitation("mind-1", "footnote-a:3")).toBe(true);
  });

  test("pinning another Citation lets the first go", () => {
    pinCitation("mind-1", "text:12");
    pinCitation("mind-1", "text:40");

    expect(takePinnedCitation("mind-1", "text:12")).toBe(false);
    expect(takePinnedCitation("mind-1", "text:40")).toBe(true);
  });

  test("stays while its Document or its own Mind is in front, and goes with any other tab", () => {
    expect(staysPinned({ kind: "document", documentId: "d" }, "mind-1")).toBe(true);
    expect(staysPinned({ kind: "mind", mindId: "mind-1" }, "mind-1")).toBe(true);
    expect(staysPinned({ kind: "mind", mindId: "mind-2" }, "mind-1")).toBe(false);
    expect(staysPinned({ kind: "library" }, "mind-1")).toBe(false);
  });
});
