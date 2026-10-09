import { describe, expect, test } from "vitest";
import {
  documentFileUrl,
  documentIdFromUrl,
  documentImageFromUrl,
  documentImageUrl,
} from "../../src/shared/documentViewer";

describe("the URLs the viewer reads Documents' files at", () => {
  test("a Document's file, and a picture beside a Markdown Document, each name only themselves", () => {
    const id = "6f1c2a9e-1111-4b6c-9a1e-2b3c4d5e6f70";
    expect(documentIdFromUrl(documentFileUrl(id))).toBe(id);
    expect(documentImageFromUrl(documentFileUrl(id))).toBeNull();

    const url = documentImageUrl(id, "figures/site map #2.png");
    expect(documentIdFromUrl(url)).toBeNull();
    expect(documentImageFromUrl(url)).toEqual({ documentId: id, path: "figures/site map #2.png" });
    expect(documentImageFromUrl("https://example.com/a.png")).toBeNull();
    expect(documentImageFromUrl(`${documentFileUrl(id)}a.png?x=1`)).toBeNull();
  });
});
