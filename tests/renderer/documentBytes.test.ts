import { afterEach, describe, expect, test, vi } from "vitest";
import {
  loadDocumentBytes,
  MissingDocumentError,
  TooLargeDocumentError,
} from "../../src/renderer/src/viewer/documentBytes";

function answerWith(status: number) {
  class FakeRequest {
    status = status;
    response = new ArrayBuffer(0);
    responseType = "";
    onload: (() => void) | null = null;
    open() {}
    abort() {}
    send() {
      this.onload?.();
    }
  }
  vi.stubGlobal("XMLHttpRequest", FakeRequest);
}

describe("reading a Document's file for a view", () => {
  afterEach(() => vi.unstubAllGlobals());

  test("a 413 is a file too large to open, and a 404 a Document that is gone", async () => {
    answerWith(413);
    await expect(loadDocumentBytes("a", new AbortController().signal)).rejects.toBeInstanceOf(
      TooLargeDocumentError,
    );
    answerWith(404);
    await expect(loadDocumentBytes("a", new AbortController().signal)).rejects.toBeInstanceOf(
      MissingDocumentError,
    );
  });
});
