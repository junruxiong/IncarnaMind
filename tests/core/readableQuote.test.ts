import { describe, expect, test } from "vitest";
import type { Core, Document } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";

/*
 * A Citation's quote as its Document's text reads it (#905), through the core's public
 * interface: a word a PDF split at a line end is whole again, a quote that stops at one runs to
 * the end of its word, and a real hyphen stays. What the Citation check finds is unchanged.
 */

const PAGE = [
  "The harbour committee will review the case for the exist-",
  "ence of tides near the pier, and the em-",
  "ployment of staff on the av-",
  "erage day. The long-",
  "term plan, a long-term view of the co-",
  "operate scheme, is to be settled. The office and the council co-operate daily.",
].join("\n");

async function setUp(contents = PAGE) {
  const core = startCore(await createTempDataFolder());
  const sources = await createTempDataFolder();
  const [document] = await addAndProcess(core, [
    await writeSourceFile(sources, "Harbour.txt", contents),
  ]);
  return { core, document: document as Document };
}

const read = (core: Core, document: Document, quote: string) =>
  core.readableQuote({
    documentId: document.id,
    contentHash: document.contentHash,
    pageFrom: null,
    pageTo: null,
    quote,
  });

describe("A Citation's quote, read as words", () => {
  test("a quote cut at a line end runs to the end of its word", async () => {
    const { core, document } = await setUp();

    expect(await read(core, document, "review the case for the exist-")).toBe(
      "review the case for the existence",
    );
  });

  test("a word split inside a quote is whole again", async () => {
    const { core, document } = await setUp();

    expect(
      await read(core, document, "ence of tides near the pier, and the em-\nployment of staff"),
    ).toBe("ence of tides near the pier, and the employment of staff");
    expect(await read(core, document, "ployment of staff on the av-erage day")).toBe(
      "ployment of staff on the average day",
    );
  });

  test("a real hyphen stays where the page writes the word with one elsewhere", async () => {
    const { core, document } = await setUp();

    expect(await read(core, document, "The long-term plan")).toBe("The long-term plan");
    expect(await read(core, document, "view of the co-operate scheme")).toBe(
      "view of the co-operate scheme",
    );
  });

  test("a quote that isn't in the Document, or a Document that isn't there, gives nothing: the quote stays as it is", async () => {
    const { core, document } = await setUp();

    expect(await read(core, document, "Neap tides are the highest.")).toBeNull();
    expect(
      await read(core, { ...document, id: "00000000-0000-4000-8000-000000000000" }, "the pier"),
    ).toBeNull();
  });

  test("a quote of a Document that has nothing to join is as it was, in Chinese too", async () => {
    const { core, document } = await setUp("潮汐每天涨落两次。\n朔望之时潮水最大。");

    expect(await read(core, document, "潮汐每天涨落两次。")).toBe("潮汐每天涨落两次。");
    expect(await read(core, document, "两次。\n朔望之时")).toBe("两次。朔望之时");
  });
});
