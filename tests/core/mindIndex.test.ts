import { randomUUID } from "node:crypto";
import { join } from "node:path";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test, vi } from "vitest";
import {
  type Core,
  DATABASE_FILE,
  type DocumentTextMatch,
  type MindTextMatch,
} from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient, typeInto } from "../helpers/mindClient";
import { editMind, heading, note, writeMind } from "../helpers/minds";

/** The words inside Minds and Documents, through the core's interface as ⌘K uses it (#116). */

async function mindWith(core: Core, blocks: JSONContent[]): Promise<[string, MindClient]> {
  const { id } = await core.createMind();
  const client = await connectToMind(core, id);
  writeMind(client, blocks);
  await client.settled();
  return [id, client];
}

/** The words a match marks, as the palette makes them bold. */
const marked = (match: Pick<MindTextMatch, "excerpt" | "highlights">) =>
  match.highlights.map((range) => match.excerpt.slice(range.start, range.end));

const bulletList = (...items: string[]): JSONContent => ({
  type: "bulletList",
  attrs: { id: randomUUID() },
  content: items.map((text) => ({
    type: "listItem",
    content: [{ type: "paragraph", content: [{ type: "text", text }] }],
  })),
});

describe("searching inside Minds", () => {
  test("a word only inside a Mind is found, with an excerpt and the heading it sits under", async () => {
    const core = startCore(await createTempDataFolder());
    const [supplier] = await mindWith(core, [
      heading(1, "Supplier review"),
      note("Aldermoor Freight runs the cold chain."),
      heading(2, "Termination"),
      note("Either party may end the agreement on ninety days' written notice."),
    ]);
    await mindWith(core, [note("Notes from the stand-up about the Leeds account.")]);

    const [match, ...others] = await core.searchMinds("ninety");
    expect(others).toEqual([]);
    expect(match).toMatchObject({ mindId: supplier, heading: "Termination" });
    expect(match?.excerpt).toBe(
      "Either party may end the agreement on ninety days' written notice.",
    );
    expect(marked(match as MindTextMatch)).toEqual(["ninety"]);

    expect(await core.searchMinds("giraffe")).toEqual([]);
    expect(await core.searchMinds("  ")).toEqual([]);
  });

  test("a Chinese word is found, written in simplified or traditional characters", async () => {
    const core = startCore(await createTempDataFolder());
    const [finance] = await mindWith(core, [note("本季度的财务报表显示利润增长了百分之十二。")]);
    await mindWith(core, [note("会议纪要：下周讨论招聘计划。")]);

    for (const query of ["财务", "財務", "利润增长"]) {
      const found = await core.searchMinds(query);
      expect(
        found.map((match) => match.mindId),
        query,
      ).toEqual([finance]);
    }
    // The words as the index splits them: "财务报表" is one, which "财务" begins.
    const [match] = await core.searchMinds("财务");
    expect(marked(match as MindTextMatch)).toEqual(["财务报表"]);
  });

  test("every word must be somewhere in the Mind, the last one maybe still being typed", async () => {
    const core = startCore(await createTempDataFolder());
    const [parks] = await mindWith(core, [
      note("Irrigated parks cool the streets around them."),
      note("The effect is strongest on windless nights."),
    ]);

    const ids = async (query: string) =>
      (await core.searchMinds(query)).map((match) => match.mindId);
    expect(await ids("parks windless")).toEqual([parks]);
    expect(await ids("irrig")).toEqual([parks]);
    expect(await ids("parks wind")).toEqual([parks]);
    expect(await ids("parks giraffe")).toEqual([]);
    // The Block with both words is the one shown.
    const [match] = await core.searchMinds("windless nights");
    expect(match?.excerpt).toBe("The effect is strongest on windless nights.");

    // A Mind with both words in one Block comes before one with them in two.
    const [together] = await mindWith(core, [note("Parks feel cooler on windless evenings.")]);
    expect(await ids("parks windless")).toEqual([together, parks]);
  });

  test("words just typed are found at once, and words taken out are not", async () => {
    const core = startCore(await createTempDataFolder());
    const [id, client] = await mindWith(core, [note("Tides in Bergen")]);
    expect(await core.searchMinds("Bergen")).toHaveLength(1);

    typeInto(client, 0, "Tides in Bergen".length, " and the moon");
    await client.settled();
    expect((await core.searchMinds("moon")).map((match) => match.mindId)).toEqual([id]);

    editMind(client, () => [note("Tides in Oslo")]);
    await client.settled();
    expect(await core.searchMinds("Bergen")).toEqual([]);
    expect(await core.searchMinds("Oslo")).toHaveLength(1);
  });

  test("the index follows the User's typing by itself, once the Mind is quiet", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const [, client] = await mindWith(core, [note("Tides")]);
    typeInto(client, 0, "Tides".length, " in Bergen");
    await client.settled();

    await vi.waitFor(
      () =>
        expect(
          queryDatabase<{ text: string }>(
            dataDir,
            "SELECT text FROM mind_blocks WHERE deleted_at IS NULL",
          ),
        ).toEqual([{ text: "Tides in Bergen" }]),
      { timeout: 10_000, interval: 50 },
    );
  });

  test("lines of a list or a table never run together, and a footnote's words are found", async () => {
    const core = startCore(await createTempDataFolder());
    await mindWith(core, [
      bulletList("apple", "banana"),
      {
        type: "paragraph",
        attrs: { id: randomUUID() },
        content: [
          { type: "text", text: "Rainfall" },
          {
            type: "footnote",
            content: [{ type: "text", text: "Measured at Schiphol" }],
          },
        ],
      },
    ]);
    expect(await core.searchMinds("banana")).toHaveLength(1);
    expect(await core.searchMinds("applebanana")).toEqual([]);
    expect(await core.searchMinds("Schiphol")).toHaveLength(1);
  });

  test("a Mind in Recently deleted isn't found until it is restored, and deleted permanently its words go", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const [id] = await mindWith(core, [note("Quarterly forecast for the Leeds office")]);
    expect(await core.searchMinds("forecast")).toHaveLength(1);

    await core.deleteMind(id);
    expect(await core.searchMinds("forecast")).toEqual([]);
    await core.restore([{ kind: "mind", id }]);
    expect((await core.searchMinds("forecast")).map((match) => match.mindId)).toEqual([id]);

    await core.deleteMind(id);
    await core.deletePermanently([{ kind: "mind", id }]);
    expect(
      queryDatabase(dataDir, "SELECT text FROM mind_blocks WHERE text <> '' OR deleted_at IS NULL"),
    ).toEqual([]);
    expect(queryDatabase(dataDir, "SELECT rowid FROM mind_blocks_fts")).toEqual([]);
  });

  test("Minds written before the index, or stored to after it last read them, are read in the background after the next start", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    await mindWith(first, [note("Harbour pilots in Rotterdam")]);
    const [later] = await mindWith(first, [note("Ferry timetables")]);
    first.close();

    const db = openDatabase(join(dataDir, DATABASE_FILE));
    try {
      // The first as if written before the index existed; the second as if the app quit before reading it again.
      db.run("DELETE FROM mind_blocks WHERE mind_id <> ?", [later]);
      db.run("DELETE FROM mind_block_reads WHERE mind_id <> ?", [later]);
      db.run("UPDATE mind_blocks SET text = 'Ferry' WHERE mind_id = ?", [later]);
      db.run("UPDATE mind_block_reads SET updated_at = '2000-01-01T00:00:00.000Z'");
    } finally {
      db.close();
    }

    const core = startCore(dataDir);
    await vi.waitFor(
      async () => {
        expect(await core.searchMinds("Rotterdam")).toHaveLength(1);
        expect(await core.searchMinds("timetables")).toHaveLength(1);
      },
      { timeout: 10_000, interval: 50 },
    );
  });
});

describe("searching the words in Documents", () => {
  test("finds Documents by words in their text, not by their name alone, one Passage each", async () => {
    const core = startCore(await createTempDataFolder());
    const folder = await createSourceFolder();
    const msa = await writeSourceFile(
      folder,
      "Aldermoor MSA.txt",
      "Either party may end this contract for convenience on ninety days' written notice.",
    );
    const handbook = await writeSourceFile(
      folder,
      "员工手册.txt",
      "试用期结束后，合同的通知期为三十天。",
    );
    const [msaDoc, handbookDoc] = await addAndProcess(core, [msa, handbook]);

    const [match, ...others] = await core.searchDocumentText("ninety notice");
    expect(others).toEqual([]);
    expect(match).toMatchObject({ documentId: msaDoc?.id, pageFrom: 1 });
    expect(marked(match as DocumentTextMatch)).toEqual(["ninety", "notice"]);

    expect(await core.searchDocumentText("Aldermoor")).toEqual([]);
    expect(await core.searchDocumentText("ninety giraffe")).toEqual([]);
    expect((await core.searchDocumentText("通知期")).map((each) => each.documentId)).toEqual([
      handbookDoc?.id,
    ]);
  });
});
