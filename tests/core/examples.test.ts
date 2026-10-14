import { access, readdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test, vi } from "vitest";
import {
  ANSWER_BLOCK,
  CITATION_NODE,
  type Core,
  type ExampleGroup,
  QUESTION_BLOCK,
} from "../../src/core";
import { EXAMPLE_SETS } from "../../src/core/exampleSets";
import type { Language } from "../../src/core/language";
import { createTempDataFolder, startCore } from "../helpers/core";
import { appendParagraph, connectToMind } from "../helpers/mindClient";
import { readMind } from "../helpers/minds";

/** The example Documents the app ships. */
const EXAMPLES = resolve(__dirname, "../../resources/examples");

async function startWithExamples(language: Language = "en") {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, {
    paths: { dataDir, examples: EXAMPLES },
    systemLanguages: () => [language === "zh-CN" ? "zh-CN" : "en-US"],
  });
  return { core, dataDir };
}

/** The example Mind as the editor would read it. */
async function exampleMind(core: Core, mindId: string): Promise<ProseMirrorNode> {
  const client = await connectToMind(core, mindId);
  return readMind(client);
}

/** Each Citation's attributes, in order. */
function citationsOf(doc: ProseMirrorNode): Record<string, unknown>[] {
  const found: Record<string, unknown>[] = [];
  doc.descendants((node) => {
    if (node.type.name === CITATION_NODE) found.push(node.attrs);
  });
  return found;
}

describe("the example Mind", { timeout: 30_000 }, () => {
  test("is made on a first run, once, and its Citations quote its Documents and are found", async () => {
    const { core } = await startWithExamples();

    const made = await core.offerExamples();
    expect(made).toMatchObject({
      available: true,
      mindId: expect.any(String),
      answerId: expect.any(String),
    });
    // Offered once only, even after the example Mind is gone.
    expect(await core.offerExamples()).toBeNull();

    const mindId = made?.mindId as string;
    const [mind] = await core.listMinds();
    expect(mind).toMatchObject({ id: mindId, title: "Where tea comes from" });
    // Its Documents appear once the folder is scanned.
    await vi.waitFor(async () => {
      const documents = await core.listDocuments({ linkedFolderId: made?.linkedFolderId });
      expect(documents.map((each) => each.name).sort()).toEqual([
        "Tea · Wikipedia",
        "茶 · 维基百科",
      ]);
    });

    // Its blocks: a Note, the Question (scoped to the examples' folder), its Answer, an empty line.
    const doc = await exampleMind(core, mindId);
    expect(doc.content.content.map((block) => block.type.name)).toEqual([
      "paragraph",
      QUESTION_BLOCK,
      ANSWER_BLOCK,
      "paragraph",
    ]);
    expect(doc.child(1).textContent).toBe(
      "Where does tea come from, and how did it spread around the world?",
    );

    // Once both Documents are read, both Citations are checked: found, in the right sections.
    await vi.waitFor(
      async () => {
        const checks = citationsOf(await exampleMind(core, mindId)).map((each) => each.check);
        expect(checks).toEqual(["found", "found"]);
      },
      { timeout: 20_000, interval: 100 },
    );
    const [english, chinese] = citationsOf(await exampleMind(core, mindId));
    expect(english).toMatchObject({
      documentName: "Tea · Wikipedia",
      quote:
        "Tea drinking may have begun in the region of Yunnan, where it was used for medicinal purposes.",
      location: { kind: "section", heading: "Early tea drinking" },
    });
    expect(chinese).toMatchObject({
      documentName: "茶 · 维基百科",
      location: { kind: "section", heading: "在世界各地的传播" },
    });
  });

  test("are removed with their Documents and the copies of their files, and can be made again", async () => {
    const { core, dataDir } = await startWithExamples();
    const made = await core.createExamples();
    const folder = join(dataDir, "Examples");
    expect((await readdir(folder)).sort()).toEqual([
      "LICENSE",
      "Tea · Wikipedia.md",
      "茶 · 维基百科.md",
    ]);

    await core.removeExamples();
    expect(await core.getExamples()).toEqual({
      group: "tea",
      available: true,
      mindId: null,
      linkedFolderId: null,
      answerId: null,
      more: expect.any(Array),
    });
    expect(await core.listMinds()).toEqual([]);
    expect(await core.listLinkedFolders()).toEqual([]);
    await expect(access(folder)).rejects.toThrow();

    const again = await core.createExamples();
    expect(again.mindId).not.toBe(made.mindId);
    expect(await core.listMinds()).toHaveLength(1);
  });

  test("deleting the example Mind on its own leaves its Linked folder marked, and making them again starts afresh", async () => {
    const { core } = await startWithExamples();
    const made = await core.createExamples();
    await core.deleteMind(made.mindId as string);
    expect(await core.getExamples()).toEqual({
      group: "tea",
      available: true,
      mindId: null,
      linkedFolderId: made.linkedFolderId,
      answerId: null,
      more: expect.any(Array),
    });

    const again = await core.createExamples();
    expect(again.mindId).not.toBe(made.mindId);
    expect(again.linkedFolderId).not.toBe(made.linkedFolderId);
    expect((await core.listLinkedFolders()).map((linked) => linked.id)).toEqual([
      again.linkedFolderId,
    ]);
    expect(await core.listMinds()).toHaveLength(1);
  });

  test("aren't offered when there are Minds already, nor when the app ships none", async () => {
    const { core } = await startWithExamples();
    await core.createMind({ title: "Mine" });
    expect(await core.offerExamples()).toBeNull();
    expect(await core.listMinds()).toHaveLength(1);

    const dataDir = await createTempDataFolder();
    const without = startCore(dataDir);
    expect(await without.offerExamples()).toBeNull();
    expect(await without.getExamples()).toEqual({
      group: "tea",
      available: false,
      mindId: null,
      linkedFolderId: null,
      answerId: null,
      more: expect.any(Array),
    });
  });
});

describe("the example Mind follows the interface language (#709)", { timeout: 60_000 }, () => {
  const foundIn = async (core: Core, mindId: string, quoteStart: string) =>
    vi.waitFor(
      async () => {
        const citations = citationsOf(await exampleMind(core, mindId));
        expect(citations.map((each) => each.check)).toEqual(["found", "found"]);
        expect(String(citations[0]?.quote)).toContain(quoteStart);
      },
      { timeout: 20_000, interval: 100 },
    );

  // protects=an example nobody changed is swapped, in place, for the same one in the new language, with its Documents, and back again; fails_when=the Mind stays in the old language, is replaced by a new Mind, keeps the old Documents, or a Citation isn't found; no_claim=the interface's own words.
  test("an untouched example becomes the other language's, in place, and back", async () => {
    const { core, dataDir } = await startWithExamples("en");
    const made = await core.createExamples();
    await foundIn(core, made.mindId as string, "Tea drinking");

    await core.updateSettings({ user: { language: "zh-CN" } });
    await vi.waitFor(async () => {
      expect((await core.listMinds()).map((mind) => mind.title)).toEqual([
        EXAMPLE_SETS.tea["zh-CN"].title,
      ]);
    });
    const doc = await exampleMind(core, made.mindId as string);
    expect(doc.child(1).textContent).toBe(EXAMPLE_SETS.tea["zh-CN"].question);
    await foundIn(
      core,
      made.mindId as string,
      EXAMPLE_SETS.tea["zh-CN"].quotes[0]?.quote.slice(0, 6) ?? "",
    );
    expect((await core.listMinds()).length).toBe(1);
    expect((await core.listLinkedFolders()).length).toBe(1);
    await expect(access(join(dataDir, "Examples"))).rejects.toThrow();
    expect((await readdir(join(dataDir, "示例文档"))).length).toBeGreaterThan(1);

    await core.updateSettings({ user: { language: "en" } });
    await vi.waitFor(async () => {
      expect((await core.listMinds()).map((mind) => mind.title)).toEqual(["Where tea comes from"]);
    });
    await foundIn(core, made.mindId as string, "Tea drinking");
    await expect(access(join(dataDir, "示例文档"))).rejects.toThrow();
  });

  // protects=an example the User wrote in or changed keeps its language and its Documents; fails_when=it is replaced and their words are lost; no_claim=none beyond protects.
  test("an example the User changed stays as it is", async () => {
    const { core } = await startWithExamples("en");
    const made = await core.createExamples();
    const client = await connectToMind(core, made.mindId as string);
    appendParagraph(client, "My own note.");
    await client.settled();
    await core.updateSettings({ user: { language: "zh-CN" } });
    expect((await core.listMinds()).map((mind) => mind.title)).toEqual(["Where tea comes from"]);
  });

  // protects=an example whose Documents a check result relies on keeps them when the language changes, so that result isn't left needing a check for a change the User didn't make (#819); fails_when=the Documents are swapped and the check goes stale; no_claim=none beyond protects.
  test("an example whose Documents a check relies on keeps its language and Documents", async () => {
    const { core } = await startWithExamples("en");
    const made = await core.createExamples();
    await foundIn(core, made.mindId as string, "Tea drinking");
    const [tea] = await core.listDocuments({ linkedFolderId: made.linkedFolderId as string });
    core.retention.retain("check-result", "a-mind-of-the-users", [
      { documentId: tea?.id as string, contentHash: tea?.contentHash as string },
    ]);

    // updateSettings returns once the examples have followed the language, or not.
    await core.updateSettings({ user: { language: "zh-CN" } });

    expect((await core.listMinds()).map((mind) => mind.title)).toEqual(["Where tea comes from"]);
    expect((await core.listDocuments()).map((each) => each.id)).toContain(tea?.id);

    // Once nothing relies on them, the next change of language swaps them.
    core.retention.release("check-result", "a-mind-of-the-users");
    await core.updateSettings({ user: { language: "en" } });
    await core.updateSettings({ user: { language: "zh-CN" } });
    await vi.waitFor(async () => {
      expect((await core.listMinds()).map((mind) => mind.title)).toEqual([
        EXAMPLE_SETS.tea["zh-CN"].title,
      ]);
    });
  });
});

const GROUPS = ["papers", "reports", "contracts", "meetings"] as const satisfies ExampleGroup[];
const LANGUAGES = ["en", "zh-CN"] as const satisfies Language[];

describe("the example Minds of each group", { timeout: 90_000 }, () => {
  for (const group of GROUPS) {
    for (const language of LANGUAGES) {
      test(`${group} in ${language}: every Citation is found in its own Document, and every Document's licence is listed`, async () => {
        const set = EXAMPLE_SETS[group][language];
        const { core, dataDir } = await startWithExamples(language);

        const made = await core.createExamples(group);
        expect(made).toMatchObject({ group, available: true, mindId: expect.any(String) });
        const [mind] = await core.listMinds();
        expect(mind).toMatchObject({ id: made.mindId, title: set.title });

        // Its Documents are what the folder holds, plus the licence that travels with them.
        const shipped = (await readdir(join(EXAMPLES, set.source))).sort();
        const copied = (await readdir(join(dataDir, set.folder))).sort();
        expect(copied).toEqual([...shipped, "LICENSE"].sort());

        // Every Document has its source and licence recorded.
        const licence = (await readFile(join(EXAMPLES, "LICENSE"), "utf8")).replace(/\s+/g, " ");
        for (const file of shipped) expect(licence, file).toContain(file);

        // All its Documents are read, and every Citation is found.
        await vi.waitFor(
          async () => {
            const documents = await core.listDocuments({ linkedFolderId: made.linkedFolderId });
            expect(documents).toHaveLength(shipped.length);
            const checks = citationsOf(await exampleMind(core, made.mindId as string)).map(
              (each) => each.check,
            );
            expect(checks).toEqual(set.quotes.map(() => "found"));
          },
          { timeout: 60_000, interval: 200 },
        );
        const citations = citationsOf(await exampleMind(core, made.mindId as string));
        expect(citations.map((each) => each.quote)).toEqual(set.quotes.map((each) => each.quote));
        expect(citations.map((each) => each.documentName)).toEqual(
          set.quotes.map((each) => each.document),
        );

        // The first Question is there for the User to ask.
        const doc = await exampleMind(core, made.mindId as string);
        expect(doc.child(1).type.name).toBe(QUESTION_BLOCK);
        expect(doc.child(1).textContent).toBe(set.question);
      });
    }
  }

  test("each group's example is its own: making one leaves the others alone", async () => {
    const { core } = await startWithExamples();
    const papers = await core.createExamples("papers");
    const meetings = await core.createExamples("meetings");
    expect(papers.mindId).not.toBe(meetings.mindId);
    expect(await core.listMinds()).toHaveLength(2);
    expect((await core.getExamples()).mindId).toBeNull();

    await core.removeExamples("papers");
    expect((await core.getExamples("papers")).mindId).toBeNull();
    expect((await core.getExamples("meetings")).mindId).toBe(meetings.mindId);
    expect(await core.listMinds()).toHaveLength(1);
  });
});
