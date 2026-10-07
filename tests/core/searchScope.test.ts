import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { Core, CoreAdapters, Document, Reranker, SearchScope } from "../../src/core";
import { translate } from "../../src/shared/i18n";
import {
  answerEnded,
  askAndFinish,
  citingModel,
  setUpWithDocuments,
  shownPassages,
} from "../helpers/citations";
import { startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import {
  answerIn,
  answerText,
  editMind,
  question,
  readMind,
  scopeAttributes,
  writeMind,
} from "../helpers/minds";
import { type ModelCall, promptOf, scriptedModel } from "../helpers/models";

/**
 * The library: a Folder tree Coast › Rivers › Deltas, a Kitchen Folder, a Tag,
 * and two unfiled Documents. Every Document mentions tides, so a search for
 * tides would find each of them.
 */
const FILES = [
  { name: "Harbour.txt", contents: "Harbour tides rise twice a day along the quay.\n" },
  { name: "Estuary.txt", contents: "Estuary tides push salt water up the river.\n" },
  { name: "Delta.txt", contents: "Delta tides spread across the mud flats.\n" },
  { name: "Moon.txt", contents: "The Moon's pull raises the tides on both sides of the Earth.\n" },
  { name: "Almanac.txt", contents: "The almanac lists the times of the tides for the year.\n" },
  { name: "Recipes.txt", contents: "Cook mussels at low tides, with garlic and white wine.\n" },
];

const ALL = ["Almanac", "Delta", "Estuary", "Harbour", "Moon", "Recipes"];

/** A model that calls Tools: it searches once for tides, then answers without citing. */
function searchingModel(): MockLanguageModelV4 {
  return scriptedModel((call: ModelCall) => {
    if (!call.tools.includes("search_documents")) return { text: "No Tools were offered." };
    if (call.results.length === 0) {
      return { calls: [{ tool: "search_documents", input: { query: "tides" } }] };
    }
    return { text: "Here is what your Documents say." };
  });
}

/**
 * The library in a core whose searches are recorded: the Documents of the
 * Passages hybrid search hands on (to the reranker), for each search, by name.
 */
async function setUpLibrary(model: MockLanguageModelV4 = searchingModel()) {
  const searches: string[][] = [];
  const reranker: Reranker = async (_query, candidates) => {
    searches.push([...new Set(candidates.map((candidate) => candidate.documentName))].sort());
    return [...candidates];
  };
  const setup = await setUpWithDocuments(model, FILES, { reranker });
  const { core } = setup;
  const documents = Object.fromEntries(
    setup.documents.map((document) => [document.name, document]),
  ) as Record<string, Document>;
  const id = (name: string) => (documents[name] as Document).id;

  const coast = await core.createFolder({ name: "Coast" });
  const rivers = await core.createFolder({ name: "Rivers", parentId: coast.id });
  const deltas = await core.createFolder({ name: "Deltas", parentId: rivers.id });
  const kitchen = await core.createFolder({ name: "Kitchen" });
  const empty = await core.createFolder({ name: "Empty" });
  await core.moveDocument(id("Harbour"), coast.id);
  await core.moveDocument(id("Estuary"), rivers.id);
  await core.moveDocument(id("Delta"), deltas.id);
  await core.moveDocument(id("Recipes"), kitchen.id);
  const astronomy = await core.createTag({ name: "Astronomy" });
  await core.addDocumentTag(id("Moon"), astronomy.id);

  return {
    ...setup,
    searches,
    reranker,
    id,
    folders: { coast, rivers, deltas, kitchen, empty },
    tags: { astronomy },
  };
}

/** Writes a Question with a Search scope, asks it, and waits for its Answer. */
async function askScoped(
  core: Core,
  client: MindClient,
  mindId: string,
  scope: Partial<SearchScope>,
  text = "What do my Documents say about tides?",
) {
  const asked = question(text, undefined, scope);
  writeMind(client, [asked]);
  await client.settled();
  const result = await core.askQuestion({ mindId, questionId: asked.attrs.id });
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  const ended = await answerEnded(core, result.answerId);
  if (ended.event !== "finished") throw new Error(JSON.stringify(ended.payload));
  return { questionId: asked.attrs.id, answerId: result.answerId, finished: ended.payload };
}

/** The Documents of the Passages the model was shown by its searches, by name. */
function shownDocuments(model: MockLanguageModelV4): string[] {
  const names = new Set<string>();
  for (const call of model.doStreamCalls) {
    for (const message of call.prompt) {
      if (message.role !== "tool") continue;
      for (const part of message.content) {
        if (part.type !== "tool-result" || part.output.type !== "text") continue;
        for (const passage of shownPassages(part.output.value)) names.add(passage.document);
      }
    }
  }
  return [...names].sort();
}

describe("Resolving a Search scope", { timeout: 30_000 }, () => {
  test("a Question with no Search scope searches every Document", async () => {
    const { core, client, mind, searches } = await setUpLibrary();

    await askScoped(core, client, mind.id, {});

    expect(searches).toEqual([ALL]);
  });

  test("a Folder covers its Documents and those of its sub-Folders, at any depth", async () => {
    const { core, client, mind, searches, folders } = await setUpLibrary();

    await askScoped(core, client, mind.id, { folderIds: [folders.coast.id] });
    await askScoped(core, client, mind.id, { folderIds: [folders.rivers.id] });
    await askScoped(core, client, mind.id, { folderIds: [folders.deltas.id] });

    expect(searches).toEqual([["Delta", "Estuary", "Harbour"], ["Delta", "Estuary"], ["Delta"]]);
  });

  test("a Tag covers the Documents that carry it, and a Document covers itself", async () => {
    const { core, client, mind, searches, tags, id } = await setUpLibrary();

    await askScoped(core, client, mind.id, { tagIds: [tags.astronomy.id] });
    await askScoped(core, client, mind.id, { documentIds: [id("Almanac")] });

    expect(searches).toEqual([["Moon"], ["Almanac"]]);
  });

  test("Folders, Tags and Documents together cover the union of their Documents", async () => {
    const { core, client, mind, searches, folders, tags, id } = await setUpLibrary();

    await askScoped(core, client, mind.id, {
      folderIds: [folders.deltas.id, folders.kitchen.id],
      tagIds: [tags.astronomy.id],
      // Delta is in a Folder of the scope too: it counts once.
      documentIds: [id("Almanac"), id("Delta")],
    });

    expect(searches).toEqual([["Almanac", "Delta", "Moon", "Recipes"]]);
  });

  test("deleted Folders, Tags and Documents are ignored", async () => {
    const { core, client, mind, searches, folders, tags, id } = await setUpLibrary();
    const comets = await core.createTag({ name: "Comets" });
    await core.addDocumentTag(id("Almanac"), comets.id);
    await core.deleteFolder(folders.kitchen.id);
    await core.deleteTag(comets.id);
    await core.deleteDocument(id("Harbour"));

    await askScoped(core, client, mind.id, {
      folderIds: [folders.kitchen.id, folders.deltas.id],
      tagIds: [comets.id, tags.astronomy.id],
      documentIds: [id("Harbour")],
    });

    // Recipes was unfiled with the Kitchen Folder; Almanac lost the deleted Tag.
    expect(searches).toEqual([["Delta", "Moon"]]);
  });

  test("the scope is resolved when the Question is asked: Documents filed since are in it, those moved out aren't", async () => {
    const { core, client, mind, searches, folders, id } = await setUpLibrary();
    await core.moveDocument(id("Almanac"), folders.deltas.id);
    await core.moveDocument(id("Estuary"), null);

    await askScoped(core, client, mind.id, { folderIds: [folders.rivers.id] });

    expect(searches).toEqual([["Almanac", "Delta"]]);
  });
});

describe("A Search scope with no Documents to search", { timeout: 30_000 }, () => {
  const EMPTY = translate("en", "scope.answer.empty");

  test("gets an Answer saying so, instead of a search of every Document: nothing is sent", async () => {
    const model = searchingModel();
    const { core, client, mind, searches, folders } = await setUpLibrary(model);

    const { answerId, finished } = await askScoped(core, client, mind.id, {
      folderIds: [folders.empty.id],
    });

    expect(answerText(client, answerId)).toBe(EMPTY);
    expect(finished).toMatchObject({ status: "done", citations: [], citationSupport: null });
    // No model wrote it: it names none, and none was asked.
    expect(answerIn(client, answerId).attrs).toMatchObject({ modelId: null, status: "done" });
    expect(model.doStreamCalls).toHaveLength(0);
    expect(searches).toEqual([]);
  });

  test("a scope whose Folders, Tags and Documents were all deleted is empty, not every Document", async () => {
    const model = searchingModel();
    const { core, client, mind, searches, folders, id } = await setUpLibrary(model);
    await core.deleteFolder(folders.kitchen.id);
    await core.deleteDocument(id("Almanac"));

    const { answerId } = await askScoped(core, client, mind.id, {
      folderIds: [folders.kitchen.id],
      documentIds: [id("Almanac")],
    });

    expect(answerText(client, answerId)).toBe(EMPTY);
    expect(model.doStreamCalls).toHaveLength(0);
    expect(searches).toEqual([]);
  });

  test("says so in the interface language", async () => {
    const { core, client, mind, folders } = await setUpLibrary();
    await core.updateSettings({ user: { language: "zh-CN" } });

    const { answerId } = await askScoped(core, client, mind.id, { folderIds: [folders.empty.id] });

    expect(answerText(client, answerId)).toBe(translate("zh-CN", "scope.answer.empty"));
  });
});

describe("Searching within a Search scope", { timeout: 30_000 }, () => {
  test("the model's searches return Passages only from the scope, and it is told what it is limited to", async () => {
    const model = searchingModel();
    const { core, client, mind, folders } = await setUpLibrary(model);

    await askScoped(core, client, mind.id, { folderIds: [folders.rivers.id] });

    expect(shownDocuments(model)).toEqual(["Delta", "Estuary"]);
    expect(promptOf(model)[0]?.text).toContain("limited this Question to 2 Documents");
  });

  test("an Answer cites a Passage from the scope", async () => {
    const model = citingModel({
      query: "tides",
      records: (passages) =>
        passages.map((passage, index) => ({
          marker: index + 1,
          passage: passage.id,
          quote: passage.text.split("\n").at(-1) ?? "",
        })),
      answer: "Your Documents describe tides [^1] [^2].",
    });
    const { core, client, mind, tags, id } = await setUpLibrary(model);

    const { finished } = await askScoped(core, client, mind.id, {
      tagIds: [tags.astronomy.id],
      documentIds: [id("Harbour")],
    });

    expect(finished.citations.map((citation) => citation.documentName).sort()).toEqual([
      "Harbour",
      "Moon",
    ]);
    expect(finished.citations.every((citation) => citation.check === "found")).toBe(true);
  });

  test("a model that can't call Tools searches once, only within the scope", async () => {
    const shown: string[] = [];
    const model = scriptedModel((call) => {
      if (call.tools.length > 0) {
        return { error: { status: 400, message: "this model does not support tools" } };
      }
      shown.push(...shownPassages(call.system).map((passage) => passage.document));
      return { text: JSON.stringify({ answer: "Tides, in short.", citations: [] }) };
    });
    const { core, client, mind, folders, searches } = await setUpLibrary(model);

    const { finished } = await askScoped(core, client, mind.id, {
      folderIds: [folders.deltas.id, folders.kitchen.id],
    });

    expect(finished.citationSupport).toBe("structured-output");
    expect(searches).toEqual([["Delta", "Recipes"]]);
    expect([...new Set(shown)].sort()).toEqual(["Delta", "Recipes"]);
  });
});

describe("A Question's Search scope", { timeout: 30_000 }, () => {
  test("is kept on the Question across a restart, and regenerating uses the scope as it is now", async () => {
    const model = searchingModel();
    const { core, client, mind, dataDir, folders, tags, reranker, searches, models } =
      await setUpLibrary(model);
    const { questionId, answerId } = await askScoped(core, client, mind.id, {
      folderIds: [folders.rivers.id],
      tagIds: [tags.astronomy.id],
    });
    core.close();

    const adapters: Partial<CoreAdapters> = { createChatModel: models.createChatModel, reranker };
    const restarted = startCore(dataDir, adapters);
    const reader = await connectToMind(restarted, mind.id);
    let stored: Record<string, unknown> | null = null;
    readMind(reader).forEach((block) => {
      if (block.attrs.id === questionId) stored = block.attrs;
    });
    expect(stored).toMatchObject(
      scopeAttributes({ folderIds: [folders.rivers.id], tagIds: [tags.astronomy.id] }),
    );

    await restarted.regenerateAnswer({ mindId: mind.id, answerId });
    await answerEnded(restarted, answerId);

    // The User narrows the scope to the Tag; regenerating searches only that.
    editMind(reader, (blocks) =>
      blocks.map((block) =>
        block.attrs?.id === questionId
          ? {
              ...block,
              attrs: { ...block.attrs, ...scopeAttributes({ tagIds: [tags.astronomy.id] }) },
            }
          : block,
      ),
    );
    await reader.settled();
    await restarted.regenerateAnswer({ mindId: mind.id, answerId });
    await answerEnded(restarted, answerId);

    expect(searches).toEqual([
      ["Delta", "Estuary", "Moon"],
      ["Delta", "Estuary", "Moon"],
      ["Moon"],
    ]);
  });

  test("an Answer to an unscoped Question still sees every Document after a scoped one", async () => {
    const { core, client, mind, searches, folders } = await setUpLibrary();

    await askScoped(core, client, mind.id, { folderIds: [folders.deltas.id] });
    await askAndFinish(core, client, mind.id, "And everything else?");

    expect(searches).toEqual([["Delta"], ALL]);
  });
});
