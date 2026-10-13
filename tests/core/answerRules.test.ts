/**
 * Two things a careful reader catches (#548). Each Passage handed to the
 * model carries its Document's date: the date it states (its reference
 * record's, else the one its file gives), else the day its file was last
 * modified. And every way of answering is told to say first when the Question
 * takes for granted what the Passages don't say, and to lead with the newer of
 * two Documents that disagree, with examples in English and Chinese.
 */
import { utimes } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import { createCitationSession } from "../../src/core/answers/citations";
import { datedDocumentsNote, documentInstructions } from "../../src/core/answers/prompt";
import { askAndFinish, setUpWithDocuments, shownPassages } from "../helpers/citations";
import { createTempDataFolder } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { type ModelCall, scriptedModel } from "../helpers/models";

/** A phrase of each rule, and of each rule's Chinese example. */
const RULES = [
  "If the Question takes for granted what the Passages don't say",
  "文档中没有董事会批准的记录",
  "give different figures for the same thing (one updates the other), lead with the newer Document by its date",
  "较早的手册（2024 年 1 月）写的是 6 小时",
];

const QUESTION = "How many hours of outage must be reported?";

/** What the model was sent: each request's system prompt, and what its searches returned. */
interface Seen {
  system: string[];
  results: string[];
}

/** A model that calls Tools: it searches once, then answers. */
function searchingModel(seen: Seen) {
  return scriptedModel((call: ModelCall) => {
    seen.system.push(call.system);
    const searches = call.results.filter((result) => result.tool === "search_documents");
    if (searches.length === 0) {
      return { calls: [{ tool: "search_documents", input: { query: "outages reported" } }] };
    }
    seen.results.push(...searches.map((result) => result.text));
    return { text: "Outages over 4 hours must be reported." };
  });
}

/** A model that can't call Tools: it answers from one search, with structured output when `json`. */
function oneSearchModel(seen: Seen, json: boolean) {
  return scriptedModel((call: ModelCall) => {
    seen.system.push(call.system);
    if (call.tools.length > 0) {
      return { error: { status: 400, message: "this model does not support tools" } };
    }
    if (call.json && !json) {
      return { error: { status: 400, message: "response_format is not supported" } };
    }
    const answer = "Outages over 4 hours must be reported.";
    return { text: call.json ? JSON.stringify({ answer, citations: [] }) : answer };
  });
}

/**
 * Three Documents that disagree: a memo whose text states its year, a
 * handbook that states none (its file was last modified on 1 March 2024),
 * and a circular whose reference record the User dated November 2023.
 */
async function setUp(model: ReturnType<typeof scriptedModel>) {
  const setup = await setUpWithDocuments(model, []);
  const sources = await createTempDataFolder();
  const memo = await writeSourceFile(
    sources,
    "Outage memo.md",
    "# Outage reporting memo\n\nIssued 14 May 2025. Outages over 4 hours must be reported.\n",
  );
  const handbook = await writeSourceFile(
    sources,
    "Operations handbook.md",
    "# Operations handbook\n\nOutages over 6 hours must be reported.\n",
  );
  const circular = await writeSourceFile(
    sources,
    "Circular.md",
    "# Circular\n\nOutages over 8 hours must be reported.\n",
  );
  const march = new Date("2024-03-01T12:00:00Z");
  await utimes(handbook, march, march);
  const added = await addAndProcess(setup.core, [memo, handbook, circular]);
  const dated = added.find((document) => document.name === "Circular");
  if (!dated) throw new Error("The circular wasn't added.");
  await setup.core.updateReference(dated.id, { issued: "2023-11" });
  return setup;
}

const datesIn = (text: string) =>
  Object.fromEntries(shownPassages(text).map((passage) => [passage.document, passage.date]));

const EXPECTED_DATES = {
  "Outage memo": "2025",
  "Operations handbook": "2024-03-01",
  Circular: "2023-11",
};

describe("The Passages handed to the model", { timeout: 30_000 }, () => {
  test("carry their Document's date: the date it states, else the day its file was modified", async () => {
    const seen: Seen = { system: [], results: [] };
    const { core, client, mind } = await setUp(searchingModel(seen));

    const { finished } = await askAndFinish(core, client, mind.id, QUESTION);

    expect(finished.citationSupport).toBe("tools");
    expect(datesIn(seen.results.join("\n\n"))).toEqual(EXPECTED_DATES);
  });

  test.each([
    ["structured-output", true],
    ["none", false],
  ] as const)("carry it in one search's Passages too (%s)", async (support, json) => {
    const seen: Seen = { system: [], results: [] };
    const { core, client, mind } = await setUp(oneSearchModel(seen, json));

    const { finished } = await askAndFinish(core, client, mind.id, QUESTION);

    expect(finished.citationSupport).toBe(support);
    expect(datesIn(seen.system.at(-1) ?? "")).toEqual(EXPECTED_DATES);
  });

  test("carry no date when none is known", async () => {
    const passage = (documentId: string, documentName: string) => ({
      passageId: `${documentId}-1`,
      documentId,
      documentName,
      documentKind: "markdown" as const,
      contentHash: "v1",
      pageFrom: null,
      pageTo: null,
      position: 0,
      seq: 0,
      windowFrom: 0,
      windowTo: 0,
      text: "Outages over 4 hours must be reported.",
    });
    const session = createCitationSession(
      {
        searchableCount: () => 2,
        search: async () => [passage("memo", "Memo"), passage("note", "Note")],
        citationSource: () => null,
        pageTexts: () => [],
        documentDate: (documentId) => (documentId === "memo" ? "2025-05-14" : null),
      },
      { onRecord: () => undefined },
    );

    const { text } = await session.tools.searchDocuments("outages");

    expect(datesIn(text)).toEqual({ Memo: "2025-05-14", Note: null });
    expect(text).toContain('<passage id="P2" document="Note">');
  });
});

describe("The Answer instructions", { timeout: 30_000 }, () => {
  test.each([
    ["English", QUESTION],
    ["Chinese", "停电超过几个小时必须报告？"],
  ])("have both rules for a Question in %s", async (_language, question) => {
    const seen: Seen = { system: [], results: [] };
    const { core, client, mind } = await setUp(searchingModel(seen));

    await askAndFinish(core, client, mind.id, question);

    const system = seen.system[0] ?? "";
    for (const rule of RULES) expect(system).toContain(rule);
  });

  test.each(["tools", "structured-output", "none"] as const)(
    "have both rules when answering with %s, with markers only where the Answer cites",
    (mode) => {
      const text = documentInstructions(mode, { total: 2, names: ["Memo", "Handbook"] }, "");

      for (const rule of RULES) expect(text).toContain(rule);
      expect(text.includes("says 6 hours[^2]")).toBe(mode !== "none");
      if (mode === "none") expect(text).not.toContain("[^");
    },
  );

  test.each(["tools", "structured-output", "none"] as const)(
    "when answering with %s, stop where the Documents stop: own knowledge only for a general Question, labelled, with no figures",
    (mode) => {
      const text = documentInstructions(mode, { total: 1, names: ["Memo"] }, "");

      expect(text).toContain("If the Documents don't cover the Question, say so plainly.");
      expect(text).toContain("never fill the gap from your own knowledge");
      expect(text).toContain('starts "Not from your Documents:" ("以下内容不是来自你的文档：")');
      expect(text).not.toContain("answer from your own knowledge if you can");
      // The premise rule's "then answer from what they say" is the Passages, never the model's own knowledge.
      expect(text).toContain("then answer from what they say");
      expect(text).toContain("You may add what the Passages do say that comes closest");
    },
  );
});

/** Passages as the search Tool formats them: each with its Document, and its date when known. */
const handed = (...documents: readonly [string, string | null][]) =>
  documents
    .map(
      ([document, date], index) =>
        `<passage id="P${index + 1}" document="${document}"${date ? ` date="${date}"` : ""} pages="1">\nOutages must be reported.\n</passage>`,
    )
    .join("\n\n");

describe("The note on Documents of different dates (#660)", { timeout: 30_000 }, () => {
  const NOTE = "These Passages come from Documents of different dates:";

  test("names each dated Document with its date, newest first, and reminds the model to name the older one", () => {
    expect(
      datedDocumentsNote(
        handed(
          ["Handbook", "2024-03-01"],
          ["Memo", "2025"],
          ["Memo", "2025"],
          ["Circular", "2023-11"],
        ),
      ),
    ).toBe(
      `${NOTE} "Memo" (2025), "Handbook" (2024-03-01), "Circular" (2023-11). Where they give different figures for the same thing, lead with the newer Document, then write one sentence naming the older one, its date and its figure, with a Citation.`,
    );
  });

  test("is left out for one Document, for Documents of the same date, and for undated ones", () => {
    expect(datedDocumentsNote(handed(["Memo", "2025"]))).toBe("");
    expect(datedDocumentsNote(handed(["Memo", "2025"], ["Memo", "2025"]))).toBe("");
    expect(datedDocumentsNote(handed(["Memo", "2024-03-01"], ["Handbook", "2024-03-01"]))).toBe("");
    expect(datedDocumentsNote(handed(["Memo", "2025"], ["Handbook", null]))).toBe("");
    expect(datedDocumentsNote("No Passages in the User's Documents match this search.")).toBe("");
  });

  test("follows a Tool-loop search's Passages when they come from Documents of different dates", async () => {
    const seen: Seen = { system: [], results: [] };
    const { core, client, mind } = await setUp(searchingModel(seen));

    await askAndFinish(core, client, mind.id, QUESTION);

    const [result = ""] = seen.results;
    expect(
      result.endsWith(
        `${NOTE} "Outage memo" (2025), "Operations handbook" (2024-03-01), "Circular" (2023-11). Where they give different figures for the same thing, lead with the newer Document, then write one sentence naming the older one, its date and its figure, with a Citation.`,
      ),
    ).toBe(true);
  });

  test("isn't added when the Passages handed are from Documents of one date", async () => {
    const seen: Seen = { system: [], results: [] };
    const setup = await setUpWithDocuments(searchingModel(seen), []);
    const sources = await createTempDataFolder();
    const day = new Date("2024-03-01T12:00:00Z");
    const paths = await Promise.all(
      ["Shift notes.md", "Night notes.md"].map(async (name) => {
        const path = await writeSourceFile(
          sources,
          name,
          `# ${name}\n\nOutages over 6 hours must be reported.\n`,
        );
        await utimes(path, day, day);
        return path;
      }),
    );
    await addAndProcess(setup.core, paths);

    await askAndFinish(setup.core, setup.client, setup.mind.id, QUESTION);

    expect(shownPassages(seen.results.join("\n\n")).map((passage) => passage.date)).toEqual([
      "2024-03-01",
      "2024-03-01",
    ]);
    expect(seen.results.join("\n")).not.toContain(NOTE);
  });

  test("follows one search's Passages in the instructions of a model that can't call Tools", async () => {
    const seen: Seen = { system: [], results: [] };
    const { core, client, mind } = await setUp(oneSearchModel(seen, true));

    await askAndFinish(core, client, mind.id, QUESTION);

    expect(seen.system.at(-1)).toContain(`${NOTE} "Outage memo" (2025)`);
  });
});
