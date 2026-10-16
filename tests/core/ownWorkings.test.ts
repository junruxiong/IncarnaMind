import { describe, expect, test } from "vitest";
import { type Core, type Document, InvalidInputError, NotFoundError } from "../../src/core";
import { checkingModel } from "../helpers/checks";
import {
  askAndFinish,
  citationsIn,
  citeFeedback,
  citingModel,
  setUpWithDocuments,
} from "../helpers/citations";
import { createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";

const REPORT = {
  name: "Annual report.md",
  contents: [
    "# Annual report",
    "",
    "Revenue was $1,482.6 million in 2025, up from $1,231.0 million in 2024.",
  ].join("\n"),
};

const WORKINGS = {
  name: "Workings.csv",
  contents: [
    "Item,2024,2025",
    "Revenue (USD million),1231.0,1482.6",
    "Operating margin,0.18,0.21",
  ].join("\n"),
};

const FIGURE = "Revenue was $1,482.6 million in 2025.";
const REPORT_QUOTE = "Revenue was $1,482.6 million in 2025";
const ONLY_IN_WORKINGS = "Operating margin was 21% in 2025.";

/** Documents by name, so a test needn't rely on the order they were indexed in. */
const named = (documents: readonly Document[]) => {
  const byName = new Map(documents.map((each) => [each.name, each]));
  const get = (name: string) => {
    const found = byName.get(name);
    if (!found) throw new Error(`${name} isn't a Document.`);
    return found;
  };
  return { report: get("Annual report"), workings: get("Workings") };
};

/** A model that searches for the figure and records a Citation for every Passage it was shown. */
const citesEverything = () =>
  citingModel({
    query: "revenue 2025",
    records: (passages) =>
      passages.map((passage, index) => ({
        marker: index + 1,
        passage: passage.id,
        quote: passage.document === "Workings" ? "1482.6" : REPORT_QUOTE,
      })),
    answer: `${FIGURE}[^1][^2]`,
  });

const evidenceFrom = async (core: Core, query: string) =>
  (await core.searchEvidence(query)).passages.map((passage) => passage.documentName);

describe("A Document the User keeps as their own workings", () => {
  test("is marked and cited again by the core, announced to every window, and refuses a choice that isn't one", async () => {
    const { core, documents } = await setUpWithDocuments(citesEverything(), [REPORT, WORKINGS]);
    const { report, workings } = named(documents);
    expect([report.notCited, workings.notCited]).toEqual([false, false]);

    const announced = nextEvent(core, "document.status");
    const marked = await core.setNotCited(workings.id, true);
    expect(marked.notCited).toBe(true);
    expect((await announced).notCited).toBe(true);
    expect((await core.listDocuments()).map((each) => [each.name, each.notCited]).sort()).toEqual([
      ["Annual report", false],
      ["Workings", true],
    ]);

    expect((await core.setNotCited(workings.id, false)).notCited).toBe(false);
    await expect(core.setNotCited(workings.id, "yes" as unknown as boolean)).rejects.toThrow(
      InvalidInputError,
    );
    await expect(core.setNotCited("no-such-document", true)).rejects.toThrow(NotFoundError);
  });

  test("keeps the mark across a restart", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const first = startCore(dataDir);
    const paths = await Promise.all(
      [REPORT, WORKINGS].map((file) => writeSourceFile(sources, file.name, file.contents)),
    );
    const { workings } = named(await addAndProcess(first, paths));
    await first.setNotCited(workings.id, true);
    first.close();

    const again = startCore(dataDir);
    const { report, workings: kept } = named(await again.listDocuments());
    expect([report.notCited, kept.notCited]).toEqual([false, true]);
    expect(await evidenceFrom(again, "revenue 2025")).toEqual(["Annual report"]);
    await again.setNotCited(kept.id, false);
    expect((await evidenceFrom(again, "revenue 2025")).sort()).toEqual([
      "Annual report",
      "Workings",
    ]);
  });

  test("is read for a Question but never cited: the Answer cites only the report", async () => {
    const model = citesEverything();
    const { core, documents, client, mind } = await setUpWithDocuments(model, [REPORT, WORKINGS]);
    const { workings } = named(documents);

    // Cited like any Document until it is marked.
    const before = await askAndFinish(core, client, mind.id, "What was revenue in 2025?");
    expect(
      citationsIn(client, before.answerId)
        .map((each) => each.documentName)
        .sort(),
    ).toEqual(["Annual report", "Workings"]);

    await core.setNotCited(workings.id, true);
    const after = await askAndFinish(core, client, mind.id, "What was revenue in 2025 again?");
    expect(citationsIn(client, after.answerId).map((each) => each.documentName)).toEqual([
      "Annual report",
    ]);

    // The model read it, told what it is, and its record for it was refused.
    const read = JSON.stringify(model.doStreamCalls.at(-1)?.prompt ?? []);
    expect(read).toContain('document=\\"Workings\\"');
    expect(read).toContain('cite=\\"never\\"');
    expect(citeFeedback(model)).toContain("own working file");
  });

  test("is still searched, but a check's evidence never comes from it, and it is named where it matched", async () => {
    const { core, documents } = await setUpWithDocuments(citesEverything(), [REPORT, WORKINGS]);
    const { workings } = named(documents);
    await core.setNotCited(workings.id, true);

    const searched = await core.searchPassages("revenue 2025");
    expect(new Set(searched.map((passage) => passage.documentName))).toEqual(
      new Set(["Annual report", "Workings"]),
    );

    const found = await core.searchEvidence("revenue 2025");
    expect(new Set(found.passages.map((passage) => passage.documentName))).toEqual(
      new Set(["Annual report"]),
    );
    expect(found.excluded).toMatchObject([
      { documentId: workings.id, documentName: "Workings", reason: "own-workings" },
    ]);

    // No setting counts it: Count it is for the files IncarnaMind wrote.
    await core.updateSettings({ user: { leaveOutOwnFiles: false } });
    expect(await evidenceFrom(core, "revenue 2025")).toEqual(["Annual report"]);

    await core.setNotCited(workings.id, false);
    expect((await evidenceFrom(core, "revenue 2025")).sort()).toEqual([
      "Annual report",
      "Workings",
    ]);
  });

  test("a figure the report and the marked file both state is Supported by the report", async () => {
    const model = checkingModel({
      [FIGURE]: {
        verdict: "supported",
        quote: REPORT_QUOTE,
        says: "The annual report gives revenue of $1,482.6 million for 2025.",
      },
    });
    const { core, documents } = await setUpWithDocuments(model, [REPORT, WORKINGS]);
    const { report, workings } = named(documents);
    await core.setNotCited(workings.id, true);

    const result = await core.checkClaim({ claim: FIGURE });

    expect(result).toMatchObject({
      verdict: "supported",
      evidence: { documentId: report.id, documentName: "Annual report" },
      versions: [{ documentId: report.id }],
      excluded: [],
    });
    const read = JSON.stringify(
      model.doStreamCalls.flatMap((call) =>
        call.prompt.filter((message) => message.role === "tool"),
      ),
    );
    expect(read).toContain('document=\\"Annual report\\"');
    expect(read).not.toContain('document=\\"Workings\\"');
  });

  test("a figure only the marked file states is Unsupported, and the result says whose it is", async () => {
    const model = checkingModel({ [ONLY_IN_WORKINGS]: { verdict: "unsupported" } });
    const { core, documents } = await setUpWithDocuments(model, [REPORT, WORKINGS]);
    const { workings } = named(documents);

    // Unmarked, the spreadsheet is evidence like any Document: nothing is left out.
    expect(await core.checkClaim({ claim: ONLY_IN_WORKINGS })).toMatchObject({
      verdict: "unsupported",
      excluded: [],
    });

    await core.setNotCited(workings.id, true);
    expect(await core.checkClaim({ claim: ONLY_IN_WORKINGS })).toMatchObject({
      verdict: "unsupported",
      excluded: [{ documentId: workings.id, documentName: "Workings", reason: "own-workings" }],
    });

    // A sentence the file merely shares a word or two with isn't said to be its.
    expect(await core.checkClaim({ claim: "Penguins nest above the harbour." })).toMatchObject({
      verdict: "unsupported",
      excluded: [],
    });
  });

  test("isn't compared: its row is left out, and a table of nothing else isn't made", async () => {
    const { core, documents, mind } = await setUpWithDocuments(citesEverything(), [
      REPORT,
      WORKINGS,
    ]);
    const { workings } = named(documents);
    await core.setNotCited(workings.id, true);
    await expect(
      core.compareDocuments({
        mindId: mind.id,
        documentIds: [workings.id],
        questions: ["Revenue 2025"],
      }),
    ).rejects.toThrow(InvalidInputError);
  });
});
