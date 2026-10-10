/**
 * The evaluation's Citation part (`npm run eval` with a chat model,
 * eval/README.md), run against a core with a scripted model in place of a
 * real one: it asks through the public interface, reads each Answer back from
 * its Mind, and sorts the Citations by what became of them.
 */
import { describe, expect, test } from "vitest";
import { runCitations } from "../../eval/lib/citations";
import type { EvalQuestion } from "../../eval/lib/evaluationSet";
import type { Library } from "../../eval/lib/library";
import { citingModel, type ShownPassage, setUpWithDocuments } from "../helpers/citations";
import { queryDatabase } from "../helpers/core";
import { buildPdf } from "../helpers/pdf";

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "Most coasts see two high tides every day."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

const SPRING = "Spring tides happen at new moon and at full moon.";

const QUESTION: EvalQuestion = {
  id: "en-01",
  language: "en",
  crossLingual: false,
  question: "When are spring tides?",
  expected: { document: "tides", pages: [2, 2], quote: SPRING },
};

/** The same fact in other words: a group of its own, never asked again for a language's count. */
const PARAPHRASE: EvalQuestion = {
  id: "para-en-01",
  language: "en",
  crossLingual: false,
  paraphrase: true,
  question: "When does the sea rise highest?",
  expected: { document: "tides", pages: [2, 2], quote: SPRING },
};

const first = (passages: ShownPassage[]) => passages[0]?.id ?? "none";

describe("The evaluation's Citation part", { timeout: 60_000 }, () => {
  test("scores each Citation of each Answer, and asks again while a language has too few", async () => {
    const model = citingModel({
      query: "spring tides",
      records: (passages) => [
        // On the cited page, word for word.
        { marker: 1, passage: first(passages), pageFrom: 2, pageTo: 2, quote: SPRING },
        // On the cited page, but not word for word: a false "not found".
        {
          marker: 2,
          passage: first(passages),
          pageFrom: 2,
          pageTo: 2,
          quote: "spring tides happen at new moon, and at full moon",
        },
        // On another page than the one cited.
        { marker: 3, passage: first(passages), pageFrom: 1, pageTo: 1, quote: SPRING },
      ],
      answer:
        "Spring tides come at new moon [^1]. And at full moon [^2]. They are the largest.[^3] Neap tides are smaller.",
    });
    const { core, dataDir } = await setUpWithDocuments(model, [
      { name: "Tides.pdf", contents: TIDES },
    ]);
    const library: Library = {
      core,
      dataDir,
      documents: new Map(),
      processingSeconds: 0,
      passageCount: 0,
      pageTexts: (documentId) =>
        queryDatabase(
          dataDir,
          "SELECT page, text FROM document_pages WHERE document_id = ? AND deleted_at IS NULL ORDER BY page",
          [documentId],
        ),
      passages: () => [],
      close: async () => {},
    };

    const run = await runCitations(
      library,
      [QUESTION, PARAPHRASE],
      { kind: "ollama", modelId: "local-model", apiKey: null, baseUrl: null },
      { minCitations: 4, maxRounds: 2, answerTimeoutMs: 30_000 },
      () => {},
    );

    // Three Citations in round 1 are fewer than 4, so the gating Question was asked again;
    // the paraphrase Question's Citations don't count for English, and it isn't asked again.
    expect(run).toMatchObject({
      model: "ollama/local-model",
      gating: false,
      rounds: 2,
      failures: [],
    });
    expect(run.answers.map((each) => [each.questionId, each.round])).toEqual([
      ["en-01", 1],
      ["para-en-01", 1],
      ["en-01", 2],
    ]);
    const [answer] = run.answers;
    expect(answer).toMatchObject({
      questionId: "en-01",
      round: 1,
      status: "done",
      citationSupport: "tools",
      searches: ["spring tides"],
      droppedMarkers: 0,
      droppedRecords: 0,
      sentences: [
        { text: "Spring tides come at new moon .", cited: true },
        { text: "And at full moon .", cited: true },
        { text: "They are the largest.", cited: true },
        { text: "Neap tides are smaller.", cited: false },
      ],
    });
    expect(answer?.citations.map(({ sentence, outcome }) => [sentence, outcome])).toEqual([
      ["Spring tides come at new moon .", "found"],
      ["And at full moon .", "false-not-found"],
      ["They are the largest.", "wrong-page"],
    ]);
    expect(run.summary.en).toMatchObject({
      answers: 2,
      citations: 6,
      foundShare: 2 / 6,
      falseNotFoundShare: 2 / 6,
      coverage: 6 / 8,
    });
    expect(run.summary.zh).toMatchObject({ answers: 0, citations: 0, foundShare: null });
    expect(run.summary.paraphrase).toMatchObject({ answers: 1, citations: 3, foundShare: 1 / 3 });
    expect(run.answers[1]).toMatchObject({ paraphrase: true });
  });
});
