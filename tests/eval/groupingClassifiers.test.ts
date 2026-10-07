/**
 * The grouping check's classifier fallback (R0) and its settings, with a fake
 * Jev server and scripted chat models: a model proposes the Topic list, and
 * Jev, Clef-Flash in Ollama (the same `/v1/systemone` request) or the chat
 * model puts each Document in one Topic. No model runs and no key is used.
 */
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import {
  chatTopicClassifier,
  chatTopicProposer,
  excerptOf,
  runClassifier,
  systemOneTopicClassifier,
  type TopicClassifier,
  topicIndex,
} from "../../eval/grouping/lib/classifiers";
import { CLEF_DEFAULT_MODEL, readGroupingConfig } from "../../eval/grouping/lib/config";
import type { CorpusDocument } from "../../eval/grouping/lib/variants";
import { JEV_KEY, startFakeJev } from "../helpers/jev";
import { replyingModel } from "../helpers/models";

const TOPICS = ["Mars", "Photosynthesis", "Inflation"];

const document = (key: string, name: string, text: string): CorpusDocument => ({
  id: `id-${key}`,
  key,
  name,
  kind: "markdown",
  pageCount: null,
  passages: [text],
});

/** Everything the model was sent in its first request, as text. */
function sent(model: MockLanguageModelV4): string {
  const call = model.doGenerateCalls[0];
  if (!call) throw new Error("The model wasn't asked.");
  return call.prompt
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => ("text" in part ? part.text : "")).join(""),
    )
    .join("\n");
}

const MARS = document("mars", "Mars · Wikipedia", "Mars is the fourth planet from the Sun.");

describe("the Topic list", () => {
  test("is proposed from the titles, without repeats, at most the number asked", async () => {
    const model = replyingModel(
      JSON.stringify({ topics: ["Planets", "planets", " Plants  and light ", "Prices", "Extra"] }),
    );
    const topics = await chatTopicProposer(model, "test/model").propose(
      ["Mars · Wikipedia", "维基百科-光合作用", "CPI-U 2015-2024"],
      3,
    );
    expect(topics).toEqual(["Planets", "Plants and light", "Prices"]);
    const prompt = sent(model);
    expect(prompt).toContain(
      "<titles>\n- Mars · Wikipedia\n- 维基百科-光合作用\n- CPI-U 2015-2024\n</titles>",
    );
    expect(prompt).toContain("Number of Topics: 3");
  });

  test("fewer than 2 Topics is an error", async () => {
    const proposer = chatTopicProposer(replyingModel(JSON.stringify({ topics: ["One"] })), "m");
    await expect(proposer.propose(["a", "b"], 4)).rejects.toThrow("fewer than 2 Topics");
  });
});

describe("the classifiers", () => {
  test("Jev (or Clef-Flash) asks one yes/no question per Topic, and the most probable wins", async () => {
    const jev = await startFakeJev();
    jev.answer({ Mars: 0.91, Photosynthesis: 0.12, Inflation: 0.03 });
    const classifier = systemOneTopicClassifier({
      name: "Jev",
      baseUrl: jev.url,
      apiKey: JEV_KEY,
      model: "jev-latest",
      local: true,
    });
    const assignment = await classifier.assign(TOPICS, excerptOf(MARS));
    expect(assignment).toEqual({ topic: 0, confidence: 0.91 });
    const [request] = jev.requests;
    expect(request?.path).toBe("/v1/systemone");
    expect(request?.authorization).toBe(`Bearer ${JEV_KEY}`);
    expect(
      Object.values(request?.body.questions ?? {}).map((question) => question.instructions),
    ).toEqual([
      "Is this Document mainly about “Mars”?",
      "Is this Document mainly about “Photosynthesis”?",
      "Is this Document mainly about “Inflation”?",
    ]);
    expect(request?.body.state).toEqual({
      name: "Mars · Wikipedia",
      type: "Markdown",
      excerpt: "Mars is the fourth planet from the Sun.",
    });
  });

  test("the chat model chooses one Topic by name, with structured output", async () => {
    const model = replyingModel(JSON.stringify({ topic: "Photosynthesis" }));
    const classifier = chatTopicClassifier(model, "test/model", false);
    expect(await classifier.assign(TOPICS, excerptOf(MARS))).toEqual({
      topic: 1,
      confidence: null,
    });
    const prompt = sent(model);
    expect(prompt).toContain("Topics:\n- Mars\n- Photosynthesis\n- Inflation");
    expect(prompt).toContain(
      "<document>\nName: Mars · Wikipedia\nType: Markdown\nText:\nMars is the fourth planet",
    );
  });

  test("a reply naming no listed Topic is no Topic; names match ignoring case and spacing", () => {
    expect(topicIndex(TOPICS, "  inflation ")).toBe(2);
    expect(topicIndex(TOPICS, "Economics")).toBeNull();
    expect(topicIndex(TOPICS, 3)).toBeNull();
  });

  test("a run times every Document, and a failed request leaves that Document out", async () => {
    let calls = 0;
    const classifier: TopicClassifier = {
      name: "flaky",
      local: true,
      async assign() {
        calls++;
        if (calls === 2) throw new Error("Ollama isn't running");
        return { topic: 0, confidence: null };
      },
    };
    const documents = ["a", "b", "c"].map((key) => document(key, key, "text"));
    const run = await runClassifier(classifier, TOPICS, documents, () => {});
    expect(run.failed).toBe(1);
    expect(run.firstError).toBe("Ollama isn't running");
    expect([...run.assignments.keys()]).toEqual(["a", "c"]);
    expect(run.seconds).toBeGreaterThanOrEqual(0);
  });

  test("a failing chat model fails the Document, not the run", async () => {
    const model = new MockLanguageModelV4({
      doGenerate: async () => {
        throw new Error("rate limited");
      },
    });
    const run = await runClassifier(
      chatTopicClassifier(model, "test/model", false),
      TOPICS,
      [MARS],
      () => {},
    );
    expect(run.failed).toBe(1);
    expect(run.assignments.size).toBe(0);
  });
});

describe("the grouping check's settings", () => {
  test("nothing set: no classifiers, no founder's library, timings on", () => {
    expect(readGroupingConfig({})).toEqual({ clef: null, jev: null, founder: null, timing: true });
    expect(readGroupingConfig({ INCARNAMIND_EVAL_GROUPING_TIMING: "0" }).timing).toBe(false);
    expect(() => readGroupingConfig({ INCARNAMIND_EVAL_GROUPING_TIMING: "yes" })).toThrow(
      "must be 0 or 1",
    );
  });

  test("Clef-Flash is Ollama's /v1/systemone on this computer, needing no key", () => {
    const { clef } = readGroupingConfig({
      INCARNAMIND_EVAL_CLEF_URL: "http://127.0.0.1:11434/v1/systemone",
    });
    expect(clef).toMatchObject({
      baseUrl: "http://127.0.0.1:11434",
      model: CLEF_DEFAULT_MODEL,
      local: true,
    });
    expect(() => readGroupingConfig({ INCARNAMIND_EVAL_CLEF_MODEL: "clef-flash" })).toThrow(
      "INCARNAMIND_EVAL_CLEF_URL",
    );
  });

  test("Jev is TypeSafe's hosted service unless a URL is given, and needs its key", () => {
    const { jev } = readGroupingConfig({ INCARNAMIND_EVAL_JEV_KEY: "key" });
    expect(jev).toMatchObject({
      baseUrl: "https://api.typesafe.ai",
      model: "jev-latest",
      local: false,
    });
    expect(() => readGroupingConfig({ INCARNAMIND_EVAL_JEV_URL: "http://localhost:9000" })).toThrow(
      "INCARNAMIND_EVAL_JEV_KEY",
    );
  });

  test("the founder's library must be a folder; a limit needs it", () => {
    expect(() =>
      readGroupingConfig({ INCARNAMIND_EVAL_GROUPING_FOLDER: "/no/such/folder/anywhere" }),
    ).toThrow("isn't a folder");
    expect(() => readGroupingConfig({ INCARNAMIND_EVAL_GROUPING_LIMIT: "300" })).toThrow(
      "needs INCARNAMIND_EVAL_GROUPING_FOLDER",
    );
    const { founder } = readGroupingConfig({
      INCARNAMIND_EVAL_GROUPING_FOLDER: import.meta.dirname,
      INCARNAMIND_EVAL_GROUPING_LIMIT: "300",
    });
    expect(founder).toEqual({ folder: import.meta.dirname, limit: 300, seed: 51 });
  });
});
