import { describe, expect, test } from "vitest";
import { diffModels, formatDiff, generateModels, isEmpty } from "../../scripts/catalogModels";

type Source = Parameters<typeof generateModels>[0];

const model = (name: string, cost: { input: number; output: number }, context = 200000) => ({
  name,
  tool_call: true,
  reasoning: false,
  modalities: { input: ["text", "image"], output: ["text"] },
  limit: { context, output: 8000 },
  cost,
});

/** models.dev as a fixture: the models of the anthropic provider by id. */
const sources = (models: Record<string, ReturnType<typeof model>>): Source => ({
  anthropic: { models },
});

const ROLES = { anthropic: { answers: "sonnet", quickTasks: "haiku" } };

const before = generateModels(
  sources({
    sonnet: model("Sonnet", { input: 3, output: 15 }),
    haiku: model("Haiku", { input: 1, output: 5 }),
    old: model("Old", { input: 2, output: 10 }),
  }),
  {},
).providers;

describe("the catalog update's diff", () => {
  test("the same sources twice give no diff", () => {
    const again = generateModels(
      sources({
        old: model("Old", { input: 2, output: 10 }),
        haiku: model("Haiku", { input: 1, output: 5 }),
        sonnet: model("Sonnet", { input: 3, output: 15 }),
      }),
      {},
    ).providers;
    const diff = diffModels(before, again, ROLES);
    expect(isEmpty(diff)).toBe(true);
    expect(formatDiff(diff)).toBe("No changes to the catalog's model facts.\n");
  });

  test("a new model is listed with its price and context", () => {
    const after = generateModels(
      sources({
        sonnet: model("Sonnet", { input: 3, output: 15 }),
        haiku: model("Haiku", { input: 1, output: 5 }),
        old: model("Old", { input: 2, output: 10 }),
        fresh: model("Fresh", { input: 4, output: 20 }, 1000000),
      }),
      {},
    ).providers;
    const diff = diffModels(before, after, ROLES);
    expect(diff.added).toEqual([
      "anthropic/fresh (Fresh, $4 in / $20 out per million tokens, context 1000000)",
    ]);
    expect(diff.removed).toEqual([]);
    expect(diff.changed).toEqual([]);
    expect(diff.defaults).toEqual([]);
  });

  test("a price change and a context change name the model and both values", () => {
    const after = generateModels(
      sources({
        sonnet: model("Sonnet", { input: 3, output: 15 }),
        haiku: model("Haiku", { input: 1, output: 5 }),
        old: model("Old", { input: 2.5, output: 10 }, 400000),
      }),
      {},
    ).providers;
    const diff = diffModels(before, after, ROLES);
    expect(diff.changed).toEqual([
      "anthropic/old: input price $2 -> $2.5",
      "anthropic/old: context 200000 -> 400000",
    ]);
    expect(diff.defaults).toEqual([]);
  });

  test("a removed model is listed", () => {
    const after = generateModels(
      sources({
        sonnet: model("Sonnet", { input: 3, output: 15 }),
        haiku: model("Haiku", { input: 1, output: 5 }),
      }),
      {},
    ).providers;
    const diff = diffModels(before, after, ROLES);
    expect(diff.removed).toEqual(["anthropic/old"]);
    expect(diff.defaults).toEqual([]);
  });

  test("a changed or removed default is called out, with the rule to cite an evaluation", () => {
    const after = generateModels(
      sources({
        sonnet: model("Sonnet", { input: 4, output: 15 }),
        old: model("Old", { input: 2, output: 10 }),
      }),
      {},
    ).providers;
    const diff = diffModels(before, after, ROLES);
    expect(diff.defaults).toEqual([
      "anthropic answers: sonnet changed (input price $3 -> $4)",
      "anthropic quickTasks: haiku is no longer in the sources",
    ]);
    const text = formatDiff(diff);
    expect(text).toContain("### A default changed");
    expect(text).toContain("needs the evaluation that cites it before this is merged");
    expect(text.indexOf("A default changed")).toBeLessThan(text.indexOf("Removed models"));
  });
});
