import { describe, expect, test } from "vitest";
import { CATALOG_PROVIDERS } from "../../src/core/providers/catalog/providers";
import { directoryGroups } from "../../src/renderer/src/components/providers/directoryGroups";

const shape = (language: "en" | "zh-CN", query?: string) =>
  directoryGroups(language, query).map(({ group, providers }) => [
    group,
    providers.map((each) => each.id),
  ]);

describe("Add a provider's directory (#179)", () => {
  test("in English: Popular, International, China, On this computer, Other", () => {
    const groups = shape("en");
    expect(groups.map(([group]) => group)).toEqual([
      "popular",
      "international",
      "china",
      "local",
      "other",
    ]);
    expect(groups[0]).toEqual([
      "popular",
      ["anthropic", "openai", "google", "ollama", "openrouter"],
    ]);
    expect(groups.find(([group]) => group === "china")?.[1]).toEqual([
      "deepseek",
      "qwen",
      "kimi",
      "glm",
      "siliconflow",
    ]);
  });

  test("in Chinese: 常用, 中国, 国际, 本机, 其他, with China's providers popular", () => {
    const groups = shape("zh-CN");
    expect(groups[0]).toEqual([
      "popular",
      ["deepseek", "qwen", "kimi", "glm", "siliconflow", "ollama"],
    ]);
    // China's every provider is popular, so its own group is empty and left out.
    expect(groups.map(([group]) => group)).toEqual(["popular", "international", "local", "other"]);
    expect(groups.find(([group]) => group === "international")?.[1]).toContain("anthropic");
  });

  test("each provider once, whatever the language", () => {
    for (const language of ["en", "zh-CN"] as const) {
      const ids = directoryGroups(language).flatMap(({ providers }) =>
        providers.map((each) => each.id),
      );
      expect(new Set(ids).size).toBe(ids.length);
      expect([...ids].sort()).toEqual(CATALOG_PROVIDERS.map((each) => each.id).sort());
    }
  });

  test("a search finds a provider by its English or Chinese name, in its group", () => {
    expect(shape("en", "通义")).toEqual([["china", ["qwen"]]]);
    expect(shape("zh-CN", "kimi")).toEqual([["popular", ["kimi"]]]);
    expect(shape("en", "studio")).toEqual([
      ["china", ["qwen"]],
      ["local", ["lmstudio", "unsloth"]],
    ]);
    expect(shape("en", "nothing like it")).toEqual([]);
  });
});
