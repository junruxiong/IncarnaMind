import { describe, expect, test } from "vitest";
import { CHOSEN_PROVIDER_FLOWS } from "../../src/core/api";
import { chosenFlowLines } from "../../src/renderer/src/chosenFlows";
import { translate } from "../../src/shared/i18n";
import { createTempDataFolder, startCore } from "../helpers/core";

describe("the line above the button that chooses a provider (ADR-0022)", () => {
  test("lists what goes to the provider, in both languages, in the order of the flows", () => {
    expect(chosenFlowLines().map((line) => translate("en", line))).toEqual([
      "your Questions, the Mind you're in, and matching passages with their Document names and pages",
      "each sentence you check",
      "the paragraph you pause on, for writing suggestions",
      "each Document's name, outline and short excerpts, to tag and file it (page images for scans with no text)",
      "the text the assistant edits when you ask it to",
    ]);
    expect(chosenFlowLines().map((line) => translate("zh-CN", line))).toEqual([
      "你的问题、当前 Mind，以及相关段落及其文档名称和页码",
      "你核对的每一句话",
      "写作建议时你停下的那一段",
      "每个文档的名称、目录和简短摘录，用于打标签和整理（没有文字的扫描件会发送页面图片）",
      "你让助手修改时被修改的文字",
    ]);
    expect(translate("en", "providers.sends", { provider: "OpenAI" })).toBe(
      "When you use it, OpenAI receives:",
    );
    expect(translate("zh-CN", "providers.sends", { provider: "OpenAI" })).toBe(
      "使用时，以下内容会发送给 OpenAI：",
    );
  });

  test.each([
    ["OpenAI", { kind: "openai", apiKey: "sk-test-openai", modelId: "gpt-5.4-mini" }],
    ["Anthropic", { kind: "anthropic", apiKey: "sk-ant-test", modelId: "claude-sonnet-4-6" }],
  ] as const)("names every flow that choosing %s turns on, and no other", async (name, input) => {
    const core = startCore(await createTempDataFolder());
    await core.saveChatProvider(input);

    const toService = (await core.listDataFlows()).filter(
      (status) => status.flow.service.name === name,
    );

    expect(toService.map((status) => status.flow.id).sort()).toEqual(
      [...CHOSEN_PROVIDER_FLOWS].sort(),
    );
    expect(toService.every((status) => status.chosen)).toBe(true);
  });
});
