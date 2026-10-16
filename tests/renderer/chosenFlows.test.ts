import { describe, expect, test } from "vitest";
import type { DataKind } from "../../src/core/api";
import { CHOSEN_PROVIDER_FLOWS } from "../../src/core/api";
import { chosenFlowLines, chosenFlowSummary } from "../../src/renderer/src/chosenFlows";
import { type MessageKey, translate } from "../../src/shared/i18n";
import type { Translate } from "../../src/shared/locations";
import { createTempDataFolder, startCore } from "../helpers/core";

type Language = "en" | "zh-CN";
const t =
  (language: Language): Translate =>
  (key, params) =>
    translate(language, key, params);

const FULL = {
  en: [
    "your Questions, the Mind you're in, matching passages with their Document names and pages, and the names of other Documents in scope",
    "each sentence you check",
    "the paragraph you pause on, for writing suggestions",
    "the text you ask the assistant to change",
    "each Document's name, outline and short excerpts, with your Folder and Tag names, to tag and file it (page images for scans with no text)",
  ],
  "zh-CN": [
    "你的问题、你所在的 Mind、相关段落及其文档名称和页码，以及范围内其他文档的名称",
    "你核对的每一句话",
    "写作建议时你停下的那一段",
    "你让助手修改的文字",
    "每个文档的名称、目录和简短摘录，以及你的文件夹和标签名称，用于打标签和归档（没有文字的扫描件会发送页面图片）",
  ],
} as const;

const SHORT = {
  en: "your Questions with the Mind you're in, matching passages and the names of Documents in scope; sentences you check; the paragraph you pause on; text you ask the assistant to change; and Document names, outlines and excerpts with your Folder and Tag names, to tag and file them (page images for scans)",
  "zh-CN":
    "你的问题连同你所在的 Mind、相关段落和范围内文档的名称；你核对的句子；写作建议时你停下的段落；你让助手修改的文字；以及文档名称、目录和摘录，连同你的文件夹和标签名称，用于打标签和归档（扫描件发送页面图片）",
} as const;

/** The kinds of data each item of the line names, so a kind a flow starts sending fails until one does. */
const NAMED_BY: Record<string, readonly DataKind[]> = {
  chat: ["blocks", "passages"],
  checks: ["verified-document"],
  suggestions: ["writing", "writing-passages"],
  edits: ["selection"],
  tagging: ["tags", "groups", "document-excerpts", "page-images"],
};

describe("the lines that say what choosing a provider sends (ADR-0022)", () => {
  test.each(["en", "zh-CN"] as const)("the list above the button, in %s", (language) => {
    expect(chosenFlowLines().map((line) => t(language)(line))).toEqual(FULL[language]);
  });

  test.each(["en", "zh-CN"] as const)("the short form, in %s", (language) => {
    expect(chosenFlowSummary(t(language))).toBe(SHORT[language]);
  });

  test("the lead names the provider", () => {
    expect(translate("en", "providers.sends", { provider: "OpenAI" })).toBe(
      "When you use it, OpenAI receives:",
    );
    expect(translate("zh-CN", "providers.sends", { provider: "OpenAI" })).toBe(
      "使用时，以下内容会发送给 OpenAI：",
    );
  });

  test("the ChatGPT lines, the API key's row and the ChatGPT plan's description say the short form", () => {
    const line = (language: Language, key: MessageKey) =>
      translate(language, key, { sends: chosenFlowSummary(t(language)) });

    expect(line("en", "setup.chatgpt.line")).toBe(
      `If you pay for ChatGPT Plus or Pro · No key needed · Signing in lets OpenAI receive, under your ChatGPT account: ${SHORT.en}.`,
    );
    expect(line("zh-CN", "setup.chatgpt.line")).toBe(
      `如果你订阅了 ChatGPT Plus 或 Pro · 无需密钥 · 登录后，以下内容会通过你的 ChatGPT 账号发送给 OpenAI：${SHORT["zh-CN"]}。`,
    );
    expect(line("en", "setup.chatgpt.line.china")).toContain(
      `(not in mainland China) · No key needed · Signing in lets OpenAI receive, under your ChatGPT account: ${SHORT.en}.`,
    );
    expect(line("zh-CN", "setup.chatgpt.line.china")).toContain(
      "（中国大陆无法使用）· 无需密钥 · ",
    );
    expect(line("en", "chatgpt.description")).toBe(
      `Use your ChatGPT Plus or Pro plan instead of an API key. OpenAI receives, under your ChatGPT account: ${SHORT.en}. It counts against your plan's limits.`,
    );
    expect(line("zh-CN", "chatgpt.description")).toBe(
      `用你的 ChatGPT Plus 或 Pro 订阅代替 API 密钥。OpenAI 会通过你的 ChatGPT 账号收到：${SHORT["zh-CN"]}。用量计入你的订阅额度。`,
    );
    for (const language of ["en", "zh-CN"] as const) {
      for (const key of [
        "providers.choice.apiKey.body",
        "providers.choice.apiKey.firstRun.body",
      ] as const) {
        const sends = chosenFlowSummary(t(language));
        const text = translate(language, key, { sends, providers: "OpenAI" });
        expect(text.endsWith(`${sends}${language === "en" ? "." : "。"}`)).toBe(true);
      }
    }
  });

  test.each([
    ["OpenAI", { kind: "openai", apiKey: "sk-test-openai", modelId: "gpt-5.4-mini" }],
    ["Anthropic", { kind: "anthropic", apiKey: "sk-ant-test", modelId: "claude-sonnet-4-6" }],
  ] as const)(
    "names every flow that choosing %s allows, and every kind of data they send",
    async (name, input) => {
      const core = startCore(await createTempDataFolder());
      await core.saveChatProvider(input);

      const toService = (await core.listDataFlows()).filter(
        (status) => status.flow.service.name === name,
      );

      expect(toService.map((status) => status.flow.id).sort()).toEqual(
        [...CHOSEN_PROVIDER_FLOWS].sort(),
      );
      expect(toService.every((status) => status.chosen)).toBe(true);
      const named = Object.values(NAMED_BY).flat();
      const sent = new Set(toService.flatMap((status) => status.flow.sends));
      // What a Connector's Tools return isn't named: with one on, the first Question asks.
      sent.delete("tool-results");
      expect([...sent].filter((kind) => !named.includes(kind))).toEqual([]);
    },
  );
});
