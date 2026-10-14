/**
 * A claim check's sentence is written in the interface's language (#701),
 * against Claude on the Anthropic key. Runs only with the key in the
 * environment (INCARNAMIND_LIVE_ANTHROPIC_KEY; the model is
 * INCARNAMIND_LIVE_ANTHROPIC_MODEL, Haiku 5.5 by default), and makes no
 * request without one. INCARNAMIND_LIVE_REPORT keeps what each run said.
 */
import { appendFile } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import { createAiSdkChatModel } from "../../src/core";
import { detectLanguage } from "../../src/core/documents/textLanguage";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";

const KEY = process.env.INCARNAMIND_LIVE_ANTHROPIC_KEY;
const MODEL =
  process.env.INCARNAMIND_LIVE_ANTHROPIC_MODEL ??
  (catalogProvider("anthropic")?.roles.quickTasks as string);

const CASES = [
  {
    language: "en",
    expected: "English",
    document: {
      name: "Halden.md",
      contents:
        "# Halden annual figures\n\n| Year | Revenue (GBP m) | Operating profit (GBP m) |\n|---|---|---|\n| FY2022 | 1,142.0 | 151.2 |\n| FY2023 | 1,180.4 | 160.9 |\n| FY2024 | 1,226.7 | 171.6 |\n",
    },
    claims: [
      "Operating profit rose to GBP 178.6 million in FY2024.",
      "Revenue in FY2023 was GBP 1,210.4 million.",
      "Operating profit in FY2022 was GBP 140.2 million.",
      "Revenue reached GBP 1,300.0 million in FY2024.",
      "Operating profit in FY2023 was GBP 165.0 million.",
    ],
  },
  {
    language: "zh-CN",
    expected: "Chinese",
    document: {
      name: "哈尔登.md",
      contents:
        "# 哈尔登年度数据\n\n| 年度 | 收入（百万英镑） | 营业利润（百万英镑） |\n|---|---|---|\n| 2022 财年 | 1,142.0 | 151.2 |\n| 2023 财年 | 1,180.4 | 160.9 |\n| 2024 财年 | 1,226.7 | 171.6 |\n",
    },
    claims: [
      "2024 财年营业利润增至 1.786 亿英镑。",
      "2023 财年收入为 12.104 亿英镑。",
      "2022 财年营业利润为 1.402 亿英镑。",
      "2024 财年收入达到 13 亿英镑。",
      "2023 财年营业利润为 1.65 亿英镑。",
    ],
  },
] as const;

describe.each(CASES)("claim check language, live: $language", (each) => {
  test.skipIf(KEY === undefined)(
    `five checks in ${each.language} come back in ${each.expected}`,
    { timeout: 600_000 },
    async () => {
      const core = startCore(await createTempDataFolder(), {
        createChatModel: createAiSdkChatModel,
      });
      core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
      await core.saveChatProvider({ kind: "anthropic", apiKey: KEY ?? "", modelId: MODEL });
      await core.updateSettings({ user: { language: each.language } });
      const sources = await createTempDataFolder();
      await addAndProcess(core, [
        await writeSourceFile(sources, each.document.name, each.document.contents),
      ]);
      const lines: string[] = [`## ${MODEL} · ${each.language}`];
      const languages: (string | null)[] = [];
      for (const claim of each.claims) {
        const result = await core.checkClaim({ claim });
        const says = result.verdict === "not-checked" ? null : result.says;
        lines.push(`- ${claim} → ${result.verdict}: ${says ?? "(none)"}`);
        if (says) languages.push(detectLanguage(says));
      }
      if (process.env.INCARNAMIND_LIVE_REPORT) {
        await appendFile(process.env.INCARNAMIND_LIVE_REPORT, `${lines.join("\n")}\n\n`);
      }
      expect(languages.length).toBeGreaterThan(0);
      for (const found of languages) expect(found ?? each.expected).toBe(each.expected);
    },
  );
});
