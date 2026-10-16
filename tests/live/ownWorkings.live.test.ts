/**
 * The User's own workings (#897), against Claude on the Anthropic key: a
 * spreadsheet marked "Don't cite this Document" beside the annual report it
 * was copied from. Runs only with the key in the environment
 * (INCARNAMIND_LIVE_ANTHROPIC_KEY; the model is INCARNAMIND_LIVE_ANTHROPIC_MODEL,
 * Haiku 5.5 by default), and makes no request without one.
 * INCARNAMIND_LIVE_REPORT keeps what each run said.
 */
import { appendFile } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import { createAiSdkChatModel } from "../../src/core";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { askAndFinish, citationsIn } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { answerText } from "../helpers/minds";
import { xlsxOf } from "../helpers/office";
import { buildPdf } from "../helpers/pdf";

const KEY = process.env.INCARNAMIND_LIVE_ANTHROPIC_KEY;
const MODEL =
  process.env.INCARNAMIND_LIVE_ANTHROPIC_MODEL ??
  (catalogProvider("anthropic")?.roles.answers as string);

const REPORT = buildPdf([
  {
    lines: [
      "Halden annual report 2025",
      "Revenue was $1,482.6 million in 2025, up from $1,231.0 million in 2024.",
    ],
  },
]);
const WORKINGS = xlsxOf([
  {
    name: "Workings",
    rows: [
      ["Item", "2024", "2025"],
      ["Revenue (USD million)", 1231.0, 1482.6],
      ["Operating margin", 0.18, 0.21],
    ],
  },
]);

describe("own workings, live", () => {
  test.skipIf(KEY === undefined)(
    "the Answer cites only the report, the check is Supported by it, and a figure only the workings state is Unsupported",
    { timeout: 600_000 },
    async () => {
      const core = startCore(await createTempDataFolder(), {
        createChatModel: createAiSdkChatModel,
      });
      core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
      await core.saveChatProvider({ kind: "anthropic", apiKey: KEY ?? "", modelId: MODEL });
      const sources = await createTempDataFolder();
      const documents = await addAndProcess(core, [
        await writeSourceFile(sources, "Halden annual report 2025.pdf", REPORT),
        await writeSourceFile(sources, "Halden workings.xlsx", WORKINGS),
      ]);
      const workings = documents.find((each) => each.name === "Halden workings");
      if (!workings) throw new Error("The workings weren't indexed.");
      await core.setNotCited(workings.id, true);
      const mind = await core.createMind({ title: "Halden" });
      const client = await connectToMind(core, mind.id);
      const lines: string[] = [`## ${MODEL}`];

      const both = await askAndFinish(core, client, mind.id, "What was revenue in 2025?");
      const cited = citationsIn(client, both.answerId);
      lines.push(
        `- Answer: ${answerText(client, both.answerId)}`,
        `  cites: ${cited.map((each) => `${each.documentName} (${each.check})`).join(", ")}`,
      );
      expect(cited.length).toBeGreaterThan(0);
      expect(new Set(cited.map((each) => each.documentName))).toEqual(
        new Set(["Halden annual report 2025"]),
      );

      const only = await askAndFinish(core, client, mind.id, "What was the operating margin?");
      lines.push(
        `- Answer: ${answerText(client, only.answerId)}`,
        `  cites: ${citationsIn(client, only.answerId)
          .map((each) => each.documentName)
          .join(", ")}`,
      );
      for (const citation of citationsIn(client, only.answerId)) {
        expect(citation.documentName).not.toBe("Halden workings");
      }

      const supported = await core.checkClaim({ claim: "Revenue was $1,482.6 million in 2025." });
      const supportedBy = supported.verdict === "supported" ? supported.evidence : null;
      lines.push(
        `- Revenue was $1,482.6 million in 2025. → ${supported.verdict} (${supportedBy?.documentName})`,
      );
      expect(supported.verdict).toBe("supported");
      expect(supportedBy?.documentName).toBe("Halden annual report 2025");

      const unsupported = await core.checkClaim({ claim: "Operating margin was 21% in 2025." });
      const left =
        unsupported.verdict === "not-checked"
          ? []
          : unsupported.excluded.map((each) => each.reason);
      lines.push(
        `- Operating margin was 21% in 2025. → ${unsupported.verdict} (${left.join(", ")})`,
      );
      expect(unsupported.verdict).toBe("unsupported");
      expect(left).toEqual(["own-workings"]);

      if (process.env.INCARNAMIND_LIVE_REPORT) {
        await appendFile(process.env.INCARNAMIND_LIVE_REPORT, `${lines.join("\n")}\n\n`);
      }
    },
  );
});
