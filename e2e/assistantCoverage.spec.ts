import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  askInAssistant,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/**
 * An Answer about the main few of the Documents says which it took, above
 * its reply, with Use others: that asks again with the rest, in a new Answer
 * under the first, whose steps line says so. The first stays as it was.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
let files: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  files = await mkdtemp(join(tmpdir(), "coverage-e2e-"));
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await rm(files, { recursive: true, force: true });
});

/** Three reports, the first the longest. */
async function reports(): Promise<string[]> {
  return Promise.all(
    [1, 2, 3].map(async (at) => {
      const path = join(files, `Report ${at}.md`);
      const lines = Array.from(
        { length: (4 - at) * 12 },
        (_, line) => `Report ${at}, finding ${line + 1}: the tide rose ${line} cm.`,
      );
      await writeFile(path, `# Report ${at}\n\n${lines.join("\n\n")}\n`);
      return path;
    }),
  );
}

test("“the two main documents” names the two it took, and Use others asks again with the rest under it; in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "en");
  await setWindowSize(app, window, 1440, 900);
  await addDocuments(window, await reports());
  await newMind(window);
  await window.keyboard.type("Reading group");

  await askInAssistant(window, "Summarise the two main documents");

  // The Answer stays the first; `askInAssistant` gives the last, which moves.
  const first = assistantAnswers(window).nth(0);
  const coverage = first.getByTestId("answer-coverage");
  await expect(coverage).toHaveText(
    "I took “Report 1” and “Report 2” as the main ones. · Use others",
  );
  // Above the reply.
  const line = await coverage.boundingBox();
  const reply = await first.getByTestId("assistant-answer-text").boundingBox();
  expect(line && reply && line.y < reply.y).toBe(true);
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "main-pick-en.png") });

  await pointAndClick(window, coverage.getByTestId("answer-coverage-rest"));
  const answers = assistantAnswers(window);
  await expect(answers).toHaveCount(2);
  const again = answers.last();
  await expect(again).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(again.getByTestId("answer-asked-again")).toHaveText(
    "Asked again with the 1 Document the first answer didn't use",
  );
  await expect(again.getByTestId("answer-coverage")).toHaveCount(0);
  await expect(coverage).toHaveText(
    "I took “Report 1” and “Report 2” as the main ones. · Use others",
  );
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "asked-again-en.png") });

  await setLanguage(window, "zh-CN");
  await expect(coverage).toHaveText("我选了“Report 1”和“Report 2”作为主要文档。· 换别的");
  await expect(again.getByTestId("answer-asked-again")).toHaveText(
    "用第一次没用到的 1 个文档重新回答",
  );
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "main-pick-zh.png") });
});
