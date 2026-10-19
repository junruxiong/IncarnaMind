import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import {
  addDocuments,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

const NAMES = ["Lease Harbour Street", "Lease Mill Lane", "Board minutes"];

/** Three Documents, added in this order, so the newest is Board minutes. */
async function setUp(app: ElectronApplication, window: Page, language: "en" | "zh-CN" = "en") {
  await setLanguage(window, language);
  await setWindowSize(app, window, 1440, 900);
  const paths = await Promise.all(
    NAMES.map(async (name) => {
      const path = join(sources, `${name}.md`);
      await writeFile(path, `# ${name}\n\nSigned in 2026.\n`);
      return path;
    }),
  );
  await addDocuments(window, paths);
  await newMind(window);
}

async function ask(window: Page, question: string) {
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type(question);
  await window.keyboard.press("Enter");
  const card = window.getByTestId("task-tray").getByTestId("question-card");
  await expect(card).toBeVisible();
  return card;
}

test("asked which Documents to summarise, the User ticks one more and goes on", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setUp(app, window);

  const card = await ask(window, "Summarise the main documents");
  await expect(card).toContainText("Which Documents should I summarise?");
  await expect(card).toContainText("Your Library has 3. The two newest are picked.");
  const rows = card.getByTestId("question-option");
  await expect(rows).toHaveText([
    /^1Board minutesRecommended/,
    /^2Lease Mill Lane/,
    /^3Lease Harbour Street/,
  ]);
  await expect(rows.nth(0)).toHaveAttribute("aria-checked", "true");
  await expect(rows.nth(1)).toHaveAttribute("aria-checked", "true");
  await expect(rows.nth(2)).toHaveAttribute("aria-checked", "false");
  await expect(rows.nth(0)).toBeFocused();

  await window.keyboard.press("3");
  await expect(rows.nth(2)).toHaveAttribute("aria-checked", "true");
  await window.keyboard.press("ControlOrMeta+Enter");

  await expect(card).toHaveCount(0);
  const answer = assistantAnswers(window).last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-asked")).toHaveText(
    "You chose: Board minutes, Lease Mill Lane, Lease Harbour Street",
  );
  const editor = window.locator('[data-testid="mind-editor"]:visible');
  await expect(editor).toContainText("The main points of “Lease Harbour Street”.");
  await app.close();
});

test("Esc skips: the Answer goes on with the two newest and says so", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setUp(app, window);

  const card = await ask(window, "Summarise the main documents");
  await window.keyboard.press("Escape");

  await expect(card).toHaveCount(0);
  const answer = assistantAnswers(window).last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-asked")).toHaveText("Skipped; went with Board minutes");
  await expect(answer).toContainText("Summarised the two newest Documents, as you didn't choose.");
  await app.close();
});

test("in Chinese, the question and its rows, at a narrow window", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setUp(app, window, "zh-CN");
  await setWindowSize(app, window, 900, 700);

  const card = await ask(window, "总结主要的文档");
  await expect(card).toContainText("要总结哪几份文档？");
  await expect(card.getByTestId("question-recommended")).toHaveText("推荐");
  await expect(card.getByTestId("question-other")).toHaveAttribute("placeholder", "其他…");
  await expect(card.getByTestId("question-continue")).toContainText("继续");
  await expect(card.getByTestId("question-skip")).toContainText("跳过");
  const box = await card.boundingBox();
  const tray = await window.getByTestId("task-tray").boundingBox();
  expect(box && tray && box.x + box.width <= tray.x + tray.width + 1).toBe(true);

  await card.getByTestId("question-other").fill("还有那份锅炉的");
  await card.getByTestId("question-other").press("Enter");
  const answer = assistantAnswers(window).last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-asked")).toHaveText(
    "你选择了：Board minutes、Lease Mill Lane、“还有那份锅炉的”",
  );
  await app.close();
});
