import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
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

/** The question and its line in full, not cut, as the short tray of a narrow window draws them. */
async function expectWhole(card: Locator, question: string, detail: string) {
  const title = card.getByText(question, { exact: true });
  await expect(title).toBeVisible();
  expect(await title.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  await expect(card.getByText(detail, { exact: true })).toBeVisible();
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

// The question as Claude Haiku 5.5 asked it: one choice, the recommended row ticked, Other… allowed.
for (const [how, act, said] of [
  [
    "a row clicked after words typed in Other… sends the row alone",
    async (card: Locator) => {
      await card.getByTestId("question-other").fill("the boiler one");
      await card.getByTestId("question-option").nth(1).click();
    },
    "You chose: All 3 Documents",
  ],
  [
    "words typed in Other… untick the ticked row and are sent alone",
    async (card: Locator) => {
      const ticked = card.getByTestId("question-option").first();
      await expect(ticked).toHaveAttribute("aria-checked", "true");
      await card.getByTestId("question-other").fill("the boiler one");
      await expect(ticked).toHaveAttribute("aria-checked", "false");
      await card.getByTestId("question-other").press("Enter");
    },
    "You chose: “the boiler one”",
  ],
] as const) {
  test(`in a single choice, ${how}`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setUp(app, window);

    const card = await ask(window, "Summarise the main ones");
    await expect(card).toContainText("Which documents should I summarise as the “main” ones?");
    await act(card);

    await expect(card).toHaveCount(0);
    const answer = assistantAnswers(window).last();
    await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(answer.getByTestId("answer-asked")).toHaveText(said);
    await expect(window.getByRole("alert")).toHaveCount(0);
    await app.close();
  });
}

test("at a narrow window the question keeps its line, and its rows and buttons stay in view", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setUp(app, window);
  await setWindowSize(app, window, 900, 700);

  const card = await ask(window, "Summarise the main documents");
  await expectWhole(
    card,
    "Which Documents should I summarise?",
    "Your Library has 3. The two newest are picked.",
  );
  await expect(card.getByTestId("question-option").first()).toBeInViewport();
  await expect(card.getByTestId("question-continue")).toBeInViewport();
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
  await expectWhole(card, "要总结哪几份文档？", "资料库里有 3 份，已选中最新的两份。");
  await expect(card.getByTestId("question-recommended")).toHaveText("推荐");
  await expect(card.getByTestId("question-other")).toHaveAttribute("placeholder", "其他…");
  await expect(card.getByTestId("question-continue")).toContainText("继续");
  await expect(card.getByTestId("question-skip")).toContainText("跳过");
  const box = await card.boundingBox();
  const tray = await window.getByTestId("task-tray").boundingBox();
  expect(box && tray && box.x + box.width <= tray.x + tray.width + 1).toBe(true);
  await expect(card.getByTestId("question-option").first()).toBeInViewport();
  await expect(card.getByTestId("question-continue")).toBeInViewport();

  await card.getByTestId("question-other").fill("还有那份锅炉的");
  await card.getByTestId("question-other").press("Enter");
  const answer = assistantAnswers(window).last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-asked")).toHaveText(
    "你选择了：Board minutes、Lease Mill Lane、“还有那份锅炉的”",
  );
  await app.close();
});
