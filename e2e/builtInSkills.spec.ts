import { expect, test } from "@playwright/test";
import {
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSettings,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

/** The Skills the app ships (resources/skills/), in name order. */
const BUILT_IN = ["literature-review", "mind-to-report", "summarise-document"];

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a fresh data folder has the three built-in Skills: labelled in Settings, and offered in a Question's slash menu", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);

  // Settings → Skills: installed on first run, turned on, and labelled built-in.
  await openSettings(window, "skills");
  const section = window.getByTestId("skills-settings");
  const items = section.getByTestId("skill-item");
  await expect(items).toHaveCount(3);
  for (const [index, name] of BUILT_IN.entries()) {
    const item = items.nth(index);
    await expect(item).toHaveAttribute("data-skill-name", name);
    await expect(item).toHaveAttribute("data-built-in", "true");
    await expect(item.getByTestId("skill-built-in")).toHaveText("Built-in");
    await expect(item.getByTestId("skill-enabled")).toBeChecked();
    await expect(item).toContainText("Licence: Apache-2.0");
  }
  const restore = section.getByTestId("skill-restore-built-ins");
  await expect(restore).toHaveCount(0);

  // Removed, a built-in Skill can be restored.
  const itemNamed = (name: string) =>
    section.locator(`[data-testid="skill-item"][data-skill-name="${name}"]`);
  await itemNamed("summarise-document").getByTestId("skill-remove").click();
  await expect(items).toHaveCount(2);
  await expect(restore).toHaveText("Restore built-in Skills");
  await restore.click();
  await expect(itemNamed("summarise-document")).toHaveAttribute("data-built-in", "true");
  await expect(items).toHaveCount(3);
  await expect(restore).toHaveCount(0);

  // Duplicated as the User's own: a copy that isn't built-in.
  await itemNamed("mind-to-report").getByTestId("skill-duplicate").click();
  const copy = itemNamed("mind-to-report-copy");
  await expect(copy).toHaveAttribute("data-built-in", "false");
  await expect(copy.getByTestId("skill-built-in")).toHaveCount(0);
  await expect(copy.getByTestId("skill-duplicate")).toHaveCount(0);
  await expect(items).toHaveCount(4);
  await closeSettings(window);

  // In a Question, "/" offers them like any other Skill.
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("/");
  const menu = window.getByTestId("slash-menu");
  for (const name of BUILT_IN) {
    await expect(menu.getByTestId(`slash-item-skill-${name}`)).toBeVisible();
  }
  await window.keyboard.type("lit");
  await expect(menu.getByTestId("slash-item-skill-literature-review")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await window.keyboard.press("Enter");
  const question = editor.getByTestId("question");
  const chip = question.getByTestId("question-skill");
  await expect(chip).toHaveText("literature-review");
  await expect(chip).toHaveAttribute("data-state", "enabled");

  // Asked, the Answer follows it, as with any forced Skill.
  await window.keyboard.type("What do my Documents say about tides?");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  const card = answer.getByTestId("answer-skill");
  await expect(card).toHaveAttribute("data-skill-name", "literature-review");
  await expect(card).toHaveAttribute("data-forced", "true");
  await expect(answer).toContainText("Following the Skill literature-review.");
  await app.close();
});
