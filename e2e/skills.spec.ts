import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  interceptSkillPicker,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

const DESCRIPTION = "Reads tide tables. Use for questions about high and low water.";

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

/** A small Skill folder: instructions, a reference and a script. */
async function writeTideSkill(): Promise<string> {
  const folder = join(sources, "tide-tables");
  await mkdir(join(folder, "references"), { recursive: true });
  await mkdir(join(folder, "scripts"), { recursive: true });
  await writeFile(
    join(folder, "SKILL.md"),
    `---\nname: tide-tables\ndescription: ${DESCRIPTION}\nlicense: MIT\n---\n\n# Tide tables\n\nRead references/ports.md first.\n`,
  );
  await writeFile(join(folder, "references", "ports.md"), "Brest, Plymouth and Saint-Malo.\n");
  await writeFile(join(folder, "scripts", "convert.py"), "print('metres to feet')\n");
  return folder;
}

test("a Skill imported from a folder in Settings is forced from the slash menu in a Question, shown as a chip, and the Answer shows it as a Skill card", async () => {
  const folder = await writeTideSkill();
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);

  // Settings → Skills: the folder (picked through the test hook) is shown first, then imported.
  await openSettings(window, "skills");
  const section = window.getByTestId("skills-settings");
  // Only the built-in Skills so far (e2e/builtInSkills.spec.ts covers them).
  const item = section.locator('[data-testid="skill-item"][data-skill-name="tide-tables"]');
  await expect(section.getByTestId("skill-item")).toHaveCount(3);
  await expect(item).toHaveCount(0);
  await interceptSkillPicker(window, folder);
  await section.getByTestId("skill-import-folder").click();
  const preview = section.getByTestId("skill-preview");
  await expect(preview.getByTestId("skill-preview-name")).toHaveText("tide-tables");
  await expect(preview.getByTestId("skill-preview-description")).toHaveText(DESCRIPTION);
  await expect(preview).toContainText("Licence: MIT");
  await preview.getByText(/^3 files/).click();
  const files = preview.getByTestId("skill-preview-file");
  await expect(files).toHaveCount(3);
  await expect(files.nth(0)).toContainText("SKILL.md");
  await expect(files.nth(1)).toContainText("references/ports.md");
  await expect(files.nth(2)).toContainText("scripts/convert.py");
  await expect(files.nth(2)).toContainText("script: runs only with your approval");
  await preview.getByTestId("skill-import-confirm").click();

  await expect(section.getByTestId("skill-item")).toHaveCount(4);
  await expect(item).toHaveAttribute("data-built-in", "false");
  await expect(item.getByTestId("skill-built-in")).toHaveCount(0);
  await expect(item.getByTestId("skill-enabled")).toBeChecked();
  await expect(item).toContainText(DESCRIPTION);
  await expect(item).toContainText("Licence: MIT · 1 script");
  await expect(section.getByTestId("skill-preview")).toHaveCount(0);
  await closeSettings(window);

  // In the composer, "/" offers the Skills only; choosing one shows it as a chip.
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("/tide");
  const menu = window.getByTestId("composer-skill-picker");
  await expect(menu.getByTestId("slash-item-skill-tide-tables")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(menu.getByTestId("slash-item-heading-1")).toHaveCount(0);
  await window.keyboard.press("Enter");
  const composer = window.getByTestId("composer");
  await expect(composer.getByTestId("composer-skill")).toHaveText("tide-tables");
  await expect(composer.getByTestId("composer-skill")).toHaveAttribute("data-state", "enabled");
  await expect(composer.getByTestId("composer-input")).toHaveValue("");

  // Asked, the Question carries the Skill, and the Answer follows it, saying so on a Skill card.
  await window.keyboard.type("When is high water in Brest?");
  await window.keyboard.press("Enter");
  const question = editor.getByTestId("question");
  const chip = question.getByTestId("question-skill");
  await expect(chip).toHaveText("tide-tables");
  await expect(chip).toHaveAttribute("data-state", "enabled");
  await expect(composer.getByTestId("composer-skill")).toHaveCount(0);
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  const card = answer.getByTestId("answer-skill");
  await expect(card).toHaveAttribute("data-skill-name", "tide-tables");
  await expect(card).toHaveAttribute("data-forced", "true");
  await expect(card).toContainText("Used the Skill tide-tables · chosen for this Question");
  await expect(answer).toContainText("Following the Skill tide-tables.");

  // A Skill chosen in the composer for the next Question, before it was turned off.
  await composer.getByTestId("composer-input").click();
  await window.keyboard.type("/tide");
  await window.keyboard.press("Enter");
  await window.keyboard.type("And in Calais?");

  // Turned off, the Skill can't be used: asking again says so, and can go ahead without it.
  await openSettings(window, "skills");
  await item.getByTestId("skill-enabled").uncheck();
  await expect(item).toHaveAttribute("data-enabled", "false");
  await closeSettings(window);
  await expect(chip).toHaveAttribute("data-state", "disabled");
  await answer.hover();
  await answer.getByTestId("answer-regenerate").click();
  const notice = question.getByTestId("question-skill-unavailable");
  await expect(notice).toContainText("which is turned off");
  await notice.getByRole("button", { name: "Ask without it" }).click();
  await expect(question.getByTestId("question-skill")).toHaveCount(0);
  await expect(notice).toHaveCount(0);
  await expect(answer).not.toContainText("Following the Skill");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer).toContainText("That is all.");
  await expect(answer.getByTestId("answer-skill")).toHaveCount(0);

  // So does asking from the composer, in its row: the Question waits there until then.
  await expect(composer.getByTestId("composer-skill")).toHaveAttribute("data-state", "disabled");
  await composer.getByTestId("composer-input").click();
  await window.keyboard.press("Enter");
  const composerNotice = composer.getByTestId("composer-skill-unavailable");
  await expect(composerNotice).toContainText("which is turned off");
  await expect(editor.getByTestId("question")).toHaveCount(1);
  await composerNotice.getByRole("button", { name: "Ask without it" }).click();
  await expect(composerNotice).toHaveCount(0);
  await expect(composer.getByTestId("composer-skill")).toHaveCount(0);
  const second = editor.getByTestId("answer").nth(1);
  await expect(second).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(editor.getByTestId("question").nth(1)).toContainText("And in Calais?");
  await expect(second).not.toContainText("Following the Skill");
  await app.close();
});

test("a Question can force a Skill and limit its Search scope at once: both chips show, and both apply", async () => {
  const folder = await writeTideSkill();
  await writeFile(join(sources, "Harbour.txt"), "High water in Brest comes twice a day.\n");
  await writeFile(join(sources, "Recipes.txt"), "Cook mussels at low water, with garlic.\n");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Harbour.txt"), join(sources, "Recipes.txt")]);
  // Imported through the core's bridge: the first test covers Settings.
  await window.evaluate(async (path) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const check = await bridge.previewSkillImport(path);
    if (!check.ok) throw new Error(check.error.message);
    await bridge.importSkill(check.preview.importId);
  }, folder);

  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("/tide");
  await expect(window.getByTestId("slash-item-skill-tide-tables")).toBeVisible();
  await window.keyboard.press("Enter");
  await window.keyboard.type("@harb");
  const choices = window.getByTestId("scope-picker").getByTestId("scope-choice");
  await expect(choices).toHaveCount(1);
  await window.keyboard.press("Enter");
  await window.keyboard.type("When is high water in Brest?");

  const composer = window.getByTestId("composer");
  await expect(composer.getByTestId("composer-skill")).toHaveText("tide-tables");
  await expect(composer.getByTestId("scope-chip")).toContainText("Harbour");
  await expect(composer.getByTestId("composer-input")).toHaveValue("When is high water in Brest?");

  // Asked: the Question carries both; the Skill was loaded up front, and the search found
  // only the Document in scope.
  await window.keyboard.press("Enter");
  const question = editor.getByTestId("question");
  await expect(question.getByTestId("question-skill")).toHaveText("tide-tables");
  const scope = question.getByTestId("scope-chip");
  await expect(scope).toHaveCount(1);
  await expect(scope).toContainText("Harbour");
  await expect(question.locator(".question-text")).toHaveText("When is high water in Brest?");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-skill")).toHaveAttribute("data-forced", "true");
  await expect(answer).toContainText("Following the Skill tide-tables.");
  await expect(answer.getByTestId("citation-chip")).toHaveAttribute("aria-label", /Harbour/);
  // Without the scope, both Documents would match.
  await answer.getByTestId("answer-tools-toggle").click();
  await expect(answer.getByTestId("answer-tool-call")).toContainText("1 Passage");

  // Regenerated: both still apply.
  await answer.hover();
  await answer.getByTestId("answer-regenerate").click();
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("answer-skill")).toHaveAttribute("data-forced", "true");
  await expect(answer).toContainText("Following the Skill tide-tables.");
  await expect(answer.getByTestId("citation-chip")).toHaveAttribute("aria-label", /Harbour/);
  await answer.getByTestId("answer-tools-toggle").click();
  await expect(answer.getByTestId("answer-tool-call")).toContainText("1 Passage");
  await app.close();
});
