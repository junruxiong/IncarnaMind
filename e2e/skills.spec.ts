import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  interceptSkillPicker,
  launchApp,
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
  await window.getByRole("button", { name: "Settings" }).click();
  const section = window.getByTestId("skills-settings");
  await expect(section).toContainText("No Skills yet.");
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
  await expect(files.nth(2)).toContainText("script: can't run yet");
  await preview.getByTestId("skill-import-confirm").click();

  const item = section.getByTestId("skill-item");
  await expect(item).toHaveAttribute("data-skill-name", "tide-tables");
  await expect(item.getByTestId("skill-enabled")).toBeChecked();
  await expect(item).toContainText(DESCRIPTION);
  await expect(item).toContainText("Licence: MIT · 1 script, which can't run yet");
  await expect(section.getByTestId("skill-preview")).toHaveCount(0);
  await window.getByTestId("settings").getByRole("button", { name: "Done" }).click();

  // In a Question, "/" offers the Skills only; choosing one shows it as a chip.
  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("/tide");
  const menu = window.getByTestId("slash-menu");
  await expect(menu.getByTestId("slash-item-skill-tide-tables")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(menu.getByTestId("slash-item-heading-1")).toHaveCount(0);
  await window.keyboard.press("Enter");
  const question = editor.getByTestId("question");
  const chip = question.getByTestId("question-skill");
  await expect(chip).toHaveText("tide-tables");
  await expect(chip).toHaveAttribute("data-state", "enabled");
  await expect(question.locator(".question-text")).toHaveText("");

  // Asked, the Answer follows the Skill, and says so on a Skill card.
  await window.keyboard.type("When is high water in Brest?");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  const card = answer.getByTestId("answer-skill");
  await expect(card).toHaveAttribute("data-skill-name", "tide-tables");
  await expect(card).toHaveAttribute("data-forced", "true");
  await expect(card).toContainText("Used the Skill tide-tables · chosen for this Question");
  await expect(answer).toContainText("Following the Skill tide-tables.");

  // Turned off, the Skill can't be used: asking says so, and can go ahead without it.
  await window.getByRole("button", { name: "Settings" }).click();
  await section.getByTestId("skill-enabled").uncheck();
  await expect(section.getByTestId("skill-item")).toHaveAttribute("data-enabled", "false");
  await window.getByTestId("settings").getByRole("button", { name: "Done" }).click();
  await expect(chip).toHaveAttribute("data-state", "disabled");
  await question.getByTestId("question-ask").click();
  const notice = question.getByTestId("question-skill-unavailable");
  await expect(notice).toContainText("which is turned off");
  await notice.getByRole("button", { name: "Ask without it" }).click();
  await expect(question.getByTestId("question-skill")).toHaveCount(0);
  await expect(notice).toHaveCount(0);
  await expect(answer).not.toContainText("Following the Skill");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer).toContainText("That is all.");
  await expect(answer.getByTestId("answer-skill")).toHaveCount(0);
  await app.close();
});
