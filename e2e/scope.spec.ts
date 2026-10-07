import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  launchApp,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

let dataDir: string;
/** Where the files a test adds are written: outside the data folder. */
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

test("typing @ in a Question limits its search to a Folder: the Answer cites only a Document in it", async () => {
  // Both answer the Question; each is filed in its own Folder.
  await writeFile(
    join(sources, "Tide tables.txt"),
    "Neap tides are the smallest of the month.\nSpring tides happen at new moon and at full moon.\n",
  );
  await writeFile(
    join(sources, "Spring menu.txt"),
    "When do spring tides happen? Spring tides happen when the market sells mussels.\n",
  );
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tide tables.txt"), join(sources, "Spring menu.txt")]);
  const ids = {
    tides: await documentIdOf(window, "Tide tables"),
    menu: await documentIdOf(window, "Spring menu"),
  };
  const oceanId = await window.evaluate(async ({ tides, menu }) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const ocean = await bridge.createFolder({ name: "Ocean" });
    const kitchen = await bridge.createFolder({ name: "Kitchen" });
    await bridge.moveDocument(tides, ocean.id);
    await bridge.moveDocument(menu, kitchen.id);
    return ocean.id;
  }, ids);

  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  const question = editor.getByTestId("question");
  await expect(question).toBeVisible();

  // "@" opens the picker: Folders, Tags and Documents, the first one chosen; arrows move.
  await window.keyboard.type("@");
  const picker = window.getByTestId("scope-picker");
  await expect(picker).toBeVisible();
  await expect(picker.getByRole("group", { name: "Folders" })).toBeVisible();
  await expect(picker.getByRole("group", { name: "Tags" })).toBeVisible();
  await expect(picker.getByRole("group", { name: "Documents" })).toBeVisible();
  const choices = picker.getByTestId("scope-choice");
  await expect(choices.first()).toHaveText("Kitchen");
  await expect(choices.first()).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("ArrowDown");
  await expect(choices.nth(1)).toHaveText("Ocean");
  await expect(choices.nth(1)).toHaveAttribute("aria-selected", "true");

  // Typing filters; Enter adds the choice as a chip, and the "@" text goes.
  await window.keyboard.type("oce");
  await expect(choices).toHaveCount(1);
  await expect(choices).toHaveAttribute("data-kind", "folder");
  await window.keyboard.press("Enter");
  await expect(picker).toHaveCount(0);
  const chip = question.getByTestId("scope-chip");
  await expect(chip).toHaveText("Ocean");
  await expect(chip).toHaveAttribute("data-id", oceanId);

  await window.keyboard.type("When do spring tides happen?");
  await expect(question.locator(".question-text")).toHaveText("When do spring tides happen?");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });

  // It searched only the Ocean Folder, and cites its Document.
  await answer.getByTestId("answer-tools-toggle").click();
  await expect(answer.getByTestId("answer-tool-call")).toContainText("1 Passage");
  const citation = answer.getByTestId("citation");
  await expect(citation).toHaveCount(1);
  await expect(citation).toHaveAttribute("data-check", "found");
  await expect(citation.getByTestId("citation-chip")).toHaveAttribute(
    "aria-label",
    /^Citation 1: Tide tables\. /,
  );

  // Once the Folder is deleted, its chip shows it, struck through; regenerating searches nothing.
  await window.evaluate(async (folderId) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.deleteFolder(folderId);
  }, oceanId);
  await expect(chip).toHaveAttribute("data-deleted", "true");
  await expect(chip).toContainText("Deleted Folder");
  await expect(chip.locator("s")).toHaveText("Deleted Folder");
  await answer.hover();
  await answer.getByTestId("answer-regenerate").click();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer).toContainText("has no Documents to search");
  await expect(answer.getByTestId("citation")).toHaveCount(0);

  // The chip's × takes it out of the scope.
  await chip.getByTestId("scope-chip-remove").click();
  await expect(question.getByTestId("question-scope")).toHaveCount(0);
  await app.close();
});
