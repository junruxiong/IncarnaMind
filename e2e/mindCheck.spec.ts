import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  acceptPlainTextKeys,
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  mindWrittenBefore,
  motionSettled,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * Check this Mind (#737): "Check this Mind" (⌘K, the Mind's menu) checks
 * every sentence that makes a claim, in one run, then the health line's
 * Check again checks only what changed. The scripted model
 * (src/main/fakeChat/claims.ts) names every sentence a claim but questions,
 * thanks and "we think"; a claim sharing three words with a line of the
 * Tides PDF is supported, or contradicted when it has a number the line
 * doesn't; any other is unsupported. Set INCARNAMIND_SCREENSHOTS to a folder
 * to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  {
    lines: [
      "Harbour times",
      "Harbours publish the times of high water every 12 months.",
      "Spring tides come at new moon and at full moon.",
      "Neap tides come at the first and last quarter moons.",
      "The harbour office opens at nine in the morning.",
    ],
  },
]);

/** Fifteen claims, three questions and two opinions: twenty sentences. */
const CLAIMS = [
  "Harbours publish the times of high water every 6 months.",
  "Spring tides come at new moon and at full moon.",
  "Neap tides come at the first and last quarter moons.",
  "The harbour office opens at nine in the morning.",
  "Penguins nest on the cliffs above the harbour.",
  "The ferry to the island leaves twice a day.",
  "Fishing boats land their catch before dawn.",
  "The lighthouse was built in 1867.",
  "The pier is closed to cars in summer.",
  "Two cranes unload the cargo ships.",
  "The harbour master keeps a log of every arrival.",
  "Storm warnings are posted at the harbour office.",
  "The sea wall was raised after the 1953 flood.",
  "Pilots guide large ships into the channel.",
  "The marina has berths for sixty yachts.",
];
const QUESTIONS = [
  "Does high water come at noon on Sundays?",
  "Who looks after the lighthouse now?",
  "Can visitors walk on the sea wall?",
];
const OPINIONS = [
  "We think the harbour is at its best in spring.",
  "I believe the new marina was worth building.",
];

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

/** The Mind of twenty sentences under a heading, opened. */
async function harbourNotes(window: Page) {
  await mindWrittenBefore(window, "Harbour notes", [
    { heading: "The harbour" },
    { note: `${CLAIMS.slice(0, 5).join(" ")} ${QUESTIONS[0]} ${OPINIONS[0]}` },
    { heading: "Ships and boats" },
    { note: `${CLAIMS.slice(5, 10).join(" ")} ${QUESTIONS[1]}` },
    { note: `${CLAIMS.slice(10).join(" ")} ${QUESTIONS[2]} ${OPINIONS[1]}` },
  ]);
  await window.getByTestId("mind-list-item").filter({ hasText: "Harbour notes" }).click();
  await expect(window.getByTestId("mind-editor")).toContainText(CLAIMS[0] ?? "");
}

const marks = (window: Page) => window.getByTestId("claim-mark");

test("Check this Mind from ⌘K: the estimate first on a cloud model, then every claim marked, the headings, questions and opinions not; an edited claim is checked again alone", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await acceptPlainTextKeys(window);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const provider = await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "sk-ant-test",
      modelId: "claude-haiku-5-5",
    });
    if (provider.service) {
      for (const flow of ["chat", "tagging", "checks"] as const) {
        await core.allowDataFlow(flow, provider.service.id);
      }
    }
  });
  await setWindowSize(app, window, 1440, 900);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await harbourNotes(window);

  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type("check this mind");
  const item = window.getByTestId("palette-item").filter({ hasText: "Check this Mind" }).first();
  await pointAndClick(window, item);

  // The model names the claims first; the line then asks with their number and the estimate,
  // its price first. Of the 20 sentences, the 15 claims are what it checks.
  const ask = window.getByTestId("mind-check-ask");
  await expect(ask).toContainText("Check this Mind’s 15 claims against your Documents?");
  await expect(ask).toContainText("About $");
  await expect(window.getByTestId("palette-search")).toBeHidden();
  await motionSettled(window);
  await screenshot(window, "mind-check-ask-en");
  await pointAndClick(window, ask.getByTestId("mind-check-check"));

  const health = window.getByTestId("mind-health");
  await expect(health.getByTestId("health-checked-all")).toContainText("Checked 15 sentences, ");
  await expect(marks(window)).toHaveCount(15);
  await expect(marks(window).and(window.locator('[data-check="checking"]'))).toHaveCount(0);
  await expect(marks(window).and(window.locator('[data-check="contradicted"]'))).toHaveCount(1);
  await expect(marks(window).and(window.locator('[data-check="supported"]'))).toHaveCount(3);
  // Nothing changed since: nothing to check again.
  await expect(health.getByTestId("health-check")).toHaveCount(0);
  await screenshot(window, "mind-check-done-en");

  // An edited claim needs a re-check, and Check again checks it alone.
  const editor = window.getByTestId("mind-editor");
  const first = editor.locator(":scope > p").first();
  await slideSelect(window, first, "Penguins");
  await window.keyboard.type("Puffins");
  await expect(marks(window).and(window.locator('[data-check="needs-recheck"]'))).toHaveCount(1);
  const again = health.getByTestId("health-check");
  await expect(again).toHaveText("Check again");
  await pointAndClick(window, again);
  await expect(marks(window).and(window.locator('[data-check="needs-recheck"]'))).toHaveCount(0);
  await expect(marks(window).and(window.locator('[data-check="unsupported"]'))).toHaveCount(11);
  await expect(marks(window)).toHaveCount(15);
});

test("In Chinese and a narrow window, from the Mind's tab menu: a model on this computer checks at once; a Folder that holds too little offers all Documents; Check all checks a sentence written since", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(
    join(sources, "Garden.pdf"),
    buildPdf([{ lines: ["Garden notes", "Roses like a sunny wall and rich soil."] }]),
  );
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf"), join(sources, "Garden.pdf")]);
  await harbourNotes(window);
  // The Mind's Folder holds only the garden notes; the Tides PDF is outside it.
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const folder = await core.createLibraryGroup({ name: "Garden", description: "" });
    const garden = (await core.listDocuments()).filter((each) => each.name === "Garden");
    const mind = (await core.listMinds()).filter((each) => each.title === "Harbour notes");
    await core.moveToFolder({
      documentIds: garden.map((each) => each.id),
      mindIds: mind.map((each) => each.id),
      folderId: folder.id,
    });
  });
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 720);

  await window.getByTestId("mind-tab").first().click({ button: "right" });
  await pointAndClick(window, window.getByRole("menuitem", { name: "核对整个 Mind" }));
  // On this computer it asks too, with the claims the model named and no price.
  const ask = window.getByTestId("mind-check-ask");
  await expect(ask).toContainText("要对照这个文件夹中的文档，核对这个 Mind 中的 15 条说法吗？");
  await expect(ask).not.toContainText("约");
  await pointAndClick(window, ask.getByTestId("mind-check-check"));
  const health = window.getByTestId("mind-health");
  await expect(health.getByTestId("health-checked-all")).toContainText("已核对 15 句，");
  await expect(marks(window).and(window.locator('[data-check="unsupported"]'))).toHaveCount(15);

  // Nothing in the Folder says any of it: checked against every Document, the Tides PDF does.
  const widen = window.getByTestId("checks-uncovered").getByTestId("checks-check-all");
  await expect(widen).toHaveText("对照全部文档核对");
  await pointAndClick(window, widen);
  // The claims aren't sorted again: the ask comes at once.
  await expect(ask).toContainText("要对照你的文档，核对这个 Mind 中的 15 条说法吗？");
  await pointAndClick(window, ask.getByTestId("mind-check-check"));
  await expect(window.getByTestId("checks-running")).toHaveCount(0);
  await expect(health.getByTestId("health-checked-all")).toBeVisible();
  await expect(marks(window).and(window.locator('[data-check="supported"]'))).toHaveCount(3);
  await expect(marks(window).and(window.locator('[data-check="contradicted"]'))).toHaveCount(1);
  await expect(widen).toHaveCount(0);
  await expect(marks(window)).toHaveCount(15);
  await expect(health.getByTestId("health-unsupported")).toHaveText("11 句无依据");
  await screenshot(window, "mind-check-done-zh-narrow");

  // A sentence written since: "全部核对" checks it, and the rest stay as they were.
  const editor = window.getByTestId("mind-editor");
  const last = editor.locator(":scope > p").last();
  const box = await last.boundingBox();
  if (!box) throw new Error("The last paragraph isn't drawn.");
  // Right of its last line: the caret goes to the paragraph's end.
  await last.click({ position: { x: box.width - 2, y: box.height - 4 } });
  await window.keyboard.type(" The harbour office opens at nine in the morning.");
  await expect(last).toContainText("worth building. The harbour office opens");
  const all = health.getByTestId("health-check");
  await expect(all).toHaveText("全部核对");
  await pointAndClick(window, all);
  await expect(ask).toContainText("要对照这个文件夹中的文档，核对这个 Mind 中的这条说法吗？");
  await pointAndClick(window, ask.getByTestId("mind-check-check"));
  await expect(marks(window)).toHaveCount(16);
  await expect(health.getByTestId("health-check")).toHaveCount(0);
});
