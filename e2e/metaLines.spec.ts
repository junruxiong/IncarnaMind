import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  dotsOf,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
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

const language = (window: Page, value: "en" | "zh-CN") =>
  window.evaluate(
    (chosen) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: chosen },
      }),
    value,
  );

test("a line of parts joined by dots wraps between parts and never starts or ends with a dot, in English and Chinese at a narrow width", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await setWindowSize(app, window, 900, 700);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://a-long-service-name-that-fills-the-line.example.com/v1",
      modelId: "example-model",
    });
    await core.setLocalOnly(true);
  });
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is IncarnaMind?");
  await window.keyboard.press("Enter");
  const line = window.getByTestId("composer-not-ready");

  for (const value of ["en", "zh-CN"] as const) {
    await language(window, value);
    {
      const row = line.locator(".meta-line").first();
      await expect(row).toBeVisible();
      const dots = await dotsOf(row);
      // The line wraps here: some part begins a line, and its dot is not shown.
      expect(dots.some((dot) => !dot.sameLine)).toBe(true);
      for (const dot of dots) {
        expect(dot.shown).toBe(dot.sameLine);
        if (dot.shown) expect(dot.between).toBe(true);
      }
      expect((await row.innerText()).trimEnd().endsWith("·")).toBe(false);
    }
  }
  await app.close();
});

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const LONG_FOLDER =
  "Supplier agreements and framework contracts reviewed by legal in the second half";

for (const value of ["en", "zh-CN"] as const) {
  test(`a problem line in the Mind's health line that wraps onto a row of its own never starts or ends with a dot, at 1,100 and 1,440 px (${value}, #887)`, async () => {
    test.setTimeout(120_000);
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true, home: sources });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, value);
    await setWindowSize(app, window, 1440, 900);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    await newMind(window);
    await window.getByTestId("mind-title").fill("Deal memo");
    // Filed in a Folder with a long name and no Document: its checks have nothing to check against.
    await window.evaluate(async (name) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const folder = await core.createLibraryGroup({ name, description: "" });
      const mind = (await core.listMinds()).find((each) => each.title === "Deal memo");
      if (mind) await core.moveToFolder({ mindIds: [mind.id], folderId: folder.id });
    }, LONG_FOLDER);
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.type("Penguins nest on the harbour cliffs.");
    const line = editor.locator(":scope > p").first();
    await slideSelect(window, line, "Penguins", { line, words: "cliffs." });
    await pointAndClick(
      window,
      window.getByTestId("selection-menu").getByTestId("selection-action-check-against-documents"),
    );
    const problem = window.getByTestId("checks-problem");
    await expect(problem).toBeVisible();

    for (const width of [1100, 1440]) {
      await setWindowSize(app, window, width, 900);
      // The row of the count: the problem line is a part of it, and wraps onto a row of its own.
      const outer = await dotsOf(window.getByTestId("mind-count"));
      expect(outer.some((dot) => !dot.sameLine)).toBe(true);
      // And the problem's own words and actions wrap between parts inside it.
      const inner = await dotsOf(problem.locator(".meta-line").first());
      for (const dot of [...outer, ...inner]) {
        expect(dot.shown).toBe(dot.sameLine);
        if (dot.shown) expect(dot.between).toBe(true);
      }
      expect((await problem.innerText()).trimEnd().endsWith("·")).toBe(false);
    }
    await app.close();
  });
}
