import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Deleting most of a Library at once (#1349): ⌘A and ⌘⌫ move every Document
 * to Recently deleted in one step, the line says so with Undo, and ⌘Z brings
 * them all back. One by one, 1,700 Documents took 90 seconds, the Library
 * draining with the boxes still ticked and the window busy behind it.
 */
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

const COUNT = 400;
const WORDS = {
  en: { deleted: `Moved ${COUNT} Documents to Recently deleted`, selected: `${COUNT} selected` },
  "zh-CN": { deleted: `已将 ${COUNT} 个文档移到回收站`, selected: `已选 ${COUNT} 个` },
} as const;

for (const [language, width] of [
  ["en", 1440],
  ["zh-CN", 900],
] as const) {
  test(`${COUNT} Documents selected with ⌘A go to Recently deleted in one step, and ⌘Z brings them all back (${language}, ${width}px)`, async () => {
    test.setTimeout(240_000);
    for (let at = 0; at < COUNT; at++) {
      await writeFile(
        join(sources, `Note ${at}.md`),
        `# Note ${at}\n\nHarbour tides, note ${at}.\n`,
      );
    }
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, width, 900);
      await dismissChatSetup(window);
      await setLanguage(window, language);
      await window.evaluate(async (path) => {
        const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
        await core.addLinkedFolder(path);
      }, sources);
      await expect
        .poll(
          () =>
            window.evaluate(
              async () =>
                (
                  await (
                    globalThis as unknown as { incarnamind: CoreBridge }
                  ).incarnamind.listDocuments()
                ).filter((each) => each.status === "ready").length,
            ),
          { timeout: 180_000 },
        )
        .toBe(COUNT);

      await window.getByTestId("open-library").click();
      const library = window.getByTestId("library");
      const count = library.getByTestId("library-count");
      await expect(count).toHaveText(String(COUNT));
      await library.getByTestId("library-document").first().locator("[data-row-main]").focus();
      await window.keyboard.press("ControlOrMeta+a");
      await expect(window.getByTestId("library-selection").getByRole("status")).toHaveText(
        WORDS[language].selected,
      );

      await window.keyboard.press("ControlOrMeta+Backspace");
      await expect(window.getByTestId("undo-line")).toContainText(WORDS[language].deleted);
      await expect(library.getByTestId("library-document")).toHaveCount(0);

      await window.keyboard.press("ControlOrMeta+z");
      await expect(count).toHaveText(String(COUNT));
    } finally {
      await app.close();
    }
  });
}
