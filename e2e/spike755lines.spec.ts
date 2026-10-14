import { readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { expect, test } from "@playwright/test";
import { readSpikeLook } from "../spikes/755-page-view/wordLayout";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  removeDataFolder,
} from "./app";

/* Spike for #755: each paragraph's line count, kerning on and off, to compare with LibreOffice's. */
test("spike 755: line counts", async () => {
  test.skip(process.env.SPIKE755 !== "1", "the #755 spike runs only when asked");
  test.setTimeout(10 * 60_000);
  const path = join(process.env.SPIKE755_FIXTURES ?? "", process.env.SPIKE755_FILES ?? "");
  const dataDir = await createDataFolder();
  const home = await createDataFolder();
  const look = await readSpikeLook(await readFile(path));
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await interceptOpenDialog(app, path);
  await window.addInitScript(
    (config) => {
      (window as unknown as { __spike755: unknown }).__spike755 = config;
    },
    {
      mode: "look",
      look,
      fontMap: { Calibri: "Carlito" },
      syncPages: 2,
      stats: {
        runs: [],
        keydownAt: 0,
        updateAt: 0,
        settledAt: 0,
        backgroundDoneAt: 0,
        composingSkips: 0,
      },
    },
  );
  await window.reload();
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-import-word").click();
  await window.getByTestId("word-import-dialog").getByTestId("word-import-confirm").click();
  await expect(window.getByTestId("mind-editor")).toHaveAttribute("contenteditable", "true", {
    timeout: 300_000,
  });
  type Api = { apply(): Promise<number>; lineCounts(): unknown; kerning(on: boolean): void };
  const api = () => (window as unknown as { __spike755Api: Api }).__spike755Api;
  await expect
    .poll(() =>
      window.evaluate(() =>
        Boolean((window as unknown as { __spike755Api?: unknown }).__spike755Api),
      ),
    )
    .toBe(true);
  await window.evaluate(() => (window as unknown as { __spike755Api: Api }).__spike755Api.apply());
  const kerned = await window.evaluate(() =>
    (window as unknown as { __spike755Api: Api }).__spike755Api.lineCounts(),
  );
  await window.evaluate(() =>
    (window as unknown as { __spike755Api: Api }).__spike755Api.kerning(false),
  );
  const plain = await window.evaluate(() =>
    (window as unknown as { __spike755Api: Api }).__spike755Api.lineCounts(),
  );
  void api;
  await writeFile(
    join(process.env.SPIKE755_OUT ?? "", `${basename(path, ".docx")}-lines.json`),
    JSON.stringify({ kerned, plain }),
  );
  await app.close();
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
});
