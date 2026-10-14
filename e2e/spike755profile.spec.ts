import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  removeDataFolder,
} from "./app";

/* Spike for #755: a CPU profile of keystrokes in a long imported Mind, with no Page view at all. */
test("spike 755: keystroke profile", async () => {
  test.skip(process.env.SPIKE755 !== "1", "the #755 spike runs only when asked");
  test.setTimeout(10 * 60_000);
  const path = join(process.env.SPIKE755_FIXTURES ?? "", process.env.SPIKE755_FILES ?? "");
  const dataDir = await createDataFolder();
  const home = await createDataFolder();
  const { app, window } = await launchApp(dataDir, { home });
  await window.setViewportSize({ width: 1600, height: 1000 }).catch(() => undefined);
  await dismissChatSetup(window);
  const mode = process.env.SPIKE755_MODES ?? "off";
  if (mode !== "off") {
    const { readFile } = await import("node:fs/promises");
    const { readSpikeLook } = await import("../spikes/755-page-view/wordLayout");
    const look = await readSpikeLook(await readFile(path));
    await window.addInitScript(
      (config) => {
        (window as unknown as { __spike755: unknown }).__spike755 = config;
      },
      {
        mode,
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
  }
  await interceptOpenDialog(app, path);
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-import-word").click();
  await window.getByTestId("word-import-dialog").getByTestId("word-import-confirm").click();
  const editor = window.getByTestId("mind-editor");
  await expect(editor).toHaveAttribute("contenteditable", "true", { timeout: 300_000 });
  await window.waitForTimeout(3000);
  if (mode !== "off") {
    await window.evaluate(() =>
      (window as unknown as { __spike755Api: { apply(): Promise<number> } }).__spike755Api.apply(),
    );
    await window.waitForTimeout(3000);
  }
  await editor.locator("p").nth(5).click();
  await window.keyboard.press("End");
  const frames: number[] = [];
  for (let index = 0; index < 4; index++) {
    const ms = await window.evaluate(
      () =>
        new Promise<number>((resolve) => {
          const start = performance.now();
          document.activeElement?.dispatchEvent(new KeyboardEvent("keydown", { key: "Shift" }));
          requestAnimationFrame(() =>
            requestAnimationFrame(() => resolve(performance.now() - start)),
          );
        }),
    );
    void ms;
    const before = await window.evaluate(() => performance.now());
    await window.keyboard.type("x");
    const after = await window.evaluate(
      () =>
        new Promise<number>((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now()))),
        ),
    );
    frames.push(after - before);
  }
  const cdp = await window.context().newCDPSession(window);
  await cdp.send("Profiler.enable");
  await cdp.send("Profiler.setSamplingInterval", { interval: 100 });
  await cdp.send("Profiler.start");
  for (let index = 0; index < 6; index++) {
    await window.keyboard.type("y");
    await window.waitForTimeout(400);
  }
  const { profile } = await cdp.send("Profiler.stop");
  await writeFile(
    join(process.env.SPIKE755_OUT ?? "", `keystroke-${mode}.cpuprofile`),
    JSON.stringify(profile),
  );
  await writeFile(
    join(process.env.SPIKE755_OUT ?? "", `keystroke-frames-${mode}.json`),
    JSON.stringify(frames),
  );
  await app.close();
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
});
