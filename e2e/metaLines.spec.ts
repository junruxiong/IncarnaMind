import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  dotsOf,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
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
