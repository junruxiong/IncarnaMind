import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
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

/**
 * By geometry, for each part after the first of a line of parts joined by
 * dots: its dot shows only where the part before it is on the same line, and
 * then sits between the two. So no line starts or ends with a dot.
 */
async function dotsOf(line: Locator) {
  return line.evaluate((element) => {
    const row = element.getBoundingClientRect();
    const parts = [...element.querySelectorAll(":scope > .meta-part")];
    return parts.slice(1).map((part, index) => {
      const before = parts[index]?.getBoundingClientRect();
      const here = part.getBoundingClientRect();
      const dot = part.querySelector(".meta-dot")?.getBoundingClientRect();
      const glyphCentre = dot ? dot.left + dot.width / 2 : Number.NaN;
      return {
        sameLine: before !== undefined && Math.abs(before.top - here.top) < 2,
        shown: glyphCentre > row.left + 4,
        between:
          before !== undefined && glyphCentre >= before.right - 1 && glyphCentre <= here.left + 1,
      };
    });
  });
}

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
