import { expect, type Page, test } from "@playwright/test";
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

const bridge = (window: Page) => ({
  /** A chat model on a server elsewhere, then local mode on. Saving sends nothing. */
  async cloudModelInLocalMode() {
    await window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await core.saveChatProvider({
        kind: "openai-compatible",
        baseUrl: "https://llm.example.com/v1",
        modelId: "example-model",
      });
      await core.setLocalOnly(true);
    });
  },
  localOnly: () =>
    window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      return (await core.getEmbeddingSettings()).localOnly;
    }),
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
});

test("in local mode a Question to a cloud model isn't sent: one line says so with a local model or turning local mode off, which can be undone; English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await setWindowSize(app, window, 900, 700);
  await bridge(window).cloudModelInLocalMode();
  await newMind(window);
  const editor = window.getByTestId("mind-editor");

  // At the top of the Mind, before anything is asked.
  const top = window.getByTestId("chat-readiness");
  await expect(top).toContainText(
    "“Keep everything on this computer” is on, so Questions aren't sent to llm.example.com",
  );

  // The model menu keeps the cloud model in sight, marked unavailable with the reason.
  await window.getByTestId("composer-model").click();
  const option = window.getByTestId("composer-model-option");
  await expect(option).toHaveAttribute("aria-disabled", "true");
  await expect(option).toContainText("Unavailable: “Keep everything on this computer” is on");
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("composer-model-menu")).toBeHidden();

  // Asking: refused where it was asked. Nothing was sent, so nothing asked consent, and the
  // scripted model, which would answer anything it was given, wrote nothing.
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is IncarnaMind?");
  await window.keyboard.press("Enter");
  const refusal = window.getByTestId("composer").getByTestId("composer-not-ready");
  await expect(refusal).toHaveText(
    "“Keep everything on this computer” is on, so this Question wasn't sent to llm.example.com · Use a local model · Turn off local mode",
  );
  await expect(refusal.getByRole("button")).toHaveText([
    "Use a local model",
    "Turn off local mode",
  ]);
  await expect(window.getByTestId("composer-input")).toHaveValue("What is IncarnaMind?");
  await expect(editor.getByTestId("question")).toHaveCount(0);
  await expect(editor.getByTestId("answer")).toHaveCount(0);
  await expect(window.getByTestId("consent-dialog")).toBeHidden();
  // The line wraps inside the composer at a narrow window.
  expect(
    await window
      .getByTestId("composer")
      .evaluate((element) => element.scrollWidth - element.clientWidth),
  ).toBeLessThanOrEqual(0);

  // In Chinese.
  await bridge(window).language("zh-CN");
  await expect(refusal).toHaveText(
    "已开启“所有内容都留在这台电脑上”，这个问题没有发送给 llm.example.com · 改用本地模型 · 关闭本地模式",
  );
  await expect(top).toContainText("因此不会向 llm.example.com 发送问题");
  await window.getByTestId("composer-model").click();
  await expect(option).toContainText("不可用：已开启“所有内容都留在这台电脑上”");
  await window.keyboard.press("Escape");
  await bridge(window).language("en");

  // Use a local model: opens the models in Settings, where a local one is set up.
  await refusal.getByTestId("local-mode-use-local").click();
  await expect(window.getByTestId("settings")).toHaveAttribute("data-page", "models");
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("settings")).toBeHidden();

  // Turn off local mode: the line goes, and Undo turns it on again.
  await refusal.getByTestId("local-mode-turn-off").click();
  await expect(refusal).toBeHidden();
  await expect(top).toBeHidden();
  expect(await bridge(window).localOnly()).toBe(false);
  const undo = window.getByTestId("undo-line");
  await expect(undo.getByTestId("undo-message")).toHaveText(
    "Turned off “Keep everything on this computer”",
  );
  await undo.getByTestId("undo-action").click();
  await expect.poll(() => bridge(window).localOnly()).toBe(true);
  await expect(top).toBeVisible();
  await app.close();
});
