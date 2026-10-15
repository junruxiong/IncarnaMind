import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { createFakeOpenAI } from "../tests/helpers/fakeOpenAI";
import {
  acceptPlainTextKeys,
  assistantAnswers,
  createDataFolder,
  interceptOpenExternal,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  urlsOpened,
} from "./app";

/*
 * The key form (#719): one row of providers, the key, a Model pick list and one "Use this key"
 * that tests the key as it saves, all in view at 900px high; and the ChatGPT sign-in's way back.
 */

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

/** The first Question with no model: the setup card waits, with the key form opened. */
async function openKeyForm(window: Page) {
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is in my Documents?");
  await window.keyboard.press("Enter");
  const card = window.getByTestId("model-setup");
  await card.waitFor();
  return card;
}

test("at 900px high the key form shows provider, key, Model and its button without scrolling, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    // Linux in CI has no keyring: its notice is accepted first, as a User does once.
    await acceptPlainTextKeys(window);
    for (const value of ["en", "zh-CN"] as const) {
      await language(window, value);
      const card = await openKeyForm(window);
      await card.getByTestId("setup-key").click();
      const form = card.getByTestId("provider-form");
      await expect(form.getByRole("radio", { checked: true })).toHaveCount(1);
      await expect(form.getByTestId("provider-model")).toBeVisible();
      await expect(form.getByTestId("provider-model-reason")).toBeVisible();
      await expect(form.getByTestId("provider-key-link")).toBeVisible();
      await expect(form.getByTestId("provider-use-key")).toBeInViewport({ ratio: 1 });
      await expect(form.getByTestId("provider-model")).toBeInViewport({ ratio: 1 });
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await window.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-form-900-${value}.png`,
        });
      }
      await card.getByTestId("setup-back").click();
    }
  } finally {
    await app.close();
  }
});

test("a way that can't be used yet says why, with the next step: Ollama not running, and Use this key before a key is pasted, in English and Chinese", async () => {
  // Nothing answers where Ollama would be.
  const { app, window } = await launchApp(dataDir, { ollama: "http://127.0.0.1:9" });
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const card = await openKeyForm(window);
    await card.getByTestId("setup-other").click();
    const local = card.getByTestId("way-local");
    await expect(local).toContainText("Ollama isn't running on this computer. It's free to get.");
    await expect(local.locator("xpath=..").getByRole("link", { name: /Get Ollama/ })).toBeVisible();
    await language(window, "zh-CN");
    await expect(local).toContainText("这台电脑上没有运行 Ollama。可以免费获取。");
    await language(window, "en");
    await card.getByTestId("setup-key").click();
    const form = card.getByTestId("provider-form");
    await expect(form.getByTestId("provider-use-key")).toBeDisabled();
    await expect(form.getByTestId("provider-use-key-why")).toHaveText("Paste the key first.");
    await form.getByLabel(/^API key/).fill("sk-something");
    await expect(form.getByTestId("provider-use-key-why")).toHaveCount(0);
    await expect(form.getByTestId("provider-use-key")).toBeEnabled();
    await form.getByLabel(/^API key/).fill("");

    await language(window, "zh-CN");
    await expect(form.getByTestId("provider-use-key-why")).toHaveText("请先粘贴密钥。");
    if (process.env.INCARNAMIND_SCREENSHOTS) {
      await window.screenshot({
        path: `${process.env.INCARNAMIND_SCREENSHOTS}/why-unavailable-zh-CN.png`,
      });
    }
  } finally {
    await app.close();
  }
});

/** An OpenAI-compatible server that refuses one key and answers with the other. */
async function server(): Promise<{ server: Server; url: string }> {
  const http = createServer((request, response) => {
    request.resume();
    request.on("end", () => {
      const good = request.headers.authorization === "Bearer good-key";
      response.writeHead(good ? 200 : 401, { "content-type": "application/json" });
      response.end(
        JSON.stringify(
          good
            ? {
                id: "c1",
                object: "chat.completion",
                created: 1,
                model: "any",
                choices: [
                  {
                    index: 0,
                    message: { role: "assistant", content: "OK" },
                    finish_reason: "stop",
                  },
                ],
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
              }
            : { error: { message: "bad key" } },
        ),
      );
    });
  });
  await new Promise<void>((resolve) => http.listen(0, "127.0.0.1", resolve));
  return { server: http, url: `http://127.0.0.1:${(http.address() as AddressInfo).port}/v1` };
}

test("Use this key shows a refused key in place, and with a good one saves and the waiting Question is asked", async () => {
  const fake = await server();
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const card = await openKeyForm(window);
    await card.getByTestId("setup-key").click();
    const form = card.getByTestId("provider-form");
    await form.getByTestId("provider-more-toggle").click();
    await form.locator("label").filter({ hasText: "OpenAI-compatible server" }).click();
    // A compatible server keeps a free-text Model.
    await expect(form.getByTestId("provider-model-text")).toBeVisible();
    await form.getByLabel("Server URL").fill(fake.url);
    await form.getByTestId("provider-model-text").fill("any");
    await form.getByLabel(/^API key/).fill("wrong-key");
    await form.getByTestId("provider-use-key").click();
    await expect(form.getByTestId("connection-test")).toContainText("The key was refused", {
      timeout: 20_000,
    });
    await expect(card).toBeVisible();

    await form.getByLabel(/^API key/).fill("good-key");
    await form.getByTestId("provider-use-key").click();
    await expect(card).toBeHidden();
    // The waiting Question was asked: the conversation holds its Answer (the server only answers
    // the connection test, so how it ends is not the point).
    await expect(assistantAnswers(window).first()).toBeVisible({ timeout: 15_000 });
  } finally {
    await app.close();
    fake.server.close();
  }
});

test("the ChatGPT wait offers the sign-in page again, and after five minutes says so in place; Cancel goes back", async () => {
  const openai = await createFakeOpenAI();
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    chatGpt: openai.endpoints.issuer,
  });
  try {
    await setWindowSize(app, window, 1100, 900);
    await interceptOpenExternal(app);
    await acceptPlainTextKeys(window);
    await window.clock.install();
    const card = await openKeyForm(window);
    await card.getByTestId("setup-chatgpt").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(1);
    await expect(card).toContainText("Finish signing in in your browser");
    await expect(card).toContainText("nothing extra to pay");
    await card.getByTestId("chatgpt-reopen").click();
    await expect.poll(() => urlsOpened(app)).toHaveLength(2);
    await expect(card.getByTestId("chatgpt-late")).toHaveCount(0);

    await window.clock.fastForward("05:01");
    await expect(card.getByTestId("chatgpt-late")).toHaveText(
      "Didn’t finish? Open the sign-in page again.",
    );
    await expect(window.getByTestId("chatgpt-error")).toHaveCount(0);

    await card.getByTestId("chatgpt-cancel").click();
    await expect(card.getByTestId("setup-chatgpt")).toBeVisible();
  } finally {
    await app.close();
  }
});
