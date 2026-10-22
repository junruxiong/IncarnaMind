import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { dismissChatSetup, launchApp, setLanguage, setWindowSize, turnOnEmbeddings } from "./app";

let dataDir = "";

test.beforeEach(async () => {
  dataDir = await mkdtemp(join(tmpdir(), "incarnamind-e2e-"));
});

test.afterEach(async () => {
  await rm(dataDir, { recursive: true, force: true });
});

const requests: { server: string; path: string }[] = [];

// A source that has none of the model's files, as each one is from a network that blocks it.
async function emptySource(name: string) {
  const server = createServer((request, response) => {
    requests.push({ server: name, path: request.url ?? "" });
    response.writeHead(404).end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { url: `http://127.0.0.1:${port}/`, close: () => server.close() };
}

const FILE = { path: "model.onnx", size: 10, sha256: "0".repeat(64) };

const WORDS = {
  en: {
    line: "Couldn’t download the search model from any source",
    modelScope: "ModelScope",
    choices: ["Automatic", "Hugging Face", "ModelScope"],
  },
  "zh-CN": {
    line: "未能从任何来源下载搜索模型",
    modelScope: "魔搭社区",
    choices: ["自动", "Hugging Face", "魔搭社区"],
  },
} as const;

// Adding a Document with embeddings on is what downloads the search model.
async function addNotes(window: Page) {
  const path = join(dataDir, "Plant notes.txt");
  await writeFile(path, "Ferns grow in shade.");
  await window.getByTestId("add-documents-input").setInputFiles([path]);
}

const modelSource = (window: Page) =>
  window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.getSettings()).device.modelSource;
  });

for (const language of ["en", "zh-CN"] as const) {
  // protects=when every source fails the search model's download, the sidebar line says so with one action, Try again, and its Details hold each source's error and "Download source: Automatic ▾"; picking ModelScope keeps it on this device and tries again at once, ModelScope's copy first, in English and Chinese (#1578, DESIGN.md); fails_when=the line names one source, the choice shows outside Details, picking doesn't save or retry, or the retry doesn't lead with the pick; no_claim=the real hosts, which the core's tests stand in for.
  test(`every source failing says so, and Details pick the source to try again from (${language})`, async () => {
    requests.length = 0;
    const pinned = await emptySource("pinned");
    const copy = await emptySource("modelscope");
    const { app, window } = await launchApp(dataDir, {
      modelDownloads: {
        source: { baseUrl: pinned.url, files: [FILE] },
        copies: { modelscope: `${copy.url}{folder}/` },
      },
    });
    await setWindowSize(app, window, 1280, 800);
    await dismissChatSetup(window);
    await setLanguage(window, language);
    await turnOnEmbeddings(window);
    await addNotes(window);

    const row = window.getByTestId("embedding-model-failed");
    await expect(row).toContainText(WORDS[language].line);
    await expect(row.getByTestId("embedding-model-retry")).toBeVisible();
    const sources = window.getByTestId("embedding-model-sources");
    await expect(sources).toHaveCount(0);

    const details = row.getByTestId("embedding-model-details");
    await details.focus();
    await window.keyboard.press("Enter");
    await expect(details).toHaveAttribute("aria-expanded", "true");
    await expect(sources).toContainText("Hugging Face");
    await expect(sources).toContainText(WORDS[language].modelScope);
    const choice = window.getByTestId("embedding-model-source");
    await expect(choice).toHaveValue("auto");
    await expect(choice.locator("option")).toHaveText([...WORDS[language].choices]);

    const before = requests.length;
    await choice.selectOption("modelscope");
    await expect.poll(() => modelSource(window)).toBe("modelscope");
    await expect.poll(() => requests.length).toBeGreaterThan(before);
    expect(requests[before]?.server).toBe("modelscope");
    await app.close();
    pinned.close();
    copy.close();
  });
}

// protects=with one source only, a failed download's line stays as it was, with no Details and no source choice (DESIGN.md: no setting shows unless there is one to make); fails_when=a choice of one source shows; no_claim=the line's words, which the other specs cover.
test("with one source only, nothing about sources shows", async () => {
  const pinned = await emptySource("pinned");
  const { app, window } = await launchApp(dataDir, {
    modelDownloads: { source: { baseUrl: pinned.url, files: [FILE] }, copies: {} },
  });
  await setWindowSize(app, window, 1280, 800);
  await dismissChatSetup(window);
  await turnOnEmbeddings(window);
  await addNotes(window);
  const row = window.getByTestId("embedding-model-failed");
  await expect(row).toContainText("Search model download failed");
  await expect(row.getByTestId("embedding-model-details")).toHaveCount(0);
  await app.close();
  pinned.close();
});
