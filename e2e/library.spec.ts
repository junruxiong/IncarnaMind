import { writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
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

test("choose starters, create and edit a custom group, move Documents and delete the group", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    const path = join(sources, "Quarterly report.txt");
    await writeFile(path, "Quarterly financial report. Revenue increased this year.");
    await addDocuments(window, [path]);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("checkbox", { name: /^Meeting notes/ }).uncheck();
    await library.getByRole("button", { name: "Add selected groups" }).click();
    await expect(
      library.getByRole("navigation", { name: "Groups" }).getByRole("button"),
    ).toHaveCount(8);
    await library.getByRole("button", { name: "New group", exact: true }).click();
    await library.getByLabel("Group name", { exact: true }).fill("Client work");
    await library.getByLabel("Description", { exact: true }).fill("Reports for my clients");
    await library.getByRole("button", { name: "Save group" }).click();
    const assignment = library.getByRole("combobox", { name: "Group for Quarterly report" });
    await assignment.selectOption({ label: "Client work" });
    await expect(library.getByTestId("library-document")).toContainText("Your choice");
    await library
      .getByRole("navigation")
      .getByRole("button", { name: /^Client work/ })
      .click();
    await library.getByRole("button", { name: "Edit group" }).click();
    await library.getByLabel("Group name", { exact: true }).fill("Client reports");
    await library.getByRole("button", { name: "Save group" }).click();
    await expect(
      library.getByRole("heading", { name: "Client reports", exact: true }),
    ).toBeVisible();
    await library.getByRole("button", { name: "Delete group", exact: true }).click();
    await library.getByRole("button", { name: "Delete group", exact: true }).last().click();
    await expect(assignment).toHaveValue("");
    await expect(library.getByTestId("library-document")).toContainText("Your choice");
    await window.setViewportSize({ width: 1000, height: 760 });
    await window.screenshot({ path: "/tmp/incarnamind-library-manual.png" });
    await library.getByRole("button", { name: "Quarterly report", exact: true }).click();
    await expect(window.getByTestId("viewer")).toBeVisible();
    await expect
      .poll(async () => library.evaluate((element) => element.scrollWidth <= element.clientWidth))
      .toBe(true);
    await window.screenshot({ path: "/tmp/incarnamind-library-viewer.png" });
  } finally {
    await app.close();
  }
});

test("classifies with a separate connected model, preserves corrections and survives reopening", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await connectLocalChatModel(window, "main-chat");
    const path = join(sources, "Research notes.txt");
    await writeFile(path, "Research about membrane separation and water filtration.");
    await addDocuments(window, [path]);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("button", { name: "New group", exact: true }).click();
    await library.getByLabel("Group name", { exact: true }).fill("Research");
    await library.getByLabel("Description", { exact: true }).fill("Research papers and notes");
    await library.getByRole("button", { name: "Save group" }).click();
    await library.locator("summary").click();
    const providerId = await window.evaluate(
      async () =>
        (
          await (
            globalThis as unknown as { incarnamind: CoreBridge }
          ).incarnamind.listChatProviders()
        )[0]?.id,
    );
    if (!providerId) throw new Error("No connected provider");
    await library.getByLabel("Connection", { exact: true }).selectOption(providerId);
    await library.getByLabel("Model name", { exact: true }).fill("small-classifier");
    await library.getByRole("checkbox", { name: /Automatically classify/ }).check();
    await library.getByRole("button", { name: "Save classification settings" }).click();
    await library.getByRole("button", { name: "Classify unsorted" }).click();
    await expect(library.getByTestId("library-document")).toContainText("Classification saved");
    await window.screenshot({ path: "/tmp/incarnamind-library-model-settings.png" });
    const assignment = library.getByRole("combobox", { name: "Group for Research notes" });
    await expect(assignment.locator("option:checked")).toHaveText("Research");
    await assignment.selectOption("");
    await expect(library.getByTestId("library-document")).toContainText("Your choice");
    await library.getByRole("button", { name: "Back to Mind" }).click();
    await window.getByTestId("open-library").click();
    await expect(assignment).toHaveValue("");
    await window.screenshot({ path: "/tmp/incarnamind-library-classifier.png" });
  } finally {
    await app.close();
  }
  const reopened = await launchApp(dataDir, { fakeChat: true });
  try {
    await reopened.window.getByTestId("open-library").click();
    await expect(reopened.window.getByTestId("library-document")).toContainText("Your choice");
    await expect(
      reopened.window.getByRole("combobox", { name: "Group for Research notes" }),
    ).toHaveValue("");
  } finally {
    await reopened.app.close();
  }
});

test("onboarding offers starter groups and the Chinese interface is complete", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true });
  try {
    await window.getByTestId("onboarding-groups").click();
    await expect(window.getByTestId("library")).toBeVisible();
    await expect(window.getByTestId("chat-setup")).not.toBeVisible();
    await window.evaluate(async () =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await window.getByTestId("library").getByRole("button", { name: "添加选中的分组" }).click();
    await expect(
      window
        .getByTestId("library")
        .getByRole("navigation")
        .getByRole("button", { name: /^研究论文/ }),
    ).toBeVisible();
    await window.screenshot({ path: "/tmp/incarnamind-library-chinese.png" });
  } finally {
    await app.close();
  }
});

test("local Clef sends PDF page images and the text-only choice persists", async () => {
  const requests: { images?: string[] }[] = [];
  const server = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    requests.push(JSON.parse(body));
    response.writeHead(200, { "content-type": "application/json" });
    response.end(
      JSON.stringify({ answers: { group: { type: "choice", choice: "__unsorted__" } } }),
    );
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    const path = join(sources, "Illustrated research.pdf");
    await writeFile(path, buildPdf([{ lines: ["Research findings"], image: true }]));
    await addDocuments(window, [path]);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("button", { name: "Add selected groups" }).click();
    await library.locator("summary").click();
    await library.getByLabel("Connection", { exact: true }).selectOption("ollama");
    await library.getByLabel("Model name", { exact: true }).fill("clef-flash");
    await library
      .getByLabel("Local Ollama address", { exact: true })
      .fill(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
    const images = library.getByRole("checkbox", { name: /Include up to 3 PDF page images/ });
    await expect(images).not.toBeChecked();
    await images.check();
    await library.getByRole("button", { name: "Save classification settings" }).click();
    await library.getByRole("button", { name: "Classify unsorted" }).click();
    await expect(library.getByTestId("library-document")).toContainText("Classification saved");
    expect(requests[0]?.images).toHaveLength(1);
    await window.screenshot({ path: "/tmp/incarnamind-library-clef-images.png" });
    await images.uncheck();
    await library.getByRole("button", { name: "Save classification settings" }).click();
    await library.getByRole("button", { name: "Back to Mind" }).click();
    await window.getByTestId("open-library").click();
    await library.locator("summary").click();
    await expect(images).not.toBeChecked();
    await library.getByRole("button", { name: "Classify unsorted" }).click();
    await expect.poll(() => requests.length).toBe(2);
    expect(requests[1]?.images).toBeUndefined();
  } finally {
    await app.close();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("Auto routes text and scans, shows a smaller-model fallback, and preserves its setting", async () => {
  const requests: { model: string; images?: string[] }[] = [];
  let models = ["tev1:0.8b", "clef-flash:latest"];
  const loaded = new Set<string>();
  const server = createServer(async (request, response) => {
    let raw = "";
    for await (const chunk of request) raw += chunk;
    const body = raw ? JSON.parse(raw) : {};
    response.writeHead(200, { "content-type": "application/json" });
    if (request.url === "/api/tags")
      response.end(JSON.stringify({ models: models.map((name) => ({ name })) }));
    else if (request.url === "/api/ps")
      response.end(JSON.stringify({ models: [...loaded].map((name) => ({ name })) }));
    else if (request.url === "/api/generate") {
      loaded.delete(body.model);
      response.end(JSON.stringify({ done: true }));
    } else {
      requests.push(body);
      loaded.add(body.model);
      response.end(
        JSON.stringify({ answers: { group: { type: "choice", choice: "__unsorted__" } } }),
      );
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    const scanPath = join(sources, "Scanned diagram.pdf");
    const textPath = join(sources, "Research notes.txt");
    await writeFile(scanPath, buildPdf([{ image: true }]));
    await writeFile(textPath, "Research about membrane separation and water filtration.");
    await window.evaluate(
      async (paths) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addDocuments(paths),
      [scanPath, textPath],
    );
    await expect
      .poll(() =>
        window.evaluate(
          async () =>
            (
              await (
                globalThis as unknown as { incarnamind: CoreBridge }
              ).incarnamind.listDocuments()
            ).filter((doc) => doc.status === "ready" || doc.status === "no-text").length,
        ),
      )
      .toBe(2);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("button", { name: "Add selected groups" }).click();
    await library.locator("summary").click();
    await library.getByLabel("Connection", { exact: true }).selectOption("auto");
    await expect(library.getByLabel("Model name", { exact: true })).toHaveCount(0);
    await library
      .getByLabel("Local Ollama address", { exact: true })
      .fill(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
    await library.getByRole("button", { name: "Save classification settings" }).click();
    await library.getByRole("button", { name: "Classify unsorted" }).click();
    const notes = library
      .getByTestId("library-document")
      .filter({ has: window.getByRole("button", { name: "Research notes", exact: true }) });
    const scan = library
      .getByTestId("library-document")
      .filter({ has: window.getByRole("button", { name: "Scanned diagram", exact: true }) });
    await expect(notes).toContainText("tev1:0.8b · Text · 4B model not installed");
    await expect(scan).toContainText("clef-flash · Text + PDF pages");
    expect(requests.map((item) => item.model)).toEqual(["tev1:0.8b", "clef-flash"]);
    expect(requests[0]?.images).toBeUndefined();
    expect(requests[1]?.images).toHaveLength(1);
    await window.screenshot({ path: "/tmp/incarnamind-library-auto.png" });
    await library.getByRole("button", { name: "Back to Mind" }).click();
    await window.getByTestId("open-library").click();
    await library.locator("summary").click();
    await expect(library.getByLabel("Connection", { exact: true })).toHaveValue("auto");
    // Installing the preferred model takes effect on the next classification.
    models = [...models, "tev1:4b"];
    await library.getByRole("button", { name: "Classify unsorted" }).click();
    await expect(notes).toContainText("tev1:4b · Text");
    await expect(notes).not.toContainText("4B model not installed");
    await expect.poll(() => requests.length).toBe(4);
    await window.evaluate(async () =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await expect(library.getByLabel("连接", { exact: true }).locator("option:checked")).toHaveText(
      "自动选择 · 本地模型",
    );
    await window.setViewportSize({ width: 1000, height: 760 });
    await expect
      .poll(() => library.evaluate((element) => element.scrollWidth <= element.clientWidth))
      .toBe(true);
    await window.screenshot({ path: "/tmp/incarnamind-library-auto-zh.png" });
  } finally {
    await app.close();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
