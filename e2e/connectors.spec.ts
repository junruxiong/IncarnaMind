import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { createDataFolder, dismissChatSetup, launchApp, removeDataFolder } from "./app";

/** The tiny MCP server the core tests use: `lookup_tide` is read-only, `book_boat` isn't. */
const TIDE_SERVER = resolve(__dirname, "../tests/fixtures/mcp-server.mjs");

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a local Connector added in Settings starts and reaches ready, and turning it off stops it", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);

  await window.getByRole("button", { name: "Settings" }).click();
  const section = window.getByTestId("connectors-settings");
  await expect(section).toContainText("No Connectors yet.");
  await section.getByTestId("connector-add").click();
  const form = section.getByTestId("connector-form");
  await form.getByLabel("Name", { exact: true }).fill("Tides");
  // Node, the way the test runs it; the app starts it with the login-shell environment.
  await form.getByLabel(/^Command/).fill(process.execPath);
  await form.getByLabel("Arguments, one per line").fill(TIDE_SERVER);
  await form.getByLabel(/^Environment variables/).fill("TIDE_TOKEN=smoke-secret");
  await form.getByRole("button", { name: "Add Connector" }).click();

  const connector = section.getByTestId("connector");
  await expect(connector).toHaveCount(1);
  await expect(connector).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  await expect(connector.getByTestId("connector-state")).toHaveText("Ready");
  await expect(connector).toContainText(TIDE_SERVER);

  // Its Tools, and which of them Answers use.
  await connector.getByText("2 Tools").click();
  const tools = connector.getByTestId("connector-tools");
  await expect(tools.locator('[data-read-only="true"]')).toHaveText("lookup_tide · read-only");
  await expect(tools.locator('[data-read-only="false"]')).toContainText("book_boat");

  // Off: its process stops, and it says so.
  await connector.getByRole("switch", { name: "Use Tides" }).uncheck();
  await expect(connector).toHaveAttribute("data-state", "off");
  await expect(connector.getByTestId("connector-state")).toHaveText("Off");
  await app.close();
});
