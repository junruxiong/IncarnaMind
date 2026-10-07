import { resolve } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

/** The tiny MCP server the core tests use: `lookup_tide` is read-only, `book_boat` isn't. */
const TIDE_SERVER = resolve(__dirname, "../tests/fixtures/mcp-server.mjs");

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** Adds the tiny server as a Connector through the core's bridge, and waits until it's ready. */
async function addTideConnector(window: Page): Promise<void> {
  await window.evaluate(
    async ({ node, server }) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const connector = await bridge.addConnector({ name: "Tides", command: node, args: [server] });
      for (let tries = 0; tries < 300; tries++) {
        const found = (await bridge.listConnectors()).find((each) => each.id === connector.id);
        if (found?.state === "ready") return;
        await new Promise((done) => setTimeout(done, 100));
      }
      throw new Error("The Connector didn't get ready.");
    },
    { node: process.execPath, server: TIDE_SERVER },
  );
}

/** Writes a Question on a new line of the Mind and asks it. */
async function ask(window: Page, text: string): Promise<void> {
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type(text);
  await window.keyboard.press("Enter");
}

test("a Tool that may change something pauses the Answer with an approval card: Allow once lets it finish, and Deny still gets a finished Answer", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addTideConnector(window);

  await window.getByTestId("new-mind").click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await ask(window, "Please book_boat from Dover");

  // The fake model calls book_boat: the Answer waits, showing the Tool, its Connector and the arguments.
  const answer = editor.getByTestId("answer").first();
  const card = answer.getByTestId("approval-card");
  await expect(card).toBeVisible({ timeout: 15_000 });
  await expect(card).toHaveAttribute("data-tool", "book_boat");
  await expect(card).toContainText("Tides wants to run book_boat");
  await expect(card).toContainText("This Tool may change something in Tides.");
  const argument = card.getByTestId("approval-argument");
  await expect(argument).toHaveCount(1);
  await expect(argument).toContainText("place");
  await expect(argument).toContainText("Dover");
  await expect(answer).toHaveAttribute("data-status", "streaming");
  await expect(answer.getByTestId("answer-writing")).toHaveText("Waiting for your approval");

  // Allow once: the call goes ahead (after the Connector's data-flow consent), and the Answer finishes.
  await card.getByTestId("approval-allow-once").click();
  await expect(card).toHaveCount(0);
  const consent = window.getByTestId("consent-dialog");
  await expect(consent).toContainText("Send data to Tides?");
  await consent.getByTestId("consent-allow").click();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer).toContainText("book_boat said: Booked a boat trip from Dover.");
  await expect(answer).toContainText("That is all.");
  const call = answer.getByTestId("answer-connector-call");
  await expect(call).toHaveAttribute("data-approval", "allowed");
  await expect(call).toContainText("Asked Tides with your approval: book_boat");
  // The card shows the arguments that were sent.
  await expect(call).toContainText('{"place":"Dover"}');

  // A second Question, on the empty line after the Answer: Deny. The call isn't made, and the
  // Answer still finishes.
  await editor.locator(":scope > p").last().click();
  await ask(window, "Now book_boat from Calais");
  const second = editor.getByTestId("answer").nth(1);
  const secondCard = second.getByTestId("approval-card");
  await expect(secondCard).toBeVisible({ timeout: 15_000 });
  await expect(secondCard.getByTestId("approval-argument")).toContainText("Calais");
  await secondCard.getByTestId("approval-deny").click();
  await expect(second).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(second).toContainText("book_boat said: The User denied this call of book_boat");
  await expect(second).toContainText("That is all.");
  const denied = second.getByTestId("answer-connector-call");
  await expect(denied).toHaveAttribute("data-approval", "denied");
  await expect(denied).toContainText("Not allowed: Tides: book_boat");
  await app.close();
});
