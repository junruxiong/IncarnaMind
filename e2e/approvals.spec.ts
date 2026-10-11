import { resolve } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import * as Y from "yjs";
import { type CoreBridge, MIND_CONTENT_FIELD, type ThirdPartySource } from "../src/core/api";
import { setTextOrigin } from "../src/shared/textOrigins";
import {
  clickEmptyLine,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
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

/** Adds the tiny server as a Connector through the core's bridge, waits until it's ready, and returns its id. */
async function addTideConnector(window: Page): Promise<string> {
  return window.evaluate(
    async ({ node, server }) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const connector = await bridge.addConnector({ name: "Tides", command: node, args: [server] });
      for (let tries = 0; tries < 300; tries++) {
        const found = (await bridge.listConnectors()).find((each) => each.id === connector.id);
        if (found?.state === "ready") return connector.id;
        await new Promise((done) => setTimeout(done, 100));
      }
      throw new Error("The Connector didn't get ready.");
    },
    { node: process.execPath, server: TIDE_SERVER },
  );
}

/**
 * A Mind holding one Note, and an empty line to ask from; with `origin`, the
 * Note is recorded as third-party text, as importing someone else's .docx
 * records it (#104). Made through the core's bridge, then closed, as a Mind
 * from before; returns its id.
 */
async function mindWithNote(
  window: Page,
  title: string,
  origin: ThirdPartySource | null,
): Promise<string> {
  const doc = new Y.Doc();
  const paragraph = (text: string) => {
    const element = new Y.XmlElement("paragraph");
    element.setAttribute("id", crypto.randomUUID());
    if (text) element.insert(0, [new Y.XmlText(text)]);
    return element;
  };
  const note = paragraph("The landlord's draft: book the survey boat from Dover on Friday.");
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([note, paragraph("")]);
  if (origin) setTextOrigin(doc, [String(note.getAttribute("id"))], origin);
  return window.evaluate(
    async ({ title, update }) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const mind = await core.createMind({ title });
      await core.applyMindUpdate(mind.id, new Uint8Array(update));
      await core.closeMind(mind.id);
      return mind.id;
    },
    { title, update: Array.from(Y.encodeStateAsUpdate(doc)) },
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

  await newMind(window);
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
  await clickEmptyLine(editor.locator(":scope > p").last());
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

test("in a Mind holding imported text, a write the User always allows asks, says why and offers no Always allow; the same Mind without it doesn't ask", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 860);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  const connectorId = await addTideConnector(window);
  await window.evaluate(async (connectorId) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.setApprovalPolicy({
      subject: { kind: "tool", connectorId, tool: "book_boat" },
      policy: "always",
    });
  }, connectorId);
  await mindWithNote(window, "Imported lease", "imported");
  await mindWithNote(window, "My own lease notes", null);
  const editor = window.getByTestId("mind-editor");
  const open = async (title: string) => {
    await pointAndClick(window, window.getByTestId("mind-list-item").filter({ hasText: title }));
    await expect(editor).toContainText("The landlord's draft");
    await clickEmptyLine(editor.locator(":scope > p").last());
  };

  // Someone else's text is in the Question context: the always-allowed write asks, and says why.
  await open("Imported lease");
  await ask(window, "Please book_boat from Dover");
  const answer = editor.getByTestId("answer").first();
  const card = answer.getByTestId("approval-card");
  await expect(card).toBeVisible({ timeout: 15_000 });
  await expect(card).toHaveAttribute("data-tool", "book_boat");
  const why = card.getByTestId("approval-third-party");
  await expect(why).toHaveText("This Mind contains imported text, so it asks you first.");
  await expect(why).toHaveAttribute("data-source", "imported");
  // "Always allow" wouldn't cover this call, so it isn't offered: Allow once, then Deny.
  await expect(card.getByTestId("approval-always-allow")).toHaveCount(0);
  const allow = card.getByTestId("approval-allow-once");
  const deny = card.getByTestId("approval-deny");
  await expect(allow).toBeVisible();
  await expect(deny).toBeVisible();
  const allowBox = await allow.boundingBox();
  const denyBox = await deny.boundingBox();
  expect(allowBox && denyBox && allowBox.x < denyBox.x).toBe(true);
  await pointAndClick(window, allow);
  const consent = window.getByTestId("consent-dialog");
  await expect(consent).toContainText("Send data to Tides?");
  await pointAndClick(window, consent.getByTestId("consent-allow"));
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer).toContainText("book_boat said: Booked a boat trip from Dover.");
  await expect(answer.getByTestId("answer-connector-call")).toHaveAttribute(
    "data-approval",
    "allowed",
  );

  // The same Note, the User's own: "always" holds, and the call runs without a card.
  await open("My own lease notes");
  await ask(window, "Please book_boat from Dover");
  const own = editor.getByTestId("answer").first();
  await expect(own).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(own).toContainText("book_boat said: Booked a boat trip from Dover.");
  await expect(own.getByTestId("approval-card")).toHaveCount(0);
  await expect(own.getByTestId("answer-connector-call")).not.toHaveAttribute("data-approval");
  await app.close();
});
