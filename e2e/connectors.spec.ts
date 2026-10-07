import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { startFakeRemote } from "../tests/helpers/remoteMcp";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenExternal,
  launchApp,
  removeDataFolder,
  urlsOpened,
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

  // Just added, its Tools are listed unfolded: which claim to be read-only, and which ask first.
  const tools = connector.getByTestId("connector-tools");
  await expect(tools).toHaveAttribute("open", "");
  await expect(tools.locator("summary")).toHaveText("2 Tools · 1 claims to be read-only");
  const lookup = tools.locator('[data-tool="lookup_tide"]');
  const book = tools.locator('[data-tool="book_boat"]');
  await expect(lookup).toHaveAttribute("data-read-only", "true");
  await expect(lookup.getByTestId("connector-tool-claim")).toContainText("claims to be read-only");
  await expect(lookup.getByTestId("connector-tool-approval")).toHaveValue("always");
  await expect(book).toHaveAttribute("data-read-only", "false");
  await expect(book.getByTestId("connector-tool-claim")).toContainText("may change things");
  await expect(book.getByTestId("connector-tool-approval")).toHaveValue("ask");

  // The claim is only a hint: the read-only Tool can be switched to ask every time, which the
  // approvals page lists, and revoking it there goes back to the default.
  const approvals = window.getByTestId("approvals-settings");
  await expect(approvals).toContainText("No Tool is set to always allow or to ask every time.");
  await lookup
    .getByRole("combobox", { name: "When an Answer calls lookup_tide" })
    .selectOption("ask");
  const policy = approvals.getByTestId("approval-policy");
  await expect(policy).toHaveCount(1);
  await expect(policy).toContainText("Tides · lookup_tide");
  await expect(policy.getByTestId("approval-policy-value")).toHaveText("Ask every time");
  await expect(lookup).toHaveAttribute("data-asks", "true");
  await policy.getByRole("button", { name: "Revoke the setting for Tides · lookup_tide" }).click();
  await expect(policy).toHaveCount(0);
  await expect(lookup.getByTestId("connector-tool-approval")).toHaveValue("always");

  // Off: its process stops, and it says so.
  await connector.getByRole("switch", { name: "Use Tides" }).uncheck();
  await expect(connector).toHaveAttribute("data-state", "off");
  await expect(connector.getByTestId("connector-state")).toHaveText("Off");
  await app.close();
});

test("a remote Connector added by URL needs a sign-in, which goes through the browser", async () => {
  // A local stand-in for the service: an OAuth authorization server and an MCP server.
  const fake = await startFakeRemote();
  try {
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    // The system browser opens nothing: the test plays the User on the sign-in page.
    await interceptOpenExternal(app);

    await window.getByRole("button", { name: "Settings" }).click();
    const section = window.getByTestId("connectors-settings");
    await section.getByTestId("connector-add").click();
    const form = section.getByTestId("connector-form");
    await form.getByLabel("Remote server (URL)").check();
    await form.getByLabel("Name", { exact: true }).fill("Wiki");
    await form.getByLabel("Server URL").fill(fake.url);
    await form.getByRole("button", { name: "Add Connector" }).click();

    const connector = section.getByTestId("connector");
    await expect(connector).toHaveAttribute("data-state", "needs-sign-in", { timeout: 30_000 });
    await expect(connector.getByTestId("connector-state")).toHaveText("Needs sign-in");
    await expect(connector.getByTestId("connector-location")).toHaveText(fake.url);
    await expect(connector.getByTestId("connector-sign-in-notice")).toContainText(
      "This server needs you to sign in.",
    );
    // Nothing opened in the browser, and IncarnaMind didn't register itself, until the User asks.
    expect(await urlsOpened(app)).toEqual([]);
    expect(fake.registrations).toEqual([]);

    // Signing in opens the authorization request and waits for the browser.
    await connector.getByTestId("connector-sign-in").click();
    await expect(connector).toHaveAttribute("data-state", "signing-in");
    await expect(connector.getByTestId("connector-signing-in")).toContainText(
      "Finish signing in in your browser",
    );
    await expect.poll(async () => (await urlsOpened(app)).length).toBe(1);
    const [authorizeUrl] = await urlsOpened(app);
    expect(authorizeUrl?.startsWith(`${fake.issuer}/authorize?`)).toBe(true);

    // The User approves; the browser comes back to the loopback redirect.
    const page = await fetch(fake.approve(authorizeUrl ?? ""));
    expect(page.status).toBe(200);
    expect(await page.text()).toContain("Signed in to Wiki");

    await expect(connector).toHaveAttribute("data-state", "ready");
    await expect(connector.getByTestId("connector-state")).toHaveText("Ready");
    await expect(connector.getByTestId("connector-sign-out")).toBeVisible();
    await expect(connector.getByText("2 Tools")).toBeVisible();
    await app.close();
  } finally {
    await fake.close();
  }
});
