import { writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { type FakeRemote, startFakeRemote } from "../tests/helpers/remoteMcp";
import {
  acceptPlainTextKeys,
  createDataFolder,
  dismissChatSetup,
  interceptOpenExternal,
  launchApp,
  openSettings,
  removeDataFolder,
  showSettingsPage,
  urlsOpened,
} from "./app";

/** The tiny MCP server the core tests use: `lookup_tide` is read-only, `book_boat` isn't. */
const TIDE_SERVER = resolve(__dirname, "../tests/fixtures/mcp-server.mjs");

/** The User approves in the browser, which comes back to the app's loopback redirect. */
async function approveInBrowser(fake: FakeRemote, authorizeUrl: string | undefined, name: string) {
  const page = await fetch(fake.approve(authorizeUrl ?? ""), {
    signal: AbortSignal.timeout(15_000),
  }).catch((error: unknown) => {
    throw new Error("The app's loopback redirect didn't answer the approval within 15 s.", {
      cause: error,
    });
  });
  expect(page.status).toBe(200);
  expect(await page.text()).toContain(`Signed in to ${name}`);
}

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

  await openSettings(window, "connectors");
  const page = window.getByTestId("settings-page-connectors");
  await expect(page.getByTestId("connectors-connected")).toHaveCount(0);
  await page.getByTestId("connector-add-custom").click();
  await page.getByRole("tab", { name: "Command" }).click();
  const form = page.getByTestId("connector-form");
  await form.getByLabel("Name", { exact: true }).fill("Tides");
  // Node, the way the test runs it; the app starts it with the login-shell environment.
  await form.getByLabel(/^Command/).fill(process.execPath);
  await form.getByLabel("Arguments, one per line").fill(TIDE_SERVER);
  await form.getByLabel(/^Environment variables/).fill("TIDE_TOKEN=smoke-secret");
  // Its variables are kept with the keys: on Linux with no keyring, once the User accepts it.
  await acceptPlainTextKeys(window, form);
  await form.getByRole("button", { name: "Add Connector" }).click();

  // Added, its page opens: its state, and its Tools, each with what it claims and whether it asks.
  await expect(page.getByTestId("connector-name")).toHaveText("Tides");
  await expect(page.getByTestId("connector-detail")).toContainText("Added by you · Not reviewed");
  const connector = page.getByTestId("connector");
  await expect(connector).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  await expect(connector.getByTestId("connector-state")).toHaveText("Connected");
  await expect(connector).toContainText(TIDE_SERVER);
  const tools = page.getByTestId("connector-tools");
  const lookup = tools.locator('[data-tool="lookup_tide"]');
  const book = tools.locator('[data-tool="book_boat"]');
  await expect(lookup).toHaveAttribute("data-read-only", "true");
  await expect(lookup.getByTestId("connector-tool-claim")).toHaveText("Reads");
  await expect(lookup.getByRole("radio", { name: "Always allow" })).toBeChecked();
  await expect(book).toHaveAttribute("data-read-only", "false");
  await expect(book.getByTestId("connector-tool-claim")).toHaveText("May change things");
  await expect(book.getByRole("radio", { name: "Ask every time" })).toBeChecked();

  // The claim is only a hint: the read-only Tool can be switched to ask every time, which the
  // Privacy page lists, and revoking it there goes back to the default.
  const approvals = window.getByTestId("approvals-settings");
  await showSettingsPage(window, "privacy");
  await expect(approvals).toContainText("No Tool is set to always allow or to ask every time.");
  await showSettingsPage(window, "connectors");
  await lookup.getByText("Ask every time").click();
  await expect(lookup).toHaveAttribute("data-asks", "true");
  await showSettingsPage(window, "privacy");
  const policy = approvals.getByTestId("approval-policy");
  await expect(policy).toHaveCount(1);
  await expect(policy).toContainText("Tides · lookup_tide");
  await expect(policy.getByTestId("approval-policy-value")).toHaveText("Ask every time");
  await policy.getByRole("button", { name: "Revoke the setting for Tides · lookup_tide" }).click();
  await expect(policy).toHaveCount(0);
  await showSettingsPage(window, "connectors");
  await expect(lookup).toHaveAttribute("data-asks", "false");
  await expect(lookup.getByRole("radio", { name: "Always allow" })).toBeChecked();

  // Back on the list, where the focus was, it is under Connected; Manage opens its page again.
  await page.getByTestId("connectors-back").click();
  await expect(page.getByTestId("connector-add-custom")).toBeFocused();
  const row = page.getByTestId("connected-connector");
  await expect(row).toHaveCount(1);
  await expect(row).toContainText("Tides");
  await expect(row.getByTestId("connected-connector-state")).toHaveText("Connected");
  await row.getByTestId("connector-manage").click();

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

    await openSettings(window, "connectors");
    const page = window.getByTestId("settings-page-connectors");
    await page.getByTestId("connector-add-custom").click();
    const form = page.getByTestId("connector-form");
    await form.getByLabel("Name", { exact: true }).fill("Wiki");
    await form.getByLabel("Server URL").fill(fake.url);
    await form.getByRole("button", { name: "Add Connector" }).click();

    const connector = page.getByTestId("connector");
    await expect(connector).toHaveAttribute("data-state", "needs-sign-in", { timeout: 15_000 });
    await expect(connector.getByTestId("connector-state")).toHaveText("Needs sign-in");
    await expect(connector.getByTestId("connector-location")).toHaveText(fake.url);
    await expect(connector.getByTestId("connector-sign-in-notice")).toContainText(
      "This server needs you to sign in.",
    );
    // Nothing opened in the browser, and IncarnaMind didn't register itself, until the User asks.
    expect(await urlsOpened(app)).toEqual([]);
    expect(fake.registrations).toEqual([]);

    // The service's tokens are kept with the keys: on Linux with no keyring, the User has
    // accepted that first (in a form that saves a key; the sign-in doesn't offer it yet, #342).
    await acceptPlainTextKeys(window);
    // Signing in opens the authorization request and waits for the browser.
    await connector.getByTestId("connector-sign-in").click();
    await expect(connector).toHaveAttribute("data-state", "signing-in");
    await expect(page.getByTestId("connector-signing-in")).toContainText(
      "Finish signing in in your browser",
    );
    await expect.poll(async () => (await urlsOpened(app)).length).toBe(1);
    const [authorizeUrl] = await urlsOpened(app);
    expect(authorizeUrl?.startsWith(`${fake.issuer}/authorize?`)).toBe(true);

    // The User approves; the browser comes back to the loopback redirect.
    await approveInBrowser(fake, authorizeUrl, "Wiki");

    await expect(connector).toHaveAttribute("data-state", "ready");
    await expect(connector.getByTestId("connector-state")).toHaveText("Connected");
    await expect(connector.getByTestId("connector-sign-out")).toBeVisible();
    await expect(page.getByTestId("connector-tool")).toHaveCount(2);
    await app.close();
  } finally {
    await fake.close();
  }
});

test("a reviewed Connector in the directory connects in one click, signing in in the browser", async () => {
  const fake = await startFakeRemote();
  // The directory as shipped would list real services: this one lists the stand-in.
  const directory = join(dataDir, "directory.json");
  await writeFile(
    directory,
    JSON.stringify([
      {
        id: "wiki",
        name: "Wiki",
        publisher: "Wiki Inc.",
        group: "files",
        summary: { en: "Search the team wiki", "zh-CN": "搜索团队维基" },
        subscription: { product: "Wiki Pro" },
        url: fake.url,
        signIn: "browser",
      },
    ]),
  );
  try {
    const { app, window } = await launchApp(dataDir, { connectorDirectory: directory });
    await dismissChatSetup(window);
    await interceptOpenExternal(app);

    await openSettings(window, "connectors");
    const page = window.getByTestId("settings-page-connectors");
    const group = page.locator('[data-testid="connectors-group"][data-group="files"]');
    await expect(group).toContainText("Where your files are");
    const card = group.getByTestId("connector-entry");
    await expect(card).toContainText("Wiki");
    await expect(card).toContainText("Search the team wiki");
    await expect(card).toContainText("Needs your Wiki Pro subscription");
    // Search finds it; a word nothing has says so.
    const search = page.getByTestId("connectors-search");
    await search.fill("zebra");
    await expect(page).toContainText("No connector matches “zebra”.");
    await search.fill("wik");
    await expect(page.getByTestId("connector-entry")).toHaveCount(1);
    // Looking at the directory reaches no service.
    expect(fake.requests).toEqual([]);

    await page.getByTestId("connector-entry").click();
    const detail = page.getByTestId("connector-detail");
    await expect(page.getByTestId("connector-name")).toHaveText("Wiki");
    await expect(detail).toContainText(
      "By Wiki Inc. · Reviewed by IncarnaMind · Needs your Wiki Pro subscription",
    );
    await expect(detail).toContainText(
      "Sign in with Wiki in your browser; you need a Wiki Pro subscription.",
    );
    await expect(page.getByTestId("connectors-back")).toBeFocused();
    expect(fake.requests).toEqual([]);

    await acceptPlainTextKeys(window);
    await page.getByTestId("connector-connect").click();
    await expect(page.getByTestId("connector-signing-in")).toContainText(
      "Finish signing in in your browser",
    );
    await expect.poll(async () => (await urlsOpened(app)).length).toBe(1);
    const [authorizeUrl] = await urlsOpened(app);
    await approveInBrowser(fake, authorizeUrl, "Wiki");

    const done = page.getByTestId("connector-connected");
    await expect(done).toContainText("Connected to Wiki.");
    await expect(page.getByTestId("connector-tool")).toHaveCount(2);
    await expect(fake.registrations).toHaveLength(1);

    // Try it in the assistant: Settings closes, and the cursor is in the composer.
    await done.getByRole("button", { name: "Try it in the assistant" }).click();
    await expect(window.getByTestId("settings")).toBeHidden();
    await expect(window.getByTestId("composer-input")).toBeFocused();

    // Settings opens again on the list, the Connector under Connected.
    await openSettings(window, "connectors");
    const row = page.getByTestId("connected-connector");
    await expect(row).toContainText("Wiki");
    await expect(row.getByTestId("connected-connector-state")).toHaveText("Connected");
    await app.close();
  } finally {
    await fake.close();
  }
});

test("a whole command line pasted as the command is split into its arguments, and a broken Connector is fixed by changing it", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await openSettings(window, "connectors");
  const page = window.getByTestId("settings-page-connectors");

  // Pasted whole, the command keeps the program and its arguments go below.
  await page.getByTestId("connector-add-custom").click();
  await page.getByRole("tab", { name: "Command" }).click();
  const form = page.getByTestId("connector-form");
  await form.getByLabel("Name", { exact: true }).fill("Tides");
  const command = form.getByLabel(/^Command/);
  await command.fill(`"${process.execPath}" ${TIDE_SERVER}`);
  await form.getByLabel(/^Arguments, one per line/).focus();
  await expect(command).toHaveValue(process.execPath);
  await expect(form.getByLabel(/^Arguments, one per line/)).toHaveValue(TIDE_SERVER);
  await expect(form).toContainText("The rest of what you pasted went into Arguments");
  await form.getByRole("button", { name: "Cancel" }).click();

  // A Connector added with a command that isn't there fails; changing its command fixes it.
  await page.getByTestId("connector-add-custom").click();
  await page.getByRole("tab", { name: "Command" }).click();
  await form.getByLabel("Name", { exact: true }).fill("Tides");
  await form.getByLabel(/^Command/).fill("tide-runtime-not-installed");
  await form.getByLabel(/^Environment variables/).fill("TIDE_TOKEN=smoke-secret");
  await acceptPlainTextKeys(window, form);
  await form.getByRole("button", { name: "Add Connector" }).click();
  const connector = page.getByTestId("connector");
  await expect(connector).toHaveAttribute("data-state", "error", { timeout: 30_000 });

  await connector.getByTestId("connector-edit").click();
  const edit = connector.getByTestId("connector-form");
  await expect(edit.getByLabel("Name", { exact: true })).toHaveValue("Tides");
  await expect(edit).toContainText("Leave empty to keep the saved values of TIDE_TOKEN.");
  await edit.getByLabel(/^Command/).fill(process.execPath);
  await edit.getByLabel(/^Arguments, one per line/).fill(TIDE_SERVER);
  await edit.getByRole("button", { name: "Save and restart" }).click();
  await expect(edit).toBeHidden();
  await expect(connector).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  await expect(connector).toContainText(TIDE_SERVER);
  await app.close();
});
