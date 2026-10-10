import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { gunzipSync } from "node:zlib";
import { expect, type Page, test } from "@playwright/test";
import { COMMON_FIELDS, SENDER_FIELDS, USAGE_EVENTS } from "../src/core/usageEvents";
import {
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openPrivacySettings,
  removeDataFolder,
  showSettingsPage,
} from "./app";

interface SentEvent {
  event: string;
  distinct_id: string;
  properties: Record<string, unknown>;
}

/** A stand-in for PostHog on this computer: it keeps every request it is sent. */
async function startFakePostHog() {
  const requests: { path: string; events: SentEvent[] }[] = [];
  const server: Server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(chunk as Buffer);
    const body = Buffer.concat(chunks);
    const text = (
      request.headers["content-encoding"] === "gzip" ? gunzipSync(body) : body
    ).toString();
    let events: SentEvent[] = [];
    try {
      events = (JSON.parse(text) as { batch?: SentEvent[] }).batch ?? [];
    } catch {
      // Not a batch: recorded with no events, and the test fails on its path.
    }
    requests.push({ path: request.url ?? "", events });
    response.writeHead(200, { "content-type": "application/json" });
    response.end("{}");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    host: `http://127.0.0.1:${port}`,
    requests,
    events: () => requests.flatMap((each) => each.events),
    close: async () => {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

type FakePostHog = Awaited<ReturnType<typeof startFakePostHog>>;

let dataDir: string;
/** A home folder for the app, so nothing it writes there is the User's. */
let home: string;
let posthog: FakePostHog;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
  posthog = await startFakePostHog();
});
test.afterEach(async () => {
  await posthog.close();
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
});

/** Long enough for anything queued to have been sent: the test build sends every half second. */
const settle = (window: Page) => window.waitForTimeout(1_500);

/** Each event holds exactly its declared fields, the common ones and PostHog's own, and nothing of the User's. */
function expectOnlyDeclared(events: readonly SentEvent[]) {
  for (const sent of events) {
    const declared = USAGE_EVENTS[sent.event as keyof typeof USAGE_EVENTS];
    expect(declared, sent.event).toBeDefined();
    expect(Object.keys(sent.properties).sort()).toEqual(
      [...Object.keys(declared), ...Object.keys(COMMON_FIELDS), ...SENDER_FIELDS].sort(),
    );
    expect(sent.properties).toMatchObject({ $ip: "0.0.0.0", $process_person_profile: false });
  }
  expect(JSON.stringify(events)).not.toContain(dataDir);
}

test("a release build asks on the first run, sends nothing when declined, and only what is declared once allowed", async () => {
  const { app, window } = await launchApp(dataDir, { home, usageData: { host: posthog.host } });

  // Asked first, with an example of what is sent; the chat setup waits.
  const dialog = window.getByTestId("usage-data-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAttribute("data-tester", "false");
  await expect(dialog).toContainText("Help improve IncarnaMind?");
  await expect(dialog.getByTestId("usage-data-examples")).toContainText("A Question was asked");
  await expect(dialog).toContainText("Never your Documents or their names");
  await expect(window.getByTestId("chat-setup")).toBeHidden();

  // Declined: nothing is sent, whatever the User does.
  await dialog.getByTestId("usage-data-decline").click();
  await expect(dialog).toBeHidden();
  await dismissChatSetup(window);
  await window.getByTestId("new-mind").click();
  const privacy = await openPrivacySettings(window);
  await expect(privacy).toContainText("collects usage data only if you agree");
  const row = privacy.locator('[data-testid="network-traffic-item"][data-traffic-id="usage-data"]');
  await expect(row).toContainText("PostHog (127.0.0.1)");
  const usageSwitch = row.getByTestId("usage-data-switch");
  await expect(usageSwitch).not.toBeChecked();
  await settle(window);
  expect(posthog.requests).toEqual([]);
  await app.close();

  // Asked once: not again.
  const second = await launchApp(dataDir, { home, usageData: { host: posthog.host } });
  await expect(second.window.getByTestId("mind-list-item")).toHaveCount(1);
  await expect(second.window.getByTestId("usage-data-dialog")).toBeHidden();
  await settle(second.window);
  expect(posthog.requests).toEqual([]);

  // Turned on in Settings → Privacy: events arrive, holding only what is declared.
  const reopened = await openPrivacySettings(second.window);
  const switchAgain = reopened.getByTestId("usage-data-switch");
  await switchAgain.check();
  await expect(switchAgain).toBeChecked();
  await showSettingsPage(second.window, "search");
  await expect
    .poll(() => posthog.events().map((each) => each.event), { timeout: 10_000 })
    .toContain("settings_page_viewed");
  expect(posthog.requests.every((each) => each.path === "/batch/")).toBe(true);
  const viewed = posthog.events().find((each) => each.event === "settings_page_viewed");
  expect(viewed?.properties).toMatchObject({ page: "search", build: "release" });
  expectOnlyDeclared(posthog.events());

  // Resetting the install ID: later events carry a new one.
  await showSettingsPage(second.window, "privacy");
  const before = viewed?.distinct_id;
  await second.window.getByTestId("usage-data-reset-id").click();
  await showSettingsPage(second.window, "general");
  await expect
    .poll(() => posthog.events().at(-1)?.distinct_id, { timeout: 10_000 })
    .not.toBe(before);

  // Turned off: nothing more is sent.
  await showSettingsPage(second.window, "privacy");
  await second.window.getByTestId("usage-data-switch").uncheck();
  await settle(second.window);
  const sent = posthog.events().length;
  await showSettingsPage(second.window, "models");
  await closeSettings(second.window);
  await second.window.getByTestId("new-mind").click();
  await settle(second.window);
  expect(posthog.events()).toHaveLength(sent);
  await second.app.close();
});

test("a test build says so on the first run, with the switch, and stops at once when it is turned off", async () => {
  const { app, window } = await launchApp(dataDir, {
    home,
    usageData: { host: posthog.host, testerBuild: true },
  });

  const dialog = window.getByTestId("usage-data-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAttribute("data-tester", "true");
  await expect(dialog).toContainText("This test version sends usage data");
  const usageSwitch = dialog.getByTestId("usage-data-switch");
  await expect(usageSwitch).toBeChecked();
  // On by default, as testers agree when they join: the app's opening is sent.
  await expect
    .poll(() => posthog.events().map((each) => each.event), { timeout: 10_000 })
    .toContain("app_opened");
  expect(posthog.events()[0]?.properties).toMatchObject({ build: "tester" });
  expectOnlyDeclared(posthog.events());

  // Turned off in the notice: nothing more is sent.
  await usageSwitch.uncheck();
  await dialog.getByTestId("usage-data-done").click();
  await expect(dialog).toBeHidden();
  await settle(window);
  const sent = posthog.events().length;
  await dismissChatSetup(window);
  await window.getByTestId("new-mind").click();
  const privacy = await openPrivacySettings(window);
  const row = privacy.locator('[data-traffic-id="usage-data"]');
  await expect(row.getByTestId("usage-data-switch")).not.toBeChecked();
  await expect(row).toContainText("this test version");
  await settle(window);
  expect(posthog.events()).toHaveLength(sent);
  await app.close();
});
