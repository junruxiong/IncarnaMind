import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { gunzipSync } from "node:zlib";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openPrivacySettings,
  removeDataFolder,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("the Privacy page lists the data flows and the update check, and offers no crash reports without a DSN", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // A chat model on a server elsewhere: the chat flow goes to it. Saving sends nothing.
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.saveChatProvider({
      kind: "openai-compatible",
      baseUrl: "https://llm.example.com/v1",
      modelId: "example-model",
    });
  });

  const privacy = await openPrivacySettings(window);
  await expect(privacy).toContainText("collects no usage data");

  // Every registered flow, with what it sends and where.
  const chat = privacy.locator('[data-testid="data-flow"][data-flow-id="chat"]');
  await expect(chat).toContainText("Chat");
  await expect(chat).toContainText("Passages from your Documents");
  await expect(privacy.locator('[data-flow-id="tagging"]')).toContainText("Automatic Tags");
  const service = chat.getByTestId("data-flow-service");
  await expect(service).toContainText("to llm.example.com");
  await expect(service).toHaveAttribute("data-consent", "not-asked");

  // Allowed from here, with the date; revoked, it asks again next time.
  await service.getByTestId("data-flow-allow").click();
  await expect(service).toHaveAttribute("data-consent", "accepted");
  await expect(service.getByTestId("data-flow-decision")).toContainText("Allowed on");
  await service.getByTestId("data-flow-revoke").click();
  await expect(service).toHaveAttribute("data-consent", "not-asked");

  // Traffic without the User's content: the update check, on by default, can be turned off.
  const updates = privacy.locator(
    '[data-testid="network-traffic-item"][data-traffic-id="update-check"]',
  );
  await expect(updates).toContainText("GitHub Releases");
  await expect(updates).toHaveAttribute("data-enabled", "true");
  const automatic = updates.getByTestId("automatic-update-checks");
  await expect(automatic).toBeChecked();
  await automatic.uncheck();
  await expect(updates).toHaveAttribute("data-enabled", "false");
  await expect(privacy.locator('[data-traffic-id="ollama-pull"]')).toContainText("Ollama");

  // Skill scripts are their own matter.
  await expect(privacy.getByTestId("skill-scripts-note")).toContainText(
    "approval you give before each run",
  );

  // A copy built without a crash-report address doesn't offer crash reports.
  await expect(privacy.getByTestId("crash-reports")).toHaveCount(0);
  expect(
    await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      return bridge.getPrivacySettings();
    }),
  ).toEqual({ crashReports: { available: false, enabled: false }, automaticUpdateChecks: false });
  await app.close();

  // The choice about update checks is kept.
  const second = await launchApp(dataDir);
  const reopened = await openPrivacySettings(second.window);
  await expect(reopened.getByTestId("automatic-update-checks")).not.toBeChecked();
  await second.app.close();
});

/** A stand-in for Sentry on this computer: it keeps every event it is sent. */
async function startFakeSentry(): Promise<{ server: Server; dsn: string; events: string[] }> {
  const events: string[] = [];
  const server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(chunk as Buffer);
    const body = Buffer.concat(chunks);
    const text = (
      request.headers["content-encoding"] === "gzip" ? gunzipSync(body) : body
    ).toString();
    // An envelope: a header line, then an item header and its payload, per item.
    const lines = text.split("\n");
    for (let index = 1; index + 1 < lines.length; index += 2) {
      if (JSON.parse(lines[index] ?? "{}").type === "event") events.push(lines[index + 1] ?? "");
    }
    response.writeHead(200, { "content-type": "application/json" });
    response.end("{}");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { server, dsn: `http://public@127.0.0.1:${port}/1`, events };
}

test("with a crash-report address, crash reports are off until opted in, scrubbed, and stop on opting out", async () => {
  const sentry = await startFakeSentry();
  try {
    const { app, window } = await launchApp(dataDir, { sentryDsn: sentry.dsn });
    await dismissChatSetup(window);
    /** An error in the main process that nothing catches, naming a file and quoting a Note. */
    const failInMainProcess = (marker: string) =>
      app.evaluate(({ app: electron }, label) => {
        const file = `${electron.getPath("userData")}/documents/Divorce papers.pdf`;
        void Promise.reject(
          new Error(`${label}: couldn't index ${file}: "Quarterly revenue fell"`),
        );
      }, marker);

    const privacy = await openPrivacySettings(window);
    const crashReports = privacy.getByTestId("crash-reports-switch");
    await expect(crashReports).not.toBeChecked();
    await expect(privacy.getByTestId("crash-reports")).toContainText("removed first");

    // Off by default: nothing is sent.
    await failInMainProcess("before opting in");
    await window.waitForTimeout(1_000);
    expect(sentry.events).toEqual([]);

    // Opted in: reports arrive, scrubbed.
    await crashReports.check();
    await expect
      .poll(
        async () => {
          await failInMainProcess("after opting in");
          return sentry.events.length;
        },
        { timeout: 15_000, intervals: [500] },
      )
      .toBeGreaterThan(0);
    const [event] = sentry.events;
    expect(event).toContain("after opting in: couldn't index [path]");
    for (const leak of [dataDir, "Divorce", "Quarterly", "before opting in"]) {
      expect(event).not.toContain(leak);
    }
    const parsed = JSON.parse(event ?? "{}");
    expect(parsed).not.toHaveProperty("user");
    expect(parsed).not.toHaveProperty("server_name");
    expect(parsed.sdk.settings).toEqual({ infer_ip: "never" });
    // Only the integrations IncarnaMind sets up: no sessions, tracing, replays or screenshots.
    expect(parsed.sdk.integrations).toEqual([
      "OnUncaughtException",
      "OnUnhandledRejection",
      "ChildProcess",
      "ElectronBreadcrumbs",
      "ElectronContext",
      "Context",
      "EventFilters",
      "FunctionToString",
      "LinkedErrors",
      "NormalizePaths",
    ]);

    // Opted out: nothing more is sent.
    await crashReports.uncheck();
    await expect(crashReports).not.toBeChecked();
    const sent = sentry.events.length;
    await failInMainProcess("after opting out");
    await window.waitForTimeout(1_500);
    expect(sentry.events.slice(sent).join("\n")).not.toContain("after opting out");
    await app.close();
  } finally {
    sentry.server.closeAllConnections();
    await new Promise((resolve) => sentry.server.close(resolve));
  }
});
