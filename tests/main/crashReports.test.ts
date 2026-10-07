import type { Breadcrumb, ErrorEvent, EventHint } from "@sentry/electron/main";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { CRASH_REPORT_INTEGRATIONS, createSentryCrashReporter } from "../../src/main/crashReports";
import { PATH } from "../../src/main/crashScrubber";

type Options = Record<string, unknown> & {
  enabled?: boolean;
  integrations: { name: string }[];
  beforeSend(event: ErrorEvent, hint: EventHint): ErrorEvent | null;
  beforeBreadcrumb(breadcrumb: Breadcrumb): Breadcrumb | null;
  transport(options: unknown): { send(envelope: unknown): Promise<unknown> };
};

/** The mocked SDK: what was loaded, initialised and sent. */
const sentry = vi.hoisted(() => {
  const scope = () => ({ clearBreadcrumbs: vi.fn() });
  return {
    loads: 0,
    options: undefined as Options | undefined,
    init: vi.fn(),
    sent: [] as unknown[],
    scopes: { current: scope(), isolation: scope(), global: scope() },
  };
});

vi.mock("@sentry/electron/main", () => {
  sentry.loads++;
  const integration = (name: string) => () => ({ name });
  return {
    IPCMode: { Classic: 1, Protocol: 2, Both: 3 },
    init: sentry.init,
    getClient: () => sentry.options && { getOptions: () => sentry.options },
    getCurrentScope: () => sentry.scopes.current,
    getIsolationScope: () => sentry.scopes.isolation,
    getGlobalScope: () => sentry.scopes.global,
    makeElectronTransport: () => ({
      send: async (envelope: unknown) => {
        sentry.sent.push(envelope);
        return {};
      },
      flush: async () => true,
    }),
    onUncaughtExceptionIntegration: integration("OnUncaughtException"),
    onUnhandledRejectionIntegration: integration("OnUnhandledRejection"),
    childProcessIntegration: integration("ChildProcess"),
    electronBreadcrumbsIntegration: integration("ElectronBreadcrumbs"),
    electronContextIntegration: integration("ElectronContext"),
    nodeContextIntegration: integration("Context"),
    eventFiltersIntegration: integration("EventFilters"),
    functionToStringIntegration: integration("FunctionToString"),
    linkedErrorsIntegration: integration("LinkedErrors"),
    normalizePathsIntegration: integration("NormalizePaths"),
  };
});

const PATHS = { homeDir: "/Users/alice", dataDir: "/Users/alice/Library/IncarnaMind" };
const DSN = "https://public@o1.ingest.example.invalid/1";

function initOptions(): Options {
  expect(sentry.init).toHaveBeenCalledTimes(1);
  const options = sentry.init.mock.calls[0]?.[0] as Options | undefined;
  if (!options) throw new Error("Sentry wasn't initialised.");
  return options;
}

const errorEvent = (message: string): ErrorEvent => ({
  type: undefined,
  exception: { values: [{ type: "Error", value: message }] },
  user: { ip_address: "203.0.113.9" },
});

beforeEach(() => {
  vi.resetModules();
  sentry.loads = 0;
  sentry.options = undefined;
  sentry.sent = [];
  sentry.init.mockReset();
  sentry.init.mockImplementation((options: Options) => {
    sentry.options = options;
  });
  for (const scope of Object.values(sentry.scopes)) scope.clearBreadcrumbs.mockClear();
});

describe("Crash reports with Sentry", () => {
  test("the SDK isn't even loaded, let alone initialised, before the User opts in", async () => {
    const reporter = createSentryCrashReporter({ dsn: DSN, paths: PATHS });
    reporter.setEnabled(false);
    await reporter.settled();

    expect(sentry.loads).toBe(0);
    expect(sentry.init).not.toHaveBeenCalled();
  });

  test("opting in initialises it once, collecting nothing about the User", async () => {
    const reporter = createSentryCrashReporter({ dsn: DSN, paths: PATHS });
    reporter.setEnabled(true);
    await reporter.settled();
    reporter.setEnabled(true);
    await reporter.settled();

    const options = initOptions();
    expect(sentry.loads).toBe(1);
    expect(options).toMatchObject({
      dsn: DSN,
      ipcMode: 1,
      defaultIntegrations: false,
      attachScreenshot: false,
      sendClientReports: false,
      traceLifecycle: "static",
      tracePropagationTargets: [],
      propagateTraceparent: false,
      dataCollection: {
        userInfo: false,
        cookies: false,
        httpHeaders: false,
        httpBodies: [],
        urlQueryParams: false,
        stackFrameVariables: false,
        frameContextLines: 0,
      },
    });
    // No tracing, sessions, replays, screenshots, minidumps, console or request breadcrumbs.
    expect(options.integrations.map((integration) => integration.name)).toEqual([
      ...CRASH_REPORT_INTEGRATIONS,
    ]);
    expect(options).not.toHaveProperty("tracesSampleRate");
    expect(options).not.toHaveProperty("sendDefaultPii");
    for (const hook of ["beforeSendTransaction", "beforeSendLog", "beforeSendMetric"]) {
      expect((options[hook] as () => unknown)()).toBeNull();
    }
  });

  test("every event is scrubbed before it is sent, attachments included", async () => {
    const reporter = createSentryCrashReporter({ dsn: DSN, paths: PATHS });
    reporter.setEnabled(true);
    await reporter.settled();
    const hint: EventHint = { attachments: [{ filename: "minidump.dmp", data: "memory" }] };

    const sent = initOptions().beforeSend(
      errorEvent("ENOENT: open /Users/alice/Library/IncarnaMind/documents/Divorce.pdf"),
      hint,
    );

    expect(sent?.exception?.values?.[0]?.value).toBe(`ENOENT: open ${PATH}`);
    expect(sent).not.toHaveProperty("user");
    expect(hint.attachments).toEqual([]);
    expect(
      initOptions().beforeBreadcrumb({ category: "console", message: "Divorce papers" }),
    ).toBeNull();
  });

  test("opting out stops reporting at once, and opting in again resumes it", async () => {
    const reporter = createSentryCrashReporter({ dsn: DSN, paths: PATHS });
    reporter.setEnabled(true);
    await reporter.settled();
    const options = initOptions();
    const transport = options.transport({});
    // Sentry treats an unset `enabled` as on.
    expect(options.enabled).not.toBe(false);

    reporter.setEnabled(false);

    // The client is off, and anything already on its way is dropped.
    expect(options.enabled).toBe(false);
    expect(options.beforeSend(errorEvent("Late"), {})).toBeNull();
    expect(options.beforeBreadcrumb({ category: "electron", message: "app.quit" })).toBeNull();
    await transport.send(["an envelope"]);
    expect(sentry.sent).toEqual([]);
    // The breadcrumbs gathered so far are forgotten.
    for (const scope of Object.values(sentry.scopes)) {
      expect(scope.clearBreadcrumbs).toHaveBeenCalled();
    }

    reporter.setEnabled(true);

    expect(options.enabled).toBe(true);
    await transport.send(["an envelope"]);
    expect(sentry.sent).toEqual([["an envelope"]]);
    expect(options.beforeSend(errorEvent("Again"), {})).not.toBeNull();
    // Never initialised twice.
    expect(sentry.init).toHaveBeenCalledTimes(1);
  });

  test("opting out while the SDK is loading means it is never initialised", async () => {
    const reporter = createSentryCrashReporter({ dsn: DSN, paths: PATHS });
    reporter.setEnabled(true);
    reporter.setEnabled(false);
    await reporter.settled();

    expect(sentry.init).not.toHaveBeenCalled();

    reporter.setEnabled(true);
    await reporter.settled();
    expect(sentry.init).toHaveBeenCalledTimes(1);
  });

  test("an SDK that fails to load is reported, not thrown", async () => {
    const reportError = vi.fn();
    const reporter = createSentryCrashReporter({
      dsn: DSN,
      paths: PATHS,
      loadSdk: () => Promise.reject(new Error("Missing chunk")),
      reportError,
    });

    expect(() => reporter.setEnabled(true)).not.toThrow();
    await reporter.settled();

    expect(reportError).toHaveBeenCalledWith(new Error("Missing chunk"));
  });
});
