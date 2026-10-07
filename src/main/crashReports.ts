/**
 * Crash reports with Sentry (`@sentry/electron`), off until the User opts in.
 *
 * Only a build made with a crash-report address offers them: the DSN comes
 * from build-time configuration (`MAIN_VITE_SENTRY_DSN`, read by electron-vite),
 * never from the code. The core turns the reporter on once the User opts in on
 * the Privacy page, and only then is the SDK loaded and initialised; opting out
 * stops reporting at once.
 *
 * What a report holds is kept small on purpose:
 * - errors and crashes in the main process and in child processes, with the
 *   app lifecycle breadcrumbs before them, and nothing else: no sessions or
 *   usage data, tracing, logs, screenshots or minidumps (which hold raw memory
 *   that can't be scrubbed), and no console or request breadcrumbs;
 * - every event is scrubbed (./crashScrubber) of file paths, Document text,
 *   Mind content, Questions and Answers before it is sent;
 * - the SDK collects no user data, so no IP address either, and no headers,
 *   cookies, bodies or local variables.
 *
 * The renderer doesn't report errors itself in v1: its crashes are reported
 * from the main process.
 *
 * `@sentry/electron` is a devDependency on purpose: electron-vite bundles the
 * part used here into a chunk loaded only on opt-in, so installers don't ship
 * its packages (about 170 MB, with the Sentry CLI). A build without a DSN
 * leaves the SDK out entirely.
 */
import type { CrashReporter } from "../core";
import { createScrubber, type ScrubPaths } from "./crashScrubber";

type SentrySdk = typeof import("@sentry/electron/main");
type SentryOptions = NonNullable<Parameters<SentrySdk["init"]>[0]>;
type Transport = ReturnType<SentrySdk["makeElectronTransport"]>;

/** The only integrations set up, in order. Everything else Sentry offers stays off. */
export const CRASH_REPORT_INTEGRATIONS = [
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
] as const;

/** Sentry collects none of these. In SDK 11 this replaces `sendDefaultPii: false`. */
const COLLECT_NOTHING: NonNullable<SentryOptions["dataCollection"]> = {
  // No user fields, so no IP address is sent or inferred.
  userInfo: false,
  cookies: false,
  httpHeaders: false,
  httpBodies: [],
  urlQueryParams: false,
  graphQL: { document: false, variables: false },
  genAI: { inputs: false, outputs: false },
  databaseQueryData: false,
  queues: false,
  stackFrameVariables: false,
  frameContextLines: 0,
};

export interface SentryCrashReporterOptions {
  /** Where reports go. */
  dsn: string;
  /** Folders whose paths are removed from every report. */
  paths: ScrubPaths;
  /** Loads the SDK. Defaults to importing `@sentry/electron/main`; tests pass a fake. */
  loadSdk?: () => Promise<SentrySdk>;
  /** Defaults to the console. */
  reportError?: (error: unknown) => void;
}

export interface SentryCrashReporter extends CrashReporter {
  /** Resolves once a start that is under way has finished (or failed). For tests. */
  settled(): Promise<void>;
}

export function createSentryCrashReporter(
  options: SentryCrashReporterOptions,
): SentryCrashReporter {
  const scrubber = createScrubber(options.paths);
  const loadSdk = options.loadSdk ?? (() => import("@sentry/electron/main"));
  const reportError =
    options.reportError ??
    ((error: unknown) => console.error("Crash reports couldn't start:", error));

  let enabled = false;
  let sdk: SentrySdk | undefined;
  let starting: Promise<void> | undefined;

  /** Nothing is sent while reports are off, even what was on its way to the transport. */
  const gated = (transport: Transport): Transport => ({
    send: (envelope) => (enabled ? transport.send(envelope) : Promise.resolve({})),
    flush: (timeout) => transport.flush(timeout),
  });

  const sentryOptions = (sentry: SentrySdk): SentryOptions => ({
    dsn: options.dsn,
    // Initialised after the app is ready, which rules out Sentry's custom
    // protocol for renderers; they don't report in v1 anyway.
    ipcMode: sentry.IPCMode.Classic,
    defaultIntegrations: false,
    integrations: [
      sentry.onUncaughtExceptionIntegration(),
      sentry.onUnhandledRejectionIntegration(),
      sentry.childProcessIntegration(),
      sentry.electronBreadcrumbsIntegration({ captureWindowTitles: false }),
      sentry.electronContextIntegration(),
      sentry.nodeContextIntegration({ culture: false, cloudResource: false }),
      sentry.eventFiltersIntegration(),
      sentry.functionToStringIntegration(),
      sentry.linkedErrorsIntegration(),
      // Last, so paths are made app-relative after the context is captured.
      sentry.normalizePathsIntegration(),
    ],
    // Sent straight away or not at all: no offline queue keeps reports on disk.
    transport: (transportOptions) => gated(sentry.makeElectronTransport(transportOptions)),
    dataCollection: COLLECT_NOTHING,
    attachScreenshot: false,
    sendClientReports: false,
    // No tracing: no traces sample rate, so no spans, and no span streaming set up for them.
    traceLifecycle: "static",
    // No tracing headers on IncarnaMind's own requests (to AI providers, GitHub…).
    tracePropagationTargets: [],
    propagateTraceparent: false,
    enhanceFetchErrorMessages: false,
    beforeSend: (event, hint) => (enabled ? scrubber.event(event, hint) : null),
    beforeBreadcrumb: (breadcrumb) => (enabled ? scrubber.breadcrumb(breadcrumb) : null),
    beforeSendTransaction: () => null,
    beforeSendLog: () => null,
    beforeSendMetric: () => null,
  });

  /** Turns the running client on or off; off also forgets the breadcrumbs gathered so far. */
  const apply = (sentry: SentrySdk) => {
    const client = sentry.getClient();
    if (client) client.getOptions().enabled = enabled;
    if (!enabled) {
      sentry.getCurrentScope().clearBreadcrumbs();
      sentry.getIsolationScope().clearBreadcrumbs();
      sentry.getGlobalScope().clearBreadcrumbs();
    }
  };

  const start = async () => {
    const sentry = await loadSdk();
    // The User opted out while the SDK was loading (and didn't opt in again): don't start it.
    if (!enabled) return;
    sentry.init(sentryOptions(sentry));
    sdk = sentry;
  };

  return {
    setEnabled(next) {
      enabled = next;
      if (sdk) {
        apply(sdk);
        return;
      }
      // A start under way reads `enabled` once the SDK has loaded.
      if (!next || starting) return;
      starting = start()
        .catch(reportError)
        .finally(() => {
          starting = undefined;
        });
    },

    async settled() {
      while (starting) await starting;
    },
  };
}
