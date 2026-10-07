/**
 * IncarnaMind collects no usage data (spec #20, "Privacy"). These guards fail
 * if a known analytics or telemetry package becomes a dependency, or if the
 * app's code reaches a known analytics service. Opt-in crash reports (Sentry)
 * are not analytics, but Sentry's usage features (session replay, tracing,
 * profiling) are, and stay out.
 */
import { readdirSync, readFileSync } from "node:fs";
import { extname, join } from "node:path";
import { describe, expect, test } from "vitest";

const root = join(__dirname, "../..");

/** Analytics, product-telemetry and session-recording packages, by name. */
const ANALYTICS_PACKAGES = new Set([
  "analytics",
  "analytics-node",
  "amplitude-js",
  "applicationinsights",
  "appcenter-analytics",
  "countly-sdk-nodejs",
  "countly-sdk-web",
  "electron-ga",
  "electron-google-analytics",
  "electron-google-analytics4",
  "electron-nucleus",
  "ga-4-react",
  "heap-api",
  "insight",
  "logrocket",
  "mixpanel",
  "mixpanel-browser",
  "newrelic",
  "nucleus-nodejs",
  "openreplay",
  "plausible-tracker",
  "posthog-js",
  "posthog-node",
  "react-ga",
  "react-ga4",
  "rudder-sdk-js",
  "smartlook-client",
  "trackjs",
  "universal-analytics",
  "@firebase/analytics",
  "@google-analytics/data",
  "@microsoft/applicationinsights-web",
  "@vercel/analytics",
  "@vercel/speed-insights",
]);

/** Scopes that only publish analytics or telemetry. */
const ANALYTICS_SCOPES = [
  "@amplitude/",
  "@aptabase/",
  "@datadog/",
  "@fullstory/",
  "@heap/",
  "@hotjar/",
  "@mixpanel/",
  "@newrelic/",
  "@openreplay/",
  "@posthog/",
  "@rudderstack/",
  "@segment/",
];

/** Sentry's usage features: never a direct dependency, and never set up. */
const SENTRY_USAGE_PACKAGES = new Set([
  "@sentry/replay",
  "@sentry/replay-canvas",
  "@sentry-internal/replay",
  "@sentry-internal/replay-canvas",
  "@sentry/profiling-node",
]);

/** Hosts of analytics services, and Sentry's usage features, as they would appear in code. */
const FORBIDDEN_IN_CODE = [
  /google-analytics\.com/,
  /googletagmanager\.com/,
  /\bgtag\(/,
  /api\.mixpanel\.com/,
  /api\.segment\.io|cdn\.segment\.com/,
  /posthog\.com/,
  /amplitude\.com/,
  /plausible\.io/,
  /aptabase\.com/,
  /browser-intake-datadoghq/,
  /\breplayIntegration\b|\bbrowserTracingIntegration\b|\breplayCanvasIntegration\b/,
  /\btracesSampleRate\b|\bprofilesSampleRate\b|\btracesSampler\b/,
];

const isAnalytics = (name: string) =>
  ANALYTICS_PACKAGES.has(name) || ANALYTICS_SCOPES.some((scope) => name.startsWith(scope));

/** The analytics packages among these names; with `direct`, Sentry's usage features too. */
function findAnalytics(names: Iterable<string>, direct = false): string[] {
  return [...names].filter(
    (name) => isAnalytics(name) || (direct && SENTRY_USAGE_PACKAGES.has(name)),
  );
}

const readJson = (file: string) => JSON.parse(readFileSync(join(root, file), "utf8"));

/** Every dependency package.json names, of every kind. */
function directDependencies(): string[] {
  const manifest = readJson("package.json");
  return ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies"].flatMap(
    (field) => Object.keys(manifest[field] ?? {}),
  );
}

/** Every package installed by the lockfile, transitive ones included. */
function installedPackages(): Set<string> {
  const lock = readJson("package-lock.json");
  return new Set(
    Object.keys(lock.packages ?? {})
      .filter((path) => path.includes("node_modules/"))
      .map((path) => path.slice(path.lastIndexOf("node_modules/") + "node_modules/".length)),
  );
}

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return [".ts", ".tsx", ".html", ".css"].includes(extname(entry.name)) ? [path] : [];
  });
}

describe("No analytics", () => {
  test("the guard recognises analytics packages", () => {
    expect(
      findAnalytics(["react", "posthog-js", "@segment/analytics-next", "@sentry/electron"]),
    ).toEqual(["posthog-js", "@segment/analytics-next"]);
    expect(findAnalytics(["@sentry/replay", "@sentry/electron"], true)).toEqual(["@sentry/replay"]);
  });

  test("no analytics or telemetry package is a dependency", () => {
    expect(findAnalytics(directDependencies(), true)).toEqual([]);
  });

  test("no analytics or telemetry package is installed, even through another package", () => {
    expect(findAnalytics(installedPackages())).toEqual([]);
  });

  test("the app's code reaches no analytics service and sets up none of Sentry's usage features", () => {
    const offending = sourceFiles(join(root, "src")).flatMap((file) => {
      const text = readFileSync(file, "utf8");
      const imports = [...text.matchAll(/(?:from\s+|import\(\s*|require\(\s*)["']([^"']+)["']/g)]
        .map((match) => match[1] ?? "")
        .map((specifier) =>
          specifier.startsWith("@")
            ? specifier.split("/").slice(0, 2).join("/")
            : (specifier.split("/")[0] ?? ""),
        );
      return [
        ...findAnalytics(imports, true).map((name) => `${file}: imports ${name}`),
        ...FORBIDDEN_IN_CODE.filter((pattern) => pattern.test(text)).map(
          (pattern) => `${file}: matches ${pattern}`,
        ),
      ];
    });

    expect(offending).toEqual([]);
  });
});
