/**
 * IncarnaMind collects usage data only with the User's consent (#187), and
 * only through one door: PostHog's Node SDK (`posthog-node`), loaded by
 * src/main/usageData.ts once the User agrees, sending only the events in
 * src/core/usageEvents.ts. These guards fail if another analytics or
 * telemetry package becomes a dependency (posthog-js too, with its
 * autocapture and session replay), if anything else loads posthog-node, or if
 * the app's code reaches an analytics service or names a project of its own:
 * the PostHog project and host come from the build, never from the code.
 * Opt-in crash reports (Sentry) are not analytics, but Sentry's usage
 * features (session replay, tracing, profiling) are, and stay out.
 */
import { readdirSync, readFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import { describe, expect, test } from "vitest";

const root = join(__dirname, "../..");

/** The one analytics SDK, and the packages it brings. */
const POSTHOG_NODE = "posthog-node";
const POSTHOG_NODE_PACKAGES = new Set([POSTHOG_NODE, "@posthog/core", "@posthog/types"]);

/** The only file that may load it. */
const USAGE_DATA_SENDER = "src/main/usageData.ts";

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
  "posthog-react-native",
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

/**
 * Hosts of analytics services, a PostHog project key, PostHog's browser
 * features, and Sentry's usage features, as they would appear in code.
 */
const FORBIDDEN_IN_CODE = [
  /google-analytics\.com/,
  /googletagmanager\.com/,
  /\bgtag\(/,
  /api\.mixpanel\.com/,
  /api\.segment\.io|cdn\.segment\.com/,
  /posthog\.com/,
  /\bphc_[A-Za-z0-9]{8,}/,
  /\bautocapture\s*:\s*true|\bsession_recording\b|\bstartSessionRecording\b/,
  /amplitude\.com/,
  /plausible\.io/,
  /aptabase\.com/,
  /browser-intake-datadoghq/,
  /\breplayIntegration\b|\bbrowserTracingIntegration\b|\breplayCanvasIntegration\b/,
  /\btracesSampleRate\b|\bprofilesSampleRate\b|\btracesSampler\b/,
];

const isAnalytics = (name: string) =>
  ANALYTICS_PACKAGES.has(name) || ANALYTICS_SCOPES.some((scope) => name.startsWith(scope));

/** The analytics packages among these names, but posthog-node's own; with `direct`, Sentry's usage features too. */
function findAnalytics(names: Iterable<string>, direct = false): string[] {
  return [...names].filter(
    (name) =>
      (isAnalytics(name) && !POSTHOG_NODE_PACKAGES.has(name)) ||
      (direct && SENTRY_USAGE_PACKAGES.has(name)),
  );
}

const readJson = (file: string) => JSON.parse(readFileSync(join(root, file), "utf8"));

const DEPENDENCY_FIELDS = [
  "dependencies",
  "devDependencies",
  "optionalDependencies",
  "peerDependencies",
] as const;

/** Every dependency package.json names, of every kind. */
function directDependencies(): string[] {
  const manifest = readJson("package.json");
  return DEPENDENCY_FIELDS.flatMap((field) => Object.keys(manifest[field] ?? {}));
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

/** The packages a file imports, by name. */
function importsOf(text: string): string[] {
  return [...text.matchAll(/(?:from\s+|import\(\s*|require\(\s*)["']([^"']+)["']/g)]
    .map((match) => match[1] ?? "")
    .map((specifier) =>
      specifier.startsWith("@")
        ? specifier.split("/").slice(0, 2).join("/")
        : (specifier.split("/")[0] ?? ""),
    );
}

describe("Usage data only through its one door", () => {
  test("the guard recognises analytics packages, and lets posthog-node's own through", () => {
    expect(
      findAnalytics([
        "react",
        "posthog-js",
        "posthog-node",
        "@posthog/core",
        "@segment/analytics-next",
        "@sentry/electron",
      ]),
    ).toEqual(["posthog-js", "@segment/analytics-next"]);
    expect(findAnalytics(["@sentry/replay", "@sentry/electron"], true)).toEqual(["@sentry/replay"]);
  });

  test("posthog-node is the only analytics package, a pinned devDependency that the build bundles", () => {
    expect(findAnalytics(directDependencies(), true)).toEqual([]);
    const manifest = readJson("package.json");
    for (const field of DEPENDENCY_FIELDS) {
      if (field === "devDependencies") continue;
      expect(Object.keys(manifest[field] ?? {}), field).not.toContain(POSTHOG_NODE);
    }
    // A devDependency is bundled into a chunk loaded only once usage data is on; pinned exactly.
    expect(manifest.devDependencies[POSTHOG_NODE]).toMatch(/^\d+\.\d+\.\d+$/);
  });

  test("no other analytics or telemetry package is installed, even through another package", () => {
    expect(findAnalytics(installedPackages())).toEqual([]);
  });

  test("only the usage data sender loads posthog-node, and nothing loads another analytics package", () => {
    const offending = sourceFiles(join(root, "src")).flatMap((file) => {
      const name = relative(root, file).replaceAll("\\", "/");
      const imports = importsOf(readFileSync(file, "utf8"));
      return [
        ...findAnalytics(imports, true).map((each) => `${name}: imports ${each}`),
        ...imports
          .filter((each) => POSTHOG_NODE_PACKAGES.has(each) && name !== USAGE_DATA_SENDER)
          .map((each) => `${name}: imports ${each}`),
      ];
    });
    expect(offending).toEqual([]);
    expect(importsOf(readFileSync(join(root, USAGE_DATA_SENDER), "utf8"))).toContain(POSTHOG_NODE);
  });

  test("the app's code names no analytics service or project, and sets up none of the usage features", () => {
    const offending = sourceFiles(join(root, "src")).flatMap((file) => {
      const text = readFileSync(file, "utf8");
      return FORBIDDEN_IN_CODE.filter((pattern) => pattern.test(text)).map(
        (pattern) => `${relative(root, file)}: matches ${pattern}`,
      );
    });
    expect(offending).toEqual([]);
  });
});
