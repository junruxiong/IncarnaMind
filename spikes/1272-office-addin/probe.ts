/**
 * Headless stand-ins for the Office hosts (spike for #1272): Playwright WebKit
 * for Office for Mac's WKWebView, Playwright Chromium for Windows' WebView2.
 * Starts the bridge, then tries each way a task pane could reach it and prints
 * what each browser did, with its exact errors. Nothing here touches Word,
 * Excel or the User's keychain: a browser given `ignoreHTTPSErrors` stands for
 * "the certificate is trusted".
 *
 *   node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON --import ./scripts/tsResolve.mjs \
 *     spikes/1272-office-addin/probe.ts
 */
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { type Browser, type BrowserContext, chromium, type Page, webkit } from "@playwright/test";
import { createOfficeHandler } from "../../src/core/office/handler.ts";
import { createPairings } from "../../src/core/office/pairing.ts";
import { memorySecrets, type RunningBridge, startBridge } from "../../tests/helpers/officeBridge.ts";
import { createAuthority, issueLoopbackCertificate } from "../../src/core/office/certificate.ts";

const HOSTED = "https://addin.example";
const OFFICE_ON_THE_WEB = "https://word-edit.officeapps.live.com";
const STATIC_DIR = resolve(import.meta.dirname, "../../addins/office");
const FAKE_OFFICE = resolve(import.meta.dirname, "fakeOffice.js");
const OFFICE_CDN = "https://officeapis.public.onecdn.static.microsoft/**";

const quiet = { info: () => undefined, warn: () => undefined, error: () => undefined };
const firstLine = (error: unknown) =>
  (error instanceof Error ? `${error.name}: ${error.message}` : String(error)).split("\n")[0];

interface Fake {
  host: "Word" | "Excel";
  platform: "Mac" | "PC" | "OfficeOnline";
  paragraphs: string[];
  values: unknown[][];
  formulas: unknown[][];
}

const LONG_SENTENCE = `The agreement lists 3 conditions: ${"each of them must be met in full before closing, ".repeat(6)}and no other.`;
const WORD: Fake = {
  host: "Word",
  platform: "Mac",
  paragraphs: [
    "Tides are caused by the moon.",
    "The senior facilities end on 14 February 2029. They are repaid in full.",
  ],
  values: [],
  formulas: [],
};
const EXCEL: Fake = {
  host: "Excel",
  platform: "PC",
  paragraphs: [],
  values: [
    ["Revenue", 10],
    ["Total", 30],
  ],
  formulas: [
    ["Revenue", 10],
    ["Total", "=SUM(B1:B2)"],
  ],
};

async function plainMount(extraOrigins: string[]) {
  const pairings = createPairings({
    secrets: memorySecrets(),
    clock: () => performance.now(),
    now: () => new Date().toISOString(),
    removed: () => undefined,
  });
  const server = createServer();
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  const port = (server.address() as { port: number }).port;
  const handler = createOfficeHandler({
    hosts: new Set([`127.0.0.1:${port}`, `localhost:${port}`]),
    origins: new Set([`http://127.0.0.1:${port}`, ...extraOrigins]),
    pairings,
    file: async () => null,
    folderName: "x",
    contentSecurityPolicy: "default-src 'none'",
    log: quiet,
  });
  server.on("request", (request, response) => void handler.request(request, response));
  server.on("upgrade", handler.upgrade);
  return { port, close: () => server.close() };
}

async function openPane(context: BrowserContext, live: RunningBridge, fake: Fake) {
  await context.route(OFFICE_CDN, (route) =>
    route.fulfill({ status: 200, contentType: "text/javascript", body: "" }),
  );
  const page = await context.newPage();
  await page.addInitScript((value) => {
    (window as unknown as { __fake: unknown }).__fake = value;
  }, fake);
  await page.addInitScript({ path: FAKE_OFFICE });
  await page.goto(`https://localhost:${live.address.port}/taskpane.html`);
  return page;
}

const status = (page: Page) => page.locator("#status").innerText();
const reportOf = async (page: Page) =>
  JSON.parse(await page.locator("#report").inputValue()) as Record<string, string>;

async function paneFlow(browser: Browser, fake: Fake, options: { blockWebSocket?: boolean } = {}) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  try {
    const page = await openPane(context, live, fake);
    if (options.blockWebSocket) {
      await page.evaluate(() => {
        window.WebSocket = class {
          constructor() {
            throw new Error("blocked for the probe");
          }
        } as unknown as typeof WebSocket;
      });
    }
    await page.waitForFunction(() => document.getElementById("status")?.textContent?.startsWith("Type the code"));
    const before = await status(page);
    const { code } = live.bridge.beginPairing();
    await page.fill("#code", "000000" === code ? "111111" : "000000");
    await page.click("#pair");
    const wrong = await status(page);
    await page.fill("#code", code);
    await page.click("#pair");
    await page.waitForFunction(() => document.getElementById("report")?.value.includes('"websocket"') || document.getElementById("report")?.value.includes('"stream"'));
    const report = await reportOf(page);
    await page.click("#run");
    await page.waitForFunction(() => {
      const text = document.getElementById("result")?.textContent ?? "";
      return text !== "" && text !== "Working…";
    });
    const result = await page.locator("#result").innerText();
    const effects = await page.evaluate(() => {
      const { comments, notes } = (window as unknown as { __fake: { comments: unknown; notes: unknown } }).__fake;
      return { comments, notes };
    });
    const [pairing] = await live.bridge.pairings();
    await live.bridge.unpair(pairing?.id ?? "");
    await page.waitForFunction(() => document.getElementById("status")?.textContent?.startsWith("This pairing ended"));
    return {
      before,
      wrongCode: wrong,
      report: {
        ping: report.ping,
        websocket: report.websocket,
        stream: report.stream,
        officeHost: report.officeHost,
        platform: report.platform,
      },
      result,
      effects,
      afterUnpair: await status(page),
    };
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

async function longSentence(browser: Browser) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  try {
    const page = await openPane(context, live, { ...WORD, paragraphs: ["Intro.", LONG_SENTENCE] });
    await page.waitForFunction(() => document.getElementById("status")?.textContent?.startsWith("Type the code"));
    await page.fill("#code", live.bridge.beginPairing().code);
    await page.click("#pair");
    await page.waitForSelector("#run:visible");
    await page.click("#run");
    await page.waitForFunction(() => {
      const text = document.getElementById("result")?.textContent ?? "";
      return text !== "" && text !== "Working…";
    });
    const effects = await page.evaluate(
      () => (window as unknown as { __fake: { comments: { on: string }[] } }).__fake.comments,
    );
    return {
      sentenceLength: LONG_SENTENCE.length,
      commentCoversWholeSentence: effects[0]?.on === LONG_SENTENCE,
    };
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

async function onTheWeb(browser: Browser) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  try {
    const page = await openPane(context, live, { ...WORD, platform: "OfficeOnline" });
    await page.waitForFunction(() => (document.getElementById("status")?.textContent ?? "") !== "Connecting…");
    return { status: await status(page), diagnosticsHidden: await page.locator("#diagnostics").isHidden() };
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

async function appClosed(browser: Browser) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  try {
    await context.route("**/api/ping", (route) => route.abort("connectionrefused"));
    const page = await openPane(context, live, WORD);
    await page.waitForFunction(() => (document.getElementById("status")?.textContent ?? "") !== "Connecting…");
    return { status: await status(page), ping: (await reportOf(page)).ping };
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

async function untrustedCertificate(browser: Browser) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.goto(`https://localhost:${live.address.port}/taskpane.html`);
    return "loaded without a warning";
  } catch (error) {
    return firstLine(error);
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

const TARGETS = (https: number, plain: number) => ({
  "fetch https://localhost": `https://localhost:${https}/api/ping`,
  "fetch https://127.0.0.1": `https://127.0.0.1:${https}/api/ping`,
  "fetch http://127.0.0.1": `http://127.0.0.1:${plain}/api/ping`,
  "fetch http://localhost": `http://localhost:${plain}/api/ping`,
  "WebSocket wss://localhost": `wss://localhost:${https}/ws`,
  "WebSocket wss://127.0.0.1": `wss://127.0.0.1:${https}/ws`,
  "WebSocket ws://127.0.0.1": `ws://127.0.0.1:${plain}/ws`,
  "WebSocket ws://localhost": `ws://localhost:${plain}/ws`,
});

async function reachFrom(browser: Browser, origin: string, trusted: boolean) {
  const live = await startBridge({ staticDir: STATIC_DIR, extraOrigins: [origin] });
  const plain = await plainMount([origin]);
  const context = await browser.newContext({ ignoreHTTPSErrors: trusted });
  const notes: string[] = [];
  try {
    const page = await context.newPage();
    page.on("console", (message) => notes.push(`console: ${message.text().split("\n")[0]}`));
    page.on("requestfailed", (request) =>
      notes.push(`requestfailed: ${request.url()} ${request.failure()?.errorText ?? ""}`),
    );
    await context.route(`${origin}/**`, (route) =>
      route.fulfill({ status: 200, contentType: "text/html", body: "<!doctype html><title>hosted</title>" }),
    );
    await page.goto(`${origin}/`);
    const results = await page.evaluate(async (targets) => {
      const attempt = async (run: () => Promise<string>) => {
        try {
          return await run();
        } catch (error) {
          const failure = error as Error;
          return `blocked: ${failure.name}: ${failure.message}`;
        }
      };
      const socket = (url: string) =>
        new Promise<string>((done) => {
          let opened: WebSocket;
          try {
            opened = new WebSocket(url);
          } catch (error) {
            const failure = error as Error;
            return done(`blocked: ${failure.name}: ${failure.message}`);
          }
          const timer = setTimeout(() => {
            opened.close();
            done("blocked: no answer in 5 s");
          }, 5000);
          opened.onopen = () => {
            clearTimeout(timer);
            opened.close();
            done("reached");
          };
          opened.onerror = () => {
            clearTimeout(timer);
            done("blocked: error event");
          };
        });
      const out: Record<string, string> = {};
      for (const [name, url] of Object.entries(targets)) {
        out[name] = await attempt(async () => {
          if (name.startsWith("WebSocket")) return socket(url);
          const response = await fetch(url, { cache: "no-store" });
          return `reached (${response.status})`;
        });
      }
      return out;
    }, TARGETS(live.address.port, plain.port));
    return { results, notes: [...new Set(notes)].slice(0, 12) };
  } finally {
    await context.close();
    plain.close();
    await live.bridge.close();
  }
}

async function iframeIn(browser: Browser, origin: string, platform: Fake["platform"]) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  try {
    await context.route(OFFICE_CDN, (route) =>
      route.fulfill({ status: 200, contentType: "text/javascript", body: "" }),
    );
    await context.route(`${origin}/**`, (route) =>
      route.fulfill({
        status: 200,
        contentType: "text/html",
        body: `<!doctype html><title>top</title><iframe id="pane" sandbox="allow-scripts allow-same-origin allow-forms allow-popups" src="https://localhost:${live.address.port}/taskpane.html"></iframe>`,
      }),
    );
    const page = await context.newPage();
    const fake = { ...WORD, platform };
    await page.addInitScript((value) => {
      (window as unknown as { __fake: unknown }).__fake = value;
    }, fake);
    await page.addInitScript({ path: FAKE_OFFICE });
    const notes: string[] = [];
    page.on("console", (message) => notes.push(`console: ${message.text().split("\n")[0]}`));
    await page.goto(`${origin}/`);
    const frame = page.frameLocator("#pane");
    try {
      await frame.locator("#status").waitFor({ timeout: 8000 });
      await frame.locator("#status").evaluate(
        (element) =>
          new Promise<void>((done) => {
            const check = () => (element.textContent !== "Connecting…" ? done() : setTimeout(check, 50));
            check();
          }),
      );
      return { loaded: true, status: await frame.locator("#status").innerText(), notes: notes.slice(0, 6) };
    } catch (error) {
      return { loaded: false, error: firstLine(error), notes: [...new Set(notes)].slice(0, 6) };
    }
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

async function customScheme(browser: Browser) {
  const live = await startBridge({ staticDir: STATIC_DIR });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  try {
    const page = await context.newPage();
    await page.goto(`https://localhost:${live.address.port}/taskpane.html`);
    const out: Record<string, string> = {};
    out["fetch incarnamind://"] = await page.evaluate(async () => {
      try {
        await fetch("incarnamind://pair");
        return "reached";
      } catch (error) {
        return `blocked: ${(error as Error).name}: ${(error as Error).message}`;
      }
    });
    out["WebSocket incarnamind://"] = await page.evaluate(() => {
      try {
        new WebSocket("incarnamind://pair");
        return "created";
      } catch (error) {
        return `blocked: ${(error as Error).name}: ${(error as Error).message}`;
      }
    });
    const second = await context.newPage();
    try {
      await second.goto("incarnamind://open");
      out["navigate incarnamind://open"] = "navigated";
    } catch (error) {
      out["navigate incarnamind://open"] = firstLine(error);
    }
    return out;
  } finally {
    await context.close();
    await live.bridge.close();
  }
}

async function rebinding(browser: Browser) {
  const plain = await plainMount([]);
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    const response = await page.goto(`http://rebind.test:${plain.port}/api/ping`);
    return { status: response?.status(), body: await response?.text() };
  } catch (error) {
    return { error: firstLine(error) };
  } finally {
    await context.close();
    plain.close();
  }
}

function macSecurityFramework() {
  const authority = createAuthority(new Date());
  const leaf = issueLoopbackCertificate(authority, new Date());
  const folder = mkdtempSync(join(tmpdir(), "office-verify-"));
  const ca = join(folder, "ca.pem");
  const server = join(folder, "leaf.pem");
  writeFileSync(ca, authority.certificatePem);
  writeFileSync(server, leaf.certificatePem);
  const other = issueLoopbackCertificate(authority, new Date(), {
    dns: ["bank.example"],
    addresses: ["127.0.0.1"],
  });
  const otherFile = join(folder, "other.pem");
  writeFileSync(otherFile, other.certificatePem);
  const verify = (certificate: string, name: string) => {
    const run = spawnSync(
      "/usr/bin/security",
      ["verify-cert", "-c", certificate, "-r", ca, "-p", "ssl", "-s", name, "-L"],
      { encoding: "utf8", env: { ...process.env, HOME: folder } },
    );
    return `${run.status === 0 ? "valid" : "invalid"}: ${(run.stdout + run.stderr).trim().split("\n").slice(-2).join(" | ")}`;
  };
  return {
    "server certificate for localhost": verify(server, "localhost"),
    "server certificate for 127.0.0.1": verify(server, "127.0.0.1"),
    "server certificate for a name outside the constraints (bank.example)": verify(otherFile, "bank.example"),
  };
}

async function runAll(name: string, browser: Browser) {
  const out: Record<string, unknown> = { version: browser.version() };
  const step = async (label: string, run: () => Promise<unknown>) => {
    try {
      out[label] = await run();
    } catch (error) {
      out[label] = `probe failed: ${firstLine(error)}`;
    }
  };
  await step("pane served by the app, certificate not trusted", () => untrustedCertificate(browser));
  await step("pane served by the app, Word flow", () => paneFlow(browser, WORD));
  await step("pane served by the app, Excel flow", () => paneFlow(browser, EXCEL));
  await step("Word flow with WebSocket blocked (falls back to a stream)", () =>
    paneFlow(browser, WORD, { blockWebSocket: true }),
  );
  await step("a sentence longer than Word's search limit", () => longSentence(browser));
  await step("Office on the web message", () => onTheWeb(browser));
  await step("app closed after the pane loaded", () => appClosed(browser));
  await step("hosted pane reaching the app, certificate trusted", () =>
    reachFrom(browser, HOSTED, true),
  );
  await step("hosted pane reaching the app, certificate not trusted", () =>
    reachFrom(browser, HOSTED, false),
  );
  await step("app-served pane framed by Office on the web (PC platform value)", () =>
    iframeIn(browser, OFFICE_ON_THE_WEB, "PC"),
  );
  await step("custom URL scheme", () => customScheme(browser));
  if (name === "chromium") await step("DNS rebinding (host resolver maps rebind.test)", () => rebinding(browser));
  return out;
}

const results: Record<string, unknown> = {};
const webkitBrowser = await webkit.launch();
results.webkit = await runAll("webkit", webkitBrowser);
await webkitBrowser.close();
const chromiumBrowser = await chromium.launch({ args: ["--host-resolver-rules=MAP rebind.test 127.0.0.1"] });
results.chromium = await runAll("chromium", chromiumBrowser);
await chromiumBrowser.close();
if (process.platform === "darwin") results["macOS Security framework (security verify-cert)"] = macSecurityFramework();
console.log(JSON.stringify(results, null, 2));
