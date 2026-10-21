/**
 * The pane over plain http://localhost (spike for #1272): whether WebKit (Office for Mac's engine)
 * and Chromium (WebView2's) load it, pair, use the WebSocket and do the Word and Excel round trips
 * against the stand-in for office.js, with no certificate at all. Only a sideload shows whether
 * Office itself loads an http:// pane.
 *
 *   PLAYWRIGHT_BROWSERS_PATH=<folder> node --disable-warning=MODULE_TYPELESS_PACKAGE_JSON \
 *     --import ./scripts/tsResolve.mjs spikes/1272-office-addin/probe-http.ts
 */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { type Browser, chromium, webkit } from "@playwright/test";
import { createOfficeBridge } from "../../src/core/office/index.ts";
import { memorySecrets } from "../../tests/helpers/officeBridge.ts";

const STATIC_DIR = resolve(import.meta.dirname, "../../addins/office");
const FAKE_OFFICE = resolve(import.meta.dirname, "fakeOffice.js");
const quiet = { info: () => undefined, warn: () => undefined, error: () => undefined };

const FAKES = {
  Word: {
    host: "Word",
    platform: "Mac",
    paragraphs: ["Tides are caused by the moon.", "The senior facilities end on 14 February 2029."],
    values: [],
    formulas: [],
  },
  Excel: {
    host: "Excel",
    platform: "Mac",
    paragraphs: [],
    values: [["Total", 30]],
    formulas: [["Total", "=SUM(B1:B2)"]],
  },
};

async function flow(browser: Browser, fake: (typeof FAKES)["Word"]) {
  const bridge = createOfficeBridge({
    adapter: {
      port: 0,
      staticDir: STATIC_DIR,
      authorityFile: join(await mkdtemp(join(tmpdir(), "probe-http-")), "unused.pem"),
      scheme: "http",
    },
    log: quiet,
    secrets: memorySecrets(),
    clock: () => performance.now(),
    now: () => new Date().toISOString(),
  });
  const { port } = await bridge.start();
  const context = await browser.newContext();
  const console: string[] = [];
  try {
    await context.route("https://officeapis.public.onecdn.static.microsoft/**", (route) =>
      route.fulfill({ status: 200, contentType: "text/javascript", body: "" }),
    );
    const page = await context.newPage();
    page.on("console", (message) => console.push(message.text()));
    await page.addInitScript((value) => {
      (window as unknown as { __fake: unknown }).__fake = value;
    }, fake);
    await page.addInitScript({ path: FAKE_OFFICE });
    await page.goto(`http://localhost:${port}/taskpane.html`);
    await page.waitForFunction(() =>
      document.getElementById("status")?.textContent?.startsWith("Type the code"),
    );
    const secure = await page.evaluate(() => window.isSecureContext);
    const { code } = bridge.beginPairing();
    await page.fill("#code", code);
    await page.click("#pair");
    await page.waitForFunction(
      () =>
        document.getElementById("report")?.value.includes('"websocket"') ||
        document.getElementById("report")?.value.includes('"stream"'),
    );
    const report = JSON.parse(await page.locator("#report").inputValue()) as Record<string, string>;
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
    const [pairing] = await bridge.pairings();
    await bridge.unpair(pairing?.id ?? "");
    await page.waitForFunction(() =>
      document.getElementById("status")?.textContent?.startsWith("This pairing ended"),
    );
    return { secure, ping: report.ping, websocket: report.websocket, stream: report.stream, result, effects, console };
  } finally {
    await context.close();
    await bridge.close();
  }
}

for (const [name, engine] of [["WebKit", webkit], ["Chromium", chromium]] as const) {
  const browser = await engine.launch();
  try {
    for (const host of ["Word", "Excel"] as const) {
      const outcome = await flow(browser, FAKES[host]).catch((error: unknown) => ({ error: String(error).split("\n")[0] }));
      process.stdout.write(`${name} ${host}: ${JSON.stringify(outcome)}\n`);
    }
  } finally {
    await browser.close();
  }
}
