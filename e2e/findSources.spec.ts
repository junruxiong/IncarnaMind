/**
 * Find sources (#296), in the built app: from the Library's "Add Documents ▾"
 * and the sidebar's "+", search OpenAlex, arXiv and PubMed (a local stand-in
 * answering with recorded responses), allow each provider once, add an
 * open-access paper as a Document in a Folder, a paywalled one as a
 * reference and one open as a web page as a web page Document, tried again
 * when its site turned IncarnaMind away; in Chinese at the narrowest window; and from an Answer that
 * found papers, "Show results". Local mode blocks every provider.
 */
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { type SourceServices, startSourceServices } from "../tests/helpers/sourceServices";
import {
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  interceptOpenExternal,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  urlsOpened,
  useLocalChatModel,
} from "./app";

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
let services: SourceServices;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  services = await startSourceServices();
});
test.afterEach(async () => {
  await services.close();
  await removeDataFolder(dataDir);
});

const bridge = (window: Page) =>
  window.evaluateHandle(() => (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind);

async function newFolder(window: Page, name: string): Promise<void> {
  const core = await bridge(window);
  await core.evaluate(
    (each, folder) => each.createLibraryGroup({ name: folder, description: "" }),
    name,
  );
}

const result = (panel: Locator, title: string) =>
  panel.getByTestId("find-sources-result").filter({ hasText: title });

/** Allows each provider the consent dialog asks about, in turn, checking what it says is sent. */
async function allowProviders(
  window: Page,
  names: readonly string[],
  sends: string,
): Promise<void> {
  const consent = window.getByTestId("consent-dialog");
  for (const name of names) {
    await expect(consent).toBeVisible();
    await expect(consent).toContainText(name);
    await expect(consent).toContainText(sends);
    await window.getByTestId("consent-allow").click();
  }
}

test("Add Documents ▾ → Find sources… asks each provider once, then adds an open-access paper as a Document in the Folder, a paywalled one as a reference and one open as a web page as a web page", async () => {
  const { app, window } = await launchApp(dataDir, { sources: services.url });
  try {
    await dismissChatSetup(window);
    await interceptOpenExternal(app);
    await newFolder(window, "Thesis");
    await window.getByTestId("open-library").click();

    // The Library's Add Documents is a menu: files, or Find sources.
    const add = window.getByTestId("library-add-documents");
    await add.hover();
    await add.click();
    await expect(
      window.getByTestId("library-add-documents-items").getByRole("menuitem"),
    ).toHaveText([/Files…/, /Find sources…/]);
    await window.getByTestId("library-find-sources").click();
    const panel = window.getByTestId("find-sources");
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId("find-sources-query")).toBeFocused();
    await expect(panel.getByTestId("find-sources-provider")).toHaveText([
      "OpenAlex",
      "arXiv",
      "PubMed",
    ]);

    // Words, a year and the Folder to add to; Enter searches.
    const since = new Date().getFullYear() - 5;
    await panel.getByTestId("find-sources-since").selectOption({ label: `Since ${since}` });
    await panel.getByTestId("find-sources-folder").selectOption({ label: "Thesis" });
    await panel.getByTestId("find-sources-query").fill("CRISPR off-target detection");
    await panel.getByTestId("find-sources-query").press("Enter");

    // Nothing goes to a provider before the User allows it; each asks once, saying what is sent.
    await expect(window.getByTestId("consent-dialog")).toBeVisible();
    expect(services.requests.filter((request) => request.path.startsWith("/openalex/"))).toEqual(
      [],
    );
    await allowProviders(
      window,
      ["OpenAlex", "arXiv", "PubMed"],
      "What you search for: the words and filters, such as years",
    );
    await expect(panel.getByTestId("find-sources-count")).toHaveText("9 results");
    const openalex = services.requests.find((request) => request.path.startsWith("/openalex/"));
    expect(new URL(openalex?.path ?? "/", services.url).searchParams.get("filter")).toBe(
      `from_publication_date:${since}-01-01`,
    );

    // Each result: title, authors, year and venue; open or not; licence; citations; provider.
    const nature = result(panel, "Improving the sensitivity of in vivo CRISPR");
    await expect(nature).toContainText("Roger S. Zou");
    await expect(nature).toContainText("2023 · Nature Methods");
    await expect(nature.getByTestId("find-sources-result-facts")).toHaveText(
      "Open access · PDF · CC BY · Cited by 51 · Found on OpenAlex",
    );
    if (SCREENSHOTS)
      await window.screenshot({ path: join(SCREENSHOTS, "find-sources-en-1280.png") });

    // An open-access paper: downloaded and added as a Document, in the Folder.
    const addPaper = nature.getByTestId("find-sources-add");
    await expect(addPaper).toHaveText("Add");
    await addPaper.hover();
    await addPaper.click();
    await expect(nature.getByTestId("find-sources-added")).toHaveText("Added to “Thesis” · Open");
    await expect(
      window
        .getByTestId("library-document")
        .filter({ hasText: "Improving the sensitivity of in vivo CRISPR" }),
    ).toBeVisible();

    // A paywalled paper: its reference only, and "Get the PDF…" opens its publisher's page.
    const closed = result(panel, "Advances in Off-Target Detection");
    await expect(closed.getByTestId("find-sources-result-facts")).toContainText(
      "No open full text",
    );
    await closed.getByTestId("find-sources-add").click();
    await expect(closed.getByTestId("find-sources-added")).toHaveText("Added as a reference");
    await closed.getByTestId("find-sources-get-pdf").click();
    await expect.poll(() => urlsOpened(app)).toEqual(["https://doi.org/10.1089/hum.2022.198"]);
    expect(services.requests.filter((request) => request.path.startsWith("/files/"))).toHaveLength(
      1,
    );

    // A paper whose open text is a web page is saved as one. While its site turns IncarnaMind
    // away, its reference is kept, with the page to open and another try.
    const page = result(panel, "CHOPOFF");
    await expect(page.getByTestId("find-sources-result-facts")).toContainText(
      "Open access · web page",
    );
    services.fail("files", 403);
    await page.getByTestId("find-sources-add").click();
    await expect(page.getByTestId("find-sources-added")).toHaveText(
      "Couldn't download the open copy, so its reference was added",
    );
    await expect(page.getByTestId("find-sources-get-pdf")).toHaveText("Open the free text…");
    services.fail("files", null);
    await expect(page.getByTestId("find-sources-add")).toHaveText("Try again");
    await page.getByTestId("find-sources-add").click();
    await expect(page.getByTestId("find-sources-added")).toHaveText("Added to “Thesis” · Open");
    await expect(page.getByTestId("find-sources-get-pdf")).toHaveCount(0);
    await expect(
      window.getByTestId("library-document").filter({ hasText: "CHOPOFF" }),
    ).toBeVisible();

    // Asked once: another search asks nothing more. Esc closes the panel.
    await panel.getByTestId("find-sources-query").fill("base editing");
    await panel.getByTestId("find-sources-search").click();
    await expect(panel.getByTestId("find-sources-count")).toHaveText("9 results");
    await expect(window.getByTestId("consent-dialog")).toBeHidden();
    await panel.getByTestId("find-sources-query").press("Escape");
    await expect(panel).toBeHidden();
  } finally {
    await app.close();
  }
});

test("in Chinese at the narrowest window, Find sources from + fits and speaks Chinese; local mode blocks every provider", async () => {
  const { app, window } = await launchApp(dataDir, { sources: services.url });
  try {
    await dismissChatSetup(window);
    const core = await bridge(window);
    await core.evaluate((each) => each.updateSettings({ user: { language: "zh-CN" } }));
    await setWindowSize(app, window, 900, 660);

    await window.getByTestId("plus-menu").click();
    await window.getByTestId("plus-find-sources").click();
    const panel = window.getByTestId("find-sources");
    await expect(panel.getByRole("heading")).toHaveText("查找资料");
    await window.keyboard.type("CRISPR off-target detection");
    await window.keyboard.press("Enter");
    await allowProviders(
      window,
      ["OpenAlex", "arXiv", "PubMed"],
      "你搜索的内容：关键词和筛选条件，例如年份",
    );
    await expect(panel.getByTestId("find-sources-count")).toHaveText("9 个结果");
    const nature = result(panel, "Improving the sensitivity of in vivo CRISPR");
    await expect(nature.getByTestId("find-sources-result-facts")).toHaveText(
      "开放获取 · PDF · CC BY · 被引 51 次 · 来自 OpenAlex",
    );
    await expect(nature.getByTestId("find-sources-add")).toHaveText("添加");
    await expect(
      result(panel, "Advances in Off-Target Detection").getByTestId("find-sources-add"),
    ).toHaveText("添加引用信息");

    // It fits in the Library: nothing runs off to the side, and titles wrap by word.
    const library = window.getByTestId("library");
    const libraryBox = await library.boundingBox();
    const panelBox = await panel.boundingBox();
    expect(panelBox && libraryBox && panelBox.x + panelBox.width).toBeLessThanOrEqual(
      (libraryBox?.x ?? 0) + (libraryBox?.width ?? 0),
    );
    const scroller = library.locator(".overflow-y-auto").first();
    expect(
      await scroller.evaluate((element) => element.scrollWidth - element.clientWidth),
    ).toBeLessThanOrEqual(0);
    if (SCREENSHOTS)
      await window.screenshot({ path: join(SCREENSHOTS, "find-sources-zh-900.png") });

    // Local mode: nothing is asked or sent, and the panel says why, with where to change it.
    const sent = services.requests.length;
    await core.evaluate((each) => each.setLocalOnly(true));
    await panel.getByTestId("find-sources-query").fill("base editing");
    await panel.getByTestId("find-sources-query").press("Enter");
    await expect(panel.getByTestId("find-sources-local-mode")).toHaveText(
      "开启“所有内容都留在这台电脑上”时无法查找资料 · 隐私设置",
    );
    expect(services.requests).toHaveLength(sent);
    await expect(window.getByTestId("consent-dialog")).toBeHidden();
  } finally {
    await app.close();
  }
});

test("an Answer asked to find papers searches the providers, and Show results opens them in the Library to add", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, sources: services.url });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  try {
    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("Find papers since 2021 on CRISPR off-target detection");
    await window.keyboard.press("Enter");

    await allowProviders(
      window,
      ["OpenAlex", "arXiv", "PubMed"],
      "What you search for: the words and filters, such as years",
    );
    const answer = assistantAnswers(window).last();
    await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(answer).toContainText("I found these papers:");
    const call = answer.getByTestId("answer-sources-call");
    await expect(call).toContainText("Found sources for “CRISPR off-target detection”");
    await expect(call).toContainText("9 found");
    if (SCREENSHOTS)
      await window.screenshot({ path: join(SCREENSHOTS, "find-sources-answer-en.png") });

    // The same results, in the Library: nothing is sent again.
    const sent = services.requests.length;
    await call.getByTestId("answer-sources-show").click();
    const panel = window.getByTestId("find-sources");
    await expect(panel.getByTestId("find-sources-count")).toHaveText("9 results");
    await expect(panel.getByTestId("find-sources-query")).toHaveValue(
      "CRISPR off-target detection",
    );
    await expect(panel.getByTestId("find-sources-since")).toHaveValue("2021");
    expect(services.requests).toHaveLength(sent);
    // The Mind stays open in its tab beside the Library.
    await expect(window.getByTestId("mind-tab")).toHaveCount(1);
  } finally {
    await app.close();
  }
});
