import { writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
} from "./app";

/*
 * Tags in the sidebar (#71): the Tags tab beside Folders and Source
 * locations, and the colours of a Document's Tags on its row. Driven as a
 * person would: the mouse slides to what it clicks, in steps, and the
 * keyboard is real key presses. Set INCARNAMIND_SCREENSHOTS to a folder to
 * save screenshots there, in English and Chinese.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Slides the mouse onto `target`, in steps, and checks it is what lies under the pointer. */
async function slideTo(window: Page, target: Locator): Promise<{ x: number; y: number }> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("Nothing to point at: the target has no box.");
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await window.mouse.move(x, y, { steps: 12 });
  await expect
    .poll(() =>
      target.evaluate(
        (element, [px, py]) => {
          const hit = document.elementFromPoint(px as number, py as number);
          return hit !== null && (element === hit || element.contains(hit));
        },
        [x, y],
      ),
    )
    .toBe(true);
  return { x, y };
}

async function slideAndClick(window: Page, target: Locator): Promise<void> {
  await slideTo(window, target);
  await window.mouse.down();
  await window.mouse.up();
}

/** Presses Tab, or Shift+Tab if `target` comes before the focus, until `target` has the focus. */
async function tabTo(window: Page, target: Locator, max = 40): Promise<void> {
  const key = (await target.evaluate((element) => {
    const focused = document.activeElement;
    return (
      focused !== null &&
      focused !== document.body &&
      (focused.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_PRECEDING) !== 0
    );
  }))
    ? "Shift+Tab"
    : "Tab";
  for (let step = 0; step < max; step++) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await window.keyboard.press(key);
  }
  await expect(target).toBeFocused();
}

/** Whether the focused control shows DESIGN.md's ring: a solid 2px accent outline. */
const focusRingShows = (window: Page) =>
  window.evaluate(() => {
    const element = document.activeElement as HTMLElement | null;
    if (!element) return false;
    const probe = document.createElement("span");
    probe.style.color = "var(--color-accent)";
    document.body.append(probe);
    const accent = getComputedStyle(probe).color;
    probe.remove();
    const style = getComputedStyle(element);
    return (
      element.matches(":focus-visible") &&
      style.outlineStyle === "solid" &&
      Number.parseFloat(style.outlineWidth) >= 2 &&
      style.outlineColor === accent
    );
  });

/** Writes text files and adds them; waits until each is ready. */
async function addTextDocuments(window: Page, files: Record<string, string>): Promise<void> {
  const paths: string[] = [];
  for (const [name, text] of Object.entries(files)) {
    const path = join(sources, name);
    await writeFile(path, text);
    paths.push(path);
  }
  await addDocuments(window, paths);
}

/** Puts Tags on Documents, by name, through the core's bridge, as the Tag picker would. */
const tagDocuments = (window: Page, wanted: Record<string, string[]>) =>
  window.evaluate(async (byName) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const tags = await bridge.listTags();
    for (const item of await bridge.listDocuments()) {
      for (const name of byName[item.name] ?? []) {
        const tag = tags.find((each) => each.name === name);
        if (!tag) throw new Error(`No Tag named ${name}.`);
        await bridge.addDocumentTag(item.id, tag.id);
      }
    }
  }, wanted);

const setLanguage = (window: Page, language: "en" | "zh-CN") =>
  window.evaluate(
    (wanted) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: wanted },
      }),
    language,
  );

const setSidebarWidth = (window: Page, width: number) =>
  window.evaluate(
    (sidebarWidth) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        device: { sidebarWidth },
      }),
    width,
  );

/**
 * A local decision server like Ollama's `/v1/systemone` (no model): the
 * Finance Folder for an invoice, Reports otherwise; Invoice in the review
 * band for an invoice, Report likely for a report, nothing else.
 */
async function decisionServer(): Promise<Server> {
  const server = createServer(async (request, response) => {
    let raw = "";
    for await (const chunk of request) raw += chunk;
    response.setHeader("content-type", "application/json");
    if (request.url !== "/v1/systemone") {
      response.end(JSON.stringify({ models: [] }));
      return;
    }
    const body = JSON.parse(raw);
    const text = JSON.stringify(body.state).toLowerCase();
    const questions = body.questions as Record<
      string,
      { type: string; instructions: string; criteria?: Record<string, string> }
    >;
    const answers = Object.fromEntries(
      Object.entries(questions).map(([id, question]) => {
        if (question.type === "choice") {
          const wanted = text.includes("invoice") ? "Finance" : "Reports";
          const choice =
            Object.entries(question.criteria ?? {}).find(([, value]) =>
              value.startsWith(wanted),
            )?.[0] ?? "__unsorted__";
          return [id, { type: "choice", choice }];
        }
        const noul =
          question.instructions.includes("“Invoice”") && text.includes("invoice")
            ? 0.65
            : question.instructions.includes("“Report”") && text.includes("report")
              ? 0.95
              : 0.01;
        return [id, { type: "noul", noul }];
      }),
    );
    response.end(JSON.stringify({ answers }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return server;
}

/** Each Tag row's name, count and dot's colour (null for "Needs review", which has the review ring). */
const tagRowsListed = (rows: Locator) =>
  rows.evaluateAll((all) =>
    all.map((row) => {
      const head = row.firstElementChild as HTMLElement;
      return [
        head.querySelector('[data-testid="row-text"]')?.textContent,
        Number(head.querySelector('[data-testid="browse-count"]')?.textContent),
        head.querySelector('[data-testid="tag-dot"]')?.getAttribute("data-colour") ?? null,
      ];
    }),
  );

/** The colours a Document row shows, in order, and its "+N" (or null). */
const marksOf = (row: Locator) =>
  row.evaluate((element) => {
    const marks = element.querySelector('[data-testid="tag-marks"]');
    return {
      colours: Array.from(marks?.querySelectorAll('[data-testid="tag-dot"]') ?? []).map((dot) =>
        dot.getAttribute("data-colour"),
      ),
      more: marks?.querySelector('[data-testid="tag-marks-more"]')?.textContent ?? null,
    };
  });

test("the Tags tab lists every Tag with its colour and count, lists its Documents, and filters the Library by it, with the mouse and the keyboard", async () => {
  const server = await decisionServer();
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.setViewportSize({ width: 1100, height: 760 });
    await addTextDocuments(window, {
      "Attention.txt": "Attention is all you need: transformers for translation.",
      "Q1 report.txt": "Quarterly report: sales grew in every region.",
      "March invoice.txt": "Invoice 2026-03 for consulting. Total due: 1,200.",
      "Field notes.txt": "Observations of tides at the harbour.",
    });
    await tagDocuments(window, { Attention: ["Paper", "Book", "Contract", "Report"] });
    // Library Folders, and Organize with a local decision server: Report on the report,
    // and Invoice on the invoice, awaiting review.
    const port = (server.address() as AddressInfo).port;
    await window.evaluate(async (url) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await core.addLibraryStarterGroups(["reports", "finance"]);
      await core.saveLibrarySettings({
        classifier: { kind: "ollama", baseUrl: url, modelId: "tev1:0.8b" },
        automatic: false,
      });
      await core.classifyDocuments();
    }, `http://127.0.0.1:${port}`);

    // Folders | Tags | Source locations, Folders first.
    const browse = window.getByTestId("browse-views");
    await expect(browse).toHaveAccessibleName("Browse Documents");
    const views = browse.locator("button[aria-pressed]");
    await expect(views).toHaveText(["Folders", "Tags", "Source locations"]);
    await expect(views.first()).toHaveAttribute("aria-pressed", "true");
    // In the Folders view, the rows show their Tags' colours too.
    const folders = window.getByTestId("library-folders");
    const attentionInFolders = folders
      .getByTestId("document-list-item")
      .filter({ hasText: "Attention" });
    await expect(attentionInFolders.getByTestId("tag-dot")).toHaveCount(3);

    // The Tags tab: "Needs review" first, then every Tag in name order, with its colour
    // and how many Documents carry it, a Tag no Document has too.
    await slideAndClick(window, browse.getByRole("button", { name: "Tags", exact: true }));
    await expect(browse.getByRole("button", { name: "Tags", exact: true })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    const tagList = window.getByTestId("sidebar-tags");
    const tagRows = tagList.getByTestId("sidebar-tag");
    const tagRow = (id: string) =>
      tagList.locator(`[data-testid="sidebar-tag"][data-tag-id="${id}"]`);
    await expect(tagRow("needs-review")).toHaveCount(1, { timeout: 20_000 });
    await expect
      .poll(() => tagRowsListed(tagRows), { timeout: 20_000 })
      .toEqual([
        ["Needs review", 1, null],
        ["Book", 1, "orange"],
        ["Contract", 1, "red"],
        ["Invoice", 1, "green"],
        ["Notes", 0, "gray"],
        ["Paper", 1, "purple"],
        ["Report", 2, "blue"],
        ["Slides", 0, "yellow"],
      ]);
    const byName = (name: string) =>
      tagRows.filter({
        has: window.locator(':scope > div [data-testid="row-text"]', { hasText: name }),
      });
    // A Tag is an 11px solid dot, as in Finder's Tags; "Needs review" a hollow amber ring.
    const paperDot = byName("Paper").locator(":scope > div").getByTestId("tag-dot");
    expect(await paperDot.evaluate(dotShape)).toEqual({
      width: 11,
      height: 11,
      round: true,
      filled: true,
      border: 1,
    });
    const reviewHead = byName("Needs review").locator(":scope > div");
    await expect(reviewHead.getByTestId("tag-dot")).toHaveCount(0);
    // Its edge is 1.5px, which a screen that isn't Retina (CI's) draws as 1px: borders are
    // drawn in whole device pixels.
    const edge = await window.evaluate(() => {
      const probe = document.createElement("span");
      probe.style.border = "1.5px solid";
      document.body.append(probe);
      const width = Number.parseFloat(getComputedStyle(probe).borderTopWidth);
      probe.remove();
      return width;
    });
    expect(await reviewHead.getByTestId("review-ring").locator("span").evaluate(dotShape)).toEqual({
      width: 9,
      height: 9,
      round: true,
      filled: false,
      border: edge,
    });
    expect(
      await reviewHead
        .getByTestId("review-ring")
        .locator("span")
        .evaluate((element) => getComputedStyle(element).borderTopColor),
    ).toBe("rgb(180, 95, 6)");
    // Rows are 28px, their marks in the icon column and their names on the text edge.
    const sidebarLeft = (await window.getByTestId("sidebar").boundingBox())?.x ?? 0;
    for (const row of await tagRows.all()) {
      const head = await row.locator(":scope > div").boundingBox();
      expect(head?.height).toBeCloseTo(28, 0);
      const text = await row.locator(':scope > div [data-testid="row-text"]').boundingBox();
      expect((text?.x ?? 0) - sidebarLeft).toBeCloseTo(40, 0);
    }
    // An empty Tag shows 0 and has nothing to expand.
    await expect(byName("Notes").getByRole("button", { name: /^Expand / })).toHaveCount(0);
    await expect(byName("Notes").getByTestId("browse-count")).toHaveText("0");

    // A Tag's chevron lists its Documents under it, one step in, with their colours.
    await slideAndClick(window, tagList.getByRole("button", { name: "Expand Report" }));
    const reportDocuments = byName("Report").getByTestId("document-list-item");
    await expect(reportDocuments).toHaveCount(2);
    await expect(reportDocuments.first()).toHaveAttribute("data-depth", "1");
    const attention = reportDocuments.filter({ hasText: "Attention" });
    expect(await marksOf(attention)).toEqual({
      colours: ["orange", "red", "purple"],
      more: "+1",
    });
    expect(await marksOf(reportDocuments.filter({ hasText: "Q1 report" }))).toEqual({
      colours: ["blue"],
      more: null,
    });

    // Clicking a Tag shows its Documents in the Library, through the shared Tag filter,
    // and marks it chosen; nothing in the list moves under the pointer.
    const library = window.getByTestId("library");
    const tagFilter = library.locator('[data-testid="library-filter"][data-facet="tag"]');
    const openTag = (name: string) => byName(name).locator(":scope > div > button").first();
    const listBefore = await tagList.boundingBox();
    await slideAndClick(window, openTag("Paper"));
    await expect(library).toBeVisible();
    await expect(tagFilter).toHaveText("Tags: Paper");
    await expect(library.getByTestId("library-document")).toHaveCount(1);
    await expect(openTag("Paper")).toHaveAttribute("aria-pressed", "true");
    await expect(openTag("Report")).toHaveAttribute("aria-pressed", "false");
    // The Tags view says the filter with its chosen Tag, so no "Tagged …" row pushes it down.
    await expect(window.getByTestId("tag-filter-active")).toHaveCount(0);
    expect(await tagList.boundingBox()).toEqual(listBefore);
    await screenshot(window, "tags-tab-en");

    // By keyboard: Tab to Report, Enter shows it, Enter again stops filtering; its chevron
    // folds and unfolds with Enter.
    await tabTo(window, openTag("Report"));
    expect(await focusRingShows(window)).toBe(true);
    await window.keyboard.press("Enter");
    await expect(tagFilter).toHaveText("Tags: Report");
    await expect(library.getByTestId("library-document")).toHaveCount(2);
    await expect(openTag("Report")).toHaveAttribute("aria-pressed", "true");
    await expect(openTag("Paper")).toHaveAttribute("aria-pressed", "false");
    await window.keyboard.press("Enter");
    await expect(tagFilter).toHaveText("Tags");
    await expect(library.getByTestId("library-document")).toHaveCount(4);
    await expect(openTag("Report")).toHaveAttribute("aria-pressed", "false");
    await window.keyboard.press("Enter");
    await expect(tagFilter).toHaveText("Tags: Report");
    await window.keyboard.press("Tab");
    const collapse = tagList.getByRole("button", { name: "Collapse Report" });
    await expect(collapse).toBeFocused();
    expect(await focusRingShows(window)).toBe(true);
    await window.keyboard.press("Enter");
    await expect(reportDocuments).toHaveCount(0);
    await expect(tagList.getByRole("button", { name: "Expand Report" })).toBeFocused();
    await window.keyboard.press("Enter");
    await expect(reportDocuments).toHaveCount(2);
    // On up to the switcher, every stop with its ring.
    await tabTo(window, browse.getByRole("button", { name: "Tags", exact: true }));
    expect(await focusRingShows(window)).toBe(true);

    // "Needs review" is reachable as in the Library's filter.
    await slideAndClick(window, openTag("Needs review"));
    await expect(tagFilter).toHaveText("Tags: Needs review");
    await expect(library.getByTestId("library-document")).toHaveCount(1);
    await expect(library.getByTestId("library-document")).toContainText("March invoice");
    // The Library's Tags filter, too: each Tag after its dot, "Needs review" after the ring.
    await slideAndClick(window, tagFilter);
    const libraryTagMenu = library.locator('[data-testid="library-filter-menu"][data-facet="tag"]');
    const reviewOption = libraryTagMenu.locator(
      '[data-testid="library-filter-option"][data-value="needs-review"]',
    );
    await expect(reviewOption.getByTestId("review-ring")).toHaveCount(1);
    await expect(reviewOption.getByTestId("tag-dot")).toHaveCount(0);
    await expect(
      libraryTagMenu
        .getByTestId("library-filter-option")
        .filter({ hasText: "Report" })
        .getByTestId("tag-dot"),
    ).toHaveAttribute("data-colour", "blue");
    await screenshot(window, "library-filter-menu-en");
    await window.keyboard.press("Escape");
    await expect(libraryTagMenu).toBeHidden();

    // The sidebar's Tags menu, the same filter: each Tag after its dot, "Needs review" after
    // the hollow ring.
    const openFilterMenu = async () => {
      await slideAndClick(window, window.getByTestId("tag-filter-menu"));
      const menu = window.getByTestId("tag-filters");
      await expect(menu).toBeVisible();
      const review = menu.locator('[data-testid="tag-filter"][data-tag-id="needs-review"]');
      await expect(review.getByTestId("review-ring")).toHaveCount(1);
      await expect(review.getByTestId("tag-dot")).toHaveCount(0);
      expect(
        await menu
          .getByTestId("tag-filter")
          .getByTestId("tag-dot")
          .evaluateAll((all) => all.map((each) => each.getAttribute("data-colour"))),
      ).toEqual(["orange", "red", "green", "gray", "purple", "blue", "yellow"]);
    };
    await openFilterMenu();
    await screenshot(window, "tag-filter-menu-en");
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("tag-filters")).toBeHidden();

    // The other views say the filter in a row under the switcher, and clear it there.
    await slideAndClick(window, browse.getByRole("button", { name: "Folders", exact: true }));
    await expect(window.getByTestId("tag-filter-active")).toHaveText("Tagged Needs review");
    await slideAndClick(window, window.getByTestId("tag-filter-clear"));
    await expect(window.getByTestId("tag-filter-active")).toHaveCount(0);
    await slideAndClick(window, browse.getByRole("button", { name: "Tags", exact: true }));

    // In Chinese.
    await setLanguage(window, "zh-CN");
    await expect(browse).toHaveAccessibleName("浏览文档");
    await expect(views).toHaveText(["文件夹", "标签", "来源位置"]);
    await expect(byName("待确认")).toHaveCount(1);
    await slideAndClick(window, tagList.getByRole("button", { name: "展开 Report" }));
    await expect(tagList.getByRole("button", { name: "收起 Report" })).toBeVisible();
    await slideAndClick(window, openTag("Paper"));
    await expect(tagFilter).toHaveText("标签：Paper");
    await expect(openTag("Paper")).toHaveAttribute("aria-pressed", "true");
    await window.mouse.move(700, 400, { steps: 6 });
    await screenshot(window, "tags-tab-zh");
    await openFilterMenu();
    await screenshot(window, "tag-filter-menu-zh");
    await window.keyboard.press("Escape");
    await slideAndClick(window, tagFilter);
    await expect(reviewOption.getByTestId("review-ring")).toHaveCount(1);
    await screenshot(window, "library-filter-menu-zh");
    await window.keyboard.press("Escape");
  } finally {
    await app.close();
    server.close();
  }
});

test("a Document row shows up to three Tag colours and +N after its name, names them on hover and on focus, and stays one 28px line", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.setViewportSize({ width: 1100, height: 760 });
    const long = "Tidal patterns in the North Sea, observed over two winters at three harbours";
    await addTextDocuments(window, {
      "Attention.txt": "Attention is all you need.",
      "Q1 report.txt": "Quarterly report: sales grew.",
      "Field notes.txt": "Observations of tides.",
      [`${long}.txt`]: "Tides.",
    });
    const items = window.getByTestId("document-list-item");
    const row = (name: string) => items.filter({ hasText: name });
    const boxes = async () => Promise.all((await items.all()).map((each) => each.boundingBox()));
    const before = await boxes();

    await tagDocuments(window, {
      Attention: ["Paper", "Book", "Contract", "Report"],
      "Q1 report": ["Report"],
      [long]: ["Paper", "Notes"],
    });
    await expect(row("Attention").getByTestId("tag-dot")).toHaveCount(3);
    expect(await marksOf(row("Attention"))).toEqual({
      colours: ["orange", "red", "purple"],
      more: "+1",
    });
    expect(await marksOf(row("Q1 report"))).toEqual({ colours: ["blue"], more: null });
    expect(await marksOf(row(long))).toEqual({ colours: ["gray", "purple"], more: null });
    await expect(row("Field notes").getByTestId("tag-marks")).toHaveCount(0);
    // Nothing moved: every row where it was, 28px, its name on one line.
    expect(await boxes()).toEqual(before);

    /**
     * The marks right after the name, inside the row: 10px round dots, each
     * overlapping the one before by 4px, ringed in the row's background.
     */
    const expectMarksInRow = async (name: string) => {
      const box = await row(name).boundingBox();
      const marks = await row(name).getByTestId("tag-marks").boundingBox();
      const text = await row(name).getByTestId("row-text").boundingBox();
      if (!box || !marks || !text) throw new Error("The row isn't visible.");
      expect(box.height).toBeCloseTo(28, 0);
      expect(text.height).toBeLessThanOrEqual(20.5);
      expect(marks.x).toBeCloseTo(text.x + text.width + 8, 0);
      expect(marks.y).toBeGreaterThanOrEqual(box.y);
      expect(marks.y + marks.height).toBeLessThanOrEqual(box.y + box.height);
      expect(marks.x + marks.width).toBeLessThanOrEqual(box.x + box.width - 8);
      const dots = await row(name).getByTestId("tag-dot").all();
      let previous: number | null = null;
      for (const each of dots) {
        expect(await each.evaluate(dotShape)).toEqual({
          width: 10,
          height: 10,
          round: true,
          filled: true,
          border: 1,
        });
        const x = (await each.boundingBox())?.x ?? 0;
        if (previous !== null) expect(x - previous).toBeCloseTo(6, 0);
        previous = x;
      }
    };
    await expectMarksInRow("Attention");
    await expectMarksInRow(long);
    // A long name gives way to the colours, not the other way round.
    expect(
      await row(long)
        .getByTestId("row-text")
        .evaluate((element) => element.scrollWidth > element.clientWidth),
    ).toBe(true);

    // Pointed at, the row's tooltip names its Tags; its description says them to a screen reader.
    const attention = row("Attention").getByTestId("open-document");
    await slideTo(window, row("Attention").getByTestId("row-text"));
    // The tooltip carries the file's name, with its extension, then the Tags.
    await expect(attention).toHaveAttribute(
      "title",
      /^Attention\nAttention\.\w+\nTags: Book, Contract, Paper, Report$/,
    );
    await expect(attention).toHaveAccessibleDescription("Tags: Book, Contract, Paper, Report");
    expect(await row("Attention").boundingBox()).toEqual(before[await indexOf(items, "Attention")]);
    // The dots' ring follows the row: the hover wash while pointed at, the sidebar's frame after.
    const ringOf = () =>
      row("Attention")
        .getByTestId("tag-dot")
        .nth(1)
        .evaluate((element) => getComputedStyle(element).boxShadow);
    await expect.poll(ringOf).toContain("rgb(235, 237, 240)");
    await window.mouse.move(700, 400, { steps: 6 });
    await expect.poll(ringOf).toContain("rgb(244, 245, 247)");

    // Reached by keyboard, the row shows its Tags' names, each after its dot, under it.
    const tip = window.getByTestId("tag-names-tip");
    await expect(tip).toBeHidden();
    await slideAndClick(
      window,
      window.getByTestId("browse-views").getByRole("button", {
        name: "Source locations",
      }),
    );
    await tabTo(window, attention);
    expect(await focusRingShows(window)).toBe(true);
    await expect(tip).toBeVisible();
    await expect(tip.getByTestId("tag-names-tip-tag")).toHaveText([
      "Book",
      "Contract",
      "Paper",
      "Report",
    ]);
    expect(
      await tip
        .getByTestId("tag-dot")
        .evaluateAll((all) => all.map((each) => each.getAttribute("data-colour"))),
    ).toEqual(["orange", "red", "purple", "blue"]);
    const rowBox = await row("Attention").boundingBox();
    const tipBox = await tip.boundingBox();
    if (!rowBox || !tipBox) throw new Error("The row or the tip isn't visible.");
    expect(tipBox.y).toBeGreaterThanOrEqual(rowBox.y + rowBox.height);
    await screenshot(window, "row-marks-en");
    // Off the row, it goes; back, it comes; Esc puts it away.
    await window.keyboard.press("Tab");
    await expect(tip).toBeHidden();
    await window.keyboard.press("Shift+Tab");
    await expect(attention).toBeFocused();
    await expect(tip).toBeVisible();
    await window.keyboard.press("Escape");
    await expect(tip).toBeHidden();
    await expect(attention).toBeFocused();
    // A row without Tags has no tip.
    await tabTo(window, row("Field notes").getByTestId("open-document"));
    await expect(tip).toBeHidden();

    // A narrow sidebar: still one 28px line each, the colours inside.
    await setSidebarWidth(window, 165);
    await expect
      .poll(async () => (await window.getByTestId("sidebar").boundingBox())?.width)
      .toBeCloseTo(165, 0);
    await expectMarksInRow("Attention");
    await expectMarksInRow(long);
    await setSidebarWidth(window, 248);
    await expect
      .poll(async () => (await window.getByTestId("sidebar").boundingBox())?.width)
      .toBeCloseTo(248, 0);

    // In Chinese.
    await setLanguage(window, "zh-CN");
    await expect(attention).toHaveAttribute(
      "title",
      "Attention\nAttention.txt\n标签：Book, Contract, Paper, Report",
    );
    await expect(attention).toHaveAccessibleDescription("标签：Book, Contract, Paper, Report");
    await tabTo(window, attention);
    await expect(tip).toBeVisible();
    await screenshot(window, "row-marks-zh");
  } finally {
    await app.close();
  }
});

/** A dot's size, whether it is round, solid or hollow, and its edge's width. */
function dotShape(element: Element) {
  const style = getComputedStyle(element);
  const box = element.getBoundingClientRect();
  return {
    width: Math.round(box.width * 10) / 10,
    height: Math.round(box.height * 10) / 10,
    round: Number.parseFloat(style.borderTopLeftRadius) >= box.width / 2,
    filled: style.backgroundColor !== "rgba(0, 0, 0, 0)",
    border: Number.parseFloat(style.borderTopWidth),
  };
}

/** Where the row with this name is among `items`. */
async function indexOf(items: Locator, name: string): Promise<number> {
  const names = await items.getByTestId("row-text").allTextContents();
  return names.indexOf(name);
}
