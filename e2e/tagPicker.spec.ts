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
 * The Tag UI (Library rows, the Tag picker, bulk tagging, review, the Tags
 * filter and Manage Tags), driven as a person would: the mouse slides to
 * what it clicks, in steps, and checks it is what lies under the pointer;
 * typing goes through the keyboard. Set INCARNAMIND_SCREENSHOTS to a folder
 * to save screenshots there.
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

/**
 * Slides the mouse to the middle of `target` in steps, checks that it is
 * what lies under the pointer (so a hover-revealed control really shows),
 * and clicks there.
 */
async function slideAndClick(
  window: Page,
  target: Locator,
  { shift = false }: { shift?: boolean } = {},
): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("Nothing to click: the target has no box.");
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
  if (shift) await window.keyboard.down("Shift");
  await window.mouse.down();
  await window.mouse.up();
  if (shift) await window.keyboard.up("Shift");
}

/** Slides the mouse onto `target`, in steps, and leaves it there (pointing at it). */
async function slideTo(window: Page, target: Locator): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("Nothing to point at: the target has no box.");
  await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 12 });
}

const bridge = (window: Page) =>
  window.evaluateHandle(() => (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind);

/** Writes text files and adds them; waits until each is ready. */
async function addTextDocuments(window: Page, files: Record<string, string>): Promise<void> {
  const paths: string[] = [];
  for (const [name, text] of Object.entries(files)) {
    const path = join(sources, name);
    await writeFile(path, text);
    paths.push(path);
  }
  await addDocuments(window, paths);
  await expect(
    window.locator('[data-testid="document-list-item"][data-status="ready"]'),
  ).toHaveCount(paths.length, { timeout: 20_000 });
}

/** The Library's row of the Document with this name. */
const rowOf = (library: Locator, name: string) =>
  library
    .getByTestId("library-document")
    .filter({ has: library.page().getByRole("button", { name, exact: true }) });

const tokens = (popover: Locator) => popover.getByTestId("tag-token");

test("Tags are added, created and removed with the keyboard, on one Document and on several at once", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await addTextDocuments(window, {
      "March invoice.txt": "Invoice 2026-03 for consulting. Total due: 1,200.",
      "Q1 review.txt": "Quarterly review of sales across regions.",
      "Q2 review.txt": "Second quarter review of sales across regions.",
    });
    await slideAndClick(window, window.getByTestId("open-library"));
    const library = window.getByTestId("library");
    const invoice = rowOf(library, "March invoice");

    // A row without Tags offers "+ Add tags"; the picker opens with the focus in its field.
    const add = invoice.getByTestId("document-tags-menu");
    await expect(add).toHaveText("+ Add tags");
    await slideAndClick(window, add);
    const picker = invoice.getByTestId("document-tags-popover");
    const field = picker.getByTestId("tag-picker-input");
    await expect(field).toBeFocused();

    // Typing filters; Enter adds the first match.
    await window.keyboard.type("inv");
    await expect(picker.getByTestId("tag-option")).toHaveCount(1);
    await window.keyboard.press("Enter");
    await expect(tokens(picker)).toHaveText(["Invoice"]);
    await expect(field).toHaveValue("");

    // A new name is offered as a new Tag; Shift+Enter gives it a description first.
    await window.keyboard.type("Receipts");
    await expect(picker.getByTestId("tag-option-create")).toHaveText(/Create “Receipts”/);
    await window.keyboard.press("Shift+Enter");
    const describe = picker.getByTestId("tag-describe");
    await expect(describe).toContainText("Automatic tagging reads it");
    await window.keyboard.type("Proof that a shop or restaurant was paid.");
    await window.keyboard.press("Enter");
    await expect(tokens(picker)).toHaveText(["Invoice", "Receipts"]);
    const created = await (await bridge(window)).evaluate(async (core) =>
      (await core.listTags()).find((tag) => tag.name === "Receipts"),
    );
    expect(created?.description).toBe("Proof that a shop or restaurant was paid.");

    // Plain Enter creates at once; Backspace picks the last Tag, and again takes it off.
    await expect(field).toBeFocused();
    await window.keyboard.type("Urgent");
    await window.keyboard.press("Enter");
    await expect(tokens(picker)).toHaveText(["Invoice", "Receipts", "Urgent"]);
    await window.keyboard.press("Backspace");
    await expect(tokens(picker)).toHaveCount(3);
    await window.keyboard.press("Backspace");
    await expect(tokens(picker)).toHaveText(["Invoice", "Receipts"]);
    // × on a Tag takes it off too.
    await slideAndClick(
      window,
      tokens(picker).filter({ hasText: "Receipts" }).getByTestId("tag-token-remove"),
    );
    await expect(tokens(picker)).toHaveText(["Invoice"]);
    await screenshot(window, "picker-one-document");
    await window.keyboard.press("Escape");
    await expect(picker).toBeHidden();
    await expect(add).toBeFocused();

    // The row shows its Tags as chips, the User's without a mark.
    const chips = invoice.getByTestId("tag-chip");
    await expect(chips).toHaveText(["Invoice"]);
    await expect(chips).toHaveAttribute("data-source", "user");

    // Several at once: rows are selected with their checkboxes, Shift-click for a range.
    const q1 = rowOf(library, "Q1 review");
    const q2 = rowOf(library, "Q2 review");
    await slideAndClick(window, q1.getByTestId("library-select"));
    await slideAndClick(window, q2.getByTestId("library-select"));
    const bar = library.getByTestId("library-selection");
    await expect(bar).toContainText("2 selected");
    await slideAndClick(window, bar.getByTestId("library-selection-tags"));
    const bulk = bar.getByTestId("selection-tags-popover");
    await expect(bulk.getByTestId("tag-picker-input")).toBeFocused();
    await window.keyboard.type("rep");
    await window.keyboard.press("Enter");
    await expect(tokens(bulk)).toHaveText(["Report"]);
    await expect(bulk.getByRole("option", { name: "Report", exact: true })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    // Invoice is on none of the two.
    await expect(bulk.getByRole("option", { name: "Invoice", exact: true })).toHaveAttribute(
      "data-coverage",
      "none",
    );
    await screenshot(window, "picker-bulk");
    await window.keyboard.press("Escape");
    await expect(q1.getByTestId("tag-chip")).toHaveText(["Report"]);
    await expect(q2.getByTestId("tag-chip")).toHaveText(["Report"]);
    await expect(chips).toHaveText(["Invoice"]);

    // Select all: a Tag on some of them is marked so, and choosing it puts it on all.
    await slideAndClick(window, bar.getByTestId("library-select-all"));
    await expect(bar).toContainText("3 selected");
    await slideAndClick(window, bar.getByTestId("library-selection-tags"));
    const report = bulk.getByRole("option", { name: "Report", exact: true });
    await expect(report).toHaveAttribute("data-coverage", "some");
    await slideAndClick(window, report);
    await expect(report).toHaveAttribute("data-coverage", "all");
    // Chosen again, a Tag on all of them comes off all of them.
    await slideAndClick(window, report);
    await expect(report).toHaveAttribute("data-coverage", "none");
    await window.keyboard.press("Escape");
    await expect(library.getByTestId("tag-chip").filter({ hasText: "Report" })).toHaveCount(0);
    await slideAndClick(window, bar.getByTestId("library-selection-clear"));
    await expect(bar).toHaveCount(0);

    // The sidebar's row offers the same picker, from its Tags button.
    const item = window
      .getByTestId("document-list-item")
      .filter({ has: window.getByText("March invoice", { exact: true }) });
    await slideTo(window, item);
    await slideAndClick(window, item.getByTestId("document-tags-menu"));
    const sidebarPicker = item.getByTestId("document-tags-popover");
    await expect(sidebarPicker.getByTestId("tag-picker-input")).toBeFocused();
    await expect(tokens(sidebarPicker)).toHaveText(["Invoice"]);
    await screenshot(window, "picker-sidebar");
    await window.keyboard.press("Escape");
    await expect(sidebarPicker).toBeHidden();
  } finally {
    await app.close();
  }
});

/**
 * A local decision server like Ollama's `/v1/systemone`: Finance for an
 * invoice, Reports otherwise; Invoice in the review band for an invoice,
 * Report likely for a report.
 */
async function decisionServer(): Promise<{ server: Server; readonly requests: number }> {
  let requests = 0;
  const server = createServer(async (request, response) => {
    let raw = "";
    for await (const chunk of request) raw += chunk;
    const body = raw ? JSON.parse(raw) : {};
    response.setHeader("content-type", "application/json");
    if (request.url !== "/v1/systemone") {
      response.end(JSON.stringify({ models: [] }));
      return;
    }
    requests++;
    const text = JSON.stringify(body.state).toLowerCase();
    const answers = Object.fromEntries(
      Object.entries(
        body.questions as Record<
          string,
          { type: string; instructions: string; criteria?: Record<string, string> }
        >,
      ).map(([id, question]) => {
        if (question.type === "choice") {
          const wanted = text.includes("invoice") ? "Finance" : "Reports";
          const choice =
            Object.entries(question.criteria ?? {}).find(([, value]) =>
              value.startsWith(wanted),
            )?.[0] ?? "__unsorted__";
          return [id, { type: "choice", choice }];
        }
        const noul = question.instructions.includes("“Invoice”")
          ? text.includes("invoice")
            ? 0.65
            : 0.01
          : question.instructions.includes("“Report”")
            ? text.includes("report")
              ? 0.95
              : 0.01
            : 0.01;
        return [id, { type: "noul", noul }];
      }),
    );
    response.end(JSON.stringify({ answers }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    server,
    get requests() {
      return requests;
    },
  };
}

test("a Tag automatic tagging wasn't sure of is confirmed or removed in one click, and Tags filter the Library", async () => {
  const decisions = await decisionServer();
  const { server } = decisions;
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await addTextDocuments(window, {
      "March invoice.txt": "Invoice 2026-03 for consulting. Total due: 1,200.",
      "April invoice.txt": "Invoice 2026-04 for consulting. Total due: 900.",
      "Q1 report.txt": "Quarterly report: sales grew in every region.",
      "Q2 report.md": "# Q2 report\n\nSales held steady.",
    });
    const port = (server.address() as AddressInfo).port;
    await (await bridge(window)).evaluate(async (core, url) => {
      await core.addLibraryStarterGroups(["reports", "finance"]);
      await core.saveLibrarySettings({
        classifier: { kind: "ollama", baseUrl: url, modelId: "tev1:0.8b" },
        automatic: false,
      });
    }, `http://127.0.0.1:${port}`);
    await slideAndClick(window, window.getByTestId("open-library"));
    const library = window.getByTestId("library");
    await slideAndClick(window, library.getByRole("button", { name: "Organize", exact: true }));
    const march = rowOf(library, "March invoice");
    const april = rowOf(library, "April invoice");
    // Applied, with an amber mark, waiting for the User.
    const marchInvoice = march.locator('[data-testid="tag-chip"][data-needs-review="true"]');
    await expect(marchInvoice).toHaveText("Invoice", { timeout: 15_000 });
    await expect(april.locator('[data-testid="tag-chip"][data-needs-review="true"]')).toHaveCount(
      1,
    );
    await expect(rowOf(library, "Q1 report").getByTestId("tag-chip")).toHaveAttribute(
      "data-source",
      "automatic",
    );
    await screenshot(window, "library-review");

    // The viewer's header shows the open Document's Tags too.
    await slideAndClick(window, march.getByRole("button", { name: "March invoice", exact: true }));
    const viewer = window.getByTestId("viewer");
    await expect(viewer).toBeVisible();
    const headerTags = viewer.getByTestId("viewer-header").getByTestId("document-tags");
    await expect(headerTags.getByTestId("tag-chip")).toHaveText(/Invoice/);
    await expect(headerTags.getByTestId("tag-chip")).toHaveAttribute("data-colour", "rose");
    // In the header, ✓ and × of a Tag awaiting review are always there, and the picker opens.
    await expect(headerTags.getByTestId("confirm-document-tag")).toBeVisible();
    await slideAndClick(window, headerTags.getByTestId("document-tags-menu"));
    await expect(headerTags.getByTestId("tag-picker-input")).toBeFocused();
    await window.keyboard.press("Escape");
    await expect(viewer).toBeVisible();
    await screenshot(window, "viewer");
    await slideAndClick(window, viewer.getByTestId("viewer-close"));
    await expect(viewer).toBeHidden();

    // ✓ and × show once the chip is pointed at; ✓ keeps the Tag, as the User's.
    await expect(marchInvoice.getByTestId("confirm-document-tag")).toBeHidden();
    await slideTo(window, marchInvoice);
    await slideAndClick(window, marchInvoice.getByTestId("confirm-document-tag"));
    const kept = march.getByTestId("tag-chip");
    await expect(kept).toHaveAttribute("data-source", "user");
    await expect(kept).not.toHaveAttribute("data-needs-review");
    // × removes it, and Organize doesn't bring it back.
    const aprilInvoice = april.getByTestId("tag-chip");
    await slideTo(window, aprilInvoice);
    await slideAndClick(window, aprilInvoice.getByTestId("reject-document-tag"));
    await expect(april.getByTestId("tag-chip")).toHaveCount(0);
    const asked = decisions.requests;
    await slideAndClick(window, library.getByRole("button", { name: "Organize", exact: true }));
    await expect.poll(() => decisions.requests, { timeout: 15_000 }).toBe(asked + 4);
    await expect(library.getByRole("button", { name: "Organize", exact: true })).toBeEnabled();
    await expect(april.getByTestId("tag-chip")).toHaveCount(0);
    await expect(march.getByTestId("tag-chip")).toHaveAttribute("data-source", "user");

    // A chip's name filters the Library by its Tag; the Tags filter says so, and so does the sidebar.
    const q1 = rowOf(library, "Q1 report");
    await slideAndClick(window, q1.getByTestId("tag-chip-filter"));
    const rows = library.getByTestId("library-document");
    await expect(rows).toHaveCount(2);
    const tagsFilter = library.locator('[data-testid="library-filter"][data-facet="tag"]');
    await expect(tagsFilter).toHaveText("Tags: Report");
    // Its options show each Tag's colour; "Needs review" has none.
    await slideAndClick(window, tagsFilter);
    const tagMenu = library.locator('[data-testid="library-filter-menu"][data-facet="tag"]');
    await expect(
      tagMenu.locator(
        '[data-testid="library-filter-option"][data-value="needs-review"] [data-testid="tag-swatch"]',
      ),
    ).toHaveCount(0);
    await expect(
      tagMenu
        .getByTestId("library-filter-option")
        .filter({ hasText: "Report" })
        .getByTestId("tag-swatch"),
    ).toHaveAttribute("data-colour", "petrol");
    await screenshot(window, "filter-colours");
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("tag-filter-active")).toContainText("Report");
    // With a Tag filter, a Question from here asks about exactly these Documents.
    await expect(library.getByTestId("library-ask")).toHaveAttribute("data-scope", "documents");
    await expect(library.getByTestId("library-ask")).toHaveText("Ask about these 2 Documents");
    // It combines with the other filters.
    await slideAndClick(
      window,
      library.locator('[data-testid="library-filter"][data-facet="format"]'),
    );
    await slideAndClick(
      window,
      library.getByRole("menuitemcheckbox", { name: "Markdown, 1 Document" }),
    );
    await window.keyboard.press("Escape");
    await expect(rows).toHaveCount(1);
    await expect(rows).toContainText("Q2 report");
    await screenshot(window, "library-filtered");
    await slideAndClick(window, library.getByTestId("library-filters-clear"));
    await expect(rows).toHaveCount(4);
    await expect(window.getByTestId("tag-filter-active")).toHaveCount(0);

    // In Chinese, the chips, picker and filter speak Chinese; Tag names are the User's.
    await (await bridge(window)).evaluate((core) =>
      core.updateSettings({ user: { language: "zh-CN" } }),
    );
    await expect(library.locator('[data-testid="library-filter"][data-facet="tag"]')).toHaveText(
      "标签",
    );
    await screenshot(window, "library-zh");
    await slideTo(window, q1);
    await slideAndClick(window, q1.getByTestId("document-tags-menu"));
    const picker = q1.getByTestId("document-tags-popover");
    await expect(picker.getByTestId("tag-picker-input")).toBeFocused();
    await window.keyboard.type("新");
    await expect(picker.getByTestId("tag-option-create")).toHaveText(/新建“新”/);
    await screenshot(window, "picker-zh");
    await window.keyboard.press("Escape");
    await slideAndClick(window, q1.getByRole("button", { name: "Q1 report", exact: true }));
    await expect(window.getByTestId("viewer")).toBeVisible();
    await screenshot(window, "viewer-zh");
  } finally {
    await app.close();
    server.closeAllConnections();
    server.close();
  }
});

test("Manage Tags shows how many Documents carry each, edits, merges and deletes, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await addTextDocuments(window, {
      "Lab notes.txt": "Notes from the lab.",
      "Field notes.txt": "Notes from the field.",
    });
    const ids = await (await bridge(window)).evaluate(async (core) => {
      const tags = await core.listTags();
      const notes = tags.find((tag) => tag.name === "Notes");
      const report = tags.find((tag) => tag.name === "Report");
      if (!notes || !report) throw new Error("A preset is missing.");
      const documents = await core.listDocuments();
      await core.addTagToDocuments(
        documents.map((document) => document.id),
        notes.id,
      );
      const lab = documents.find((document) => document.name === "Lab notes");
      if (lab) await core.addDocumentTag(lab.id, report.id);
      return { Notes: notes.id, Report: report.id };
    });
    await slideAndClick(window, window.getByTestId("open-library"));
    const library = window.getByTestId("library");
    await slideAndClick(window, library.getByRole("button", { name: "Manage tags" }));
    const dialog = window.getByTestId("tags-dialog");
    const row = (name: keyof typeof ids) =>
      dialog.locator(`[data-testid="tag-row"][data-tag-id="${ids[name]}"]`);
    await expect(row("Notes").getByTestId("tag-count")).toHaveText("2 Documents");
    await expect(row("Report").getByTestId("tag-count")).toHaveText("1 Document");

    // The explanation names Organize's routes as they are.
    await expect(dialog).toContainText("your connected chat model, Auto · local models");
    // Each Tag has its colour; editing says the description guides automatic tagging, and
    // offers the palette.
    await expect(row("Report").getByTestId("tag-row-name")).toHaveAttribute(
      "data-colour",
      "petrol",
    );
    await slideAndClick(window, row("Report").getByTestId("edit-tag"));
    await expect(dialog).toContainText("Automatic tagging goes by the name and description");
    const swatches = row("Report").getByTestId("tag-colour");
    await expect(swatches).toHaveCount(8);
    await slideAndClick(window, row("Report").locator('label:has([data-colour="rose"])'));
    await expect(
      row("Report").locator('[data-testid="tag-colour"][data-colour="rose"]'),
    ).toBeChecked();
    await screenshot(window, "manage-tags-colour");
    await slideAndClick(window, row("Report").getByTestId("save-tag"));
    await expect(row("Report").getByTestId("tag-row-name")).toHaveAttribute("data-colour", "rose");
    await window.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
    // The Library's chips take it.
    await expect(
      library.locator('[data-testid="tag-chip"]').filter({ hasText: "Report" }),
    ).toHaveAttribute("data-colour", "rose");
    await slideAndClick(window, library.getByRole("button", { name: "Manage tags" }));

    // Merging Report into Notes: its Document keeps one Tag, Notes.
    await slideAndClick(window, row("Report").getByTestId("merge-tag"));
    await row("Report").getByTestId("merge-target").selectOption({ label: "Notes" });
    await expect(row("Report")).toContainText("Its Documents get “Notes” instead");
    await slideAndClick(window, row("Report").getByTestId("confirm-merge-tag"));
    await expect(row("Report")).toHaveCount(0);
    await expect(row("Notes").getByTestId("tag-count")).toHaveText("2 Documents");
    await screenshot(window, "manage-tags");

    // The count shows its Documents in the Library.
    await slideAndClick(window, row("Notes").getByTestId("tag-count"));
    await expect(dialog).toBeHidden();
    await expect(library.locator('[data-testid="library-filter"][data-facet="tag"]')).toHaveText(
      "Tags: Notes",
    );

    // Deleting says how many Documents lose it.
    await slideAndClick(window, library.getByRole("button", { name: "Manage tags" }));
    await slideAndClick(window, row("Notes").getByTestId("delete-tag"));
    await expect(row("Notes")).toContainText("It is taken off 2 Documents.");
    await slideAndClick(window, row("Notes").getByTestId("confirm-delete-tag"));
    await expect(row("Notes")).toHaveCount(0);
    await slideAndClick(window, dialog.getByRole("button", { name: "Done" }));
    // The filter by the deleted Tag goes with it.
    await expect(library.getByTestId("library-document")).toHaveCount(2);

    // In Chinese.
    await (await bridge(window)).evaluate((core) =>
      core.updateSettings({ user: { language: "zh-CN" } }),
    );
    await slideAndClick(window, library.getByRole("button", { name: "管理标签" }));
    await expect(dialog.getByTestId("tag-row").first().getByTestId("tag-count")).toHaveText(
      "0 个文档",
    );
    await screenshot(window, "manage-tags-zh");
  } finally {
    await app.close();
  }
});
