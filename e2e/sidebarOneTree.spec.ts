import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { translate } from "../src/shared/i18n";
import type { MessageKey, MessageParams } from "../src/shared/i18n/types";
import {
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dragOnto,
  launchApp,
  pointAt,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  sidebarFolder,
} from "./app";

/*
 * The sidebar as one tree (#233): on every screen the header with its one +,
 * Search and the Library, then the sections that have something in them
 * (Folders, Not in a Folder, Tags), each with its heading. Minds and
 * Documents move onto a Folder or out of every Folder by dragging, one or a
 * mixed selection, or by Move to… (⇧⌘M), each with ⌘Z; the moved Mind's next
 * Question searches its new Folder, and the tree's counts are the core's.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots there.
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
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const makeFolder = (window: Page, name: string) =>
  window.evaluate(
    async (folderName) =>
      (
        await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.createLibraryGroup(
          {
            name: folderName,
            description: "",
          },
        )
      ).id,
    name,
  );

const makeMind = (window: Page, title: string) =>
  window.evaluate(
    async (name) =>
      (
        await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.createMind({
          title: name,
        })
      ).id,
    title,
  );

const addFiles = (window: Page, paths: string[]) =>
  window.evaluate(async (files) => {
    await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addDocuments(files);
  }, paths);

/** Each Document's id, by name. */
const documentIds = (window: Page): Promise<Record<string, string>> =>
  window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return Object.fromEntries((await bridge.listDocuments()).map((each) => [each.name, each.id]));
  });

/** Files Documents in a Folder (null: none), through the core. */
const file = (window: Page, documents: string[], folder: string | null) =>
  window.evaluate(
    async (move) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      for (const id of move.documents) await bridge.assignDocumentGroup(id, move.folder);
    },
    { documents, folder },
  );

/** Moves a Mind to a Folder through the core, as another window would. */
const moveMind = (window: Page, mind: string, folder: string | null) =>
  window.evaluate(
    async (move) => {
      await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.moveToFolder({
        mindIds: [move.mind],
        documentIds: [],
        conversationIds: [],
        folderId: move.folder,
      });
    },
    { mind, folder },
  );

/** Where the core has a Mind or a Document now: its Folder's id, or null. */
const folderOfMind = (window: Page, id: string) =>
  window.evaluate(async (mindId) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.listMinds()).find((each) => each.id === mindId)?.folderId ?? null;
  }, id);
const folderOfDocument = (window: Page, id: string) =>
  window.evaluate(async (documentId) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const { assignments } = await bridge.getLibrary();
    return assignments.find((each) => each.documentId === documentId)?.groupId ?? null;
  }, id);

/** Tags a Document through the core, with a new Tag. */
const tagDocument = (window: Page, documentId: string, name: string) =>
  window.evaluate(
    async (tagging) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const tag = await bridge.createTag({ name: tagging.name });
      await bridge.addTagToDocuments([tagging.documentId], tag.id);
    },
    { documentId, name },
  );

/** Each Folder's count in the tree is the core's: its Minds and its Documents. */
async function expectCountsOfCore(window: Page): Promise<void> {
  const counts = await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const [library, minds] = await Promise.all([bridge.getLibrary(), bridge.listMinds()]);
    return library.groups.map((group) => ({
      name: group.name,
      count:
        minds.filter((mind) => mind.folderId === group.id).length +
        library.assignments.filter((each) => each.groupId === group.id).length,
    }));
  });
  for (const { name, count } of counts) {
    await expect(sidebarFolder(window, name).getByTestId("browse-count")).toHaveText(String(count));
  }
}

/** Clicks the empty foot of the sidebar's tree, where ⌘Z goes to the app's changes. */
async function leaveText(window: Page): Promise<void> {
  const box = await window.getByTestId("sidebar-tree").boundingBox();
  if (!box) throw new Error("The sidebar isn't showing.");
  await window.mouse.move(box.x + box.width / 2, box.y + box.height - 8, { steps: 8 });
  await window.mouse.down();
  await window.mouse.up();
}

async function undo(window: Page): Promise<void> {
  await leaveText(window);
  await window.keyboard.press("ControlOrMeta+z");
}

const mindRow = (window: Page, title: string) =>
  window.getByTestId("mind-row").filter({ hasText: title });
const documentRow = (scope: Page | Locator, name: string) =>
  scope.getByTestId("document-list-item").filter({ hasText: name });
const notInAFolder = (window: Page) => window.getByTestId("not-in-a-folder");
const notInAFolderHeading = (window: Page) => window.getByTestId("not-in-a-folder-section-heading");

/** The composer's scope for the open Mind's next Question: its Folder's chip, or every Document. */
async function expectScope(window: Page, folder: string | null): Promise<void> {
  await window.keyboard.press("ControlOrMeta+j");
  const composer = window.getByTestId("composer");
  if (folder === null) {
    await expect(composer.getByTestId("scope-chip")).toHaveCount(0);
    await expect(composer.getByTestId("composer-scope-all")).toBeVisible();
  } else {
    await expect(composer.getByTestId("scope-chip")).toContainText(folder);
  }
}

const opacityOf = (target: Locator) =>
  target.evaluate((element) => getComputedStyle(element.parentElement ?? element).opacity);

for (const [language, width] of [
  ["en", 1440],
  ["en", 900],
  ["zh-CN", 1440],
  ["zh-CN", 900],
] as const) {
  test(`a fresh sidebar is the header, Search, the Library and the foot; each section comes with its first item, and a heading's + shows on hover and focus (${language}, ${width}px)`, async () => {
    const t = (key: MessageKey, params?: MessageParams) => translate(language, key, params);
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, width, 660);
      await setLanguage(window, language);
      const sidebar = window.getByTestId("sidebar");

      // Fresh: the header's +, Search and the Library, then nothing until there is something.
      await expect(sidebar.getByTestId("plus-menu")).toBeVisible();
      await expect(sidebar.getByTestId("sidebar-search")).toContainText(t("sidebar.search"));
      await expect(sidebar.getByTestId("open-library")).toContainText(t("library.title"));
      for (const section of [
        "folders-section",
        "not-in-a-folder",
        "linked-folders-section",
        "sidebar-tags",
        "browse-views",
      ]) {
        await expect(sidebar.getByTestId(section)).toHaveCount(0);
      }
      await expect(sidebar.getByRole("button", { name: t("assistant.newChat") })).toHaveCount(0);
      await expect(sidebar.getByTestId("sidebar-sections")).toHaveText("");
      // Search, the Library and the foot share the icon and text columns.
      const left = (await sidebar.boundingBox())?.x ?? 0;
      for (const row of [
        sidebar.getByTestId("sidebar-search"),
        sidebar.getByTestId("open-library"),
        sidebar.getByTestId("sidebar-footer").getByRole("button", { name: t("sidebar.settings") }),
      ]) {
        const icon = await row.locator("svg").first().boundingBox();
        expect((icon?.x ?? 0) - left).toBeCloseTo(16, 0);
      }
      await screenshot(window, `one-tree-fresh-${language}-${width}`);

      // A Mind: it is listed under "Minds" while there are no Folders.
      await makeMind(window, "Shortlist");
      const minds = notInAFolder(window);
      await expect(minds.getByTestId("not-in-a-folder-title")).toHaveText(t("sidebar.minds"));
      await expect(minds.getByTestId("mind-list-item")).toHaveText("Shortlist");

      // A Folder: the Folders section comes, and the Minds in none are "Not in a Folder".
      await makeFolder(window, "Finance");
      const folders = window.getByTestId("folders-section");
      await expect(folders.getByTestId("folders-title")).toHaveText(t("library.groups"));
      await expect(minds.getByTestId("not-in-a-folder-title")).toHaveText(t("library.unsorted"));
      await expect(notInAFolderHeading(window)).toHaveAttribute(
        "title",
        t("sidebar.notInFolder.hint"),
      );
      // Folders come before Not in a Folder.
      const foldersBox = await folders.boundingBox();
      const mindsBox = await minds.boundingBox();
      expect((foldersBox?.y ?? 0) + (foldersBox?.height ?? 0)).toBeLessThanOrEqual(
        mindsBox?.y ?? 0,
      );

      // A Tag on a Document: Tags comes last, its rows as many as the Tags in use.
      await writeFile(join(sources, "Budget.txt"), "Budget for the harbour survey.\n");
      await addFiles(window, [join(sources, "Budget.txt")]);
      await expect(documentRow(window, "Budget")).toHaveAttribute("data-status", "ready");
      await tagDocument(window, (await documentIds(window)).Budget ?? "", "Harbour");
      const tags = window.getByTestId("sidebar-tags");
      await expect(tags.getByTestId("tags-toggle")).toContainText(t("tags.title"));
      await expect(tags.getByTestId("sidebar-tag")).toHaveCount(1);
      const tagsBox = await tags.boundingBox();
      const notInBox = await minds.boundingBox();
      expect(tagsBox?.y ?? 0).toBeGreaterThanOrEqual((notInBox?.y ?? 0) + (notInBox?.height ?? 0));
      // Its heading folds it.
      await tags.getByTestId("tags-toggle").click();
      await expect(tags.getByTestId("sidebar-tag")).toHaveCount(0);
      await tags.getByTestId("tags-toggle").click();
      await expect(tags.getByTestId("sidebar-tag")).toHaveCount(1);

      // At rest only the header's + shows; the Folders heading's shows while it is pointed at.
      const add = folders.getByTestId("folders-add");
      await window.mouse.move(width - 40, 300);
      await expect.poll(() => opacityOf(add)).toBe("0");
      await pointAt(window, folders.getByTestId("folders-section-heading"));
      await expect.poll(() => opacityOf(add)).toBe("1");
      await expect(add).toHaveAccessibleName(
        t("sidebar.newFolderKeys", {
          keys: process.platform === "darwin" ? "⇧⌘N" : "Ctrl+Shift+N",
        }),
      );
      await window.mouse.move(width - 40, 300, { steps: 6 });
      await expect.poll(() => opacityOf(add)).toBe("0");
      // And while it has the keyboard's focus.
      await sidebar.getByTestId("open-library").focus();
      for (let step = 0; step < 6; step++) {
        if (await add.evaluate((element) => element === document.activeElement)) break;
        await window.keyboard.press("Tab");
      }
      await expect(add).toBeFocused();
      await expect.poll(() => opacityOf(add)).toBe("1");
      await screenshot(window, `one-tree-sections-${language}-${width}`);

      // It makes a Folder, named in place.
      await window.keyboard.press("Enter");
      const field = window.getByTestId("folder-rename-field");
      await expect(field).toBeFocused();
      await field.fill("Research");
      await field.press("Enter");
      await expect(sidebarFolder(window, "Research")).toHaveCount(1);
    } finally {
      await app.close();
    }
  });
}

for (const [language, width] of [
  ["en", 1440],
  ["zh-CN", 900],
] as const) {
  test(`Minds and Documents drag onto Folders and out of them, alone or selected together, and Move to… does the same by keyboard, each with ⌘Z; the Mind's next Question follows, and the counts are the core's (${language}, ${width}px)`, async () => {
    const t = (key: MessageKey, params?: MessageParams) => translate(language, key, params);
    for (const name of ["Budget", "Ledger", "Survey"]) {
      await writeFile(join(sources, `${name}.txt`), `${name} for the harbour project.\n`);
    }
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    try {
      await setWindowSize(app, window, width, 660);
      await setLanguage(window, language);
      await connectLocalChatModel(window);
      const finance = await makeFolder(window, "Finance");
      const research = await makeFolder(window, "Research");
      const shortlist = await makeMind(window, "Shortlist");
      await addFiles(
        window,
        ["Budget", "Ledger", "Survey"].map((name) => join(sources, `${name}.txt`)),
      );
      await expect(
        window.locator('[data-testid="document-list-item"][data-status="ready"]'),
      ).toHaveCount(3);
      const ids = await documentIds(window);
      const budget = ids.Budget ?? "";
      const ledger = ids.Ledger ?? "";
      await file(window, [budget, ledger], research);
      const financeFolder = sidebarFolder(window, "Finance");
      const researchFolder = sidebarFolder(window, "Research");
      await expectCountsOfCore(window);
      // The Mind is open, as the User works in it.
      await mindRow(window, "Shortlist").getByTestId("mind-list-item").click();
      const undoLine = window.getByTestId("undo-message");

      // One Mind dragged onto a folded Folder: only it lights, with the "into" box and no line between
      // rows, and held there it opens; let go, the Mind is in it.
      await dragOnto(
        window,
        mindRow(window, "Shortlist"),
        financeFolder.getByTestId("folder-row"),
        async () => {
          await expect(financeFolder).toHaveAttribute("data-drop-target", "true");
          await expect(researchFolder).not.toHaveAttribute("data-drop-target", "true");
          const row = financeFolder.locator(":scope > div");
          await expect(row).toHaveCSS("box-shadow", /inset/);
          await expect(row).toHaveCSS("background-color", "rgb(231, 237, 251)");
          await expect(financeFolder).toHaveAttribute("data-expanded", "true");
        },
      );
      await expect(financeFolder.getByTestId("mind-list-item")).toHaveText("Shortlist");
      await expect(undoLine).toHaveText(t("move.done", { name: "Shortlist", folder: "Finance" }));
      expect(await folderOfMind(window, shortlist)).toBe(finance);
      await expectCountsOfCore(window);
      await expectScope(window, "Finance");
      await screenshot(window, `one-tree-dragged-${language}-${width}`);
      // ⌘Z takes it back.
      await undo(window);
      await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");
      expect(await folderOfMind(window, shortlist)).toBeNull();
      await expectCountsOfCore(window);
      await expectScope(window, null);

      // A Mind and two Documents, chosen with ⌘-click, drag together onto another Folder.
      await researchFolder.getByTestId("folder-row").click();
      await expect(researchFolder).toHaveAttribute("data-expanded", "true");
      const shortlistRow = mindRow(window, "Shortlist");
      await shortlistRow.getByTestId("mind-list-item").click();
      for (const name of ["Budget", "Ledger"]) {
        await documentRow(researchFolder, name)
          .getByTestId("open-document")
          .click({ modifiers: ["ControlOrMeta"] });
      }
      await expect(window.getByTestId("sidebar-selection")).toHaveText(
        t("sidebar.selected", { count: 3 }),
      );
      await dragOnto(
        window,
        documentRow(researchFolder, "Ledger"),
        financeFolder.getByTestId("folder-row"),
      );
      await expect(undoLine).toHaveText(t("move.done.items", { count: 3, folder: "Finance" }));
      // Dropped, they are no longer selected.
      await expect(window.getByTestId("sidebar-selection")).toHaveText("");
      expect(await folderOfMind(window, shortlist)).toBe(finance);
      expect(await folderOfDocument(window, budget)).toBe(finance);
      expect(await folderOfDocument(window, ledger)).toBe(finance);
      await expect(financeFolder).toHaveAttribute("data-expanded", "true");
      await expect(financeFolder.getByTestId("mind-list-item")).toHaveText("Shortlist");
      await expect(financeFolder.getByTestId("document-list-item")).toHaveCount(2);
      await expectCountsOfCore(window);
      await screenshot(window, `one-tree-dragged-together-${language}-${width}`);
      await undo(window);
      await expect.poll(() => folderOfDocument(window, budget)).toBe(research);
      expect(await folderOfDocument(window, ledger)).toBe(research);
      expect(await folderOfMind(window, shortlist)).toBeNull();
      await expectCountsOfCore(window);

      // Dragged onto Not in a Folder, a Mind leaves its Folder, and searches everything.
      await moveMind(window, shortlist, finance);
      await expect(financeFolder.getByTestId("mind-list-item")).toHaveText("Shortlist");
      // Clicked, it is the one selected.
      await financeFolder.getByTestId("mind-list-item").click();
      await expect(window.getByTestId("sidebar-selection")).toHaveText("");
      await expectScope(window, "Finance");
      await dragOnto(
        window,
        mindRow(window, "Shortlist"),
        notInAFolderHeading(window),
        async () => {
          await expect(notInAFolderHeading(window)).toHaveAttribute("data-drop-target", "true");
          await expect(notInAFolderHeading(window)).toHaveCSS("box-shadow", /inset/);
        },
      );
      await expect(notInAFolder(window).getByTestId("mind-list-item")).toHaveText("Shortlist");
      await expect(undoLine).toHaveText(
        t("move.done", { name: "Shortlist", folder: t("library.unsorted") }),
      );
      expect(await folderOfMind(window, shortlist)).toBeNull();
      await expectCountsOfCore(window);
      await expectScope(window, null);
      await undo(window);
      await expect.poll(() => folderOfMind(window, shortlist)).toBe(finance);
      await expectScope(window, "Finance");

      // Move to… by keyboard: ⇧⌘M on the row, type, Enter, and the folded Folder opens with it;
      // ⌘Z takes it back.
      const picker = window.getByTestId("move-to-picker");
      if ((await researchFolder.getAttribute("data-expanded")) === "true") {
        await researchFolder.getByTestId("folder-row").click();
      }
      await expect(researchFolder).toHaveAttribute("data-expanded", "false");
      await mindRow(window, "Shortlist").getByTestId("mind-list-item").focus();
      await window.keyboard.press("ControlOrMeta+Shift+KeyM");
      await expect(picker.getByTestId("move-to-search")).toBeFocused();
      await window.keyboard.type("Res");
      await window.keyboard.press("Enter");
      await expect.poll(() => folderOfMind(window, shortlist)).toBe(research);
      await expect(researchFolder).toHaveAttribute("data-expanded", "true");
      await expectCountsOfCore(window);
      await expectScope(window, "Research");
      await undo(window);
      await expect.poll(() => folderOfMind(window, shortlist)).toBe(finance);

      // A name no Folder has: Move to… makes that Folder, which opens with the Mind inside; one ⌘Z
      // takes both back.
      await mindRow(window, "Shortlist").getByTestId("mind-list-item").focus();
      await window.keyboard.press("ControlOrMeta+Shift+KeyM");
      await expect(picker.getByTestId("move-to-search")).toBeFocused();
      await window.keyboard.type("Harbour");
      await expect(picker.getByTestId("move-to-option")).toHaveText([
        t("move.newFolder", { name: "Harbour" }),
      ]);
      await window.keyboard.press("Enter");
      const harbour = sidebarFolder(window, "Harbour");
      await expect(harbour).toHaveAttribute("data-expanded", "true");
      await expect(harbour.getByTestId("mind-list-item")).toHaveText("Shortlist");
      await expect(undoLine).toHaveText(
        t("folder.made.one", { name: "Harbour", item: "Shortlist" }),
      );
      await expectCountsOfCore(window);
      await expectScope(window, "Harbour");
      await undo(window);
      await expect(harbour).toHaveCount(0);
      await expect.poll(() => folderOfMind(window, shortlist)).toBe(finance);

      // …for a Mind and two Documents selected together, and out of every Folder.
      await mindRow(window, "Shortlist").getByTestId("mind-list-item").click();
      await researchFolder.getByTestId("folder-row").focus();
      if ((await researchFolder.getAttribute("data-expanded")) !== "true") {
        await window.keyboard.press("ArrowRight");
      }
      for (const name of ["Budget", "Ledger"]) {
        await documentRow(researchFolder, name)
          .getByTestId("open-document")
          .click({ modifiers: ["ControlOrMeta"] });
      }
      await documentRow(researchFolder, "Ledger").getByTestId("open-document").focus();
      await window.keyboard.press("ControlOrMeta+Shift+KeyM");
      await expect(picker).toContainText(t("move.title", { name: t("move.items", { count: 3 }) }));
      await window.keyboard.type(language === "en" ? "Not in" : "不在");
      await window.keyboard.press("Enter");
      await expect.poll(() => folderOfMind(window, shortlist)).toBeNull();
      expect(await folderOfDocument(window, budget)).toBeNull();
      expect(await folderOfDocument(window, ledger)).toBeNull();
      await expect(undoLine).toHaveText(
        t("move.done.items", { count: 3, folder: t("library.unsorted") }),
      );
      await expectCountsOfCore(window);
      await expect(notInAFolder(window)).toHaveAttribute("data-count", "3");
      await expect(notInAFolder(window)).toHaveAttribute("data-minds", "1");
      await undo(window);
      await expect.poll(() => folderOfDocument(window, budget)).toBe(research);
      expect(await folderOfMind(window, shortlist)).toBe(finance);
      await expectCountsOfCore(window);
    } finally {
      await app.close();
    }
  });
}

for (const [language, width] of [
  ["en", 1440],
  ["zh-CN", 900],
] as const) {
  test(`every row kind in the tree has the shared menu, renames in place and selects the shared way (${language}, ${width}px)`, async () => {
    await writeFile(join(sources, "Budget.txt"), "Budget for the harbour project.\n");
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, width, 660);
      await setLanguage(window, language);
      await makeFolder(window, "Finance");
      await makeMind(window, "Shortlist");
      await addFiles(window, [join(sources, "Budget.txt")]);
      const budget = documentRow(window, "Budget");
      await expect(budget).toHaveAttribute("data-status", "ready");
      const folder = sidebarFolder(window, "Finance");
      const rows: { row: Locator; look: Locator; main: Locator; actions: string }[] = [
        {
          row: folder,
          look: folder.locator(":scope > div"),
          main: folder.getByTestId("folder-row"),
          actions: "folder-actions",
        },
        {
          row: mindRow(window, "Shortlist"),
          look: mindRow(window, "Shortlist"),
          main: mindRow(window, "Shortlist").getByTestId("mind-list-item"),
          actions: "mind-actions",
        },
        {
          row: budget,
          look: budget,
          main: budget.getByTestId("open-document"),
          actions: "document-file-actions",
        },
      ];
      for (const { row, main, actions } of rows) {
        // The shared menu, by a right-click and by Shift+F10, Delete last.
        await main.click({ button: "right" });
        const menu = row.getByTestId(actions);
        await expect(menu).toBeVisible();
        await expect(menu.getByRole("menuitem").last()).toHaveAttribute("data-testid", /\.delete$/);
        await window.keyboard.press("Escape");
        await main.focus();
        await window.keyboard.press("Shift+F10");
        await expect(menu).toBeVisible();
        await window.keyboard.press("Escape");
        // F2 renames in place; Esc leaves the name as it was.
        await main.focus();
        await window.keyboard.press("F2");
        const field = window.getByTestId("sidebar").locator("input:not([type=file])");
        await expect(field).toBeFocused();
        await window.keyboard.press("Escape");
        await expect(field).toHaveCount(0);
        // ⌘-click selects it with the rest, without opening it.
        await main.click({ modifiers: ["ControlOrMeta"] });
        await expect(main).toHaveAttribute("aria-selected", "true");
      }
      await expect(window.getByTestId("sidebar-selection")).toHaveText(
        translate(language, "sidebar.selected", { count: 3 }),
      );
      // Several selected look alike, on the accent's wash.
      for (const { look } of rows)
        await expect(look).toHaveCSS("background-color", "rgb(231, 237, 251)");
      await window.keyboard.press("Escape");
      await expect(window.getByTestId("sidebar-selection")).toHaveText("");
      // Enter on a focused row opens it: a Mind in its tab, a Folder in place.
      await mindRow(window, "Shortlist").getByTestId("mind-list-item").focus();
      await window.keyboard.press("Enter");
      await expect(window.getByTestId("mind-tab").getByTestId("tab-title")).toHaveText([
        "Shortlist",
      ]);
      await folder.getByTestId("folder-row").focus();
      await window.keyboard.press("Enter");
      await expect(folder).toHaveAttribute("data-expanded", "true");
    } finally {
      await app.close();
    }
  });
}

/** The New menu's four rows, then what is under More, each as ⌘K names its command; Record a meeting is off until turned on. */
const NEW_MENU_ROWS: readonly MessageKey[] = [
  "menu.newMind",
  "sidebar.plus.addDocuments",
  "menu.newFolder",
  "sidebar.plus.more",
];
const MORE_COMMANDS: readonly MessageKey[] = [
  "sidebar.plus.linkFolder",
  "command.compareDocuments",
  "verify.command",
  "sidebar.plus.addWebPage",
  "wordImport.command",
  "library.chooseStarters",
  "command.sources.find",
];

for (const [language, width] of [
  ["en", 1440],
  ["zh-CN", 900],
] as const) {
  test(`the New menu's four rows and what is under More open by mouse and keyboard, and each is in ⌘K too (${language}, ${width}px)`, async () => {
    const t = (key: MessageKey) => translate(language, key);
    const { app, window } = await launchApp(dataDir);
    try {
      await setWindowSize(app, window, width, 660);
      await setLanguage(window, language);
      const plus = window.getByTestId("plus-menu");
      const rows = window.getByTestId("plus-menu-items").getByRole("menuitem");
      const more = window.getByTestId("plus-more-menu").getByRole("menuitem");

      // By mouse: four rows in order, and More lists the rest.
      await pointAt(window, plus);
      await window.mouse.down();
      await window.mouse.up();
      await expect(rows).toHaveCount(NEW_MENU_ROWS.length);
      for (const [index, key] of NEW_MENU_ROWS.entries()) {
        await expect(rows.nth(index)).toContainText(t(key));
      }
      await pointAt(window, window.getByTestId("plus-more"));
      await expect(more).toHaveCount(MORE_COMMANDS.length);
      await window.keyboard.press("Escape");
      await window.keyboard.press("Escape");
      await expect(rows).toHaveCount(0);

      // By keyboard: Enter opens it on its first row, ↓ walks to More, → opens it, Esc closes each.
      await plus.focus();
      await window.keyboard.press("Enter");
      await expect(window.getByTestId("plus-new-mind")).toBeFocused();
      for (let step = 1; step < NEW_MENU_ROWS.length; step++)
        await window.keyboard.press("ArrowDown");
      await expect(window.getByTestId("plus-more")).toBeFocused();
      await window.keyboard.press("ArrowRight");
      await expect(more.first()).toBeFocused();
      await window.keyboard.press("Escape");
      await expect(window.getByTestId("plus-more")).toBeFocused();
      await window.keyboard.press("Escape");
      await expect(plus).toBeFocused();

      for (const key of [...NEW_MENU_ROWS.slice(0, -1), ...MORE_COMMANDS]) {
        const label = t(key).replace(/…$/, "");
        await window.keyboard.press("ControlOrMeta+k");
        const search = window.getByTestId("palette-search");
        await expect(search).toBeFocused();
        await window.keyboard.type(`>${label}`);
        await expect(
          window.getByTestId("palette-item").filter({ hasText: label }).first(),
        ).toBeVisible();
        await window.keyboard.press("Escape");
        await expect(search).toHaveCount(0);
      }
    } finally {
      await app.close();
    }
  });
}
