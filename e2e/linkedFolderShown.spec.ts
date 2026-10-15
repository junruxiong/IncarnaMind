import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  linkedFolderRow,
  linkFolderFromSidebar,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/**
 * After linking a folder, the sidebar shows where it is (#763): the view with
 * the Linked folder's folders as on disk, "Folders on disk" (磁盘上的文件夹),
 * which is not the User's own Folders, and the folder's row is in sight and
 * washed for a moment. Folders and Folders on disk never share a name.
 */
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

const WORDS = {
  en: { views: ["Folders", "Folders on disk"], name: "contracts" },
  "zh-CN": { views: ["文件夹", "磁盘上的文件夹"], name: "contracts" },
} as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`linking a folder switches to the view that shows it and washes its row (${language})`, async () => {
    test.setTimeout(120_000);
    const words = WORDS[language];
    const linked = join(sources, words.name);
    await mkdir(join(linked, "2026"), { recursive: true });
    await writeFile(join(linked, "Lease.txt"), "The lease term is five years.\n");
    await writeFile(join(linked, "2026", "Renewal.txt"), "The renewal term is two years.\n");
    await writeFile(join(sources, "Own.txt"), "A file of my own.\n");
    const { app, window } = await launchApp(dataDir, { home: sources });
    await setWindowSize(app, window, 1440, 900);
    await dismissChatSetup(window);
    await setLanguage(window, language);
    // Something to browse besides Minds, so the switcher is there, and Folders is its first view.
    await addDocuments(window, [join(sources, "Own.txt")]);
    await window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await core.createLibraryGroup({ name: "Mine", description: "" });
    });
    const views = window.getByTestId("browse-views");
    await expect(views.getByRole("button").first()).toHaveAttribute("aria-pressed", "true");

    await linkFolderFromSidebar(app, window, linked);

    // The sidebar is on the view that shows it, and the two views have two names.
    const onDisk = views.getByRole("button", { name: words.views[1], exact: true });
    await expect(onDisk).toHaveAttribute("aria-pressed", "true");
    await expect(views.getByRole("button", { name: words.views[0], exact: true })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    const row = linkedFolderRow(window, words.name);
    await expect(row).toBeVisible();
    await expect(row).toHaveAttribute("data-just-linked", "true");
    // A moment later the wash is gone, and the view stays where it was put.
    await expect(row).not.toHaveAttribute("data-just-linked", "true", { timeout: 10_000 });
    await expect(onDisk).toHaveAttribute("aria-pressed", "true");
    await app.close();
  });
}
