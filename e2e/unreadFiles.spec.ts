import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { encryptedOfficeFile } from "../tests/helpers/office";
import { buildPdf } from "../tests/helpers/pdf";
import {
  askInAssistant,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setGroupBeside,
  setLanguage,
  setWindowSize,
} from "./app";

/**
 * Every Answer says which files in its scope couldn't be read, and why (#740):
 * a Folder with a .doc, a password-protected .docx and a scan, besides a file
 * that reads. "3 files couldn't be read · Show" lists each with its reason,
 * and the Library shows the file in a format not read. Nothing shows when
 * everything was read. English and Chinese.
 */
let dataDir: string;
let library: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  library = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(library);
});

const FILES: Record<string, string | Uint8Array> = {
  "Tides.txt": "Spring tides happen at new moon and at full moon.\n",
  "Old/Contract.doc": "binary",
  "Locked.docx": encryptedOfficeFile(),
  "Scan.pdf": buildPdf([{ image: true }]),
};

async function start(language: "en" | "zh-CN", files: Record<string, string | Uint8Array> = FILES) {
  for (const [name, contents] of Object.entries(files)) {
    const path = join(library, name);
    await mkdir(join(path, ".."), { recursive: true });
    await writeFile(path, contents);
  }
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home: library });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await connectLocalChatModel(window);
  await setLanguage(window, language);
  await window.evaluate(async (path) => {
    await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addLinkedFolder(path);
  }, library);
  // Read to the end: the scan has no text, the .docx wants a password.
  await expect
    .poll(() =>
      window.evaluate(
        async () =>
          (
            await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments()
          ).filter((each) => each.status === "queued" || each.status === "extracting").length,
      ),
    )
    .toBe(0);
  await newMind(window);
  return { app, window };
}

const WORDS = {
  en: {
    count: "3 files couldn't be read",
    reasons: [
      ["Contract.doc", /Format not read yet \(\.doc\)/],
      ["Locked.docx", /Password-protected/],
      ["Scan.pdf", /A scan with no text yet/],
    ],
  },
  "zh-CN": {
    count: "3 个文件未能读取",
    reasons: [
      ["Contract.doc", /暂不支持此格式（\.doc）/],
      ["Locked.docx", /已设置密码/],
      ["Scan.pdf", /扫描件，还没有文字/],
    ],
  },
} as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`an Answer says which files in its scope couldn't be read, and why (${language})`, async () => {
    test.setTimeout(120_000);
    const { app, window } = await start(language);
    const answer = await askInAssistant(window, "What do tides do?");
    const line = answer.getByTestId("unread-files");
    await expect(line).toContainText(WORDS[language].count);
    await expect(line.getByTestId("unread-file")).toHaveCount(0);
    await line.getByTestId("unread-files-show").click();
    const rows = line.getByTestId("unread-file");
    await expect(rows).toHaveCount(3);
    for (const [name, reason] of WORDS[language].reasons) {
      await expect(rows.filter({ hasText: name })).toHaveText(reason);
    }
    await expect(rows.filter({ hasText: "Contract.doc" })).toHaveAttribute("data-reason", "format");
    await expect(rows.filter({ hasText: "Locked.docx" })).toHaveAttribute(
      "data-reason",
      "password",
    );
    await expect(rows.filter({ hasText: "Scan.pdf" })).toHaveAttribute("data-reason", "scan");

    // The Library shows the file in a format not read, with the same reason.
    await setGroupBeside(window, true);
    const unread = window.getByTestId("library-unread");
    await expect(unread).toContainText(
      language === "en" ? "1 file couldn't be read" : "1 个文件未能读取",
    );
    await unread.getByTestId("library-unread-show").click();
    await expect(unread.getByTestId("unread-file")).toHaveText(
      language === "en" ? /Contract\.doc: Format not read yet/ : /Contract\.doc：暂不支持此格式/,
    );
    await app.close();
  });
}

test("nothing shows when everything in scope was read", async () => {
  const { app, window } = await start("en", { "Tides.txt": FILES["Tides.txt"] as string });
  const answer = await askInAssistant(window, "What do tides do?");
  await expect(answer.getByTestId("unread-files")).toHaveCount(0);
  await app.close();
});
