import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  frontMind,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * "This agreement" and "这份劳动合同" mean the one Document the User points at (#1334): with a
 * Chinese labour contract and an English employment agreement in the Library, a short summary of
 * either, asked in an empty Mind, is written from that one alone into the page, with its line, in
 * English and Chinese. The model is the smoke tests' fake.
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

const CONTRACTS = [
  [
    "劳动合同",
    "# 劳动合同书\n\n乙方每年享有带薪年休假十天。任何一方提前三十日书面通知对方，可以解除本合同。\n",
  ],
  [
    "Employment agreement",
    "# Employment agreement\n\nTwenty-five days of paid leave a year. Either party may end this agreement with one month's written notice.\n",
  ],
] as const;

for (const [language, request, meant, other, wrote] of [
  [
    "en",
    "Write a short summary of this agreement.",
    "Employment agreement",
    "劳动合同",
    "Wrote 1 paragraph",
  ],
  [
    "zh-CN",
    "请为这份劳动合同写一个简短的摘要。",
    "劳动合同",
    "Employment agreement",
    "写入了 1 个段落",
  ],
] as const) {
  test(`“${request}” is written from “${meant}” alone into the empty Mind (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    const paths = await Promise.all(
      CONTRACTS.map(async ([name, text]) => {
        const path = join(sources, `${name}.md`);
        await writeFile(path, text);
        return path;
      }),
    );
    await addDocuments(window, paths);
    await newMind(window);

    await askInAssistant(window, request);

    const editor = frontMind(window).getByTestId("mind-editor");
    await expect(editor).toContainText(`Written from ${meant}`);
    await expect(editor).not.toContainText(other);
    await expect(editor.getByTestId("applied-count")).toHaveText(wrote);
    await expect(window.getByTestId("mind-tab")).toHaveCount(1);
    await app.close();
  });
}
