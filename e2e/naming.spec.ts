import { expect, test } from "@playwright/test";
import {
  addToNote,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  exportFromMenu,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
  useOutputLocation,
} from "./app";

let dataDir: string;
let saved: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  saved = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(saved);
});

test("a Mind with no title is named by its first line, in the sidebar and the tab, until a title is typed", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);

  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  const title = window.getByTestId("mind-title");
  await expect(title).toBeFocused();
  const row = window.getByTestId("mind-list-item");
  const tab = window.getByTestId("mind-tab");
  await expect(row).toContainText("Untitled");
  await expect(title).toHaveAttribute("placeholder", "Untitled");

  await enterContent(window);
  await window.keyboard.type("Tides in Bergen");
  await expect(row).toContainText("Tides in Bergen");
  await expect(tab).toContainText("Tides in Bergen");
  await expect(title).toHaveAttribute("placeholder", "Tides in Bergen");
  await expect(title).toHaveValue("");

  await window.keyboard.type(" and the moon");
  await expect(row).toContainText("Tides in Bergen and the moon");

  await title.click();
  await window.keyboard.type("My own title");
  await expect(row).toContainText("My own title");
  await expect(row).not.toContainText("Tides");
  await expect(tab).toContainText("My own title");

  await title.press("ControlOrMeta+a");
  await title.press("Backspace");
  await expect(row).toContainText("Tides in Bergen and the moon");
});

test("a long first line is cut at a word, and a Chinese one at a character", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);

  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  await enterContent(window);
  await window.keyboard.type(
    "How much cooler are large irrigated parks than the surrounding built-up streets during clear windless summer nights",
  );
  const row = window.getByTestId("mind-list-item");
  await expect(row).toContainText("irrigated parks");
  await expect(row).not.toContainText("than");

  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  await enterContent(window);
  await window.keyboard.type("夜间降温".repeat(20));
  await expect(window.getByTestId("mind-list-item").first()).toContainText("夜间降温".repeat(15));
  await expect(window.getByTestId("mind-list-item").first()).not.toContainText(
    "夜间降温".repeat(16),
  );
});

const FROM_QUESTION = [
  {
    language: "en",
    width: 1440,
    question:
      "How many days of paid holiday does a full-time employee get in a year, and how much notice after probation?",
    name: "How many days of paid holiday does a full-time employee get",
  },
  {
    language: "en",
    width: 1100,
    question:
      "How many days of paid holiday does a full-time employee get in a year, and how much notice after probation?",
    name: "How many days of paid holiday does a full-time employee get",
  },
  {
    language: "zh-CN",
    width: 1440,
    question:
      "员工每年享有多少天带薪年假，试用期结束后需要提前多久通知公司解除劳动合同，以及需要支付哪些补偿，还有哪些情形下公司可以不提前通知直接解除合同？",
    name: "员工每年享有多少天带薪年假，试用期结束后需要提前多久通知公司解除劳动合同，以及需要支付哪些补偿",
  },
  {
    language: "zh-CN",
    width: 1100,
    question:
      "员工每年享有多少天带薪年假，试用期结束后需要提前多久通知公司解除劳动合同，以及需要支付哪些补偿，还有哪些情形下公司可以不提前通知直接解除合同？",
    name: "员工每年享有多少天带薪年假，试用期结束后需要提前多久通知公司解除劳动合同，以及需要支付哪些补偿",
  },
] as const;

for (const { language, width, question, name } of FROM_QUESTION) {
  test(`Add to note into an empty Mind names it by the Question; the grey name is one line with the whole name in its tooltip, and the exported file takes it (${language}, ${width}px)`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await useOutputLocation(window, saved);
    await setLanguage(window, language);
    await setWindowSize(app, window, width, 700);
    await newMind(window);
    const title = window.getByTestId("mind-title");
    await expect(title).toHaveAttribute("placeholder", language === "en" ? "Untitled" : "未命名");

    const answer = await askInAssistant(window, question);
    await addToNote(window, answer);
    const editor = window.getByTestId("mind-editor");
    const answerText = (await editor.locator(":scope > p").first().innerText()).trim();
    expect(answerText).not.toBe("");

    // The Mind's name is the Question's, cut cleanly: not the Answer's first line, and not typed.
    const added = answer.getByTestId("answer-added");
    await expect(added).toContainText(name);
    // However long the name, the line keeps its words and its Undo on one row.
    const [words, undo] = await Promise.all([
      added.locator('[role="status"]').boundingBox(),
      added.getByTestId("answer-added-undo").boundingBox(),
    ]);
    expect(Math.abs((undo?.y ?? 0) - (words?.y ?? 100))).toBeLessThan(8);
    const row = window.getByTestId("mind-list-item");
    await expect(row).toContainText(name);
    await expect(row).not.toContainText(answerText.slice(0, 20));
    await expect(title).toHaveValue("");

    // The grey name above the note is one line, cut at a word with an ellipsis, and the whole
    // name is in its tooltip.
    const shown = (await title.getAttribute("placeholder")) ?? "";
    expect(shown.endsWith("…")).toBe(true);
    const start = shown.slice(0, -1);
    expect(name.startsWith(start)).toBe(true);
    if (language === "en") expect(name.charAt(start.length)).toBe(" ");
    await expect(title).toHaveAttribute("title", name);
    const oneLine = await title.evaluate((element) => {
      const lineHeight = Number.parseFloat(getComputedStyle(element).lineHeight);
      return element.getBoundingClientRect().height <= lineHeight * 1.5;
    });
    expect(oneLine).toBe(true);

    // The exported file is named by it, and ends at the same clean cut.
    await exportFromMenu(window);
    const done = window.getByTestId("export-result").getByTestId("export-done");
    await expect(done).toContainText(`${name}.docx`, { timeout: 30_000 });
    await app.close();
  });
}

test("a title the User typed is never replaced when an Answer is added to the Mind", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 700);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await newMind(window);
  const title = window.getByTestId("mind-title");
  await title.fill("My own title");

  const answer = await askInAssistant(window, "How many days of paid holiday?");
  await addToNote(window, answer);

  await expect(title).toHaveValue("My own title");
  await expect(window.getByTestId("mind-list-item")).toContainText("My own title");
  await expect(window.getByTestId("mind-list-item")).not.toContainText("How many days");
  await app.close();
});
