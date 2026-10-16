import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setCheck,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

const SPRING = ["Spring and neap tides", "Spring tides happen at new moon and at full moon."];

/** A cover, then three pages whose footers print 1, 2 and 3: the spring tides page is PDF page 3, printed 2. */
const ANNUAL_REPORT = buildPdf([
  { lines: ["Tides and the Moon", "Annual report"] },
  { lines: ["The Moon raises two bulges of water on the Earth.", "1"] },
  { lines: [...SPRING, "2"] },
  { lines: ["Harbours publish the times of high water every year.", "3"] },
]);

/** Two pages of front matter in roman numerals, then 1, 2, 3: the spring tides page is PDF page 4, labelled 2. */
const THESIS = buildPdf(
  [
    { lines: ["Tides and the Moon"] },
    { lines: ["Contents"] },
    { lines: ["The Moon raises two bulges of water on the Earth."] },
    { lines: SPRING },
    { lines: ["Harbours publish the times of high water every year."] },
  ],
  {
    pageLabels: [
      { page: 1, style: "r" },
      { page: 3, style: "D" },
    ],
  },
);

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

test("a PDF whose footers print numbers one lower says the printed page, and both in the card, and still opens the PDF's page", async () => {
  await writeFile(join(sources, "Annual Report.pdf"), ANNUAL_REPORT);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Annual Report.pdf")]);
  await setWindowSize(app, window, 1200, 800);
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  const answer = await askInAssistant(window, "When do spring tides happen?");

  const chip = answer.getByTestId("citation").getByTestId("citation-chip");
  await expect(chip).toHaveAttribute(
    "aria-label",
    "Citation 1: Annual Report, p. 2. Quote found on p. 2",
  );
  await chip.hover();
  const card = window.getByTestId("citation-card");
  await expect(card.getByTestId("citation-badge")).toHaveText("Quote found on p. 2");
  await expect(card.getByTestId("citation-pdf-page")).toHaveText("p. 2 · page 3 of the PDF");

  await card.getByTestId("citation-open").click();
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("3");
  await expect(
    window.getByTestId("viewer").locator('[data-page-number="3"] [data-quote-highlight]'),
  ).toHaveText(["Spring tides happen at new moon and at full moon."]);
  await app.close();
});

test("in Chinese, a PDF with page labels says the labelled page, and the PDF's page beside it", async () => {
  await writeFile(join(sources, "论文.pdf"), THESIS);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "zh-CN");
  await addDocuments(window, [join(sources, "论文.pdf")]);
  await setWindowSize(app, window, 1200, 800);
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  const answer = await askInAssistant(window, "When do spring tides happen?");

  const chip = answer.getByTestId("citation").getByTestId("citation-chip");
  await chip.hover();
  const card = window.getByTestId("citation-card");
  await expect(card.getByTestId("citation-badge")).toHaveText("已在第 2 页找到引文");
  await expect(card.getByTestId("citation-pdf-page")).toHaveText("第 2 页 · PDF 第 4 页");

  await card.getByTestId("citation-open").click();
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("4");
  await app.close();
});

test("a PDF whose printed numbers don't keep step shows its PDF pages, and the card says nothing more", async () => {
  const odd = buildPdf([
    { lines: ["Tides and the Moon", "7"] },
    { lines: ["The Moon raises two bulges of water on the Earth.", "2"] },
    { lines: [...SPRING, "9"] },
    { lines: ["Harbours publish the times of high water every year.", "3"] },
  ]);
  await writeFile(join(sources, "Notes.pdf"), odd);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Notes.pdf")]);
  await setWindowSize(app, window, 1200, 800);
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  const answer = await askInAssistant(window, "When do spring tides happen?");

  await answer.getByTestId("citation").getByTestId("citation-chip").hover();
  const card = window.getByTestId("citation-card");
  await expect(card.getByTestId("citation-badge")).toHaveText("Quote found on p. 3");
  await expect(card.getByTestId("citation-pdf-page")).toHaveCount(0);
  await app.close();
});

test("where the card doesn't say the page in its badge, the page shows once, in the line under the name", async () => {
  await writeFile(join(sources, "Annual Report.pdf"), ANNUAL_REPORT);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Annual Report.pdf")]);
  await setWindowSize(app, window, 1200, 800);
  await newMind(window);
  await window.getByTestId("mind-editor").click();
  const answer = await askInAssistant(window, "When do spring tides happen?");
  const chip = answer.getByTestId("citation").getByTestId("citation-chip");
  const card = window.getByTestId("citation-card");

  for (const check of ["unsupported", "contradicted", "cant-check"]) {
    await setCheck(answer, { check });
    await window.mouse.move(0, 0);
    await chip.hover();
    await expect(card.getByTestId("citation-pdf-page")).toHaveText("p. 2 · page 3 of the PDF");
    await expect(card).not.toContainText("Annual Report, p.");
    expect((await card.innerText()).split("p. 2").length - 1).toBe(1);
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
  }
  await app.close();
});
