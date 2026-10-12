import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge, Document } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  addToNote,
  askInAssistant,
  clickEndOf,
  createDataFolder,
  dismissChatSetup,
  expectReadyDocuments,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  stubClipboard,
  useLocalChatModel,
} from "./app";

/**
 * Copying Blocks with Citations (#102): HTML that Word and Google Docs paste as footnotes, a
 * numbered-marker fallback with a Sources list, plain text with the same, and the Bibliography.
 * The payload is read from the editor's own copy event, so the computer's clipboard stays as it
 * was. Set INCARNAMIND_SAMPLES to a folder to also save the HTML samples there.
 */
const SAMPLES = process.env.INCARNAMIND_SAMPLES;

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

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

const bridge = (window: Page) => ({
  documents: () =>
    window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
    ),
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
  saveWebPage: (html: string) =>
    window.evaluate(
      (page) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveWebPage({
          address: "https://encyclopedia.example/tea",
          html: page,
        }),
      html,
    ),
});

/** What the editor's copy puts on the clipboard for the current selection. */
const copyPayload = (editor: Locator) =>
  editor.evaluate((dom) => {
    const data = new DataTransfer();
    dom.dispatchEvent(
      new ClipboardEvent("copy", { clipboardData: data, bubbles: true, cancelable: true }),
    );
    return { html: data.getData("text/html"), text: data.getData("text/plain") };
  });

/** The footnotes a pasted HTML holds, as Word reads them: each one's marker text and its note. */
const footnotesIn = (window: Page, html: string) =>
  window.evaluate((source) => {
    const page = new DOMParser().parseFromString(source, "text/html");
    return {
      markers: [
        ...page.querySelectorAll('span[data-type="citation"] a[style*="mso-footnote-id"]'),
      ].map((each) => each.textContent ?? ""),
      notes: [...page.querySelectorAll('div[style*="mso-element:footnote"][id]')].map((each) => ({
        id: each.getAttribute("id"),
        text: each.textContent?.replace(/\s+/g, " ").trim() ?? "",
        links: [...each.querySelectorAll("a[href^='http']")].map((a) => a.getAttribute("href")),
      })),
    };
  }, html);

async function screenshot(target: Page | Locator, name: string): Promise<void> {
  const folder = process.env.INCARNAMIND_SCREENSHOTS;
  if (folder) await target.screenshot({ path: join(folder, `${name}.png`) });
}

async function save(name: string, content: string): Promise<void> {
  if (!SAMPLES) return;
  await mkdir(SAMPLES, { recursive: true });
  await writeFile(join(SAMPLES, name), content);
}

/** A Citation of a Document, as the core writes one into an Answer. */
function citationOf(document: Document, location: object, extra: object = {}) {
  return {
    type: "citation",
    attrs: {
      passageId: `passage-${document.id}`,
      documentId: document.id,
      documentName: document.name,
      contentHash: document.contentHash,
      location,
      quote: "q",
      check: "found",
      ...extra,
    },
  };
}

test("copying Blocks with Citations puts footnotes in the HTML, and markers with Sources in the text; the Bibliography lists each Document once", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "Lease agreement.txt"), "Spring rent is due at the new moon.");
  const article = await readFile(join(__dirname, "../tests/fixtures/web/article.html"), "utf8");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1200, 800);
  await addDocuments(window, [join(sources, "Tides.pdf"), join(sources, "Lease agreement.txt")]);
  const saved = await bridge(window).saveWebPage(article);
  expect(saved.outcome).toBe("saved");
  await expectReadyDocuments(window, 3);
  const documents = await bridge(window).documents();
  const named = (text: string) => {
    const found = documents.find((each) => each.name.includes(text));
    if (!found) throw new Error(`No Document named ${text}.`);
    return found;
  };

  // A Mind with the Answer's Citation of Tides, one of a web page, and one a pending change adds.
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, editor);
  await addToNote(window, await askInAssistant(window, "When do spring tides happen?"));
  await expect(editor.getByTestId("citation")).toHaveCount(1);
  await editor.evaluate(
    (dom, blocks) => {
      const { editor: tiptap } = dom as unknown as { editor: Editor };
      tiptap.commands.insertContentAt(tiptap.state.doc.content.size, blocks);
    },
    [
      {
        type: "paragraph",
        content: [
          { type: "text", text: "Tea began in the Han era " },
          citationOf(named("Tea"), {
            kind: "section",
            heading: "History",
            paragraphs: { from: 2, to: 2 },
          }),
          { type: "text", text: " and cakes " },
          citationOf(
            named("Lease"),
            { kind: "lines", from: 1, to: 1 },
            { proposal: { id: "pending-1", request: "request-1", kind: "insert" } },
          ),
          { type: "text", text: "." },
        ],
      },
    ],
  );
  await expect(editor.getByTestId("citation")).toHaveCount(3);

  // Copy with everything selected (from a line with no Citation, whose chip would open its Document).
  await pointAndClick(window, editor.locator(":scope > p").nth(1));
  await window.keyboard.press("ControlOrMeta+a");
  const copied = await copyPayload(editor);
  await save("copy-with-citations-en.html", copied.html);
  await save("copy-with-citations-en.txt", copied.text);

  // The HTML: a marker in the text and a footnote for each Citation, as Word writes footnotes,
  // each Citation still a Citation to paste back into IncarnaMind.
  expect(copied.html).toContain('data-type="citation"');
  expect(copied.html).toContain("mso-element:footnote-list");
  const held = await footnotesIn(window, copied.html);
  expect(held.markers).toEqual(["[1]", "[2]"]);
  expect(held.notes.map((note) => note.id)).toEqual(["ftn1", "ftn2"]);
  expect(held.notes[0]?.text).toBe("[1] Tides, p. 2");
  // A web page: its title without the site repeated, its site, its address and when it was fetched.
  expect(held.notes[1]?.text).toMatch(
    /^\[2\] Tea, The Open Encyclopedia, § History, ¶.2, https:\/\/encyclopedia\.example\/tea, accessed \d{1,2} \w+ \d{4}$/,
  );
  expect(held.notes[1]?.links).toEqual(["https://encyclopedia.example/tea"]);
  // The pending change's Citation is no part of the copy.
  expect(copied.html).not.toContain("Lease");
  expect(copied.text).not.toContain("Lease");

  // The text: markers, and a Sources list.
  expect(copied.text).toMatch(/Your Documents answer this.*\[1\]/s);
  expect(copied.text).toMatch(/Tea began in the Han era \[2\] and cakes \./);
  expect(copied.text).toMatch(
    /\n\nSources\n\[1\] Tides, p\. 2\n\[2\] Tea, The Open Encyclopedia, § History, ¶.2, https:\/\/encyclopedia\.example\/tea, accessed \d{1,2} \w+ \d{4}$/,
  );

  // Pasted back into the Mind, the Citations are Citations and the footnote list adds no text.
  await clickEndOf(window, editor.locator(":scope > p").last());
  await window.keyboard.press("Enter");
  await editor.evaluate(
    (dom, html) => (dom as unknown as { editor: Editor }).editor.view.pasteHTML(html),
    copied.html,
  );
  await expect(editor.getByTestId("citation")).toHaveCount(5);
  await expect(editor).not.toContainText("[1] Tides");

  // In Chinese, the Sources list and the Locations are Chinese.
  await bridge(window).language("zh-CN");
  await pointAndClick(window, editor.locator(":scope > p").nth(1));
  await window.keyboard.press("ControlOrMeta+a");
  const chinese = await copyPayload(editor);
  await save("copy-with-citations-zh.txt", chinese.text);
  expect(chinese.text).toContain("来源\n[1] Tides，第 2 页");
  expect(chinese.text).toContain("访问于");
  await bridge(window).language("en");

  // The Bibliography: "/" Bibliography, at the end of the Mind. A Numbered Mind has none until asked.
  const references = window.getByTestId("references");
  await expect(references).toHaveCount(0);
  await clickEndOf(window, editor.locator(":scope > p").last());
  await window.keyboard.press("Enter");
  await window.keyboard.type("/biblio");
  const item = window.getByTestId("slash-item-bibliography");
  await expect(item).toBeVisible();
  await expect(item).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await expect(references).toBeVisible();
  await expect(references).toHaveAttribute("data-style", "numbered");
  await expect(references.getByRole("heading")).toHaveText("Bibliography");
  const entries = references.getByTestId("reference-entry");
  // Each Document once; the pending change's Citation isn't counted.
  await expect(entries).toHaveCount(2);
  await expect(entries.nth(0)).toContainText(
    "Tea, The Open Encyclopedia, https://encyclopedia.example/tea, accessed",
  );
  await expect(entries.nth(0)).toContainText("Cited: § History");
  await expect(entries.nth(1)).toHaveText("Tides. Cited: p. 2");
  await expect(references).not.toContainText("Lease");

  await references.scrollIntoViewIfNeeded();
  await screenshot(window, "bibliography-numbered-en");

  // It is copied as text and HTML, and updates as Citations come and go.
  const clipboard = await stubClipboard(window);
  await pointAndClick(window, references.getByTestId("references-copy"));
  await expect(references.getByTestId("references-copy")).toHaveText("Copied");
  const bibliography = await clipboard();
  expect(bibliography["text/plain"]).toContain("Tides. Cited: p. 2");
  expect(bibliography["text/html"]).toContain("<p>Tides. Cited: p. 2</p>");

  // Hide takes it away; "/" Bibliography brings it back.
  await pointAndClick(window, references.getByTestId("bibliography-hide"));
  await expect(references).toHaveCount(0);
  await app.close();
});
