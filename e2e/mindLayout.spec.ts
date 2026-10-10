import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import type { TestHooks } from "../src/shared/testHooks";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  clickEmptyLine,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

/** Three short pages: the line about spring tides is on page 2. */
const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
]);

/** The tiny MCP server the core tests use: `book_boat` may change something, so it asks. */
const TIDE_SERVER = resolve(__dirname, "../tests/fixtures/mcp-server.mjs");

/** DESIGN.md, Layout: the Mind's left margin for controls, its measure, its right margin for checks. */
const LEFT_MARGIN = 56;
const MEASURE = 680;
const RIGHT_MARGIN = 48;

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

/** The size the approved mockups were drawn at. */
async function useMockupSize(app: ElectronApplication, window: Page) {
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setSize(1440, 900));
  await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBeGreaterThan(1200);
  await window.evaluate(() => document.fonts.ready.then(() => undefined));
}

/** Saves a screenshot with the test's results (test-results/…), and attaches it to the report. */
async function screenshot(target: Page | Locator, name: string) {
  const path = test.info().outputPath(name);
  await target.screenshot({ path });
  await test.info().attach(name, { path, contentType: "image/png" });
}

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  if (!found) throw new Error("The element isn't visible.");
  return found;
}

/** The editor's text column: its box reaches over the left margin, its padding keeps the text edge. */
function textColumn(editor: Locator) {
  return editor.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    const left = Number.parseFloat(style.paddingLeft);
    const right = Number.parseFloat(style.paddingRight);
    return { x: rect.left + left, width: rect.width - left - right };
  });
}

const middle = (rect: { y: number; height: number }) => rect.y + rect.height / 2;

/** `actual` is within `tolerance` px of `expected` (a soft check: the test goes on). */
function expectNear(actual: number, expected: number, tolerance: number, what: string) {
  expect
    .soft(
      Math.abs(actual - expected),
      `${what}: ${actual} should be within ${tolerance}px of ${expected}`,
    )
    .toBeLessThanOrEqual(tolerance);
}

/** Opens or closes the Document viewer through the test hook, as a Citation would. */
function setViewer(window: Page, open: boolean) {
  return window.evaluate((show) => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off: launch with INCARNAMIND_TEST_HOOKS=1.");
    if (show) hooks.openViewer();
    else hooks.closeViewer();
  }, open);
}

/**
 * Adds a sentence with two more Citations to the Answer, after its first
 * paragraph, as copies of its own Citation: one found, and one whose quote
 * wasn't found, on a later line. Like an Answer that cited three times.
 */
async function citeMore(editor: Locator) {
  await editor.evaluate((dom) => {
    const { view, commands } = (dom as unknown as { editor: Editor }).editor;
    let at = -1;
    let cited: Record<string, unknown> | null = null;
    view.state.doc.forEach((node, offset) => {
      if (node.type.name !== "answer" || at >= 0 || !node.firstChild) return;
      at = offset + 1 + node.firstChild.nodeSize;
      node.descendants((child) => {
        if (!cited && child.type.name === "citation") cited = { ...child.attrs };
        return cited === null;
      });
    });
    if (at < 0 || !cited) throw new Error("No Answer with a Citation.");
    const citation = (check: string, checkReason: string | null) => ({
      type: "citation",
      attrs: { ...(cited as Record<string, unknown>), check, checkReason },
    });
    commands.insertContentAt(at, {
      type: "paragraph",
      content: [
        { type: "text", text: "They come when the Sun and the Moon pull in line." },
        citation("found", null),
        {
          type: "text",
          text: " The pull is strongest at new moon and at full moon, when the two line up with the Earth, and weakest in between, at the quarter moons, when the smaller neap tides come instead.",
        },
        citation("not-found", "quote-not-on-pages"),
      ],
    });
  });
}

test("every text in a Mind starts at one edge, its controls sit in the left margin, and each Citation's check is level with its marker in the right margin", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await useMockupSize(app, window);

  // A title, a Note, a heading, a list, then a Question.
  await newMind(window);
  const title = window.getByTestId("mind-title");
  await title.fill("Reading notes: tides");
  await title.press("Enter");
  const editor = window.getByTestId("mind-editor");
  await window.keyboard.type(
    "Goal for this week: find out what drives spring tides, and when they come. Start with the tide tables, then the notes on the Moon.",
  );
  await window.keyboard.press("Enter");
  await window.keyboard.type("## Spring tides");
  await window.keyboard.press("Enter");
  await window.keyboard.type("- Twice a month, at new moon and full moon");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Neap tides fall in between");
  await window.keyboard.press("Enter");
  await window.keyboard.press("Enter");
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("When do spring tides happen, and what makes them stronger?");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await citeMore(editor);
  await clickEmptyLine(editor.locator(":scope > p").last());
  await window.keyboard.type(
    "So spring tides follow the Moon. Check whether the tables put numbers on how much higher they are.",
  );

  // One text edge: the title, a Note, a heading, a list item, a Question's text and an Answer's text.
  const edge = (await textColumn(editor)).x;
  const texts = {
    title: title,
    note: editor.locator(":scope > p").first(),
    heading: editor.locator(":scope > h2"),
    "list item": editor.locator(":scope > ul > li").first(),
    question: editor.getByTestId("question").locator(".question-text"),
    answer: answer.locator(".answer-content p").first(),
  };
  for (const [name, locator] of Object.entries(texts)) {
    expectNear((await box(locator)).x, edge, 1, `the ${name} starts at the text edge`);
  }
  // The measure is 680px, with its margins inside the pane.
  expectNear((await textColumn(editor)).width, MEASURE, 1, "the measure");

  // A Question's spark and fold chevron and the block handle sit in the left margin.
  const inLeftMargin = async (locator: Locator, name: string) => {
    const rect = await box(locator);
    expect
      .soft(rect.x, `${name} is right of the margin's start`)
      .toBeGreaterThanOrEqual(edge - LEFT_MARGIN - 1);
    expect.soft(rect.x + rect.width, `${name} is left of the text`).toBeLessThanOrEqual(edge + 1);
  };
  await inLeftMargin(editor.getByTestId("question-fold"), "the Question's fold chevron");
  // It is level with the Question's one 20px line; the Answer has no label.
  const foldMiddle = middle(await box(editor.getByTestId("question-fold")));
  expectNear(foldMiddle, (await box(texts.question)).y + 10, 1, "the fold chevron's middle");
  await expect(answer.locator(".answer-label")).toHaveCount(0);
  // The composer, pinned under the Mind, is the measure wide at the text edge.
  const composer = await box(window.getByTestId("composer"));
  expectNear(composer.x, edge, 1, "the composer's edge");
  expectNear(composer.width, MEASURE, 1, "the composer's width");
  await texts.note.hover();
  const handle = window.getByTestId("block-handle");
  await expect(handle).toBeVisible();
  await inLeftMargin(handle, "the block handle");

  // The Citations are numbered within their Answer; the not-found one is amber in the text.
  const markers = editor.getByTestId("citation-chip");
  await expect(markers).toHaveText(["1", "2", "3"]);
  await expect(markers.nth(0)).toHaveAttribute("aria-label", /Tides, p\. 2\. Quote found on p\. 2/);
  await expect(markers.nth(2)).toHaveAttribute("data-check", "not-found");

  // A mark for each in the right margin, with its state, level with its marker's line.
  const marks = window.getByTestId("margin-check");
  await expect(marks).toHaveCount(3);
  await expect(marks).toHaveText(["1", "2", "3"]);
  for (const [index, check] of ["found", "found", "not-found"].entries()) {
    const mark = marks.nth(index);
    await expect(mark).toHaveAttribute("data-check", check);
    const markRect = await box(mark);
    const markerRect = await box(markers.nth(index));
    expectNear(middle(markRect), middle(markerRect), 4, `mark ${index + 1}'s middle`);
    expect.soft(markRect.x).toBeGreaterThanOrEqual(edge + MEASURE);
    expect.soft(markRect.x + markRect.width).toBeLessThanOrEqual(edge + MEASURE + RIGHT_MARGIN);
  }
  await screenshot(window, "mind.png");

  // Clicking somewhere empty, right of the margin column, closes a card.
  const away = { x: edge + MEASURE + RIGHT_MARGIN + 24, y: (await box(title)).y };
  const card = window.getByTestId("citation-card");

  // A marker opens its card (and its page: closed again here, to see the card at full width).
  await markers.nth(0).click();
  await expect(card.getByTestId("citation-badge")).toHaveText("Quote found on p. 2");
  await expect(markers.nth(0)).toHaveAttribute("aria-expanded", "true");
  await setViewer(window, false);
  await expect(window.getByTestId("viewer")).toHaveCount(0);
  await expect(marks.nth(0)).toHaveAttribute("data-open", "true");
  expectNear((await box(card.locator(".citation-card"))).width, 340, 1, "the card's width");
  await screenshot(window, "mind-card-found.png");
  await window.mouse.click(away.x, away.y);
  await expect(card).toHaveCount(0);

  // So does its mark: here the one whose quote wasn't found, which doesn't open the page.
  await marks.nth(2).click();
  await expect(card.getByTestId("citation-badge")).toHaveText("Quote not found on p. 2");
  await expect(card.getByTestId("citation-reason")).toContainText("isn't in the text");
  await expect(window.getByTestId("viewer")).toHaveCount(0);
  await screenshot(window, "mind-card-not-found.png");
  await window.mouse.click(away.x, away.y);
  await expect(card).toHaveCount(0);

  // Under 900px (here, the viewer open beside it) the margins fold away: the markers show the checks.
  await setViewer(window, true);
  await expect(window.getByTestId("margin-checks")).toBeHidden();
  await expect(markers.nth(0).locator(".citation-marker-icon")).toBeVisible();
  expectNear((await box(texts.note)).x, (await textColumn(editor)).x, 1, "narrow: the Note's edge");
  expectNear(
    (await box(window.getByTestId("composer"))).x,
    (await textColumn(editor)).x,
    1,
    "narrow: the composer's edge",
  );
  expectNear(
    (await box(texts.answer)).x,
    (await textColumn(editor)).x,
    1,
    "narrow: the Answer's edge",
  );
  await screenshot(window, "mind-narrow.png");
  await setViewer(window, false);
  await expect(window.getByTestId("margin-checks")).toBeVisible();

  // Two markers on one line: their marks stack, the first level with the line.
  await editor.evaluate((dom) => {
    const { view, commands } = (dom as unknown as { editor: Editor }).editor;
    const found: Record<string, unknown>[] = [];
    view.state.doc.descendants((child) => {
      if (found.length === 0 && child.type.name === "citation") found.push({ ...child.attrs });
      return found.length === 0;
    });
    const cited = found[0] ?? {};
    commands.insertContentAt(view.state.doc.content.size, {
      type: "paragraph",
      content: [
        { type: "text", text: "Copied from the Answer:" },
        { type: "citation", attrs: cited },
        { type: "citation", attrs: cited },
      ],
    });
  });
  await expect(marks).toHaveCount(5);
  const [first, second] = [await box(marks.nth(3)), await box(marks.nth(4))];
  expectNear(middle(first), middle(await box(markers.nth(3))), 4, "the first stacked mark");
  expectNear(second.y, first.y + first.height + 4, 1, "the second stacked mark");
  await app.close();
});

test("a quote, a highlight, a code block and a formula keep the text edge too; their boxes reach into the margin", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await useMockupSize(app, window);
  await newMind(window);
  const title = window.getByTestId("mind-title");
  await title.fill("Tide tables");
  await title.press("Enter");
  const editor = window.getByTestId("mind-editor");

  await window.keyboard.type("> High water comes about 50 minutes later each day.");
  await window.keyboard.press("Enter");
  await window.keyboard.press("Enter");
  await window.keyboard.type("==Spring tides== are the strongest of the month.");
  await window.keyboard.press("Enter");
  await window.keyboard.type("``` ");
  await window.keyboard.type("high_water = previous + minutes(50)");
  for (const _ of [1, 2, 3]) await window.keyboard.press("Enter");
  // The slash menu: 28px rows in the interface type.
  await window.keyboard.type("/");
  const menu = window.getByTestId("slash-menu");
  await expect(menu).toBeVisible();
  const row = await box(menu.getByTestId("slash-item-text"));
  expectNear(row.height, 28, 0.5, "a slash menu row");
  await screenshot(window, "slash-menu.png");
  await window.keyboard.type("math");
  await window.keyboard.press("Enter");
  await window.getByTestId("math-editor").fill("h = h_0 + a \\cos(\\omega t)");
  await window.getByTestId("math-editor").press("Enter");
  await expect(editor.locator('[data-type="block-math"] .katex')).toBeVisible();

  const edge = (await textColumn(editor)).x;
  const quote = editor.locator(":scope > blockquote");
  expectNear((await box(quote.locator("p"))).x, edge, 1, "the quote's text");
  expect((await box(quote)).x, "the quote's rule hangs in the margin").toBeLessThan(edge - 8);
  const highlighted = editor.locator(":scope > p").filter({ has: window.locator("mark") });
  expectNear((await box(highlighted)).x, edge, 1, "the highlighted paragraph");
  expectNear((await box(highlighted.locator("mark"))).x, edge, 1, "the highlight");
  const code = editor.locator(":scope pre");
  expectNear((await box(code.locator("code"))).x, edge, 1, "the code's text");
  expectNear((await box(code)).x, edge - 16, 1, "the code block's band");
  expectNear((await box(editor.locator('[data-type="block-math"]'))).x, edge, 1, "the formula");
  await screenshot(window, "blocks.png");
  await app.close();
});

/** Adds the tiny server as a Connector through the core's bridge, and waits until it's ready. */
async function addTideConnector(window: Page): Promise<void> {
  await window.evaluate(
    async ({ node, server }) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const connector = await bridge.addConnector({ name: "Tides", command: node, args: [server] });
      for (let tries = 0; tries < 300; tries++) {
        const found = (await bridge.listConnectors()).find((each) => each.id === connector.id);
        if (found?.state === "ready") return;
        await new Promise((done) => setTimeout(done, 100));
      }
      throw new Error("The Connector didn't get ready.");
    },
    { node: process.execPath, server: TIDE_SERVER },
  );
}

/** A Skill with one tiny JavaScript script, which greets its first argument. */
async function importGreeterSkill(window: Page): Promise<void> {
  const folder = join(sources, "greeter");
  await mkdir(join(folder, "scripts"), { recursive: true });
  await writeFile(
    join(folder, "SKILL.md"),
    "---\nname: greeter\ndescription: Greets people. Use to say hello.\n---\n\nRun scripts/hello.js with the name.\n",
  );
  await writeFile(
    join(folder, "scripts", "hello.js"),
    'console.log("Hello, " + process.argv[2] + "!");\n',
  );
  await window.evaluate(async (path) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const check = await bridge.previewSkillImport(path);
    if (!check.ok) throw new Error(check.error.message);
    await bridge.importSkill(check.preview.importId);
  }, folder);
}

/** The text of every section of an approval card starts at the same x. */
async function expectOneTextEdge(card: Locator) {
  const starts = [
    card.locator(".approval-title-text"),
    card.locator(".approval-explanation").first(),
    card.locator(".approval-label"),
    card.locator(".approval-args dt").first(),
    card.locator(".approval-actions > :first-child"),
  ];
  const x = (await box(starts[0] as Locator)).x;
  for (const start of starts) expectNear((await box(start)).x, x, 1, "a section's text");
}

test("an approval card is a ruled block whose sections share one text edge, for a Tool and for a Skill script", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addTideConnector(window);
  await importGreeterSkill(window);
  await useMockupSize(app, window);

  await newMind(window);
  const title = window.getByTestId("mind-title");
  await title.fill("Boat trips");
  await title.press("Enter");
  const editor = window.getByTestId("mind-editor");
  await window.keyboard.type("Book the morning boat from Dover, and greet the harbour master.");
  await window.keyboard.press("Enter");
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Please book_boat from Dover");
  await window.keyboard.press("Enter");

  const answer = editor.getByTestId("answer").first();
  const card = answer.getByTestId("approval-card");
  await expect(card).toBeVisible({ timeout: 15_000 });
  await expect(answer.getByTestId("answer-writing")).toHaveText("Waiting for your approval");
  await expect(card.locator(".approval-title")).toHaveText("Tides wants to run book_boat");
  await expect(card.locator(".approval-title code")).toHaveText("book_boat");
  await expectOneTextEdge(card);
  // Deny is pushed to the right.
  const deny = await box(card.getByTestId("approval-deny"));
  const always = await box(card.getByTestId("approval-always-allow"));
  expect(deny.x - (always.x + always.width)).toBeGreaterThan(100);
  await screenshot(window, "approval.png");
  await screenshot(answer, "approval-card.png");
  await card.getByTestId("approval-deny").click();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });

  // A Skill script: the same three sections; "Always run" warns in the last one.
  await clickEmptyLine(editor.locator(":scope > p").last());
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Run greeter scripts/hello.js for Calais");
  await window.keyboard.press("Enter");
  const second = editor.getByTestId("answer").nth(1);
  const scriptCard = second.getByTestId("approval-card");
  await expect(scriptCard).toBeVisible({ timeout: 15_000 });
  await expect(scriptCard.locator(".approval-title code")).toHaveText("scripts/hello.js");
  await expectOneTextEdge(scriptCard);
  await scriptCard.scrollIntoViewIfNeeded();
  await screenshot(second, "approval-script-card.png");
  await scriptCard.getByTestId("approval-always-run").click();
  await expect(scriptCard.getByTestId("always-run-warning")).toBeVisible();
  await screenshot(second, "approval-script-warning.png");
  await scriptCard.getByTestId("always-run-cancel").click();
  await scriptCard.getByTestId("approval-deny").click();
  await expect(second).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await app.close();
});
