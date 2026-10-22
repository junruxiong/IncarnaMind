import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  checkCard,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  tokenColor,
  useLocalChatModel,
} from "./app";

/*
 * Every sentence check says which sentence it is about (#1621): its card opens with the sentence in
 * quotation marks, the sentence is outlined while the card is open, the mark sits on the sentence's
 * first line, and Next in the Mind's health card ends where a click on the mark does. The checks are
 * written into the page, so no model is involved.
 */

const TIDES = buildPdf([
  { lines: ["Harbour times", "Harbours publish the times of high water every 12 months."] },
]);

const SENTENCES = {
  en: {
    supported:
      "Spring tides happen at new moon and at full moon, when the Sun, the Moon and the Earth line up, and harbours along the whole coast publish the times of high water for every day of the year.",
    unsupported: "Staff may work from home two days a week.",
    contradicted: "The agreement says the facilities end in 2027.",
    notChecked: "The model was not able to check this one.",
    recheck: "Gulls circle the pier at dusk.",
    recheckWas: "Gulls circled the pier at dusk.",
  },
  "zh-CN": {
    supported:
      "春潮发生在新月和满月，那时太阳、月球和地球排成一线，沿岸所有港口每年都会公布每一天的高潮时间，让船员能在离开码头之前很早就计划好每一次出海，保证每一条船都按时回港。",
    unsupported: "员工每周可以在家办公两天。",
    contradicted: "协议规定这些贷款在 2027 年到期。",
    notChecked: "模型这次没能核对这一句。",
    recheck: "海鸥黄昏时在码头上空盘旋。",
    recheckWas: "海鸥黄昏时曾在码头上空盘旋。",
  },
} as const;

const KINDS = ["supported", "unsupported", "contradicted", "not-checked", "needs-recheck"] as const;

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

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  if (!found) throw new Error("The element isn't visible.");
  return found;
}

for (const language of ["en", "zh-CN"] as const) {
  const words = SENTENCES[language];
  test(`each check card names its sentence and outlines it while open; Next opens a sentence's card (${language})`, async () => {
    test.setTimeout(180_000);
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 900, 900);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    const [tides] = await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
    );
    const passage = {
      passageId: "passage-1",
      documentId: tides?.id,
      documentName: "Tides",
      contentHash: tides?.contentHash,
      pageFrom: 1,
      pageTo: 1,
      location: null,
      quote: "Harbours publish the times of high water every 12 months.",
      check: "found",
      checkReason: null,
    };
    const checked = (
      id: string,
      state: string,
      text: string,
      extra: Record<string, unknown> = {},
    ) => ({
      type: "text",
      text,
      marks: [
        {
          type: "claimCheck",
          attrs: {
            id,
            request: "check-1",
            action: "check-against-documents",
            state,
            text,
            scope: { folderId: null, widened: false },
            versions: [],
            excluded: [],
            checkedAt: "2026-10-22T09:00:00.000Z",
            ...extra,
          },
        },
      ],
    });
    const paragraph = (...content: object[]) => ({ type: "paragraph", content });

    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.evaluate(
      (dom, content) => {
        (dom as unknown as { editor: Editor }).editor.commands.insertContentAt(0, content);
      },
      [
        paragraph(checked("check-1", "supported", words.supported, { evidence: passage })),
        paragraph(checked("check-2", "unsupported", words.unsupported)),
        paragraph(
          checked("check-3", "contradicted", words.contradicted, {
            evidence: passage,
            says: "The agreement says the facilities end on 14 February 2029.",
          }),
        ),
        paragraph(checked("check-4", "not-checked", words.notChecked, { reason: "refused" })),
        paragraph({
          ...checked("check-5", "supported", words.recheckWas, { evidence: passage }),
          text: words.recheck,
        }),
      ],
    );
    const marks = window.getByTestId("claim-mark");
    const card = window.getByTestId("claim-card");
    const outlined = window.getByTestId("claim-outline");
    const paragraphOf = (sentence: string) =>
      editor.locator(":scope > p").filter({ hasText: sentence });
    // The boxes of the outline are one to a line of the paragraph's words, and enclose them.
    const outlinesWords = async (paragraph: Locator) => {
      const words = await paragraph.evaluate((element) => {
        const range = document.createRange();
        range.selectNodeContents(element);
        const rect = range.getBoundingClientRect();
        const tops = Array.from(range.getClientRects(), (each) => Math.round(each.top));
        return { ...rect.toJSON(), lines: new Set(tops).size };
      });
      const boxes = await outlined.evaluateAll((all) =>
        all.map((element) => element.getBoundingClientRect().toJSON()),
      );
      return {
        boxes: boxes.length,
        lines: words.lines,
        encloses:
          boxes.length > 0 &&
          Math.min(...boxes.map((each) => each.left)) <= words.left &&
          Math.min(...boxes.map((each) => each.top)) <= words.top &&
          Math.max(...boxes.map((each) => each.right)) >= words.right &&
          Math.max(...boxes.map((each) => each.bottom)) >= words.bottom,
      };
    };
    await expect(marks).toHaveCount(5);

    const metaInk = await tokenColor(window, "--color-ink-meta");
    const accent = await tokenColor(window, "--color-accent");
    const sentences = [
      words.supported,
      words.unsupported,
      words.contradicted,
      words.notChecked,
      words.recheck,
    ];

    for (const [at, kind] of KINDS.entries()) {
      const sentence = sentences[at] ?? "";
      const mark = marks.nth(at);
      await expect(mark).toHaveAttribute("data-check", kind);
      await expect(outlined).toHaveCount(0);

      await pointAndClick(window, mark);
      await expect(card).toHaveAttribute("data-check", kind);
      const line = card.getByTestId("claim-sentence");
      await expect(line).toHaveText(`“${sentence}”`);
      // It is the first thing under the verdict's title.
      expect(
        await card
          .getByTestId("claim-badge")
          .evaluate((title) => title.nextElementSibling?.getAttribute("data-testid")),
      ).toBe("claim-sentence");
      await expect(line).toHaveCSS("font-size", "13px");
      await expect(line).toHaveCSS("color", metaInk);
      const shape = await line.evaluate((element) => {
        const lineHeight = Number.parseFloat(getComputedStyle(element).lineHeight);
        return {
          lines: Math.round(element.getBoundingClientRect().height / lineHeight),
          clipped: element.scrollHeight > element.clientHeight,
        };
      });
      expect(shape.lines, "at most two lines").toBeLessThanOrEqual(2);
      expect(shape.clipped, "only a long sentence is cut").toBe(at === 0);

      // While the card is open the sentence has the accent outline, and the mark is the open one.
      await expect(outlined.first()).toBeVisible();
      const around = await outlinesWords(paragraphOf(sentence));
      expect(around.encloses, "it encloses the sentence").toBe(true);
      expect(around.boxes, "a box on each line").toBe(around.lines);
      if (at === 0) expect(around.lines).toBeGreaterThanOrEqual(3);
      await expect(outlined.first()).toHaveCSS("border-top-style", "solid");
      await expect(outlined.first()).toHaveCSS("border-top-width", "2px");
      await expect(outlined.first()).toHaveCSS("border-top-color", accent);
      await expect(mark).toHaveAttribute("data-open", "true");
      const cardBox = await box(card);
      expect(cardBox.x).toBeGreaterThanOrEqual(0);
      expect(cardBox.x + cardBox.width).toBeLessThanOrEqual(900);
      expect(await window.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        900,
      );
      await screenshot(window, `card-${kind}-${language}-900`);

      // Esc closes the card and takes the outline off.
      await window.keyboard.press("Escape");
      await expect(card).toHaveCount(0);
      await expect(outlined).toHaveCount(0);
      await expect(mark).not.toHaveAttribute("data-open", "true");
    }

    // A sentence edited after its check shows its words as they are now.
    const edited = `${words.recheck.slice(0, 2)}EDITED${words.recheck.slice(2)}`;
    await editor.evaluate((dom, sentence) => {
      const { editor: current } = dom as unknown as { editor: Editor };
      current.state.doc.descendants((node, pos) => {
        if (node.isText && node.text === sentence) {
          current.commands.insertContentAt(pos + 2, "EDITED");
        }
      });
    }, words.recheck);
    await pointAndClick(window, marks.nth(4));
    await expect(card.getByTestId("claim-sentence")).toHaveText(`“${edited}”`);
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);

    // A mark on a sentence that wraps over three lines sits on its first line.
    const first = paragraphOf(words.supported);
    const lineHeight = await first.evaluate((element) =>
      Number.parseFloat(getComputedStyle(element).lineHeight),
    );
    const page = await box(first);
    expect(page.height).toBeGreaterThanOrEqual(lineHeight * 3 - 1);
    const middle = await box(marks.first()).then((rect) => rect.y + rect.height / 2);
    expect(middle).toBeGreaterThanOrEqual(page.y);
    expect(middle).toBeLessThanOrEqual(page.y + lineHeight);

    // Hovering or focusing a mark shows its card, and the sentence is outlined as long as it shows.
    await marks.nth(1).focus();
    await expect(card).toHaveAttribute("data-check", "unsupported");
    expect((await outlinesWords(paragraphOf(words.unsupported))).encloses).toBe(true);
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await expect(outlined).toHaveCount(0);
    await window.mouse.move(2, 2);
    await marks.nth(2).hover();
    await expect(card).toHaveAttribute("data-check", "contradicted");
    expect((await outlinesWords(paragraphOf(words.contradicted))).encloses).toBe(true);
    await window.mouse.move(2, 2);
    await expect(card).toHaveCount(0);
    await expect(outlined).toHaveCount(0);

    // Next: the unsupported sentence is pushed out of sight, and Next brings it back with its card open.
    await editor.evaluate(
      (dom, filler) => {
        const { editor: current } = dom as unknown as { editor: Editor };
        const after = current.state.doc.firstChild?.nodeSize ?? 0;
        current.commands.insertContentAt(after, filler);
      },
      Array.from({ length: 30 }, (_, at) => paragraph({ type: "text", text: `Note ${at + 1}` })),
    );
    const scroller = window.locator(".mind-scroller:visible");
    await scroller.evaluate((element) => {
      element.scrollTop = 0;
    });
    const unsupported = paragraphOf(words.unsupported);
    // Shown clear of the page's top and of the fade and the badge at its foot.
    const inView = async () => {
      const [view, target] = await Promise.all([box(scroller), box(unsupported)]);
      return target.y >= view.y + 64 && target.y + target.height <= view.y + view.height - 64;
    };
    expect(await inView()).toBe(false);
    const health = window.getByTestId("check-card");
    await pointAndClick(window, (await checkCard(window)).getByTestId("health-unsupported"));
    await expect(card).toHaveAttribute("data-check", "unsupported");
    await expect(health).toHaveCount(0);
    await expect.poll(inView).toBe(true);
    expect((await outlinesWords(unsupported)).encloses).toBe(true);
    await expect(marks.nth(1)).toHaveAttribute("data-open", "true");
    await expect(card.getByTestId("claim-sentence")).toHaveText(`“${words.unsupported}”`);
    await screenshot(window, `next-unsupported-${language}-900`);
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await expect(outlined).toHaveCount(0);

    // The same from the keyboard alone: the badge, then a row.
    await window.getByTestId("check-badge-open").focus();
    await window.keyboard.press("Enter");
    await expect(health).toBeVisible();
    await health.getByTestId("health-contradicted").focus();
    await window.keyboard.press("Enter");
    await expect(card).toHaveAttribute("data-check", "contradicted");
    await expect(health).toHaveCount(0);
    expect((await outlinesWords(paragraphOf(words.contradicted))).encloses).toBe(true);
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await expect(outlined).toHaveCount(0);
    await app.close();
  });
}
