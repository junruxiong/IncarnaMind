import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  type CrossrefWork,
  type ReferenceServices,
  startReferenceServices,
} from "../tests/helpers/referenceServices";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Checking a verified document's reference list (#711): the line under its
 * claims says what looking the references up sends, by its button, so
 * clicking it is the consent; then it flags the reference Crossref has no
 * work for. A local stand-in answers for Crossref. Set INCARNAMIND_SCREENSHOTS
 * to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const LSTM: CrossrefWork = {
  DOI: "10.1162/neco.1997.9.8.1735",
  title: ["Long Short-Term Memory"],
  author: [
    { family: "Hochreiter", given: "Sepp" },
    { family: "Schmidhuber", given: "Jürgen" },
  ],
  issued: { "date-parts": [[1997, 11, 1]] },
};
const DEEP: CrossrefWork = {
  DOI: "10.1038/nature14539",
  title: ["Deep learning"],
  author: [
    { family: "LeCun", given: "Yann" },
    { family: "Bengio", given: "Yoshua" },
    { family: "Hinton", given: "Geoffrey" },
  ],
  issued: { "date-parts": [[2015, 5, 27]] },
};
const INVENTED =
  "Marlowe, T. (2019). Quantum gradient folding in recurrent attention lattices. J. Imagined Results, 4.";

/** A short paper someone else wrote, with a reference list holding one invented work. */
const PAPER = buildPdf([
  { lines: ["Memory in networks", "Recurrent networks remember what came before."] },
  {
    lines: [
      "References",
      "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.",
      "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature. doi:10.1038/nature14539.",
      INVENTED,
    ],
  },
]);

let dataDir: string;
let sources: string;
let services: ReferenceServices;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
  services = await startReferenceServices({ works: [LSTM, DEEP] });
});
test.afterEach(async () => {
  await services.close();
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

/** Launches with the scripted model and the Crossref stand-in, and verifies the paper. */
async function verifyPaper(width: number) {
  await writeFile(join(sources, "Memory paper.pdf"), PAPER);
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    referenceLookup: services.url,
  });
  await setWindowSize(app, window, width, 900);
  await dismissChatSetup(window);
  // The scripted model, as the Ollama model the setup screen would save, finds the claims.
  await window.evaluate(async () => {
    await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveChatProvider({
      kind: "ollama",
      modelId: "fake-model",
    });
  });
  await interceptOpenDialog(app, join(sources, "Memory paper.pdf"));
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-verify-document"));
  await expect(window.getByTestId("mind-title")).toHaveValue("Memory paper");
  return { app, window };
}

/** Records each consent request the core raises from now on. */
async function recordConsentRequests(window: Page) {
  await window.evaluate(() => {
    const scope = globalThis as unknown as { incarnamind: CoreBridge; asked: string[] };
    scope.asked = [];
    scope.incarnamind.on("consent.requested", (request) => scope.asked.push(request.flow.id));
  });
  return () => window.evaluate(() => (globalThis as unknown as { asked: string[] }).asked);
}

test("a verified document's references are looked up, and the invented one is flagged", async () => {
  const { app, window } = await verifyPaper(1440);
  const asked = await recordConsentRequests(window);

  const line = window.getByTestId("references-line");
  await expect(line.getByTestId("references-sends")).toHaveText(
    "Looks up the list's 3 references in Crossref, sending only each reference as written.",
  );
  await screenshot(window, "references-start-en");
  await pointAndClick(window, line.getByTestId("references-check"));

  await expect(line.getByTestId("references-done")).toHaveText("Checked 3 references·1 not found");
  // The one that needs a look comes first, saying how much its mark can say.
  const listed = window.getByTestId("reference-check");
  await expect(listed).toHaveCount(3);
  const flagged = listed.first();
  await expect(flagged).toHaveAttribute("data-mark", "not-found");
  await expect(flagged).toContainText("We couldn't find this work");
  await expect(flagged).toContainText("Quantum gradient folding");
  await expect(flagged).toContainText("Crossref has no record like it.");
  // The found ones follow, each with the record it matched and its DOI, which opens the work.
  await expect(listed.nth(1)).toHaveAttribute("data-mark", "found");
  const record = window.getByTestId("reference-record").first();
  await expect(record).toContainText(
    "Long Short-Term Memory · Hochreiter, Sepp; Schmidhuber, Jürgen · 1997",
  );
  await expect(record.getByTestId("reference-doi")).toHaveAttribute(
    "href",
    "https://doi.org/10.1162/neco.1997.9.8.1735",
  );
  await screenshot(window, "references-flagged-en");
  // Clicking it is the consent: nothing asked; only the references went, one a request.
  expect(await asked()).toEqual([]);
  expect(services.requests.length).toBeGreaterThanOrEqual(3);
  for (const request of services.requests) {
    expect(decodeURIComponent(request.path)).not.toContain("Recurrent networks remember");
  }

  // Shown by keyboard too: the entry moves to its Block.
  await flagged.focus();
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("mind-editor")).toBeFocused();
  await app.close();
});

test("in Chinese and a narrow window, and in local mode, where it says why and offers the way out", async () => {
  const { app, window } = await verifyPaper(900);
  await setLanguage(window, "zh-CN");
  await window.evaluate(async () => {
    await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.setLocalOnly(true);
  });

  const local = window.getByTestId("references-local-mode");
  await expect(local).toContainText("无法查找这 3 条参考文献");
  await screenshot(window, "references-local-zh");
  await pointAndClick(window, local.getByTestId("references-local-mode-off"));

  const line = window.getByTestId("references-line");
  await expect(line.getByTestId("references-sends")).toHaveText(
    "在 Crossref 中查找列表里的 3 条参考文献，只发送每条参考文献的原文。",
  );
  await pointAndClick(window, line.getByTestId("references-check"));
  await expect(line.getByTestId("references-done")).toHaveText("已核对 3 条参考文献·1 条未找到");
  await expect(window.getByTestId("reference-check").first()).toContainText("找不到这部作品");
  await screenshot(window, "references-flagged-zh");
  // Narrow, the line wraps rather than overflowing the Mind.
  const lineBox = await line.boundingBox();
  const editorBox = await window.getByTestId("mind-editor").boundingBox();
  expect((lineBox?.x ?? 0) + (lineBox?.width ?? 0)).toBeLessThanOrEqual(
    (editorBox?.x ?? 0) + (editorBox?.width ?? 0) + 1,
  );
  await app.close();
});
