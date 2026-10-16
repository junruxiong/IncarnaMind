import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import { claimSpans } from "../../src/renderer/src/editor/claimChecks";
import { checkingModel, type Plan } from "../helpers/checks";
import { type SourceFile, setUpWithDocuments } from "../helpers/citations";
import { waitForProcessing } from "../helpers/documents";
import { readMind, writeMind } from "../helpers/minds";

const TIDES: SourceFile = {
  name: "Tides.md",
  contents: "# Tides\n\nHarbours publish the times of high water every year.",
};

const OPINION: SourceFile = {
  name: "Hallmark Builders v Rio Grande Supply.md",
  contents: [
    "# Hallmark Builders, Inc. v. Rio Grande Supply Co.",
    "",
    "512 S.W.3d 101 (Tex. 2017)",
    "",
    "A supplier lien attaches when the materials are delivered to the site.",
  ].join("\n"),
};

const CITATION = "Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017)";
const INVENTED = `See ${CITATION}.`;
const LIEN = "A supplier lien attaches when the materials are delivered to the site";
const LIEN_CITED = `${LIEN}, see ${CITATION}.`;
const LIEN_PLAIN = `${LIEN}.`;
const PENGUINS = "Penguins nest on the cliffs above the harbour.";
const PENGUINS_CITED = `Penguins nest on cliffs, as held in ${CITATION}.`;

const FILLER = Array.from(
  { length: 6 },
  () =>
    "<p>Construction law lets suppliers of materials secure payment against the property improved with them, provided they give the owner written notice within the time the statute allows.</p>",
).join("");

const supports: Plan = { verdict: "supported", quote: `${LIEN}.`, says: "It says so." };
const PLANS: Record<string, Plan> = {
  [INVENTED]: { verdict: "unsupported" },
  [PENGUINS]: { verdict: "unsupported" },
  [PENGUINS_CITED]: { verdict: "unsupported" },
  [LIEN_CITED]: supports,
  [LIEN_PLAIN]: supports,
};

/** Checks the sentences in one paragraph, and gives each check as it is on its mark. */
async function checked(sentences: readonly string[], files: readonly SourceFile[], pages = false) {
  const setup = await setUpWithDocuments(checkingModel(PLANS), files);
  if (pages) {
    const saved = await setup.core.saveWebPage({
      address: "https://reader.example/hallmark",
      html: `<html><head><title>Hallmark Builders v. Rio Grande Supply</title></head><body><article><h1>Hallmark Builders v. Rio Grande Supply</h1><p>${LIEN}. ${CITATION}.</p>${FILLER}</article></body></html>`,
    });
    if (saved.outcome !== "saved")
      throw new Error(`The page wasn't saved: ${JSON.stringify(saved)}`);
    await waitForProcessing(setup.core, [saved.document.id]);
  }
  const body = sentences.join(" ");
  const block: JSONContent & { attrs: { id: string } } = {
    type: "paragraph",
    attrs: { id: "block-court" },
    content: [{ type: "text", text: body }],
  };
  writeMind(setup.client, [block]);
  await setup.client.settled();
  await setup.core.checkSentences({
    requestId: crypto.randomUUID(),
    mindId: setup.mind.id,
    action: "find-support",
    from: { blockId: "block-court", offset: 0 },
    to: { blockId: "block-court", offset: body.length },
  });
  await setup.client.settled();
  return claimSpans(readMind(setup.client)).map((span) => span.check);
}

describe("A sentence that cites a court decision", () => {
  test("no Document holds is unsupported, and its check says it cites a decision; the sentence beside it doesn't", async () => {
    const checks = await checked([INVENTED, PENGUINS], [TIDES]);

    expect(checks.map((each) => [each.text, each.state, each.citesDecision])).toEqual([
      [INVENTED, "unsupported", true],
      [PENGUINS, "unsupported", false],
    ]);
  });

  test("is checked against its opinion when it is among the Documents, as any sentence is", async () => {
    const checks = await checked([LIEN_CITED, PENGUINS_CITED], [TIDES, OPINION]);

    expect(checks.map((each) => [each.text, each.state, each.citesDecision])).toEqual([
      [LIEN_CITED, "supported", false],
      [PENGUINS_CITED, "unsupported", false],
    ]);
    expect(checks[0]?.evidence?.quote).toBe(`${LIEN}.`);
  });

  test("is never supported by a page saved from the web, though the same words are, uncited", async () => {
    const checks = await checked([LIEN_CITED, LIEN_PLAIN], [TIDES], true);

    const [cited, plain] = checks;
    expect(plain?.state).toBe("supported");
    expect(cited?.state).not.toBe("supported");
    expect(cited?.evidence).toBeNull();
  });
});
