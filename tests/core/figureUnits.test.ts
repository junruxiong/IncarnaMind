/**
 * A figure keeps exactly the units its Document states (#1332): a currency the
 * Document doesn't state, added to a figure it does, leaves the sentence
 * unsupported, naming the currency as the part not found; and the assistant is
 * told never to add one.
 */
import { describe, expect, test } from "vitest";
import { addedCurrency } from "../../src/core/answers/checks/units";
import { documentInstructions } from "../../src/core/answers/prompt";
import { checkingModel } from "../helpers/checks";
import { setUpWithDocuments } from "../helpers/citations";

const CONTRACT = {
  name: "Employment agreement.txt",
  contents:
    "Employment agreement between Harbourline Ltd and the Employee.\nThe annual salary is 54,000, paid monthly in arrears.\nThe notice period is three months.\n",
};
const STATED = "The annual salary is 54,000, paid monthly in arrears";

describe("A currency the Document doesn't state", () => {
  // protects=a currency added to a figure the cited text states as a plain number, which the Document never names, is found, as the sentence writes it, in English and Chinese; fails_when="£54,000" against "54,000" passes; no_claim=other units, which the model judges.
  test.each([
    ["The annual salary is £54,000.", "£"],
    ["年薪为 54,000英镑。", "英镑"],
    ["The salary is GBP 54,000.", "GBP"],
    ["Salary: $54k a year.", "$"],
  ])("%s adds %s", (claim, currency) => {
    expect(addedCurrency(claim, STATED, CONTRACT.contents)).toEqual({ currency, amount: "54,000" });
  });

  // protects=no false alarm: a figure written as the Document writes it, a currency the Document names anywhere, or a number it doesn't state is left to the model; fails_when=any of these is flagged; no_claim=none beyond protects.
  test.each([
    ["the figure as written", "The annual salary is 54,000.", STATED, CONTRACT.contents],
    [
      "a currency the Document names elsewhere",
      "The annual salary is £54,000.",
      STATED,
      `${CONTRACT.contents}All amounts are in GBP.\n`,
    ],
    [
      "a currency on the cited figure",
      "The annual salary is £54,000.",
      "The annual salary is £54,000",
      "The annual salary is £54,000.",
    ],
    ["a number the cited text doesn't state", "The bonus is £5,000.", STATED, CONTRACT.contents],
  ])("%s isn't flagged", (_, claim, cited, document) => {
    expect(addedCurrency(claim, cited, document)).toBeNull();
  });
});

describe("Checking a sentence that adds a currency", { timeout: 30_000 }, () => {
  const plans = (claim: string) => ({
    [claim]: { verdict: "supported" as const, quote: STATED, says: "The salary is 54,000." },
  });

  // protects=a cited sentence "£54,000" against the text "54,000" checks Unsupported even when the model calls it supported, naming the currency as the part not found, in English and Chinese (#1332); fails_when=it shows as supported, or names nothing; no_claim=a real model's verdict.
  test("is Unsupported, naming the currency as the part not found", async () => {
    const claim = "The annual salary is £54,000.";
    const { core } = await setUpWithDocuments(checkingModel(plans(claim)), [CONTRACT]);
    const cited = [{ documentName: "Employment agreement", quote: STATED }];

    expect(await core.checkClaim({ claim, cited })).toMatchObject({
      verdict: "unsupported",
      unsupported: {
        reason: "The Document gives 54,000 without a currency.",
        covered: "the amount, 54,000",
        uncovered: "the currency (£)",
      },
    });

    await core.updateSettings({ user: { language: "zh-CN" } });
    expect(await core.checkClaim({ claim, cited })).toMatchObject({
      verdict: "unsupported",
      unsupported: { covered: "金额 54,000", uncovered: "币种（£）" },
    });
  });

  // protects=the same sentence written as the Document writes it stays supported; fails_when=the rule flags a figure that adds nothing; no_claim=none beyond protects.
  test("stays supported when it writes the figure as the Document does", async () => {
    const claim = "The annual salary is 54,000.";
    const { core } = await setUpWithDocuments(checkingModel(plans(claim)), [CONTRACT]);

    expect(await core.checkClaim({ claim })).toMatchObject({ verdict: "supported" });
  });
});

describe("What the assistant is told", () => {
  // protects=every way of citing tells the model to keep each figure's currency, unit and scale as its Passage states them and never add one, with the example in English and Chinese; fails_when=a mode leaves it out; no_claim=what a real model writes, which the pull request's hand check covers.
  test.each(["tools", "structured-output", "none"] as const)("in %s", (mode) => {
    const told = documentInstructions(mode, { total: 1, names: ["Employment agreement"] });
    expect(told).toContain("Give a figure only the currency, unit and scale its Passage states");
    expect(told).toContain("54,000（未注明币种）");
  });
});
