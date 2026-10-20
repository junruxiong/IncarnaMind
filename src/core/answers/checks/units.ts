/**
 * A currency a sentence gives a figure that its Document states without one
 * (#1332): "a salary of £54,000" against "The annual salary is 54,000". A
 * model reads "54,000" in a UK contract as pounds and calls the sentence
 * supported, so a program decides.
 */
import { type Figure, readFigures } from "../../../shared/figures";
import { documentFigureContext } from "../calculations";

export interface AddedCurrency {
  /** The currency as the sentence writes it: "£", "英镑", "GBP". */
  currency: string;
  /** The Document's figure as it writes it: "54,000". */
  amount: string;
}

/** What a figure writes besides its number: its currency sign, code or word. */
function currencyAsWritten(figure: Figure): string {
  let rest = figure.chars;
  for (const part of [figure.scaleWord, figure.footnote]) if (part) rest = rest.replace(part, "");
  return rest.replace(/[\d.,'’\s()+\-−–]/gu, "");
}

const sameValue = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));

/**
 * The currency `claim` adds to a figure that `cited` (the Units its Citation points to) states as a
 * plain number, when the Document's whole text (`document`) never names that currency; null when
 * it adds none. A figure the cited text doesn't state, or states with a unit of its own, is the
 * model's to judge.
 */
export function addedCurrency(
  claim: string,
  cited: string,
  document: string,
): AddedCurrency | null {
  const context = documentFigureContext([{ text: document }]);
  const there = readFigures(cited, context).filter((figure) => figure.ambiguous.length === 0);
  const inDocument = readFigures(document, context);
  for (const figure of readFigures(claim, { ...context, header: null })) {
    const { unit } = figure;
    if (unit.kind !== "currency" || figure.ambiguous.length > 0) continue;
    const same = there.filter((each) => sameValue(each.value, figure.value));
    if (same.length === 0 || same.some((each) => each.unit.kind !== "plain")) continue;
    const written = currencyAsWritten(figure);
    const named =
      inDocument.some((each) => each.unit.kind === "currency" && each.unit.code === unit.code) ||
      (written !== "" && document.includes(written)) ||
      new RegExp(`\\b${unit.code}\\b`, "i").test(document);
    if (named) continue;
    return { currency: written || unit.code, amount: same[0]?.chars ?? String(figure.value) };
  }
  return null;
}
