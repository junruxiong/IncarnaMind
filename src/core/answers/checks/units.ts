/** A model reads "54,000" in a UK contract as pounds and calls "£54,000" supported (#1332), so a program decides. */
import {
  currenciesOf,
  type Figure,
  type FigureContext,
  type FigureUnit,
  namedCurrencies,
  type Reading,
  readFigures,
} from "../../../shared/figures";
import { documentFigureContext } from "../calculations";

export type UnitNotStated =
  /** `currency` as the sentence writes it ("£", "英镑"), `amount` as the Document does ("54,000"). */
  | { kind: "currency"; currency: string; amount: string }
  /** A percent where the Document gives percentage points, or points where it gives a percent. */
  | { kind: "unit"; figure: string; amount: string };

/** The text a sentence is checked against, read only when it is needed. */
export interface CheckedText {
  /** The Units its Citation points to, or its quote. */
  cited(): string;
  /** Its Document's whole text. */
  document(): string;
}

const sameValue = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));

const isShare = (unit: FigureUnit) => unit.kind === "percent" || unit.kind === "points";

const readingsOf = (figure: Figure): Reading[] => [
  figure,
  ...figure.ambiguous.flatMap((each) => each.readings),
];

const hasUnit = (figure: Figure) => currenciesOf(figure).length > 0 || isShare(figure.unit);

/**
 * The sentence's figures that carry a currency, a percent or points, and nothing else unsettled
 * but their separators: read with the cited text's context, the same digits read the same.
 */
function figuresToCheck(claim: string, context: FigureContext): Figure[] {
  return readFigures(claim, context).filter(
    (figure) =>
      hasUnit(figure) &&
      figure.ambiguous.every((each) => each.kind === "currency" || each.kind === "separator"),
  );
}

/**
 * What `claim` gives a figure that its cited text states without it: a currency on a plain
 * number, which neither the cited Units nor the Document's words name; a percent where the text
 * gives percentage points; or points where it gives a percent. Null when it gives none. A figure
 * the cited text doesn't state, or states with another currency or unit, is the model's to judge.
 */
export function unitNotStated(claim: string, text: CheckedText): UnitNotStated | null {
  const own = documentFigureContext([{ text: claim }]);
  if (!readFigures(claim, own).some(hasUnit)) return null;
  const cited = text.cited();
  const context = documentFigureContext([{ text: cited }]);
  const there = readFigures(cited, context);
  let named: Set<string> | null = null;
  let inWords: Set<string> | null = null;
  for (const figure of figuresToCheck(claim, own.language ? own : context)) {
    const same = there.flatMap((each) =>
      readingsOf(each)
        .filter((reading) => sameValue(reading.value, figure.value))
        .map((reading) => ({ chars: each.chars, unit: reading.unit })),
    );
    if (same.length === 0) continue;
    if (isShare(figure.unit)) {
      if (same.some(({ unit }) => unit.kind === figure.unit.kind)) continue;
      const other = same.find(({ unit }) => isShare(unit));
      if (other) return { kind: "unit", figure: figure.chars, amount: other.chars };
      continue;
    }
    if (same.some(({ unit }) => unit.kind !== "plain")) continue;
    const codes = currenciesOf(figure);
    named ??= namedCurrencies(cited, context);
    if (codes.some((code) => named?.has(code))) continue;
    inWords ??= namedCurrencies(text.document(), context, { wordsOnly: true });
    if (codes.some((code) => inWords?.has(code))) continue;
    return {
      kind: "currency",
      currency: figure.currency ?? codes[0] ?? "",
      amount: same[0]?.chars ?? figure.chars,
    };
  }
  return null;
}
