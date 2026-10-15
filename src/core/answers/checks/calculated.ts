/**
 * A figure a sentence states that no Passage does, worked out from numbers
 * the Passages state (#707): which number of the sentence the model named as
 * that figure, and whether it agrees with the figure IncarnaMind calculated
 * (ADR-0021), to the sentence's own rounding. A program decides, not the
 * model: a wrong derived number is what an analyst fears most.
 */
import {
  type Figure,
  type FigureContext,
  type FigureUnit,
  readFigures,
} from "../../../shared/figures";
import type { ResultUnit } from "../../../shared/formula";
import type { CalculatedFigure } from "../../api";
import { documentFigureContext } from "../calculations";

/**
 * The number of `sentence` that `stated` names ("1.1x" in "Net debt to
 * EBITDA fell to 1.1x"), as the sentence writes it; null when the sentence
 * has no such number, or it can be read more than one way. A sentence too
 * short to tell its language by writes its numbers as `document` does, the
 * Document its figure was worked out from.
 */
export function statedFigure(
  sentence: string,
  stated: string,
  document: FigureContext,
): Figure | null {
  const own = documentFigureContext([{ text: sentence }]);
  const context = own.language ? own : { ...document, header: null };
  const [named] = readFigures(stated, context);
  if (!named) return null;
  return (
    readFigures(sentence, context).find(
      (figure) =>
        figure.ambiguous.length === 0 &&
        figure.value === named.value &&
        (named.unit.kind === "plain" || sameUnit(figure.unit, named.unit)),
    ) ?? null
  );
}

const sameUnit = (a: FigureUnit, b: FigureUnit): boolean =>
  a.kind === b.kind &&
  (a.kind !== "currency" || (b.kind === "currency" && a.code === b.code)) &&
  (a.kind !== "other" || (b.kind === "other" && a.word === b.word));

/** Whether a sentence's number and a result measure the same kind of thing, so they can be compared. */
function comparable(stated: FigureUnit, result: ResultUnit): boolean {
  switch (stated.kind) {
    case "percent":
      return result.kind === "percent" || result.kind === "ratio";
    case "points":
      return result.kind === "points";
    case "times":
      return result.kind === "times" || result.kind === "ratio" || result.kind === "plain";
    case "currency":
      return (result.kind === "currency" && result.code === stated.code) || result.kind === "plain";
    case "other":
      return (result.kind === "other" && result.word === stated.word) || result.kind === "plain";
    // A bare number isn't a ratio's: "42" could be 0.42 or 42%, so it is compared only with an amount.
    case "plain":
      return result.kind === "plain" || result.kind === "currency" || result.kind === "other";
  }
}

/**
 * How far a number may be from the value it was rounded from: half its last
 * place as written ("1.1x", 0.05; "25%", half a percent; "£131.9m", £50,000).
 * One written as approximate is looser: "about £130m" was rounded to its last
 * digit that isn't a zero, and "about 1.3x" is a place either way more.
 */
function tolerance(stated: Figure): number {
  const place =
    (stated.unit.kind === "percent" || stated.unit.kind === "points" ? 0.01 : 1) * stated.scale;
  let step = place * 10 ** -stated.decimals;
  let more = 0;
  if (stated.approximate && stated.decimals > 0) more = step;
  else if (stated.approximate) {
    let written = Math.round(Math.abs(stated.value) / place);
    while (written >= 10 && written % 10 === 0) {
      written /= 10;
      step *= 10;
    }
  }
  return step / 2 + more + 1e-9 * Math.max(1, Math.abs(stated.value));
}

/**
 * Whether the sentence's number agrees with the figure worked out for it, to
 * the sentence's rounding: a range agrees when the figure is in it. Null when
 * they measure different things, or the figure is a yes or no.
 */
export function agreesWith(
  stated: Figure,
  figure: Pick<CalculatedFigure, "value" | "unit">,
): boolean | null {
  const { value } = figure;
  if (typeof value === "boolean" || !comparable(stated.unit, figure.unit)) return null;
  const within = tolerance(stated);
  const low = Math.min(stated.value, stated.rangeTo ?? stated.value) - within;
  const high = Math.max(stated.value, stated.rangeTo ?? stated.value) + within;
  return value >= low && value <= high;
}
