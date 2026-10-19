/**
 * What the app names in a document the pane reads: the sentence to comment on in
 * Word, the cell to add a note to in Excel. The spike picks the first figure
 * (a sentence with a digit, a cell with a formula) so the round trip is
 * checkable; the real checks (#96) replace it.
 */

export interface SentenceFinding {
  paragraph: number;
  sentence: string;
  comment: string;
}

export interface CellFinding {
  row: number;
  column: number;
  note: string;
}

const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });

export function nameSentence(paragraphs: readonly string[]): SentenceFinding | null {
  let firstSentence: SentenceFinding | null = null;
  for (const [paragraph, text] of paragraphs.entries()) {
    for (const { segment } of segmenter.segment(text)) {
      const sentence = segment.trim();
      if (sentence.length === 0) continue;
      const found = { paragraph, sentence, comment: `Checked by IncarnaMind: ${sentence}` };
      if (/\d/.test(sentence)) return found;
      firstSentence ??= found;
    }
  }
  return firstSentence;
}

export function nameCell(
  values: readonly (readonly unknown[])[],
  formulas: readonly (readonly unknown[])[],
): CellFinding | null {
  let firstNumber: CellFinding | null = null;
  for (const [row, cells] of formulas.entries()) {
    for (const [column, formula] of cells.entries()) {
      const value = values[row]?.[column];
      if (typeof formula === "string" && formula.startsWith("=")) {
        return { row, column, note: `Checked by IncarnaMind: ${formula} gives ${String(value)}` };
      }
      if (typeof value === "number") {
        firstNumber ??= { row, column, note: `Checked by IncarnaMind: ${value}` };
      }
    }
  }
  return firstNumber;
}
