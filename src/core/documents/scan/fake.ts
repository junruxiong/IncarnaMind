/**
 * A stand-in for the text recognition model (#397), for tests and the smoke
 * tests: on page n it "reads" the lines its script gives for page n,
 * whatever the picture shows, laid out down the page as a scan's lines are.
 * It needs no files. Words the script calls unsure are read with a low
 * confidence, the rest with a high one; a page it says fails, fails.
 */
import type { RecognizedLine, TextRecognizer } from "../../adapters";

/**
 * A line the fake reads: its text, or its text with its box (as fractions of
 * the picture: x, y of top-left, top-right, bottom-right, bottom-left) and
 * the words in it to read as unsure, as tests/fixtures/scans/read.json
 * records what the real model read.
 */
export type FakeScanLine = string | { text: string; quad?: number[]; unsure?: string[] };

/** What the fake reads on each page, by page number; or why it fails on one. */
export interface FakeScanScript {
  pages: Record<string, { lines?: FakeScanLine[]; unsure?: string[]; fail?: string }>;
}

/** Where the fake puts line `index` of a page that gives no box, as fractions of its size. */
const lineBox = (index: number, text: string): number[] => {
  const left = 0.1;
  const top = 0.08 + index * 0.035;
  const right = Math.min(0.9, left + 0.012 * [...text].length);
  const bottom = top + 0.025;
  return [left, top, right, top, right, bottom, left, bottom];
};

export function createFakeTextRecognizer(
  options: { script?: FakeScanScript; delayMs?: number } = {},
): TextRecognizer {
  let loaded = false;
  return {
    async load() {
      loaded = true;
    },
    async read(picture, page) {
      if (!loaded) throw new Error("The text recognition model isn't loaded.");
      if (options.delayMs) await new Promise((resolve) => setTimeout(resolve, options.delayMs));
      const script = options.script?.pages[String(page)];
      if (script?.fail) throw new Error(script.fail);
      return (script?.lines ?? []).map((line, index): RecognizedLine => {
        const text = typeof line === "string" ? line : line.text;
        const unsure = new Set([
          ...(script?.unsure ?? []),
          ...(typeof line === "string" ? [] : (line.unsure ?? [])),
        ]);
        const fractions = (typeof line === "string" ? null : line.quad) ?? lineBox(index, text);
        const at = (corner: number): [number, number] => [
          (fractions[corner * 2] ?? 0) * picture.width,
          (fractions[corner * 2 + 1] ?? 0) * picture.height,
        ];
        // Where each character starts in the text, and whether an unsure word covers it.
        const pieces = [...text];
        const starts: number[] = [];
        let offset = 0;
        for (const piece of pieces) {
          starts.push(offset);
          offset += piece.length;
        }
        const doubtful = new Set<number>();
        for (const word of unsure) {
          if (!word) continue;
          for (let found = text.indexOf(word); found >= 0; found = text.indexOf(word, found + 1)) {
            for (let at = found; at < found + word.length; at++) doubtful.add(at);
          }
        }
        const characters = pieces.map((piece, position) => ({
          text: piece,
          confidence: doubtful.has(starts[position] ?? -1) ? 0.4 : 0.98,
          at: (position + 0.5) / pieces.length,
        }));
        return {
          text,
          quad: [at(0), at(1), at(2), at(3)],
          confidence: characters.reduce((sum, each) => sum + each.confidence, 0) / pieces.length,
          characters,
        };
      });
    },
    close() {
      loaded = false;
    },
  };
}
