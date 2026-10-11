/**
 * A stand-in for the text recognition model (#397), for tests and the smoke
 * tests: on page n it "reads" the lines its script gives for page n,
 * whatever the picture shows, laid out down the page as a scan's lines are.
 * It needs no files. Words the script calls unsure are read with a low
 * confidence, the rest with a high one.
 */
import type { RecognizedLine, TextRecognizer } from "../../adapters";

/** What the fake reads on each page, by page number. */
export interface FakeScanScript {
  pages: Record<string, { lines: string[]; unsure?: string[] }>;
}

/** Where the fake puts line `index` of a page, as fractions of its width and height. */
export const fakeLineBox = (index: number, text: string) => {
  const left = 0.1;
  const top = 0.08 + index * 0.035;
  return { left, top, right: Math.min(0.9, left + 0.012 * [...text].length), bottom: top + 0.025 };
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
      const unsure = new Set(script?.unsure ?? []);
      return (script?.lines ?? []).map((text, index): RecognizedLine => {
        const box = fakeLineBox(index, text);
        const x0 = box.left * picture.width;
        const x1 = box.right * picture.width;
        const y0 = box.top * picture.height;
        const y1 = box.bottom * picture.height;
        const pieces = [...text];
        // Each character's confidence is its word's.
        const confidences: number[] = [];
        for (const word of text.split(/(\s+)/)) {
          for (const _ of word) confidences.push(unsure.has(word) ? 0.4 : 0.98);
        }
        const characters = pieces.map((piece, at) => ({
          text: piece,
          confidence: confidences[at] ?? 0.98,
          at: (at + 0.5) / pieces.length,
        }));
        return {
          text,
          quad: [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
          ],
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
