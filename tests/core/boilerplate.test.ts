import { describe, expect, test } from "vitest";
import { stripBoilerplate } from "../../src/core/documents/boilerplate";

/** Body lines that differ from page to page in their words, not only their numbers. */
const BODIES = [
  "Tides rise twice a day.",
  "Spring tides follow the full moon.",
  "Neap tides are the smallest.",
  "Harbours publish tide tables.",
  "Sailors read them every morning.",
  "Storm surges add to the tide.",
  "The Moon pulls on the oceans.",
  "The Sun pulls on them too.",
  "Estuaries amplify the range.",
  "Some seas barely have tides.",
  "Tidal power turns turbines.",
  "Charts give heights above datum.",
];

describe("Running headers, footers and page numbers", () => {
  test("lines repeated at the edges of enough pages are removed, page numbers included", () => {
    const pages = [1, 2, 3, 4].map((number) => ({
      page: number,
      text: [
        "Annual Report 2026",
        BODIES[number] as string,
        "Every page has this line in its middle.",
        BODIES[number + 4] as string,
        `Page ${number} of 4`,
      ].join("\n"),
    }));

    expect(stripBoilerplate(pages)).toEqual(
      [1, 2, 3, 4].map((number) => ({
        page: number,
        text: `${BODIES[number]}\nEvery page has this line in its middle.\n${BODIES[number + 4]}`,
      })),
    );
  });

  test("several header and footer lines are peeled from each edge, and bare or roman page numbers go too", () => {
    const romans = ["i", "ii", "iii", "iv"];
    const pages = romans.map((roman, index) => ({
      page: index + 1,
      text: [
        "ACME Corp",
        "Confidential",
        BODIES[index] as string,
        "INTRODUCTION",
        "APPENDICES",
        roman,
      ].join("\n"),
    }));

    expect(stripBoilerplate(pages).map((each) => each.text)).toEqual(BODIES.slice(0, 4));
  });

  test("nothing is removed from Documents with too few pages, long lines, or text without pages", () => {
    const twoPages = [1, 2].map((number) => ({
      page: number,
      text: `Header\n${BODIES[number]}\n${number}`,
    }));
    expect(stripBoilerplate(twoPages)).toEqual(twoPages);

    const long =
      "This long sentence repeats at the top of every page, but it is far too long to be a running header or footer of any kind, so it stays where it is, as body text does, on each and every page it appears on.";
    expect(long.length).toBeGreaterThan(200);
    const longLines = [1, 2, 3].map((number) => ({
      page: number,
      text: `${long}\n${BODIES[number]}`,
    }));
    expect(stripBoilerplate(longLines)).toEqual(longLines);

    const text = [{ page: null, text: "Notes\nNotes\nNotes" }];
    expect(stripBoilerplate(text)).toEqual(text);
  });

  test("a page that says nothing but such a line keeps it", () => {
    const pages = Array.from({ length: 6 }, (_, index) => ({
      page: index + 1,
      text: index === 2 ? `Page 3 of 6\n${BODIES[2]}` : `Page ${index + 1} of 6`,
    }));

    expect(stripBoilerplate(pages).map((each) => each.text)).toEqual([
      "Page 1 of 6",
      "Page 2 of 6",
      BODIES[2],
      "Page 4 of 6",
      "Page 5 of 6",
      "Page 6 of 6",
    ]);
  });

  test("a line repeated on only a few of many pages stays", () => {
    const pages = BODIES.map((body, index) => ({
      page: index + 1,
      text: index < 3 ? `Chapter One\n${body}` : body,
    }));

    expect(stripBoilerplate(pages)).toEqual(pages);
  });
});
