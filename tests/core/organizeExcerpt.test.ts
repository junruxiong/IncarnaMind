import { describe, expect, test } from "vitest";
import {
  documentOutline,
  type OrganizeUnit,
  organizeExcerpt,
  organizeNeedsPageImages,
} from "../../src/core/library/excerpt";

const slide = (page: number, title: string | null, text = "Some points"): OrganizeUnit => ({
  page,
  kind: "slide",
  label: title === null ? null : { title },
  text: title === null ? text : `${title}\n${text}`,
});

describe("what Organize reads of a Document", () => {
  test("a deck's outline numbers its slides by title, or by first line without one", () => {
    const units = [
      slide(1, "Q2 Business Review"),
      slide(2, "Agenda"),
      slide(3, null, "Revenue up 12%\nmore"),
    ];
    expect(documentOutline({ kind: "pptx", units })).toBe(
      "3 slides: 1. Q2 Business Review; 2. Agenda; 3. Revenue up 12%",
    );
  });

  test("a workbook's outline names its sheets once each, in order", () => {
    const units: OrganizeUnit[] = ["Instructions", "Expenses", "Expenses", "Totals"].map(
      (sheet, index) => ({ page: index + 1, kind: "rows", label: { sheet }, text: "a\tb" }),
    );
    expect(documentOutline({ kind: "xlsx", units })).toBe(
      "3 sheets: Instructions; Expenses; Totals",
    );
  });

  test("long outlines say how many more; Word, Markdown, PDFs and plain text have none", () => {
    const slides = Array.from({ length: 20 }, (_, index) => slide(index + 1, `Part ${index + 1}`));
    const outline = documentOutline({ kind: "pptx", units: slides });
    expect(outline).toMatch(/^20 slides: 1\. Part 1; 2\. Part 2;/);
    expect(outline).toMatch(/\(\+4 more\)$/);
    // Their headings are in their text already.
    expect(
      documentOutline({
        kind: "docx",
        units: [{ page: 1, kind: "section", label: { path: ["Report", "Method"] }, text: "x" }],
      }),
    ).toBeNull();
  });

  test("an outline cuts long entries", () => {
    expect(documentOutline({ kind: "pdf", units: [] })).toBeNull();
    expect(documentOutline({ kind: "text", units: [] })).toBeNull();
    const long =
      "A very long slide title that goes on and on well past what an outline needs to show";
    expect(documentOutline({ kind: "pptx", units: [slide(1, long)] })).toMatch(/…$/);
  });

  test("the excerpt keeps the name, kind and page count, and reads at most six Passages", () => {
    const excerpt = organizeExcerpt({
      name: "Q2 review",
      kind: "pdf",
      pageCount: 3,
      passages: ["one", "two", "three", "four", "five", "six", "seven"],
      units: [],
    });
    expect(excerpt).toMatchObject({ name: "Q2 review", kind: "pdf", pageCount: 3 });
    expect(excerpt.text).toContain("six");
    expect(excerpt.text).not.toContain("seven");
  });

  test("a PDF with little text on its first pages needs page images; other formats never do", () => {
    const page = (text: string, n: number): OrganizeUnit => ({
      page: n,
      kind: "page",
      label: null,
      text,
    });
    expect(organizeNeedsPageImages({ kind: "pdf", pageCount: 1, units: [page("Scan", 1)] })).toBe(
      true,
    );
    expect(
      organizeNeedsPageImages({ kind: "pdf", pageCount: 1, units: [page("word ".repeat(200), 1)] }),
    ).toBe(false);
    expect(organizeNeedsPageImages({ kind: "docx", pageCount: null, units: [] })).toBe(false);
  });
});
