import { describe, expect, test } from "vitest";
import { documentTitle } from "../../src/core/documents/title";
import { displayName, fileNameOf, looksMachineMade } from "../../src/shared/documentNames";

describe("machine-made Document names", () => {
  test.each([
    "49ed72de-a284-4da6-b515-1ddbb7c0a8f1",
    "49ED72DEA2844DA6B5151DDBB7C0A8F1",
    "3f9a1c0b7d2e4a68",
    "20261010142701",
    "2026-10-10 14.27.01",
    "Untitled",
    "untitled 2",
    "document",
    "Document (3)",
    "IMG_0042",
    "Scan 12",
    "a8f3k2m9q1x7z4c6v5b0n8l2",
    "  ",
  ])("%j is machine-made", (name) => {
    expect(looksMachineMade(name)).toBe(true);
  });

  test.each([
    "Quarterly report",
    "Reading notes",
    "37-38UpperGrosvenorStreet",
    "13Clapeyron-EL",
    "Contract 2025",
    "Tide notes v2",
    "Documentation",
    "关于潮汐的笔记",
  ])("%j is a name a person gave", (name) => {
    expect(looksMachineMade(name)).toBe(false);
  });
});

describe("the name shown for a Document", () => {
  test("a machine-made name gives way to the title; a name given never does", () => {
    expect(displayName({ name: "49ed72de-a284-4da6-b515-1ddbb7c0a8f1", title: "Rent roll" })).toBe(
      "Rent roll",
    );
    expect(displayName({ name: "Untitled", title: "Rent roll" })).toBe("Rent roll");
    expect(displayName({ name: "Quarterly report", title: "Something else" })).toBe(
      "Quarterly report",
    );
  });

  test("without a title, the name stays", () => {
    expect(displayName({ name: "Untitled", title: null })).toBe("Untitled");
    expect(displayName({ name: "Untitled" })).toBe("Untitled");
    expect(displayName({ name: "Untitled", title: "  " })).toBe("Untitled");
  });

  test("the file's name is the last part of its path", () => {
    expect(fileNameOf("/Users/me/Reports/Q3.xlsx")).toBe("Q3.xlsx");
    expect(fileNameOf("C:\\Users\\me\\Q3.xlsx")).toBe("Q3.xlsx");
  });
});

describe("a Document's own title", () => {
  test("the title in its properties comes first", () => {
    expect(documentTitle("  Rent roll  2025 ", "# Heading\nText")).toBe("Rent roll 2025");
  });

  test("a property that says nothing, or is machine-made, gives way to the first heading", () => {
    expect(documentTitle("Untitled", "Intro line\n# Real heading\nText")).toBe("Real heading");
    expect(documentTitle("", "## **Bold heading** ##\nText")).toBe("Bold heading");
  });

  test("without a heading, the first line of words", () => {
    expect(documentTitle(null, "\n\n  Minutes of the tenants' meeting\nSecond line")).toBe(
      "Minutes of the tenants' meeting",
    );
  });

  test("a long first line is cut at a word", () => {
    const title = documentTitle(null, `${"word ".repeat(40)}end`);
    expect(title?.endsWith("…")).toBe(true);
    expect(title?.length).toBeLessThanOrEqual(101);
  });

  test("nothing to take gives null", () => {
    expect(documentTitle(null, null)).toBeNull();
    expect(documentTitle(null, "   \n  ")).toBeNull();
    expect(documentTitle(null, "49ed72de-a284-4da6-b515-1ddbb7c0a8f1")).toBeNull();
  });
});
