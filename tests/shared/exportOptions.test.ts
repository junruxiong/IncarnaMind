import { describe, expect, test } from "vitest";
import type { ExportPreferences } from "../../src/core/api";
import { optionsFor } from "../../src/shared/exportOptions";

const preferences = (markAiText: "mark" | "leave" | null): ExportPreferences => ({
  format: "docx",
  docx: { includeQuestions: false, trackChanges: true, markAiText, commentFailedChecks: true },
  pdf: { includeQuestions: false },
  html: { includeQuestions: false },
  "share-page": { includeQuestions: false, citationCards: true, checkedLine: false },
});

describe("What an export asks the core for", () => {
  test("a Word export waits until the User has said whether to mark the text IncarnaMind wrote", () => {
    expect(optionsFor("docx", preferences(null))).toBeNull();
  });

  test.each([
    ["mark", true],
    ["leave", false],
  ] as const)("a Word export that %s the text IncarnaMind wrote asks for %s", (answer, marked) => {
    expect(optionsFor("docx", preferences(answer))).toEqual({
      format: "docx",
      includeQuestions: false,
      trackChanges: true,
      markAiText: marked,
      commentFailedChecks: true,
    });
  });

  test("the other formats don't wait for it", () => {
    for (const format of ["pdf", "html", "markdown", "source-pack", "share-page"] as const) {
      expect(optionsFor(format, preferences(null))).not.toBeNull();
    }
  });
});
