import { describe, expect, test } from "vitest";
import { withoutSearchTalk } from "../../src/core/answers/searchTalk";

describe("Text written into a Mind, without talk of searching", () => {
  // protects=sentences about the assistant's searching stay out of the Mind, in English and Chinese; fails_when=one is written into the page; no_claim=every wording a model may use.
  test.each([
    "Where the searches found little, that is stated.",
    "The Passages retrieved did not include their content.",
    "My searches returned no passages from either tea article.",
    "The search limit cut me off before the last three files.",
    "检索到的段落中没有关于方法的内容。",
  ])("%s", (sentence) => {
    const { markdown, left } = withoutSearchTalk(
      `## Summary\n\nTides follow the moon. ${sentence}`,
    );
    expect(markdown).toBe("## Summary\n\nTides follow the moon.");
    expect(left).toBe(1);
  });

  // protects=the Documents' own words about search, figures and lists stay as written; fails_when=content is cut; no_claim=none beyond protects.
  test.each([
    "Search results returned by the engine show a 3.5× lift [^1].",
    "The paper found that leverage is capped at 3.5× (clause 7.1).",
    "- Revenue rose 3% to £27.4bn.\n- Net debt was £2.6bn.",
    "| Search | Found |\n|---|---|\n| a | b |",
  ])("%s", (text) => {
    expect(withoutSearchTalk(text)).toEqual({ markdown: text, left: 0 });
  });

  // protects=a list item or paragraph left empty goes, and the rest keeps its shape; fails_when=an empty bullet or paragraph is written; no_claim=none beyond protects.
  test("an item or paragraph left with nothing goes", () => {
    const { markdown, left } = withoutSearchTalk(
      "- Tides follow the moon.\n- My searches found nothing on neap tides.\n\nThe Passages retrieved didn't cover the methods.\n\nSpring tides are higher.",
    );
    expect(markdown).toBe("- Tides follow the moon.\n\nSpring tides are higher.");
    expect(left).toBe(2);
  });
});
