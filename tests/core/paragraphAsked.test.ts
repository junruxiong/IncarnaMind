import { describe, expect, test } from "vitest";
import { paragraphAsked } from "../../src/core/answers/paragraphAsked";

describe("The paragraph a request names by its place", () => {
  // protects=a request naming one paragraph by its place gives that paragraph, in English and Chinese; fails_when=another paragraph or none is given; no_claim=paragraphs named by their words.
  test.each([
    ["Make the second paragraph more formal and leave the other paragraphs as they are", 1],
    ["Shorten the second paragraph and keep the rest unchanged", 1],
    ["Shorten the first paragraph", 0],
    ["make my third paragraph clearer", 2],
    ["Rewrite paragraph 2", 1],
    ["Fix the 4th paragraph", 3],
    ["Tighten the last paragraph", "last"],
    ["把第二段改得更正式，其他段落保持不变", 1],
    ["把第三段缩短", 2],
    ["把第10段删掉", 9],
    ["可以把第一段改短吗？", 0],
    ["最后一段写长一点", "last"],
  ] as const)("%s names paragraph %s", (request, paragraph) => {
    expect(paragraphAsked(request)).toBe(paragraph);
  });

  // protects=a request that names no place, several, or one to leave alone names none, so nothing is narrowed; fails_when=one paragraph is picked out; no_claim=none beyond protects.
  test.each([
    "Make my notes more formal",
    "Shorten the paragraphs",
    "Move the second paragraph after the third",
    "Make the second paragraph more like the first",
    "Merge the first and second paragraphs",
    "Shorten everything except the second paragraph",
    "Rewrite it all but the last paragraph",
    "Keep the first paragraph and shorten the rest",
    "把第二段和第三段合并",
    "除了第二段，其他都缩短",
    "第二段保持不变，其他的改正式一点",
    "把第二段移到最后",
    "把第十一段删掉",
    "Don't change the second paragraph, shorten the rest",
    "第二段不要改，其他的缩短",
    "The second paragraph is fine, tighten the others",
    "Make the whole thing more formal, especially the second paragraph",
    "整体改得更正式，尤其是第二段",
  ])("%s names none", (request) => {
    expect(paragraphAsked(request)).toBeNull();
  });
});
