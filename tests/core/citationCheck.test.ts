/**
 * The Citation check, case by case, in English and Chinese (#31): the text
 * PDFs give and the quotes models write differ in hyphenation, ligatures,
 * full-width punctuation and CJK spacing, quotes run across page breaks, and
 * models cite page ranges that break the page-range rule. eval/README.md
 * lists these cases with the end-to-end tests that cover them too.
 *
 * Each case is a Document's stored page text (as processing stores it, see
 * `pageTexts`), a Citation's pages and the pages of its Passage, and the quote.
 */
import { describe, expect, test } from "vitest";
import type { CitationCheckReason } from "../../src/core";
import { type CheckResult, checkCitation } from "../../src/core/answers/citations";

interface Case {
  language: "en" | "zh";
  topic: string;
  name: string;
  /** The Document's pages, from 1. */
  pages: string[];
  /** The cited pages. */
  cited: [number, number];
  /** The cited Passage's pages. */
  passage: [number, number];
  quote: string;
  /** "found", or why not. */
  expected: "found" | CitationCheckReason;
}

const cases: Case[] = [
  // Hyphenation at line ends
  {
    language: "en",
    topic: "hyphenation",
    name: "a word split by a hyphen at a line end is joined",
    pages: ["Trade figures\nGrowth in inter-\nnational trade slowed in 2022."],
    cited: [1, 1],
    passage: [1, 1],
    quote: "Growth in international trade slowed in 2022.",
    expected: "found",
  },
  {
    language: "en",
    topic: "hyphenation",
    name: "a hyphenated compound broken after its hyphen keeps the hyphen",
    pages: ["On the WMT 2014 English-\nto-German translation task, the big model wins."],
    cited: [1, 1],
    passage: [1, 1],
    quote: "On the WMT 2014 English-to-German translation task",
    expected: "found",
  },
  {
    language: "zh",
    topic: "hyphenation",
    name: "an English term split at a line end inside Chinese text is joined",
    pages: ["用梯度下降法优化 Rosen-\nbrock 函数时，收敛非常缓慢。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: "用梯度下降法优化Rosenbrock函数时,收敛非常缓慢。",
    expected: "found",
  },

  // Ligatures and other compatibility characters
  {
    language: "en",
    topic: "ligatures",
    name: "fi, fl and ffi ligatures on the page match plain letters in the quote",
    pages: ["The ﬁrst ﬂoor was ofﬁcially opened in May."],
    cited: [1, 1],
    passage: [1, 1],
    quote: "The first floor was officially opened in May.",
    expected: "found",
  },
  {
    language: "zh",
    topic: "ligatures",
    name: "a ligature in an English term inside Chinese text",
    pages: ["模型经过 ﬁne-tuning 之后，效果明显更好。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: "模型经过fine-tuning之后,效果明显更好",
    expected: "found",
  },

  // Full-width forms and punctuation
  {
    language: "en",
    topic: "full-width punctuation",
    name: "full-width letters, digits and hyphen in the quote match ASCII on the page",
    pages: ["GPT-3 has 175 billion parameters in total."],
    cited: [1, 1],
    passage: [1, 1],
    quote: "ＧＰＴ－３ has １７５ billion parameters",
    expected: "found",
  },
  {
    language: "zh",
    topic: "full-width punctuation",
    name: "full-width colon, comma and full stop on the page match ASCII in the quote",
    pages: ["该目标要求：到 2030 年，让全球因为交通事故而伤亡的人数减少一半。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: "该目标要求:到2030年,让全球因为交通事故而伤亡的人数减少一半.",
    expected: "found",
  },
  {
    language: "zh",
    topic: "full-width punctuation",
    name: "Chinese quotation marks and full-width brackets match their ASCII forms",
    pages: ["注意力机制的灵活性来自于它的“软权重”特性（soft weights）。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: '注意力机制的灵活性来自于它的"软权重"特性(soft weights)',
    expected: "found",
  },

  // CJK text
  {
    language: "en",
    topic: "CJK",
    name: "a Chinese term in English text matches whatever the spacing around it",
    pages: ["The Chinese term 大语言模型 appears in most survey titles."],
    cited: [1, 1],
    passage: [1, 1],
    quote: "The Chinese term大语言模型appears in most survey titles",
    expected: "found",
  },
  {
    language: "zh",
    topic: "CJK",
    name: "radical look-alikes and the spaces pdf.js adds around Latin words and numbers",
    pages: ["例如 GPT-3 含 1750 亿参数，是一个 ⼤型语⾔模型。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: "GPT-3含1750亿参数,是一个大型语言模型",
    expected: "found",
  },
  {
    language: "zh",
    topic: "CJK",
    name: "simplified characters don't match traditional ones (ADR-0009)",
    pages: ["歐盟因此於 2021 年 3 月制定永續財務揭露法規。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: "欧盟因此于2021年3月制定永续财务揭露法规",
    expected: "quote-not-on-pages",
  },

  // Quotes across a page break
  {
    language: "en",
    topic: "page break",
    name: "a sentence running from one page onto the next, citing both",
    pages: [
      "Tides are the rise and fall of the sea.",
      "High water comes twice a day.\nNeap tides occur when the Sun and the Moon",
      "pull at right angles to each other.\nTheir range is the smallest.",
    ],
    cited: [2, 3],
    passage: [1, 3],
    quote: "Neap tides occur when the Sun and the Moon pull at right angles to each other.",
    expected: "found",
  },
  {
    language: "en",
    topic: "page break",
    name: "a word hyphenated across the page break",
    pages: ["Delegates from every region and the inter-", "national community agreed on the text."],
    cited: [1, 2],
    passage: [1, 2],
    quote: "the international community agreed",
    expected: "found",
  },
  {
    language: "en",
    topic: "page break",
    name: "citing only the first of the two pages a quote runs across",
    pages: ["Neap tides occur when the Sun and the Moon", "pull at right angles to each other."],
    cited: [1, 1],
    passage: [1, 2],
    quote: "Neap tides occur when the Sun and the Moon pull at right angles to each other.",
    expected: "quote-not-on-pages",
  },
  {
    language: "zh",
    topic: "page break",
    name: "a Chinese sentence running onto the next page, citing both",
    pages: ["潮汐是海水的周期性涨落。\n月球的引力使地球两侧的海水隆起，", "形成两个潮汐隆起。"],
    cited: [1, 2],
    passage: [1, 2],
    quote: "月球的引力使地球两侧的海水隆起，形成两个潮汐隆起。",
    expected: "found",
  },

  // The page-range rule: at most two consecutive pages, within the Passage's pages
  {
    language: "en",
    topic: "page range",
    name: "three pages cited",
    pages: ["Tides rise and fall.", "Spring tides happen at full moon.", "Neap tides are small."],
    cited: [1, 3],
    passage: [1, 3],
    quote: "Spring tides happen at full moon.",
    expected: "too-many-pages",
  },
  {
    language: "en",
    topic: "page range",
    name: "a page outside the cited Passage",
    pages: ["Tides rise and fall.", "Spring tides happen at full moon.", "Neap tides are small."],
    cited: [2, 2],
    passage: [3, 3],
    quote: "Spring tides happen at full moon.",
    expected: "pages-outside-passage",
  },
  {
    language: "zh",
    topic: "page range",
    name: "three pages cited",
    pages: ["潮汐是海水的周期性涨落。", "大潮发生在新月和满月时。", "小潮的潮差最小。"],
    cited: [1, 3],
    passage: [1, 3],
    quote: "大潮发生在新月和满月时。",
    expected: "too-many-pages",
  },
  {
    language: "zh",
    topic: "page range",
    name: "a page outside the cited Passage",
    pages: ["潮汐是海水的周期性涨落。", "大潮发生在新月和满月时。", "小潮的潮差最小。"],
    cited: [2, 2],
    passage: [3, 3],
    quote: "大潮发生在新月和满月时。",
    expected: "pages-outside-passage",
  },

  // The match is exact
  {
    language: "en",
    topic: "exact match",
    name: "a paraphrase isn't found",
    pages: ["Revenue grew by ten percent in the third quarter."],
    cited: [1, 1],
    passage: [1, 1],
    quote: "Revenue rose 10% in the third quarter.",
    expected: "quote-not-on-pages",
  },
  {
    language: "zh",
    topic: "exact match",
    name: "a paraphrase isn't found",
    pages: ["优化过程是之字形的向极小值点靠近，速度非常缓慢。"],
    cited: [1, 1],
    passage: [1, 1],
    quote: "优化过程以之字形缓慢地接近极小值",
    expected: "quote-not-on-pages",
  },
];

/** The check as the core runs it: on the stored text of the cited pages only. */
function check({ pages, cited, passage, quote }: Case): CheckResult {
  const [from, to] = cited;
  return checkCitation({
    quote,
    range: { pageFrom: from, pageTo: to },
    passage: { pageFrom: passage[0], pageTo: passage[1] },
    documentDeleted: false,
    pages: pages
      .map((text, index) => ({ page: index + 1, text }))
      .filter(({ page }) => page >= from && page <= to),
  });
}

describe("The Citation check, case by case", () => {
  test.each(cases)("$language, $topic: $name", (each) => {
    expect(check(each)).toEqual(
      each.expected === "found"
        ? { check: "found", checkReason: null }
        : { check: "not-found", checkReason: each.expected },
    );
  });

  test("covers every topic in both languages", () => {
    const covered = (language: Case["language"]) =>
      new Set(cases.filter((each) => each.language === language).map((each) => each.topic));
    expect([...covered("en")].sort()).toEqual([...covered("zh")].sort());
  });
});
