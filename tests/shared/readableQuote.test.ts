import { describe, expect, test } from "vitest";
import { findQuote } from "../../src/shared/quoteMatch";
import { readableQuote, readableQuoteText } from "../../src/shared/readableQuote";
import { normaliseText } from "../../src/shared/text";

/*
 * A quote read as words (#905): the hyphens a PDF puts where its lines break are joined, a quote
 * that stops at one runs to the end of its word, and real hyphens stay. What the Citation check
 * matches is unchanged: these tests keep both in step.
 */

const whole = (text: string) => [{ start: 0, end: text.length }];

describe("a word split at a line end", () => {
  test.each([
    ["em-\nploy the staff", "employ the staff"],
    ["the av-\nerage day", "the average day"],
    ["both for exist-\nence", "both for existence"],
    ["inter\u00ad\nnational", "international"],
    ["em-  \r\n   ploy", "employ"],
  ])("%j reads %j", (raw, read) => {
    expect(readableQuote(raw, whole(raw))).toBe(read);
    expect(readableQuoteText(raw)).toBe(read);
  });

  test("keeps its real hyphen where the page writes it elsewhere, and joins where it doesn't", () => {
    const quote = "a long-\nterm plan";
    const page = `${quote} and the long-term view`;
    expect(readableQuote(page, [{ start: 0, end: quote.length }])).toBe("a long-term plan");
    expect(readableQuoteText(quote)).toBe("a longterm plan");

    const cooperate = "to co-\noperate";
    expect(
      readableQuote(`${cooperate}. We co-operate daily.`, [{ start: 0, end: cooperate.length }]),
    ).toBe("to co-operate");

    const spelledJoined = "the long-\nterm view";
    expect(
      readableQuote(`${spelledJoined} of longterm plans`, [
        { start: 0, end: spelledJoined.length },
      ]),
    ).toBe("the longterm view");
  });

  test("keeps the hyphen of a compound by the rule matching uses", () => {
    expect(readableQuoteText("an English-\nto-German list")).toBe("an English-to-German list");
    expect(readableQuoteText("the state-of-\nthe-art")).toBe("the state-of-the-art");
    expect(readableQuoteText("a non-\nEnglish word")).toBe("a non-English word");
  });

  test("a hyphen or dash before a digit stays, with the break gone", () => {
    expect(readableQuoteText("COVID-\n19 cases")).toBe("COVID-19 cases");
    expect(readableQuoteText("in 1990–\n1995")).toBe("in 1990–1995");
  });

  test("a hyphen that isn't at a line end is left alone", () => {
    expect(readableQuoteText("a long-term plan")).toBe("a long-term plan");
    expect(readableQuoteText("an em-ploy case")).toBe("an em-ploy case");
  });
});

describe("a quote that stops at a split word", () => {
  const page = "The committee will review the case for the exist-\nence of tides near the pier.";

  test("runs to the end of the word, when it ends on the hyphen", () => {
    const end = page.indexOf("exist-") + "exist-".length;
    expect(readableQuote(page, [{ start: 0, end }])).toBe(
      "The committee will review the case for the existence",
    );
  });

  test("and when it ends just before it", () => {
    const end = page.indexOf("exist") + "exist".length;
    expect(readableQuote(page, [{ start: 0, end }])).toBe(
      "The committee will review the case for the existence",
    );
  });

  test("stays as it is where it ends on a whole word", () => {
    const end = page.indexOf("tides") + "tides".length;
    expect(readableQuote(page, [{ start: 0, end }])).toBe(
      "The committee will review the case for the existence of tides",
    );
  });

  test("is found as before: the same range the check decides with", () => {
    const quote = "review the case for the exist-";
    const found = findQuote(page, quote);
    expect(found).not.toBeNull();
    expect(normaliseText(page)).toContain("existence");
    expect(readableQuote(page, found ?? [])).toBe("review the case for the existence");
  });
});

describe("the rest of a quote", () => {
  test("a break that isn't a hyphen's is a space, and none between Chinese or Japanese", () => {
    expect(readableQuoteText("The tides\nrise   twice\n\na day")).toBe(
      "The tides rise twice a day",
    );
    expect(readableQuoteText("含\n1750 亿参数")).toBe("含1750 亿参数");
    expect(readableQuoteText("潮水\n每天涨落两次")).toBe("潮水每天涨落两次");
  });

  test("a space the writer typed next to Chinese stays, a line break there doesn't", () => {
    const written = "使用 Mind 创建文档，第 3 页";
    expect(readableQuoteText(written)).toBe(written);
    expect(readableQuote(written, whole(written))).toBe(written);
    expect(readableQuoteText("使用 Mind\n创建文档，第\n3 页")).toBe("使用 Mind创建文档，第3 页");
  });

  test("the parts of a quote with an ellipsis come back with one between them", () => {
    const page = "Spring tides come at new moon, and then, after em-\nploy ment ends, neap.";
    const first = page.indexOf("Spring");
    const second = page.indexOf("after");
    expect(
      readableQuote(page, [
        { start: first, end: first + "Spring tides come at new moon".length },
        { start: second, end: page.indexOf("ends") + "ends".length },
      ]),
    ).toBe("Spring tides come at new moon … after employ ment ends");
  });

  test("a quote with nothing to change is untouched, accents and quote marks included", () => {
    const quote = "Les marées “vives” s’élèvent à l’équinoxe.";
    expect(readableQuoteText(quote)).toBe(quote);
  });
});
