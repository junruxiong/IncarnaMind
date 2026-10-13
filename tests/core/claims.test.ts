/**
 * A verified document's text as paragraphs, and the claims the model names
 * (#97, src/core/answers/checks/claims.ts).
 */
import { describe, expect, test } from "vitest";
import { paragraphsOf, readClaimNumbers } from "../../src/core/answers/checks/claims";

const page = (number: number, ...lines: string[]) => ({
  page: number,
  text: lines.join("\n"),
  kind: "page" as const,
  label: null,
});

describe("A PDF's lines become paragraphs", () => {
  test("a sentence runs on across lines; a short line before a fresh one ends its paragraph; a word broken at a hyphen is mended", () => {
    const paragraphs = paragraphsOf([
      page(
        1,
        "Riverside Library roof and insulation works: progress",
        "report 3",
        "1 Summary",
        "This report covers the roof and insulation package from 1 to 31 March 2027. Roof works were programmed to",
        "start on 4 January 2027. Under our contract, the package is due to be com-",
        "pleted in April 2027.",
      ),
      page(2, "Forty-two operatives worked on site during March."),
    ]);
    expect(paragraphs).toEqual([
      { page: 1, text: "Riverside Library roof and insulation works: progress report 3" },
      { page: 1, text: "1 Summary" },
      {
        page: 1,
        text: "This report covers the roof and insulation package from 1 to 31 March 2027. Roof works were programmed to start on 4 January 2027. Under our contract, the package is due to be completed in April 2027.",
      },
      { page: 2, text: "Forty-two operatives worked on site during March." },
    ]);
  });

  test("a numbered heading starts a paragraph of its own, after a line as long as the page's", () => {
    const paragraphs = paragraphsOf([
      page(
        1,
        "Meridian Health Staffing · for the clinic network's workforce team · 2 November 2026",
        "1 Your plan as we understand it",
        "Your 2027 staffing plan sets a total budget of £50,929,200 across the network and its clinics.",
      ),
    ]);
    expect(paragraphs.map((each) => each.text)).toEqual([
      "Meridian Health Staffing · for the clinic network's workforce team · 2 November 2026",
      "1 Your plan as we understand it",
      "Your 2027 staffing plan sets a total budget of £50,929,200 across the network and its clinics.",
    ]);
  });

  test("Chinese runs on across lines with no space, until a sentence ends or a heading starts", () => {
    const paragraphs = paragraphsOf([
      page(
        1,
        "老年人持卡乘车继续免费。第三季度共处理乘客投诉1,250",
        "件。",
        "二、库存情况",
        "萧山冷链仓有机鲜牛奶库存量为4200，高于3000的安全库存。",
      ),
    ]);
    expect(paragraphs.map((each) => each.text)).toEqual([
      "老年人持卡乘车继续免费。第三季度共处理乘客投诉1,250件。",
      "二、库存情况",
      "萧山冷链仓有机鲜牛奶库存量为4200，高于3000的安全库存。",
    ]);
  });
});

describe("The claims the model names", () => {
  test.each([
    ['{"claims": [1, 3, 4]}', [1, 3, 4]],
    ['Here they are:\n```json\n{"claims": ["2", 5]}\n```', [2, 5]],
    ['{"claims": []}', []],
    ['{"claims": [1.5, "x", 2]}', [2]],
  ])("reads %j", (reply, numbers) => {
    expect(readClaimNumbers(reply)).toEqual(numbers);
  });

  test.each(["No claims.", '{"sentences": [1]}', '{"claims": 3}', "{not json}"])(
    "reads nothing from %j",
    (reply) => {
      expect(readClaimNumbers(reply)).toBeNull();
    },
  );
});
