/**
 * Amounts in brackets (#978). Accountants write a negative as "(12,400)", so
 * a figure read from accounts is minus; a year, a note, a page or a list's
 * "(a)" in brackets is not an amount at all, and what the text can't settle
 * (a lone "(250)", a debit) is a question, never a guess. Each case names every
 * figure the text holds, as the reader gives it.
 */
import { describe, expect, test } from "vitest";
import {
  type AmbiguityKind,
  type Figure,
  type FigureContext,
  figureContextOf,
  readFigures,
} from "../../src/shared/figures";

/** The characters of a figure as written, its value, and what is left a question. */
type Expected = [chars: string, value: number, questions?: AmbiguityKind[]];
type Case = [name: string, text: string, figures: Expected[], context?: FigureContext];

const EN: FigureContext = { language: "en" };
const ZH: FigureContext = { language: "zh-CN" };

const NEGATIVE_EN: Case[] = [
  ["an amount in brackets", "(12,400)", [["(12,400)", -12400]]],
  ["with the currency inside", "(£12,400)", [["(£12,400)", -12400]]],
  ["with the currency outside", "£(12,400)", [["£(12,400)", -12400]]],
  ["with pence", "(£12,500.00)", [["(£12,500.00)", -12500]]],
  ["with spaces inside", "( 12,500.00 )", [["( 12,500.00 )", -12500]]],
  ["with the scale after", "(12.4)m", [["(12.4)m", -12400000]]],
  ["with a dollar sign and a scale inside", "($1.2bn)", [["($1.2bn)", -1200000000]]],
  ["with a scale word inside", "(£1.2 million)", [["(£1.2 million)", -1200000]]],
  ["a loss in a sentence", "Net loss (£1.2m) for the year.", [["(£1.2m)", -1200000]]],
  ["a deduction in a statement", "Less: agent's fee   (4,620.00)", [["(4,620.00)", -4620]]],
  [
    "two cells of a row",
    "Interest paid\t(£340)\t(£280)",
    [
      ["(£340)", -340],
      ["(£280)", -280],
    ],
  ],
  [
    "a column after a figure",
    "Profit 12,400 (3,100)",
    [
      ["12,400", 12400],
      ["(3,100)", -3100],
    ],
  ],
  ["a small amount with its currency", "Fee (£25)", [["(£25)", -25]]],
  ["a debit after brackets", "(12,500.00) DR", [["(12,500.00) DR", -12500]]],
  [
    "a whole number, once the Document writes brackets",
    "Fees (1,500.00) and legal (250)",
    [
      ["(1,500.00)", -1500],
      ["(250)", -250],
    ],
  ],
  ["a whole number under a unit header", "£'000\nRent (250)", [["(250)", -250, ["scale"]]]],
  [
    "a percentage, once the Document writes brackets",
    "Margin (3.2%) and costs (1,250)",
    [
      ["(3.2%)", -0.032],
      ["(1,250)", -1250],
    ],
  ],
  [
    "a whole number, when the Document is said to be accounts",
    "(250)",
    [["(250)", -250]],
    { ...EN, accounts: true },
  ],
];

const NEGATIVE_ZH: Case[] = [
  ["a full-width bracket", "（12,400）", [["（12,400）", -12400]]],
  ["with pence", "（12,400.50）", [["（12,400.50）", -12400.5]]],
  ["with a yuan sign inside", "（¥12,400）", [["（¥12,400）", -12400]]],
  ["with a dollar sign and a scale inside", "（$1.2bn）", [["（$1.2bn）", -1200000000]]],
  [
    "a table under a unit header",
    "单位：万元\n管理费用 （250） （300）",
    [
      ["（250）", -250, ["scale"]],
      ["（300）", -300, ["scale"]],
    ],
  ],
  ["a deduction in a statement", "减：中介费 （46,200.00）", [["（46,200.00）", -46200]]],
];

const NOT_NEGATIVE_EN: Case[] = [
  ["a year", "(2023)", []],
  ["a note", "(note 4)", []],
  ["a page", "(p. 12)", []],
  ["a section", "(see section 12)", []],
  ["list markers", "(a) the buyer; (iii) the seller", []],
  ["footnote numbers", "(1) and (2)", []],
  ["a count with its unit", "(5 users)", [["5 users", 5]]],
  ["a number inside a longer aside", "(250 of them children)", [["250", 250]]],
  ["a room", "the meeting (room 250)", [["250", 250]]],
  [
    "an amount restated in another currency",
    "revenue of RMB 1.2 billion (US$170 million)",
    [
      ["RMB 1.2 billion", 1200000000],
      ["US$170 million", 170000000],
    ],
  ],
  [
    "sterling restated in dollars",
    "£5m ($6.3m)",
    [
      ["£5m", 5000000],
      ["$6.3m", 6300000],
    ],
  ],
  ["a doctor, not a debit", "balance 1,200.50 Dr. Smith", [["1,200.50", 1200.5]]],
];

const NOT_NEGATIVE_ZH: Case[] = [
  ["a year", "（2023年）", []],
  ["a note", "（见附注4）", []],
  ["a page", "（第12页）", []],
  ["list markers", "（一）买方；（二）卖方", []],
  ["a count with its unit", "（5家）", [["5家", 5]]],
  [
    "an amount restated in another currency",
    "营业收入12亿元（1.7亿美元）",
    [
      ["12亿元", 1200000000],
      ["1.7亿美元", 170000000],
    ],
  ],
  [
    "an approximate restatement",
    "营业收入12亿元（约1.7亿美元）",
    [
      ["12亿元", 1200000000],
      ["1.7亿美元", 170000000],
    ],
  ],
];

const QUESTIONS: Case[] = [
  ["a lone whole number", "(250)", [["(250)", 250, ["sign"]]], EN],
  ["a lone percentage", "(12.5%)", [["(12.5%)", 0.125, ["sign"]]], EN],
  ["a lone whole number in Chinese", "（250）", [["（250）", 250, ["sign"]]], ZH],
  [
    "an amount with a word for its currency",
    "管理费用（12.4万元）",
    [["（12.4万元）", 124000, ["sign"]]],
    ZH,
  ],
  ["a debit", "12,500.00 DR", [["12,500.00 DR", 12500, ["sign"]]], EN],
  ["a credit", "12,500.00 CR", [["12,500.00 CR", 12500, ["sign"]]], EN],
  ["a debit stuck to its amount", "£12,500.00Dr", [["£12,500.00Dr", 12500, ["sign"]]], EN],
  [
    "a whole number the context says isn't accounts",
    "(1,250) and (250)",
    [
      ["(1,250)", -1250],
      ["(250)", 250, ["sign"]],
    ],
    { ...EN, accounts: false },
  ],
];

const GROUPS: { name: string; context: FigureContext; cases: Case[] }[] = [
  { name: "English accounts: negative", context: EN, cases: NEGATIVE_EN },
  { name: "Chinese accounts: negative", context: ZH, cases: NEGATIVE_ZH },
  { name: "English: not negative", context: EN, cases: NOT_NEGATIVE_EN },
  { name: "Chinese: not negative", context: ZH, cases: NOT_NEGATIVE_ZH },
  { name: "what the text can't settle is a question", context: EN, cases: QUESTIONS },
];

/** A figure in the shape cases are written in. */
const shape = (figure: Figure): Expected =>
  figure.ambiguous.length > 0
    ? [figure.chars, figure.value, figure.ambiguous.map((each) => each.kind)]
    : [figure.chars, figure.value];

describe("amounts in brackets", () => {
  for (const group of GROUPS) {
    describe(group.name, () => {
      test.each(group.cases)("%s: %s", (_name, text, figures, context) => {
        const read = readFigures(text, context ?? group.context);
        expect(read.map(shape)).toEqual(figures);
        for (const each of read) expect(text.slice(each.start, each.end)).toBe(each.chars);
      });
    });
  }
});

describe("a Document that is accounts", () => {
  test.each([
    ["writes an amount in brackets", "Fees (1,500.00)", true],
    ["writes pence in brackets", "Fees (12.50)", true],
    ["states its scale in the English way", "Revenue (in millions)", true],
    ["states its unit in the Chinese way", "单位：万元", true],
    ["states its currency in the heading", "£'000", true],
    ["says nothing of money in brackets", "As agreed (see below) on 4 June (2026).", false],
    ["only a percentage heading", "Growth (%)", false],
    ["a lone whole number", "As agreed (250)", false],
  ])("%s: %s", (_name, text, accounts) => {
    expect(figureContextOf(text, "en").accounts).toBe(accounts);
  });

  test("its context reads a lone whole number in a Passage of it as negative", () => {
    const document = figureContextOf("Fees (1,500.00)\nLegal (250)", "en");
    expect(readFigures("Legal (250)", document).map((each) => each.value)).toEqual([-250]);
    expect(readFigures("Legal (250)", EN).map((each) => each.ambiguous.map((x) => x.kind))).toEqual(
      [["sign"]],
    );
  });
});
