/**
 * The hard tier's Answer rules (eval/hard/lib/answerRules.ts, #548): whether
 * an Answer says first that a premise isn't in the Documents, and whether the
 * newer of two Documents that disagree leads; and how they are counted and
 * reported per difficulty.
 */
import { describe, expect, test } from "vitest";
import { flagsPremise, newerLeads } from "../../eval/hard/lib/answerRules";
import { answerRuleResult, summariseDifficulty } from "../../eval/hard/lib/citations";
import type { HardQuestion } from "../../eval/hard/lib/questions";
import type { AnswerRecord, CitationRecord } from "../../eval/lib/citations";

describe("A premise the Documents don't support", () => {
  test.each([
    "The Documents don't record a Board approval; the 14 May memo asks for one [^1].",
    "The Federal Reserve didn't cut the rate in the first half of 2024: the FOMC held it at 5¼ to 5½ percent [^1].",
    "The guidance hasn't been finalized. It is a draft, issued for comment on January 7, 2025 [^1].",
    "The agreement's term is ten years, not twenty — the Question assumes a 20-year term [^1].",
    "None: the label says safety and effectiveness have not been established in children under 10 [^1].",
    "## Short answer\n\nIn fact, revenue rose 7.09% [^1].",
    "文档中没有董事会批准的记录；5 月 14 日的备忘录是在申请批准[^1]。",
    "五粮液 2024 年营业收入并没有下降，而是同比增长 7.09%[^1]。",
    "泸州老窖 2024 年没有亏损：归属于上市公司股东的净利润约 134.73 亿元[^1]。",
    "不是上调，而是下调：2024 年两次下调存款准备金率各 0.5 个百分点[^1]。",
    "实际上，世卫组织建议每日盐摄入量低于 5 克[^1]。",
  ])("is flagged: %s", (text) => {
    expect(flagsPremise(text)).toBe(true);
  });

  test.each([
    "The Board approved USD 310 million for Berth 7 [^1].",
    "Revenue fell because of weaker demand in its core markets [^1].\n\nThe Documents don't say more.",
    "五粮液 2024 年营业收入下降，主要因为市场需求减弱[^1]。",
    "世卫组织建议成人每天至少摄入 5 克盐，以满足身体需要[^1]。",
  ])("isn't flagged in the Answer's opening: %s", (text) => {
    expect(flagsPremise(text)).toBe(false);
  });
});

describe("The newer of two Documents that disagree", () => {
  const newer = { document: "memo-2025", says: ["4 hours", "four hours", "4 小时"] };
  const older = { document: "handbook-2024", says: ["6 hours", "six hours", "6 小时"] };

  test("leads when what it says comes first, and the older one is named after", () => {
    expect(
      newerLeads(
        "Outages over 4 hours must be reported (memo, May 2025) [^1]; the older handbook (2024) says 6 hours [^2].",
        newer,
        older,
      ),
    ).toEqual({ leads: true, namesOlder: true });
    expect(newerLeads("超过 4 小时的停电须报告[^1]。", newer, older)).toEqual({
      leads: true,
      namesOlder: false,
    });
  });

  test("doesn't lead when the older one comes first, or the newer one is missing", () => {
    expect(
      newerLeads(
        "The handbook says 6 hours [^1] and the memo 4 hours [^2]; probably 4 hours.",
        newer,
        older,
      ),
    ).toEqual({ leads: false, namesOlder: true });
    expect(newerLeads("Report outages over six hours [^1].", newer, older)).toEqual({
      leads: false,
      namesOlder: true,
    });
  });

  test("reads grouped digits alike, and finds a figure only where it stands on its own", () => {
    const staff = { document: "a", says: ["116,000"] };
    const before = { document: "b", says: ["114,000"] };
    expect(newerLeads("About 116 000 people [^1], up from 114000 [^2].", staff, before)).toEqual({
      leads: true,
      namesOlder: true,
    });
    const growth = { document: "a", says: ["4.9"] };
    const earlier = { document: "b", says: ["4.8"] };
    expect(newerLeads("Poverty fell to 14.8% and growth is 4.85%.", growth, earlier)).toEqual({
      leads: false,
      namesOlder: false,
    });
    const functions = { document: "a", says: ["six"] };
    expect(newerLeads("Sixty controls [^1].", functions, earlier).leads).toBe(false);
  });
});

describe("Counting the Answer rules", () => {
  const cited = (documentName: string): CitationRecord => ({
    sentence: "A claim.",
    quote: "a quote",
    documentName,
    pageFrom: 1,
    pageTo: 1,
    check: "found",
    checkReason: null,
    passagePages: null,
    quoteOn: null,
    outcome: "found",
  });
  const answer = (
    questionId: string,
    text: string,
    citations: CitationRecord[] = [],
  ): AnswerRecord => ({
    questionId,
    language: "en",
    crossLingual: false,
    round: 1,
    question: questionId,
    status: "done",
    error: null,
    citationSupport: "tools",
    searches: [],
    droppedMarkers: 0,
    droppedRecords: 0,
    rejectedRecords: [],
    text,
    sentences: [],
    citations,
    seconds: 1,
  });
  const premise: HardQuestion = {
    id: "fp",
    language: "en",
    domain: "reports",
    difficulty: "false-premise",
    question: "Why did revenue fall?",
    expected: [{ document: "r23", pages: [1, 1], quote: "x" }],
    premise: "That revenue fell.",
  };
  const disagreeing: HardQuestion = {
    id: "dis",
    language: "en",
    domain: "reports",
    difficulty: "disagreeing",
    question: "What is revenue?",
    expected: [
      { document: "r23", pages: [1, 1], quote: "x" },
      { document: "r22", pages: [1, 1], quote: "y" },
    ],
    newer: { document: "r23", says: ["12 million"] },
    older: { document: "r22", says: ["10 million"] },
  };
  const questions = new Map([
    ["fp", premise],
    ["dis", disagreeing],
  ]);
  const library = { documents: [] };
  const names = (key: string) => ({ r23: "Report 2023", r22: "Report 2022" })[key];

  test("per difficulty: the premise flagged, the newer source leading and cited, the older named", () => {
    expect(
      summariseDifficulty(
        [answer("fp", "Revenue didn't fall; it rose."), answer("fp", "Revenue fell on demand.")],
        questions,
        names,
        library,
      ),
    ).toMatchObject({ answers: 2, premiseFlagged: 1 });
    expect(
      summariseDifficulty(
        [
          answer("dis", "12 million in 2023; the 2022 report said 10 million.", [
            cited("Report 2023"),
            cited("Report 2022"),
          ]),
          answer("dis", "12 million.", [cited("Report 2023")]),
          answer("dis", "From what I know, 12 million; the 2022 report said 10 million.", [
            cited("Report 2022"),
          ]),
          answer("dis", "10 million, or probably 12 million.", [
            cited("Report 2022"),
            cited("Report 2023"),
          ]),
        ],
        questions,
        names,
        library,
      ),
    ).toMatchObject({ answers: 4, newerLeads: 2, namesOlder: 3 });
  });

  test("each Answer's result keeps how it opens, for a reviewer", () => {
    expect(
      answerRuleResult(answer("fp", "# Revenue\n\nIt didn't fall.\n\nMore."), premise, names),
    ).toEqual({
      questionId: "fp",
      difficulty: "false-premise",
      followed: true,
      namesOlder: false,
      opening: "It didn't fall.",
    });
    expect(
      answerRuleResult(answer("x", "Text."), { ...premise, difficulty: "easy" }, names),
    ).toBeNull();
  });
});
