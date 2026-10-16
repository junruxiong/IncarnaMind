/**
 * Whether a Question asks to calculate, or to check figures or claims: only
 * such a Question gets the larger thinking budget (`thinking: "careful"`), as
 * an Answer that writes or summarizes starts sooner with the modest one
 * (#899). Checking and verifying count only beside what is checked, so "check
 * my grammar" and "cloud computing" don't. English and Chinese.
 */
const CALCULATE =
  /\b(?:calculat\w*|comput(?:e|es|ed|ation|ations)|add(?:s|ed)?\s+up|ratios?|percent(?:age)?s?|growth\s+rate|cagr|headroom)\b/i;

const CHECKED =
  "figures?|numbers?|totals?|sums?|calculations?|maths?|arithmetic|claims?|facts?|statements?|amounts?";

const CHECK = new RegExp(
  `\\b(?:check|verify|double-check|reconcile)(?:s|ed|ing|d)?\\b(?:\\W+\\w+){0,4}?\\W+(?:${CHECKED})\\b`,
  "i",
);

const CHINESE_CALCULATE =
  /(?<!云)计算(?!机)|核算|对账|比率|占比|百分比|增长率|同比|环比|算一下|算出/;

const CHINESE_CHECK =
  /(?:核对|核实|校验|验证|检查)[^。？！?!\n]{0,8}?(?:数字|数据|数值|金额|总数|总额|合计|结果|说法|事实|表述|论断|计算)/;

export function asksToCheckOrCalculate(question: string): boolean {
  return (
    CALCULATE.test(question) ||
    CHECK.test(question) ||
    CHINESE_CALCULATE.test(question) ||
    CHINESE_CHECK.test(question)
  );
}
