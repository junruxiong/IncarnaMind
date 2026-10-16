/**
 * Whether a Question asks to check something or to calculate: only such a
 * Question gets the larger thinking budget (`thinking: "careful"`), as an
 * Answer that writes or summarizes starts sooner with the modest one (#899).
 * The words name checking and working out figures, in English and Chinese.
 */
const ENGLISH =
  /\b(?:calculat\w*|comput(?:e|es|ed|ing|ation)|check(?:s|ed|ing)?|verif(?:y|ies|ied|ying)|validat\w*|double-check\w*|reconcil\w*|add(?:s|ed)?\s+up|ratios?|percent(?:age)?s?|growth\s+rate|cagr|headroom)\b/i;

const CHINESE = /计算|核算|核对|核实|检查|校验|验证|对账|比率|占比|百分比|增长率|同比|环比|算一下|算出/;

export function asksToCheckOrCalculate(question: string): boolean {
  return ENGLISH.test(question) || CHINESE.test(question);
}
