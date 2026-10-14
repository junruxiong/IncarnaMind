/**
 * Periods: the labels of a table that say when, which a chart puts on its
 * x axis. Years ("2023", "FY2024", "2024年", "Year 1"), quarters and halves
 * ("Q1 2025", "2025Q1", "第一季度", "2024年第一季度", "H1 2025"), months
 * ("Jan 2025", "2025-03", "3月", "2025年1-6月") and dates ("2025-03-31",
 * "30 June 2025", "12月31日").
 *
 * Pure, with no Node or DOM imports: the core, the renderer and a later phone
 * build share it.
 */

const YEAR = String.raw`(?:19|20)\d{2}`;
const YEAR_SHORT = String.raw`(?:${YEAR}|\d{2})`;
const FISCAL = String.raw`(?:fy|cy|fiscal(?:\s+year)?)`;
const ESTIMATE = `[eapf]?`;
const MONTH = String.raw`(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?`;
const MONTH_NUMBER = `(?:0?[1-9]|1[0-2])`;
const DAY_NUMBER = String.raw`(?:0?[1-9]|[12]\d|3[01])`;
const CHINESE_YEAR = String.raw`(?:${YEAR}\s?年\s?)?`;
const CHINESE_NUMBER = `[一二三四1-4]`;
const CHINESE_MONTH = `(?:0?[1-9]|1[0-2]|十[一二]?|[一二三四五六七八九])`;
const SEPARATOR = String.raw`\s?[-–/.]\s?`;

const PERIODS: readonly RegExp[] = [
  // Years: 2023, FY2024, Fiscal 2024, 2024E, 2023/24, FY24, 2024年, 2024财年, 二〇二四年.
  String.raw`(?:${FISCAL}\s?)?${YEAR}(?:${SEPARATOR}${YEAR_SHORT})?\s?${ESTIMATE}`,
  String.raw`(?:fy|cy)\s?'?\d{2}${ESTIMATE}`,
  String.raw`${YEAR}\s?(?:年度?|财年|財年)`,
  String.raw`[〇零一二三四五六七八九]{4}\s?年`,
  // Years of a plan: Year 1, Yr 2, 第1年, 第三年.
  String.raw`(?:year|yr)\.?\s?\d{1,2}`,
  String.raw`第\s?(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s?年`,
  // Quarters and halves: Q1, Q1 2025, 2025Q1, 2024年Q1, 1Q25, Q1 FY25, H1 2025, 1H25, Quarter 1.
  String.raw`q\s?[1-4](?:\s?[-/,'’.]?\s?(?:${FISCAL}\s?)?${YEAR_SHORT})?${ESTIMATE}`,
  String.raw`(?:${FISCAL}\s?)?${YEAR_SHORT}(?:\s?年)?\s?[-/. ]?\s?q\s?[1-4]${ESTIMATE}`,
  String.raw`[1-4]q(?:\s?'?\s?(?:${FISCAL}\s?)?${YEAR_SHORT})?`,
  String.raw`h\s?[12](?:\s?[-/']?\s?(?:${FISCAL}\s?)?${YEAR_SHORT})?`,
  String.raw`[12]h(?:\s?'?${YEAR_SHORT})?`,
  String.raw`(?:quarter|qtr)\.?\s?[1-4](?:\s?,?\s?${YEAR})?`,
  // Chinese quarters and halves: 一季度, 第一季度, 2024年第一季度, 上半年, 全年.
  String.raw`${CHINESE_YEAR}第?\s?${CHINESE_NUMBER}\s?季度?`,
  `${CHINESE_YEAR}(?:上半年|下半年|全年|前三季度?)`,
  // Months: Jan, January 2025, Jan-25, 2025-03, 03/2025, 2025年3月, 2025年1-6月, 3月, 十二月.
  String.raw`${MONTH}(?:[\s,'’-]*${YEAR_SHORT})?`,
  `${YEAR}${SEPARATOR}${MONTH_NUMBER}`,
  String.raw`${MONTH_NUMBER}\s?[-/]\s?${YEAR}`,
  String.raw`${CHINESE_YEAR}${CHINESE_MONTH}(?:\s?[-–~～至]\s?${CHINESE_MONTH})?\s?月份?`,
  // Dates: 2025-03-31, 31/03/2025, 30 June 2025, June 30, 2025, 2024年12月31日.
  `${YEAR}${SEPARATOR}${MONTH_NUMBER}${SEPARATOR}${DAY_NUMBER}`,
  String.raw`\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\.\d{1,2}\.\d{4}`,
  String.raw`${DAY_NUMBER}(?:st|nd|rd|th)?\s+${MONTH},?(?:\s*${YEAR})?`,
  String.raw`${MONTH}\s+${DAY_NUMBER}(?:st|nd|rd|th)?,?(?:\s*${YEAR})?`,
  String.raw`${CHINESE_YEAR}${CHINESE_MONTH}\s?月\s?${DAY_NUMBER}\s?[日号]`,
].map((pattern) => new RegExp(`^(?:${pattern})$`, "i"));

/** A footnote mark or a note in brackets after a label: "2024 (restated)", "FY2025¹". */
const NOTE = /\s*(?:[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡]+|[（(][^)）]*[)）])$/u;

/** Whether a label says when: a year, a quarter, a month or a date, in English or Chinese. */
export function isPeriod(label: string): boolean {
  const text = label.replace(NOTE, "").replace(/\s+/g, " ").trim();
  return text !== "" && PERIODS.some((pattern) => pattern.test(text));
}

/**
 * Whether labels are periods: at least two of them, and more than half of the
 * labels that aren't empty, so a "Total" or "Change" beside the years doesn't
 * hide that they are years.
 */
export function holdsPeriods(labels: readonly string[]): boolean {
  const shown = labels.map((label) => label.trim()).filter((label) => label !== "");
  const periods = shown.filter(isPeriod).length;
  return periods >= 2 && periods * 2 > shown.length;
}
