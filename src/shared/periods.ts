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

const YEARS = [
  String.raw`(?:${FISCAL}\s?)?${YEAR}(?:${SEPARATOR}${YEAR_SHORT})?\s?${ESTIMATE}`,
  String.raw`(?:fy|cy)\s?'?\d{2}${ESTIMATE}`,
  String.raw`${YEAR}\s?(?:年度?|财年|財年)`,
  String.raw`[〇零一二三四五六七八九]{4}\s?年`,
];

const PLAN_YEARS = [
  String.raw`(?:year|yr)\.?\s?\d{1,2}`,
  String.raw`第\s?(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s?年`,
];

const QUARTERS_AND_HALVES = [
  String.raw`q\s?[1-4](?:\s?[-/,'’.]?\s?(?:${FISCAL}\s?)?${YEAR_SHORT})?${ESTIMATE}`,
  String.raw`(?:${FISCAL}\s?)?${YEAR_SHORT}(?:\s?年)?\s?[-/. ]?\s?q\s?[1-4]${ESTIMATE}`,
  String.raw`[1-4]q(?:\s?'?\s?(?:${FISCAL}\s?)?${YEAR_SHORT})?`,
  String.raw`h\s?[12](?:\s?[-/']?\s?(?:${FISCAL}\s?)?${YEAR_SHORT})?`,
  String.raw`[12]h(?:\s?'?${YEAR_SHORT})?`,
  String.raw`(?:quarter|qtr)\.?\s?[1-4](?:\s?,?\s?${YEAR})?`,
  String.raw`${CHINESE_YEAR}第?\s?${CHINESE_NUMBER}\s?季度?`,
  `${CHINESE_YEAR}(?:上半年|下半年|全年|前三季度?)`,
];

const MONTHS = [
  String.raw`${MONTH}(?:[\s,'’-]*${YEAR_SHORT})?`,
  `${YEAR}${SEPARATOR}${MONTH_NUMBER}`,
  String.raw`${MONTH_NUMBER}\s?[-/]\s?${YEAR}`,
  String.raw`${CHINESE_YEAR}${CHINESE_MONTH}(?:\s?[-–~～至]\s?${CHINESE_MONTH})?\s?月份?`,
];

const DATES = [
  `${YEAR}${SEPARATOR}${MONTH_NUMBER}${SEPARATOR}${DAY_NUMBER}`,
  String.raw`\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\.\d{1,2}\.\d{4}`,
  String.raw`${DAY_NUMBER}(?:st|nd|rd|th)?\s+${MONTH},?(?:\s*${YEAR})?`,
  String.raw`${MONTH}\s+${DAY_NUMBER}(?:st|nd|rd|th)?,?(?:\s*${YEAR})?`,
  String.raw`${CHINESE_YEAR}${CHINESE_MONTH}\s?月\s?${DAY_NUMBER}\s?[日号]`,
];

const PERIODS: readonly RegExp[] = [
  ...YEARS,
  ...PLAN_YEARS,
  ...QUARTERS_AND_HALVES,
  ...MONTHS,
  ...DATES,
].map((pattern) => new RegExp(`^(?:${pattern})$`, "i"));

const TRAILING_NOTE = /\s*(?:[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡]+|[（(][^)）]*[)）])$/u;

export function isPeriod(label: string): boolean {
  const text = label.replace(TRAILING_NOTE, "").replace(/\s+/g, " ").trim();
  return text !== "" && PERIODS.some((pattern) => pattern.test(text));
}

/**
 * At least two labels, and more than half of those that aren't empty, are periods: a
 * "Total" or "Change" beside the years doesn't hide that they are years.
 */
export function holdsPeriods(labels: readonly string[]): boolean {
  const shown = labels.map((label) => label.trim()).filter((label) => label !== "");
  const periods = shown.filter(isPeriod).length;
  return periods >= 2 && periods * 2 > shown.length;
}
