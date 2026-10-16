/**
 * A reference record from what a PDF's first page prints: its title, in the page's largest
 * type; its authors, under the title or after "作者："; the university a thesis or the
 * organisation a report comes from; the year on its cover or in its running header; and the kind
 * of work its cover says it is. Pure. What isn't plainly there is left out, and what is read is a
 * suggestion the User checks (`fieldsToCheck` in ../../shared/references), never a fact.
 */

import { parseName } from "../../shared/references";
import { foldRadicals } from "../../shared/text";
import type { ReferenceFields, ReferenceName } from "../api";
import type { PageLine } from "../documents/extract";

export type FirstPageReading = Partial<
  Pick<ReferenceFields, "type" | "title" | "authors" | "issued">
> & {
  /** The university or organisation it names: a thesis's or a report's publisher. */
  institution?: string;
};

const HAN = /\p{Script=Han}/u;
const LETTER = /\p{L}/gu;

/** The top of a page, where a running header and a cover's first lines are. */
const TOP = 0.15;

/** Headings that open a page but name no work. */
const GENERIC =
  /^(?:abstract|summary|executive summary|contents|table of contents|preface|foreword|introduction|acknowledge?ments?|appendix|preprint|draft|chapter\s+\d+|摘\s*要|目\s*录|前\s*言|序\s*言|引\s*言|致\s*谢|附\s*录|第[一二三四五六七八九十\d]+章)(?:$|[\s:：.])/iu;
/**
 * A thesis cover's heading, which says what kind of work it is and is never its title:
 * "硕士学位论文", "博士研究生学位论文", "华东师范大学硕士专业学位论文", "Doctoral Dissertation".
 */
const THESIS_HEADING =
  /^(?:\S{0,16}(?:大学|学院|研究院|研究所)\s*)?(?:(?:硕\s*士|博\s*士|学\s*士)\s*(?:研\s*究\s*生\s*)?(?:专\s*业\s*)?)?学\s*位\s*论\s*文$|^毕\s*业\s*(?:论\s*文|设\s*计)$|^(?:an?\s+)?(?:(?:master|doctoral|bachelor)(?:'s)?\s+|ph\.?\s?d\.?\s+)?(?:thesis|dissertation)$/iu;
/** What only a thesis's cover says: "A dissertation submitted to…", "in partial fulfillment of…". */
const THESIS_STATEMENT =
  /^an?\s+(?:thesis|dissertation)\s+submitted\b|partial\s+fulfil+ment\s+of\s+the\s+requirements/iu;
const THESIS = /学位论文|毕业论文|\b(?:thesis|dissertation)\b/iu;
const REPORT =
  /工作报告|研究报告|调查报告|年度报告|白皮书|\b(?:annual|technical|research|working)\s+(?:report|paper)\b|\bwhite\s+paper\b/iu;
/**
 * What a journal prints on an article's first page: a Chinese journal's article marks, a DOI, or
 * the volume and issue in its running header.
 */
const JOURNAL =
  /文献标识码|中图分类号|文章编号|\bdoi\s*[:：]?\s*10\.|doi\.org\/10\.|第\s*\d+\s*卷|第\s*\d+\s*期|\bvol(?:ume)?\.?\s*\d+\s*,?\s*no\.\s*\d/iu;
/** A cover's label before its title: "题目：", "论文题目：", "Title:". */
const TITLE_LABEL = /^(?:论\s*文\s*)?题\s*目\s*[:：]\s*|^title\s*:\s*/iu;
/** A journal's running header ("第 41 卷 第 3 期…中国科学院院刊"), which names no institution of the work's. */
const JOURNAL_HEADER =
  /第\s*\d+\s*[卷期]|\bvol(?:ume)?\.?\s*\d|\bno\.\s*\d|\bissn\b|学报|院刊|杂志|期刊|\bjournal\b|\bproceedings\b/iu;
/** A cover's university or organisation, set as large as a title: "华东师范大学". */
const ONLY_AN_INSTITUTION = /^.{0,14}(?:大学|学院|研究所|研究院|university|college|institute)$/iu;

/**
 * Chinese family names, the common single ones and the compound ones, which a line of names
 * under a title starts each name with: a heading such as "要点" or "摘要" isn't read as a name.
 */
const SURNAMES = new Set(
  "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付傅方白邹孟熊秦邱江尹薛闫阎段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤常温康施文牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚符舒阮柯纪梅童凌毕单季裴霍涂成苗谷盛曲翁冉骆蓝路游辛靳管柴蒙鲍华喻祁蒲房滕屈饶解牟艾尤阳时穆农司卓古吉缪简车项连芦麦褚娄窦戚岑景党宫费卜冷晏席卫米柏宗瞿桂全佟应臧闵苟邬边卞姬师和仇栾隋商刁沙荣巫寇桑郎甄丛仲虞敖巩明佘池查麻苑迟邝官封谈匡鞠惠荆乐冀郁胥南班储原栗燕楚鄢劳谌奚皮粟冼蔺楼盘满闻位厉伊仝区郜海阚花权强帅屠豆朴盖练廉禹井祖漆巴丰支卿国狄平计索宣晋相初门云容敬来扈晁芮都普阙浦戈伏鹿薄邸雍辜羊乌母裘亓修邰赫杭况那宿鲜印逯隆茹诸战慕危玉银亢嵇公哈湛宾戎勾茅利於呼居揭干但尉冶斯元束檀衣信展阴昝智幸奉植衡富尧闭由钮寿竺",
);
const COMPOUND_SURNAMES = new Set(
  "欧阳 司马 上官 诸葛 东方 皇甫 尉迟 公孙 慕容 长孙 宇文 司徒 夏侯 令狐 端木 独孤 南宫 西门 轩辕 澹台 申屠 呼延 闻人 钟离 赫连".split(
    " ",
  ),
);
const hasSurname = (name: string) =>
  SURNAMES.has(name[0] ?? "") || COMPOUND_SURNAMES.has(name.slice(0, 2));

/** Words in a heading or a sentence that no person's name has. */
const NOT_A_NAME = /\b(?:of|the|for|in|on|to|with|from|at)\b/iu;
/** Words that end a heading under a title, never a name: "文献综述", "研究进展". */
const NOT_A_CHINESE_NAME =
  /(?:综述|研究|进展|分析|报告|论文|摘要|引言|概述|简介|绪论|目录|前言|要点|专论|述评|评论)$/u;
/** A date on the line ("2026年10月15日"), whose numbers are no affiliation marks. */
const DATE_ON_LINE = /\d\s*[年月日]/u;

/** A label before a value on a cover: "作者：张三", "Author: Jane Smith". */
const AUTHOR_LABEL =
  /^(?:作\s*者(?:\s*姓\s*名)?|著\s*者|研\s*究\s*生|学\s*生\s*姓\s*名|姓\s*名|申\s*请\s*人|authors?|by)\s*[:：]?\s*(.+)$/iu;
/** Labels of a cover's institution, the degree-granting one first. */
const INSTITUTION_LABELS = [
  /^(?:学\s*位\s*)?授\s*予\s*单\s*位\s*[:：]\s*(.+)$/u,
  /^(?:学\s*校|university)\s*[:：]\s*(.+)$/iu,
  /^(?:培\s*养\s*单\s*位|单\s*位|institution)\s*[:：]\s*(.+)$/iu,
];
const DATE_LABEL =
  /^(?:完\s*成\s*日\s*期|答\s*辩\s*日\s*期|提\s*交\s*日\s*期|论\s*文\s*日\s*期|日\s*期|date)\s*[:：]/iu;
/** Dates that aren't the work's: when a paper was received or accepted, or a preprint's version posted. */
const OTHER_DATE =
  /收稿|修回|录用|接收|基金|received|revised|accepted|submitted|funding|arxiv\s*:/iu;
const INSTITUTION =
  /大学|学院|研究所|研究院|研究中心|科学院|实验室|university|institute|college|academy|school of|laborator/iu;
const UNIVERSITY =
  /^.*?大学|[^,，]*\buniversity\b[^,，]*|[^,，]*\binstitute of technology\b[^,，]*/iu;

/** Superscript affiliation numbers and marks after a name: "曲清莉 1，2", "Asai†", "Smith1,2". */
const CJK_MARKS = /(?:\d+\s*[,，]\s*)*\d+\s*\*?|[*†‡§¶#∗♢]/gu;
const LATIN_MARKS = /(?<=\p{L})\d+(?:,\d+)*|[*†‡§¶#∗♢]/gu;

const MONTHS =
  "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec";
const YEAR_IN_DATE = new RegExp(
  [
    "((?:19|20)\\d{2})\\s*年",
    "\\b((?:19|20)\\d{2})[-/.](?:0?[1-9]|1[0-2])\\b",
    `\\b(?:${MONTHS})\\.?\\s+(?:\\d{1,2},?\\s+)?((?:19|20)\\d{2})\\b`,
    `\\b\\d{1,2}\\s+(?:${MONTHS})\\.?\\s+((?:19|20)\\d{2})\\b`,
    "([〇零一二三四五六七八九]{4})\\s*年",
  ].join("|"),
  "giu",
);
/** The end of a range of years before a date's year: "2020—" in "2020—2024 年". */
const RANGE_BEFORE = /(?:19|20)\d{2}\s*年?\s*[—–\-~～至到]\s*$/u;
const CHINESE_DIGITS: Readonly<Record<string, string>> = {
  〇: "0",
  零: "0",
  一: "1",
  二: "2",
  三: "3",
  四: "4",
  五: "5",
  六: "6",
  七: "7",
  八: "8",
  九: "9",
};

const meaningful = (text: string) => (text.match(LETTER)?.length ?? 0) >= 2;
const isHeading = (text: string) => GENERIC.test(text) || THESIS_HEADING.test(text);

function joinLines(texts: readonly string[]): string {
  return texts.reduce((joined, text) => {
    if (joined === "") return text;
    const glue = HAN.test(joined.slice(-1)) && HAN.test(text[0] ?? "") ? "" : " ";
    return `${joined}${glue}${text}`;
  }, "");
}

/**
 * The size the page's running text is set in: the smallest that holds a fifth of its characters,
 * so a cover whose title has more characters than its few labelled lines still has its labels'.
 */
function bodySize(lines: readonly PageLine[]): number {
  const weight = new Map<number, number>();
  let total = 0;
  for (const line of lines) {
    weight.set(line.size, (weight.get(line.size) ?? 0) + line.text.length);
    total += line.text.length;
  }
  const sizes = [...weight].sort(([a], [b]) => a - b);
  return sizes.find(([, chars]) => chars >= total / 5)?.[0] ?? 0;
}

function titleOf(lines: readonly PageLine[]): { title: string; start: number; end: number } | null {
  const body = bodySize(lines);
  const sizes = [...new Set(lines.filter((line) => meaningful(line.text)).map((l) => l.size))]
    .filter((size) => size >= body * 1.15)
    .sort((a, b) => b - a)
    .slice(0, 3);
  const sized = (line: PageLine | undefined, size: number) =>
    line !== undefined && Math.abs(line.size - size) < 0.5;
  for (const size of sizes) {
    const start = lines.findIndex((line) => sized(line, size) && meaningful(line.text));
    let end = start;
    while (sized(lines[end + 1], size)) end++;
    const title = joinLines(lines.slice(start, end + 1).map((line) => line.text)).replace(
      TITLE_LABEL,
      "",
    );
    if (!isHeading(title) && !ONLY_AN_INSTITUTION.test(title) && title.length <= 300) {
      return { title, start, end };
    }
  }
  return null;
}

/**
 * "张三，李四" or "裴 栋 1，2*": Chinese names, two to four characters each starting with a
 * family name (unless a label said they are names), or names in parts joined by "·".
 */
function chineseNames(text: string, labelled: boolean): ReferenceName[] | null {
  if (DATE_ON_LINE.test(text)) return null;
  const tokens = text
    .replace(CJK_MARKS, " ")
    .split(/[，,、；;]/u)
    .flatMap((token) => {
      const parts = token.trim().split(/\s+/u).filter(Boolean);
      // "裴 栋" is one name spaced out to the width of three; "张三 李四" is two.
      return parts.every((part) => part.length >= 2) ? parts : [parts.join("")];
    })
    .filter((token) => token !== "");
  const valid = (name: string) =>
    /^\p{Script=Han}{2,6}(?:·\p{Script=Han}{2,6})+$/u.test(name) ||
    (/^\p{Script=Han}{2,4}$/u.test(name) &&
      !NOT_A_CHINESE_NAME.test(name) &&
      (labelled || hasSurname(name)));
  if (tokens.length === 0 || tokens.length > 30 || !tokens.every(valid)) return null;
  return tokens.map((literal) => ({ literal }));
}

/** "Akari Asai†, Zeqiu Wu† and Avirup Sil‡": people's names, two to four words each. */
function latinNames(text: string): ReferenceName[] | null {
  if (/@|\d/u.test(text.replace(LATIN_MARKS, ""))) return null;
  const tokens = text
    .replace(LATIN_MARKS, "")
    .split(/\s*,\s*(?:and\s+|&\s+)?|\s+(?:and|&)\s+/iu)
    .map((token) => token.trim())
    .filter(Boolean);
  if (tokens.length === 0 || tokens.length > 30) return null;
  const names: ReferenceName[] = [];
  for (const token of tokens) {
    const words = token.split(/\s+/u);
    if (words.length < 2 || words.length > 4) return null;
    if (INSTITUTION.test(token) || NOT_A_NAME.test(token)) return null;
    // A heading in capitals ("MIT CSAIL"); a romanised family name in capitals has a given name.
    if (words.every((word) => word === word.toUpperCase())) return null;
    if (
      !words.every((word) => /^(?:\p{Lu}[\p{L}'’.-]*|van|von|de|der|da|di|du|la|le)$/u.test(word))
    )
      return null;
    const name = parseName(token);
    if (!name || "literal" in name) return null;
    names.push(name);
  }
  return names;
}

const namesIn = (text: string, labelled = false) =>
  HAN.test(text) ? chineseNames(text, labelled) : latinNames(text);

/**
 * Authors written as a file's author property writes them, without the affiliation numbers and
 * marks a publisher leaves in: "曲清莉1,2， 王小芳3*" as "曲清莉， 王小芳".
 */
export const withoutAffiliationMarks = (text: string): string =>
  HAN.test(text)
    ? text.replace(CJK_MARKS, "").replace(/[^\p{Script=Han}·\s,，、;；/]/gu, "")
    : text.replace(LATIN_MARKS, "");

function authorsOf(lines: readonly PageLine[], titleEnd: number): ReferenceName[] | null {
  for (const line of lines) {
    const value = AUTHOR_LABEL.exec(line.text)?.[1];
    const names = value ? namesIn(value, true) : null;
    if (names) return names;
  }
  if (titleEnd < 0) return null;
  for (const line of lines.slice(titleEnd + 1, titleEnd + 4)) {
    const names = namesIn(line.text);
    if (names) return names;
  }
  return null;
}

/**
 * The institution a thesis is from is its university, as the degree-granting body: a labelled
 * department ("培养单位：计算机科学与技术系") gives way to the university the cover names.
 */
function institutionOf(lines: readonly PageLine[]): string | null {
  const tidy = (text: string) =>
    text
      // An affiliation's number, "1 中国科学院大学", "（1. …", or its mark.
      .replace(/^[（(]?\s*\d{1,2}\s*[.．、)）]?\s*(?=\D)|^[*†‡§¶#∗♢]+/u, "")
      .split(/[，,；;]|\s{2,}|[*†‡§¶#∗♢]/u)[0]
      ?.trim() ?? "";
  const universityIn = (text: string) => UNIVERSITY.exec(text)?.[0]?.trim() || null;
  const labelled = INSTITUTION_LABELS.map((label) =>
    lines.map((line) => label.exec(line.text)?.[1]?.trim()).find(Boolean),
  );
  // A line of the body that mentions one ("国家实验室体系建设…。") names no institution of the work's.
  const named = lines
    .map((line) => line.text)
    .filter(
      (text) => INSTITUTION.test(text) && !JOURNAL_HEADER.test(text) && !/[@。！？!?]/u.test(text),
    )
    .map(tidy)
    .filter((text) => INSTITUTION.test(text) && text.length <= 60 && /^[\p{L}\p{N}]/u.test(text));
  const [granting, school, unit] = labelled;
  for (const value of [granting, school]) {
    if (value) return universityIn(tidy(value)) ?? tidy(value);
  }
  const university = [unit, ...named]
    .map((text) => (text ? universityIn(tidy(text)) : null))
    .find(Boolean);
  return university ?? (unit ? tidy(unit) : null) ?? named[0] ?? null;
}

/**
 * The year from a cover's or a running header's date, a labelled one first: a date on a line
 * that is mostly the date, or at the top of the page. Not a date in a sentence of the abstract,
 * nor the end of a range ("2020—2024 年"), nor when the paper was received or accepted. Years in
 * Chinese numerals ("二〇二六年五月") count.
 */
function yearOf(lines: readonly PageLine[]): string | null {
  const yearIn = (line: PageLine): string | null => {
    if (OTHER_DATE.test(line.text)) return null;
    for (const match of line.text.matchAll(YEAR_IN_DATE)) {
      if (RANGE_BEFORE.test(line.text.slice(0, match.index))) continue;
      const written = match.slice(1).find((group) => group !== undefined) ?? "";
      const year = [...written].map((digit) => CHINESE_DIGITS[digit] ?? digit).join("");
      const rest = line.text
        .replace(match[0], "")
        .replace(DATE_LABEL, "")
        .replace(/[\d\s月日.,，]/gu, "");
      if (line.top < TOP || rest.length <= 12 || DATE_LABEL.test(line.text)) return year;
    }
    return null;
  };
  const labelled = lines.filter((line) => DATE_LABEL.test(line.text));
  for (const line of [...labelled, ...lines]) {
    const year = yearIn(line);
    if (year) return year;
  }
  return null;
}

/**
 * What its cover says the work is: a thesis by a cover's heading or statement; else a journal
 * article by a journal's marks, whatever its abstract says of theses; else a thesis by a line at
 * the top of the page that isn't the title (a title about theses is no thesis's); a report by its
 * title or the top of the page.
 */
function typeOf(
  lines: readonly PageLine[],
  title: { start: number; end: number } | null,
): FirstPageReading["type"] {
  const inTitle = (index: number) => title !== null && index >= title.start && index <= title.end;
  if (lines.some((line) => THESIS_HEADING.test(line.text) || THESIS_STATEMENT.test(line.text))) {
    return "thesis";
  }
  if (lines.some((line) => JOURNAL.test(line.text))) return "article-journal";
  if (lines.some((line, index) => line.top < TOP && !inTitle(index) && THESIS.test(line.text))) {
    return "thesis";
  }
  const report = lines.some(
    (line, index) => (inTitle(index) || line.top < TOP) && REPORT.test(line.text),
  );
  return report ? "report" : undefined;
}

/** What a PDF's first page gives of its reference record: only what it plainly prints. */
export function readFirstPage(printed: readonly PageLine[]): FirstPageReading {
  const lines = printed
    .map((line) => ({ ...line, text: foldRadicals(line.text).trim() }))
    .filter((line) => line.text !== "");
  const read: FirstPageReading = {};
  const found = titleOf(lines);
  if (found) read.title = found.title;
  const authors = authorsOf(lines, found?.end ?? -1);
  if (authors) read.authors = authors;
  const institution = institutionOf(lines);
  if (institution) read.institution = institution;
  const year = yearOf(lines);
  if (year) read.issued = year;
  const type = typeOf(lines, found);
  if (type) read.type = type;
  return read;
}
