/**
 * A reference record from what a PDF's first page prints: its title, in the page's largest
 * type; its authors, under the title or after "作者："; the institution a thesis or report comes
 * from; the year on its cover or in its running header; and the kind of work it says it is.
 * Pure. What isn't plainly there is left out, and what is read is a suggestion the User checks
 * (`fieldsToCheck` in ../../shared/references), never a fact.
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

/** Headings that open a page but name no work, a thesis cover's "硕士学位论文" among them. */
const GENERIC =
  /^(?:abstract|summary|executive summary|contents|table of contents|preface|foreword|introduction|acknowledge?ments?|appendix|preprint|draft|chapter\s+\d+|(?:master|doctoral|bachelor)(?:'s)?\s+(?:thesis|dissertation)|摘\s*要|目\s*录|前\s*言|序\s*言|引\s*言|致\s*谢|附\s*录|第[一二三四五六七八九十\d]+章|(?:硕\s*士|博\s*士|学\s*士)?\s*学\s*位\s*论\s*文|毕\s*业\s*(?:论\s*文|设\s*计))(?:$|[\s:：.])/iu;
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
/** A date on the line ("2026年10月15日"), whose numbers are no affiliation marks. */
const DATE_ON_LINE = /\d\s*[年月日]/u;

/** A label before a value on a cover: "作者：张三", "Author: Jane Smith". */
const AUTHOR_LABEL =
  /^(?:作\s*者(?:\s*姓\s*名)?|著\s*者|研\s*究\s*生|学\s*生\s*姓\s*名|姓\s*名|申\s*请\s*人|authors?|by)\s*[:：]?\s*(.+)$/iu;
const INSTITUTION_LABEL =
  /^(?:培\s*养\s*单\s*位|学\s*位\s*授\s*予\s*单\s*位|授\s*予\s*单\s*位|学\s*校|单\s*位|institution|university)\s*[:：]\s*(.+)$/iu;
const DATE_LABEL =
  /^(?:完\s*成\s*日\s*期|答\s*辩\s*日\s*期|提\s*交\s*日\s*期|论\s*文\s*日\s*期|日\s*期|date)\s*[:：]/iu;
/** Dates that aren't the work's: when a paper was received or accepted, or a preprint's version posted. */
const OTHER_DATE =
  /收稿|修回|录用|接收|基金|received|revised|accepted|submitted|funding|arxiv\s*:/iu;
const INSTITUTION =
  /大学|学院|研究所|研究院|研究中心|科学院|实验室|university|institute|college|academy|school of|laborator/iu;

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
    "\\b\\d{1,2}\\s+(?:" + MONTHS + ")\\.?\\s+((?:19|20)\\d{2})\\b",
  ].join("|"),
  "iu",
);
const CHINESE_YEAR = /([〇零一二三四五六七八九]{4})\s*年/u;
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

const THESIS =
  /学位论文|毕业论文|毕业设计|(?:master|doctoral|ph\.?\s?d\.?|bachelor)(?:'s)?\s+(?:thesis|dissertation)|\bdissertation\b|partial fulfil+ment/iu;
const REPORT =
  /工作报告|研究报告|调查报告|年度报告|白皮书|\b(?:annual|technical|research|working)\s+(?:report|paper)\b|\bwhite\s+paper\b/iu;
/** What a Chinese journal prints on an article's first page. */
const JOURNAL = /文献标识码|中图分类号|文章编号/u;

const meaningful = (text: string) => (text.match(LETTER)?.length ?? 0) >= 2;

/** Lines joined as one: no space between two Han characters, one between anything else. */
function joinLines(texts: readonly string[]): string {
  return texts.reduce((joined, text) => {
    if (joined === "") return text;
    const glue = HAN.test(joined.slice(-1)) && HAN.test(text[0] ?? "") ? "" : " ";
    return `${joined}${glue}${text}`;
  }, "");
}

/** The size most of the page's text is set in. */
function bodySize(lines: readonly PageLine[]): number {
  const weight = new Map<number, number>();
  for (const line of lines) {
    weight.set(line.size, (weight.get(line.size) ?? 0) + line.text.length);
  }
  let body = 0;
  let most = 0;
  for (const [size, chars] of weight) {
    if (chars > most) [body, most] = [size, chars];
  }
  return body;
}

/** The title, in the largest type set well above the body's, and the index of its last line. */
function titleOf(lines: readonly PageLine[]): { title: string; end: number } | null {
  const body = bodySize(lines);
  const sizes = [...new Set(lines.filter((line) => meaningful(line.text)).map((l) => l.size))]
    .filter((size) => size >= body * 1.15)
    .sort((a, b) => b - a)
    .slice(0, 3);
  for (const size of sizes) {
    const start = lines.findIndex(
      (line) => Math.abs(line.size - size) < 0.5 && meaningful(line.text),
    );
    let end = start;
    while (end + 1 < lines.length && Math.abs((lines[end + 1] as PageLine).size - size) < 0.5) {
      end++;
    }
    const title = joinLines(lines.slice(start, end + 1).map((line) => line.text));
    if (!GENERIC.test(title) && !ONLY_AN_INSTITUTION.test(title) && title.length <= 300) {
      return { title, end };
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
    (/^\p{Script=Han}{2,4}$/u.test(name) && (labelled || hasSurname(name)));
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

/** The authors: after a label anywhere, else on the lines just under the title. */
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

/** The institution, as a cover labels it or as the first line naming one gives it. */
function institutionOf(lines: readonly PageLine[]): string | null {
  const tidy = (text: string) =>
    text
      .replace(/^[（(]?\s*\d+\s*[.．、]\s*|^[*†‡§¶#∗♢]+/u, "")
      .split(/[，,；;]|\s{2,}|[*†‡§¶#∗♢]/u)[0]
      ?.trim() ?? "";
  for (const line of lines) {
    const value = INSTITUTION_LABEL.exec(line.text)?.[1];
    if (value) return tidy(value) || null;
  }
  for (const line of lines) {
    // A line of the body that mentions one ("国家实验室体系建设…。") names no institution of the work's.
    if (!INSTITUTION.test(line.text) || /[@。！？!?]/u.test(line.text)) continue;
    const named = tidy(line.text);
    if (INSTITUTION.test(named) && named.length <= 60 && /^[\p{L}\p{N}]/u.test(named)) {
      return named;
    }
  }
  return null;
}

/**
 * The year, from a date on a short line (a cover's, a running header's) that isn't when the
 * paper was received or accepted; a labelled date first. Years written in Chinese numerals
 * ("二〇二六年五月") count.
 */
function yearOf(lines: readonly PageLine[]): string | null {
  const yearIn = (text: string): string | null => {
    const chinese = CHINESE_YEAR.exec(text)?.[1];
    if (chinese) return [...chinese].map((digit) => CHINESE_DIGITS[digit]).join("");
    const match = YEAR_IN_DATE.exec(text);
    return match?.slice(1).find((group) => group !== undefined) ?? null;
  };
  const dated = lines.filter((line) => line.text.length <= 40 && !OTHER_DATE.test(line.text));
  for (const line of dated) {
    if (DATE_LABEL.test(line.text)) {
      const year = yearIn(line.text);
      if (year) return year;
    }
  }
  for (const line of dated) {
    const year = yearIn(line.text);
    if (year) return year;
  }
  return null;
}

/** The kind of work the page says it is, if it says. */
function typeOf(text: string): FirstPageReading["type"] {
  if (THESIS.test(text)) return "thesis";
  if (JOURNAL.test(text)) return "article-journal";
  if (REPORT.test(text)) return "report";
  return undefined;
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
  const type = typeOf(lines.map((line) => line.text).join("\n"));
  if (type) read.type = type;
  return read;
}
