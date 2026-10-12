/**
 * Chinese text in a printed PDF that copies and searches as it reads (#100). Chromium writes
 * the text of each glyph (a font's ToUnicode map) as the first code point the font maps to it.
 * CJK fonts draw a Kangxi radical and its ideograph with one glyph, and the radical's code point
 * comes first, so 大, 月, 人 and two hundred others would copy as ⼤, ⽉, ⼈ (U+2F24…), and 页 as
 * ⻚ (U+2EDA). This maps them back to the ideographs Unicode gives as equivalent, in those maps
 * only: what the page shows doesn't change. A PDF not laid out as Chromium lays one out (one
 * classic cross-reference table, maps as compressed streams) is left as it is.
 */
import { deflateSync, inflateSync } from "node:zlib";

/** The CJK radicals (supplement) and the Kangxi radicals. */
const RADICAL = /[⺀-⻿⼀-⿟]/u;

/**
 * The supplement's radicals and their ideographs, in pairs, from Unicode's
 * EquivalentUnifiedIdeograph.txt: few have a compatibility form, and Noto CJK (Linux) draws
 * 页, 见, 长, 门, 车, 马 and others with their radicals' glyphs.
 */
const SUPPLEMENT = new Map(
  [
    ..."⺁厂⺂乛⺃乚⺄乙⺅亻⺆冂⺇𠘨⺈刀⺉刂⺊卜⺋㔾⺌小⺍小⺎兀⺏尣⺐尢⺑𡯂⺒巳⺓幺⺔彑⺕𫜹⺖忄⺗心⺘扌⺙攵⺛旡⺜日⺝月⺞歺⺟母⺠民⺡氵⺢氺⺣灬⺤爫⺥爫⺦丬⺧牛⺨犭⺩王⺪𤴔⺫目⺬示⺭礻⺮𥫗⺯糹⺰纟⺱罓⺲罒⺳㓁⺴冗⺵𦉫⺶羊⺷𦍌⺸𦍋⺹耂⺺肀⺻聿⺼肉⺽𦥑⺾艹⺿艹⻀艹⻁虎⻂衤⻃覀⻄西⻅见⻆角⻇𧢲⻈讠⻉贝⻊𧾷⻋车⻌辶⻍辶⻎辶⻏邑⻐钅⻑長⻒镸⻓长⻔门⻕𨸏⻖阝⻗雨⻘青⻙韦⻚页⻛风⻜飞⻝食⻞𩙿⻟飠⻠饣⻡𩠐⻢马⻣骨⻤鬼⻥鱼⻦鸟⻧卤⻨麦⻩黄⻪黾⻫斉⻬齐⻭歯⻮齿⻯竜⻰龙⻱龜⻲亀⻳龟",
  ].flatMap((char, index, chars) =>
    index % 2 === 0 ? [[char, chars[index + 1] ?? char] as const] : [],
  ),
);

/** A code point's ideograph, if it is a radical that stands for one. */
function ideographOf(text: string): string {
  return text.replace(
    new RegExp(RADICAL, "gu"),
    (radical) => SUPPLEMENT.get(radical) ?? radical.normalize("NFKC"),
  );
}

const hexToText = (hex: string): string => {
  let text = "";
  for (let at = 0; at + 4 <= hex.length; at += 4) {
    text += String.fromCharCode(Number.parseInt(hex.slice(at, at + 4), 16));
  }
  return text;
};

const textToHex = (text: string): string =>
  [...text]
    .flatMap((char) => {
      const units: string[] = [];
      for (let at = 0; at < char.length; at++) {
        units.push((char.charCodeAt(at) ?? 0).toString(16).toUpperCase().padStart(4, "0"));
      }
      return units;
    })
    .join("");

/** One code's text in a ToUnicode map: `<0B0E> <5927>`. */
interface Mapping {
  code: string;
  text: string;
}

/** The mappings of a ToUnicode CMap, its single characters and its ranges spelled out. */
function mappingsOf(cmap: string): Mapping[] | null {
  const mappings: Mapping[] = [];
  for (const [, body] of cmap.matchAll(/beginbfchar([\s\S]*?)endbfchar/g)) {
    for (const [, code, text] of (body ?? "").matchAll(/<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>/g)) {
      mappings.push({ code: code ?? "", text: hexToText(text ?? "") });
    }
  }
  for (const [, body] of cmap.matchAll(/beginbfrange([\s\S]*?)endbfrange/g)) {
    const entry = /<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?:<([0-9A-Fa-f]*)>|\[([^\]]*)\])/g;
    for (const [, low = "", high = "", single, list] of (body ?? "").matchAll(entry)) {
      const from = Number.parseInt(low, 16);
      const to = Number.parseInt(high, 16);
      if (!(to >= from) || to - from > 0xffff) return null;
      const texts = list === undefined ? null : [...list.matchAll(/<([0-9A-Fa-f]*)>/g)];
      for (let code = from; code <= to; code++) {
        let text: string;
        if (texts) {
          text = hexToText(texts[code - from]?.[1] ?? "");
        } else {
          // A range's text goes up with its code, in its last UTF-16 unit.
          const base = hexToText(single ?? "");
          text =
            base.slice(0, -1) + String.fromCharCode(base.charCodeAt(base.length - 1) + code - from);
        }
        mappings.push({ code: code.toString(16).toUpperCase().padStart(low.length, "0"), text });
      }
    }
  }
  return mappings;
}

/** The CMap with its mappings written again as single characters, the radicals as ideographs. */
function unifiedCmap(cmap: string): string | null {
  if (!RADICAL.test(cmap.replace(/<[0-9A-Fa-f]*>/g, (hex) => hexToText(hex.slice(1, -1))))) {
    return null;
  }
  const first = cmap.search(/\d+\s+begin(bfchar|bfrange)/);
  const last = Math.max(cmap.lastIndexOf("endbfchar") + 9, cmap.lastIndexOf("endbfrange") + 10);
  const mappings = mappingsOf(cmap);
  if (first < 0 || last <= first || !mappings) return null;
  let body = "";
  for (let at = 0; at < mappings.length; at += 100) {
    const block = mappings.slice(at, at + 100);
    body += `${block.length} beginbfchar\n`;
    for (const { code, text } of block) body += `<${code}> <${textToHex(ideographOf(text))}>\n`;
    body += "endbfchar\n";
  }
  return cmap.slice(0, first) + body + cmap.slice(last).replace(/^\n/, "");
}

interface Entry {
  offset: number;
  generation: string;
  used: boolean;
}

/** The PDF with its radicals' text mapped to their ideographs; the same bytes if none or unsure. */
export function unifyIdeographs(pdf: Uint8Array): Uint8Array {
  try {
    return unify(pdf) ?? pdf;
  } catch {
    return pdf;
  }
}

function unify(pdf: Uint8Array): Uint8Array | null {
  const bytes = Buffer.from(pdf.buffer, pdf.byteOffset, pdf.byteLength);
  const text = bytes.toString("latin1");
  const start = /startxref\s+(\d+)\s+%%EOF\s*$/.exec(text);
  if (!start) return null;
  const xref = Number(start[1]);
  const trailer = text.indexOf("trailer", xref);
  if (!text.startsWith("xref", xref) || trailer < 0 || /\/Prev\b/.test(text.slice(trailer))) {
    return null;
  }

  // The cross-reference table: one subsection or more, each "first count" and its entries.
  const entries = new Map<number, Entry>();
  const table = text.slice(xref + 4, trailer);
  const lines = table.split(/\r?\n|\r/).filter((line) => line.trim() !== "");
  let next = 0;
  for (const line of lines) {
    const subsection = /^(\d+) (\d+)$/.exec(line.trim());
    if (subsection) {
      next = Number(subsection[1]);
      continue;
    }
    const entry = /^(\d{10}) (\d{5}) ([nf])/.exec(line);
    if (!entry) return null;
    entries.set(next++, {
      offset: Number(entry[1]),
      generation: entry[2] ?? "00000",
      used: entry[3] === "n",
    });
  }

  const replaced: { from: number; to: number; with: Buffer }[] = [];
  const maps = new Set(
    [...text.matchAll(/\/ToUnicode\s+(\d+)\s+0\s+R/g)].map((match) => Number(match[1])),
  );
  for (const number of maps) {
    const entry = entries.get(number);
    if (!entry?.used || !text.startsWith(`${number} 0 obj`, entry.offset)) continue;
    const keyword = text.indexOf("stream", entry.offset);
    const dictionary = text.slice(entry.offset, keyword);
    const length = /\/Length\s+(\d+)(?!\s+\d+\s+R)/.exec(dictionary);
    if (!length || !/\/Filter\s*\/FlateDecode/.test(dictionary)) continue;
    const dataStart = keyword + 6 + (text.startsWith("\r\n", keyword + 6) ? 2 : 1);
    const dataEnd = dataStart + Number(length[1]);
    const end = text.indexOf("endobj", dataEnd);
    if (end < 0) continue;
    const cmap = unifiedCmap(inflateSync(bytes.subarray(dataStart, dataEnd)).toString("latin1"));
    if (cmap === null) continue;
    const data = deflateSync(Buffer.from(cmap, "latin1"));
    const head = dictionary.replace(/\/Length\s+\d+/, `/Length ${data.length}`);
    replaced.push({
      from: entry.offset,
      to: end + 6,
      with: Buffer.concat([
        Buffer.from(`${head}stream\n`, "latin1"),
        data,
        Buffer.from("\nendstream\nendobj", "latin1"),
      ]),
    });
  }
  if (replaced.length === 0) return null;

  replaced.sort((a, b) => a.from - b.from);
  const parts: Buffer[] = [];
  let at = 0;
  for (const each of replaced) {
    parts.push(bytes.subarray(at, each.from), each.with);
    at = each.to;
  }
  parts.push(bytes.subarray(at, xref));
  const moved = (offset: number) =>
    offset +
    replaced
      .filter((each) => each.from < offset)
      .reduce((sum, each) => sum + each.with.length - (each.to - each.from), 0);

  // The table again, every object where it is now.
  const numbers = [...entries.keys()].sort((a, b) => a - b);
  let rebuilt = "xref\n";
  for (let index = 0; index < numbers.length; ) {
    let run = 1;
    while (numbers[index + run] === (numbers[index] ?? 0) + run) run++;
    rebuilt += `${numbers[index]} ${run}\n`;
    for (const number of numbers.slice(index, index + run)) {
      const entry = entries.get(number) as Entry;
      const offset = entry.used ? moved(entry.offset) : entry.offset;
      rebuilt += `${String(offset).padStart(10, "0")} ${entry.generation} ${entry.used ? "n" : "f"}\r\n`;
    }
    index += run;
  }
  const newXref = parts.reduce((sum, part) => sum + part.length, 0);
  const tail = text.slice(trailer, start.index);
  parts.push(Buffer.from(`${rebuilt}${tail}startxref\n${newXref}\n%%EOF\n`, "latin1"));
  return new Uint8Array(Buffer.concat(parts));
}
