/** The résumé and contract of the Word-Look-Resume-ZH and Word-Look-Contract boards, with synthetic text. */
import { escapeXml, wordFile } from "./word";

const REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

const run = (text: string, rPr = "") =>
  `<w:r>${rPr ? `<w:rPr>${rPr}</w:rPr>` : ""}<w:t xml:space="preserve">${escapeXml(text)}</w:t></w:r>`;

const tabbed = (parts: readonly string[], rPr = "") =>
  parts
    .map(
      (part, index) =>
        `${index > 0 ? `<w:r>${rPr ? `<w:rPr>${rPr}</w:rPr>` : ""}<w:tab/></w:r>` : ""}${run(part, rPr)}`,
    )
    .join("");

const p = (content: string, pPr = "") =>
  `<w:p>${pPr ? `<w:pPr>${pPr}</w:pPr>` : ""}${content}</w:p>`;

const cell = (paragraphs: string, tcPr = "") =>
  `<w:tc>${tcPr ? `<w:tcPr>${tcPr}</w:tcPr>` : ""}${paragraphs}</w:tc>`;

/** A4 with 2 cm margins: 11906 − 2 × 1134 twips. */
export const RESUME_TEXT_WIDTH_TWIPS = 9638;

const RESUME_STYLES = `<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="华文中宋"/><w:sz w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="SectionTitle"><w:name w:val="Section title"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:color w:val="1F3864"/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:tblPr><w:tblCellMar><w:left w:w="108" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>`;

const RESUME_NUMBERING = `<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:pPr><w:ind w:left="420" w:hanging="210"/></w:pPr></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>`;

const entry = (parts: readonly string[], dates: string) =>
  p(
    tabbed([...parts, dates]),
    `<w:tabs><w:tab w:val="left" w:pos="2400"/><w:tab w:val="left" w:pos="4200"/><w:tab w:val="right" w:pos="${RESUME_TEXT_WIDTH_TWIPS}"/></w:tabs>`,
  );

const bullet = (text: string) =>
  p(run(text), '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>');

const layoutTable = (cells: readonly string[], widths: readonly number[]) =>
  `<w:tbl><w:tblPr><w:tblW w:w="${RESUME_TEXT_WIDTH_TWIPS}" w:type="dxa"/><w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders></w:tblPr><w:tblGrid>${widths.map((width) => `<w:gridCol w:w="${width}"/>`).join("")}</w:tblGrid><w:tr>${cells.map((text, index) => cell(p(run(text, '<w:sz w:val="21"/>'), index === cells.length - 1 ? '<w:jc w:val="right"/>' : ""), `<w:tcW w:w="${widths[index]}" w:type="dxa"/>`)).join("")}</w:tr></w:tbl>`;

const skills = `<w:tbl><w:tblPr><w:tblW w:w="${RESUME_TEXT_WIDTH_TWIPS}" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="4472C4"/><w:left w:val="single" w:sz="4" w:color="4472C4"/><w:bottom w:val="single" w:sz="4" w:color="4472C4"/><w:right w:val="single" w:sz="4" w:color="4472C4"/><w:insideH w:val="single" w:sz="4" w:color="4472C4"/><w:insideV w:val="single" w:sz="4" w:color="4472C4"/></w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="2000"/><w:gridCol w:w="7638"/></w:tblGrid>${[
  ["语言", "英语 CET-6、雅思 7.5"],
  ["工具", "Wind、Python、Excel VBA"],
  ["证书", "CFA 二级候选人、证券从业资格"],
]
  .map(
    ([name, value]) =>
      `<w:tr>${cell(p(run(name ?? "")), '<w:tcW w:w="2000" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>')}${cell(p(run(value ?? "")), '<w:tcW w:w="7638" w:type="dxa"/>')}</w:tr>`,
  )
  .join("")}</w:tbl>`;

export function resumeFile(): Buffer {
  const title = (text: string) => p(run(text), '<w:pStyle w:val="SectionTitle"/>');
  return wordFile({
    styles: RESUME_STYLES,
    numbering: RESUME_NUMBERING,
    body: [
      p(
        run("林　晓", '<w:color w:val="1F3864"/><w:spacing w:val="40"/><w:sz w:val="44"/>'),
        '<w:jc w:val="center"/>',
      ),
      p(
        run("电话 138-0000-0000 ｜ 邮箱 linxiao@example.com ｜ 上海", '<w:sz w:val="18"/>'),
        '<w:jc w:val="center"/>',
      ),
      title("教育背景"),
      entry(["复旦大学", "金融学", "硕士"], "2023.09 – 2026.06"),
      p(
        run(
          "GPA 3.8/4.0（前 5%）｜ 主修：公司金融、计量经济学、固定收益证券",
          '<w:sz w:val="18"/>',
        ),
      ),
      entry(["上海财经大学", "会计学", "学士"], "2019.09 – 2023.06"),
      title("实习经历"),
      layoutTable(["某证券公司研究所　食品饮料行业研究实习生", "2025.06 – 2025.09"], [7138, 2500]),
      bullet("跟踪 12 家上市公司季报，搭建毛利率与费用率拆分模型，撰写点评 6 篇"),
      bullet("整理行业数据库，核对研报中的财务数据与公告原文"),
      layoutTable(["某资产管理公司　固定收益部实习生", "2024.07 – 2024.09"], [7138, 2500]),
      bullet("协助信用债尽调，阅读募集说明书并整理偿债指标"),
      title("技能与证书"),
      skills,
      title("荣誉奖项"),
      entry(["国家奖学金"], "2025"),
    ].join(""),
    sectPr:
      '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/>',
  });
}

const CONTRACT_THEME = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"><a:themeElements><a:fontScheme name="Contract"><a:majorFont><a:latin typeface="Book Antiqua"/><a:ea typeface=""/><a:cs typeface=""/><a:font script="Jpan" typeface="游ゴシック Light"/><a:font script="Hang" typeface="맑은 고딕"/><a:font script="Hans" typeface="等线 Light"/><a:font script="Hant" typeface="新細明體"/></a:majorFont><a:minorFont><a:latin typeface="Book Antiqua"/><a:ea typeface=""/><a:cs typeface=""/><a:font script="Jpan" typeface="游明朝"/><a:font script="Hang" typeface="맑은 고딕"/><a:font script="Hans" typeface="宋体"/><a:font script="Hant" typeface="新細明體"/></a:minorFont></a:fontScheme></a:themeElements></a:theme>`;

const CONTRACT_STYLES = `<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:asciiTheme="minorHAnsi" w:hAnsiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia"/><w:sz w:val="24"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="259" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="0"/></w:pPr><w:rPr><w:b/><w:smallCaps/><w:spacing w:val="40"/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Clause"><w:name w:val="Clause"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="both"/><w:ind w:left="567" w:hanging="567"/></w:pPr></w:style><w:style w:type="character" w:styleId="Defined"><w:name w:val="Defined term"/><w:rPr><w:b/></w:rPr></w:style>`;

/**
 * `eastAsia`: the file's East Asian language, as Word records it in its
 * settings ("ja-JP"); `settings`: the settings part's whole text instead.
 */
export function contractFile({
  eastAsia,
  settings = eastAsia
    ? `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:themeFontLang w:val="en-GB" w:eastAsia="${eastAsia}"/></w:settings>`
    : undefined,
}: {
  eastAsia?: string;
  settings?: string;
} = {}): Buffer {
  const clause = (number: string, text: string) =>
    p(tabbed([number, text]), '<w:pStyle w:val="Clause"/>');
  const bordered = (side: string) =>
    `<w:${side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>`;
  const definitions = `<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders>${["top", "left", "bottom", "right", "insideH", "insideV"].map(bordered).join("")}</w:tblBorders><w:tblCellMar><w:top w:w="57" w:type="dxa"/><w:bottom w:w="57" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid><w:gridCol w:w="2600"/><w:gridCol w:w="6426"/></w:tblGrid><w:tr>${cell(p(run("Term", "<w:b/>")), '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>')}${cell(p(run("Meaning", "<w:b/>")), '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>')}</w:tr><w:tr>${cell(p(run("“Products”", '<w:rStyle w:val="Defined"/>')))}${cell(p(run("the milled flour and bakery fats listed in Schedule 2")))}</w:tr><w:tr>${cell(p(run("“Term”", '<w:rStyle w:val="Defined"/>')))}${cell(p(run("three years from 1 January 2027, unless ended earlier under clause 9")), '<w:vAlign w:val="center"/>')}</w:tr></w:tbl>`;
  const signatures = `<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/></w:tblPr><w:tblGrid><w:gridCol w:w="4513"/><w:gridCol w:w="4513"/></w:tblGrid><w:tr>${[
    "Signed for Northwind Foods Ltd",
    "Signed for Corran Bakeries Ltd",
  ]
    .map((who) =>
      cell(
        `${p(run(who))}${p("")}`,
        `<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>`,
      ),
    )
    .join("")}</w:tr></w:tbl>`;
  return wordFile({
    styles: CONTRACT_STYLES,
    rels: `<Relationship Id="rIdTheme" Type="${REL}/theme" Target="theme/theme1.xml"/>${
      settings
        ? `<Relationship Id="rIdSettings" Type="${REL}/settings" Target="settings.xml"/>`
        : ""
    }`,
    parts: [
      { name: "word/theme/theme1.xml", data: CONTRACT_THEME },
      ...(settings ? [{ name: "word/settings.xml", data: settings }] : []),
    ],
    body: [
      p(run("Supply agreement"), '<w:pStyle w:val="Title"/>'),
      p(
        run("between Northwind Foods Ltd and Corran Bakeries Ltd", "<w:i/>"),
        '<w:jc w:val="center"/>',
      ),
      p(run("Dated 3 September 2026"), '<w:jc w:val="right"/>'),
      p(tabbed(["1.", "Definitions"]), '<w:pStyle w:val="Heading1"/>'),
      definitions,
      p(tabbed(["2.", "Prices"]), '<w:pStyle w:val="Heading1"/>'),
      clause(
        "2.1",
        "Prices are fixed for the first twelve months at the rates in Schedule 2. After that, Northwind may raise its prices once a year by up to 6%, on 45 days’ notice.",
      ),
      p(tabbed(["3.", "Orders and payment"]), '<w:pStyle w:val="Heading1"/>'),
      clause(
        "3.1",
        "Corran pays each invoice within 30 days of the invoice date. Late payments carry interest at 4% a year above the Bank of England base rate.",
      ),
      signatures,
    ].join(""),
    sectPr:
      '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>',
  });
}
