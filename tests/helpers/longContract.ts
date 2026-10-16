/**
 * A long contract saved as Word, for tests that need a Mind of hundreds of
 * pages (#812): a title, then numbered clauses of a heading and eleven
 * justified paragraphs, with a footnote in every other clause, a header and a
 * page-number footer. 240 clauses come to about 300 A4 pages in LibreOffice,
 * the "304-page contract" of the Page view spike (#755).
 */
import { escapeXml, wordFile } from "./word";

const WORDS =
  "the supplier shall deliver the services with reasonable skill and care in accordance with the specification and good industry practice and the customer shall pay each undisputed invoice within thirty days of receipt subject to clause nine and to any set off permitted by law where the parties agree otherwise in writing".split(
    " ",
  );

function sentence(seed: number, length: number): string {
  const out: string[] = [];
  for (let i = 0; i < length; i++) out.push(WORDS[(seed * 7 + i * 3) % WORDS.length] ?? "");
  const text = out.join(" ");
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

const run = (text: string) => `<w:r><w:t xml:space="preserve">${escapeXml(text)}</w:t></w:r>`;

const STYLES = [
  '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:widowControl/><w:spacing w:after="160" w:line="259" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>',
  '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>',
  '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>',
  '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>',
  '<w:style w:type="paragraph" w:styleId="FootnoteText"><w:name w:val="footnote text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="20"/></w:rPr></w:style>',
  '<w:style w:type="character" w:styleId="FootnoteReference"><w:name w:val="footnote reference"/><w:rPr><w:vertAlign w:val="superscript"/></w:rPr></w:style>',
  '<w:style w:type="paragraph" w:styleId="Header"><w:name w:val="header"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0"/></w:pPr></w:style>',
  '<w:style w:type="paragraph" w:styleId="Footer"><w:name w:val="footer"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="0"/></w:pPr></w:style>',
].join("");

const CLAUSE_STYLE =
  '<w:style w:type="paragraph" w:styleId="Clause"><w:name w:val="Clause"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="both"/><w:ind w:left="567" w:hanging="567"/></w:pPr></w:style>';

/** The clauses that make the 304-page contract. */
export const LONG_CONTRACT_CLAUSES = 240;

/** `tabbed`: each clause's number is followed by a tab to a hanging indent, as a contract's often is (#746). */
export function longContract(
  clauses: number = LONG_CONTRACT_CLAUSES,
  { tabbed = false }: { tabbed?: boolean } = {},
): Buffer {
  const body: string[] = [
    `<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>${run("Framework Services Agreement")}</w:p>`,
  ];
  const notes: string[] = [];
  let note = 0;
  for (let clause = 1; clause <= clauses; clause++) {
    body.push(
      `<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>${run(`${clause}. Clause ${clause}`)}</w:p>`,
    );
    for (let k = 1; k <= 11; k++) {
      const seed = clause * 31 + k;
      const text = `${sentence(seed, 18 + (seed % 23))} ${sentence(seed + 1, 12 + (seed % 17))} ${sentence(seed + 2, 9 + (seed % 13))}`;
      let runs = tabbed
        ? `${run(`${clause}.${k}`)}<w:r><w:tab/></w:r>${run(text)}`
        : run(`${clause}.${k} ${text}`);
      if (k === 4 && clause % 2 === 0) {
        note++;
        runs += `<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="${note}"/></w:r>`;
        notes.push(
          `<w:footnote w:id="${note}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/></w:pPr><w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>${run(` ${sentence(seed + 5, 14 + (seed % 20))}`)}</w:p></w:footnote>`,
        );
      }
      body.push(
        `<w:p><w:pPr>${tabbed ? '<w:pStyle w:val="Clause"/>' : '<w:jc w:val="both"/>'}</w:pPr>${runs}</w:p>`,
      );
    }
  }
  const header = `<w:p><w:pPr><w:pStyle w:val="Header"/></w:pPr>${run("Framework Services Agreement")}</w:p>`;
  const footer = `<w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr>${run("Page ")}<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>${run("1")}<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>`;
  return wordFile({
    body: body.join(""),
    styles: tabbed ? STYLES + CLAUSE_STYLE : STYLES,
    footnotes: notes.join(""),
    headers: { rIdH1: header },
    footers: { rIdF1: footer },
    sectPr:
      '<w:headerReference w:type="default" r:id="rIdH1"/><w:footerReference w:type="default" r:id="rIdF1"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>',
  });
}
