/** Spike for #755: a long contract fixture (about 300 pages at 240 clauses in LibreOffice). */
import { escapeXml, wordFile } from "../../tests/helpers/word";

const WORDS =
  "the supplier shall deliver the services with reasonable skill and care in accordance with the specification and good industry practice and the customer shall pay each undisputed invoice within thirty days of receipt subject to clause nine and to any set off permitted by law where the parties agree otherwise in writing".split(
    " ",
  );

function sentence(seed: number, length: number): string {
  const out: string[] = [];
  for (let i = 0; i < length; i++) out.push(WORDS[(seed * 7 + i * 3) % WORDS.length] as string);
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

export function contract(clauses: number): Buffer {
  const body: string[] = [`<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>${run("Framework Services Agreement")}</w:p>`];
  const notes: string[] = [];
  let note = 0;
  for (let clause = 1; clause <= clauses; clause++) {
    body.push(`<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>${run(`${clause}. Clause ${clause}`)}</w:p>`);
    for (let k = 1; k <= 11; k++) {
      const seed = clause * 31 + k;
      let runs = run(
        `${clause}.${k} ${sentence(seed, 18 + (seed % 23))} ${sentence(seed + 1, 12 + (seed % 17))} ${sentence(seed + 2, 9 + (seed % 13))}`,
      );
      if (k === 4 && clause % 2 === 0) {
        note++;
        runs += `<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="${note}"/></w:r>`;
        notes.push(
          `<w:footnote w:id="${note}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/></w:pPr><w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>${run(` ${sentence(seed + 5, 14 + (seed % 20))}`)}</w:p></w:footnote>`,
        );
      }
      body.push(`<w:p><w:pPr><w:jc w:val="both"/></w:pPr>${runs}</w:p>`);
    }
  }
  const header = `<w:p><w:pPr><w:pStyle w:val="Header"/></w:pPr>${run("Framework Services Agreement")}</w:p>`;
  const footer = `<w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr>${run("Page ")}<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>${run("1")}<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>`;
  return wordFile({
    body: body.join(""),
    styles: STYLES,
    footnotes: notes.join(""),
    headers: { rIdH1: header },
    footers: { rIdF1: footer },
    sectPr:
      '<w:headerReference w:type="default" r:id="rIdH1"/><w:footerReference w:type="default" r:id="rIdF1"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>',
  });
}
