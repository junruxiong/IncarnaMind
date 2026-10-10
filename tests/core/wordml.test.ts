/**
 * The WordprocessingML pieces every Word writer shares (src/core/exports/wordml.ts,
 * #77): what each one writes, that the ids they are numbered with are unique
 * and below 0x80000000, and that a document made of them is one Word reads.
 * The .docx export's own tests (exports.test.ts) cover the writer that uses them.
 */
import { deflateRawSync } from "node:zlib";
import { describe, expect, test } from "vitest";
import { extractDocx } from "../../src/core/documents/formats/docx";
import { writeZip } from "../../src/core/documents/formats/zip";
import {
  COMMENTS_CONTENT_TYPE,
  COMMENTS_RELATIONSHIP,
  comment,
  commented,
  commentsPart,
  deletedRun,
  deletion,
  escapeXml,
  footnote,
  footnoteReference,
  footnotesPart,
  insertion,
  MAX_ID,
  R_NAMESPACE,
  run,
  W_NAMESPACE,
  WordIds,
  XML_DECLARATION,
} from "../../src/core/exports/wordml";
import { checkWellFormed, footnotesOf, paragraphsOf, textOf } from "../helpers/docx";

const AT = "2026-10-06T09:00:00.000Z";

/** A part's XML, its root declaring the `w:` namespace, checked to be well-formed. */
const wellFormed = (xml: string) => {
  checkWellFormed(xml);
  return xml;
};

describe("Runs", () => {
  test("a run carries its format in the order the schema requires, and its text escaped", () => {
    expect(
      run('Tides & "neap" <tides>', {
        style: "Hyperlink",
        monospace: true,
        bold: true,
        italic: true,
        strike: true,
        highlight: true,
        underline: true,
      }),
    ).toBe(
      '<w:r><w:rPr><w:rStyle w:val="Hyperlink"/><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:b/><w:i/><w:strike/><w:highlight w:val="yellow"/><w:u w:val="single"/></w:rPr><w:t xml:space="preserve">Tides &amp; &quot;neap&quot; &lt;tides&gt;</w:t></w:r>',
    );
    expect(run("Plain")).toBe('<w:r><w:t xml:space="preserve">Plain</w:t></w:r>');
  });

  test("tabs and line breaks are Word's, characters XML can't hold are left out, and no text is no run", () => {
    expect(run("High\twater\nLow\r\nwater\u0001￾")).toBe(
      '<w:r><w:t xml:space="preserve">High</w:t><w:tab/><w:t xml:space="preserve">water</w:t><w:br/><w:t xml:space="preserve">Low</w:t><w:br/><w:t xml:space="preserve">water</w:t></w:r>',
    );
    expect(run("")).toBe("");
    expect(escapeXml("a\u0000b\u000Bc\td\ne")).toBe("abc\td\ne");
  });
});

describe("Footnotes", () => {
  test("a footnote and its reference share its id, in Word's footnote styles", () => {
    expect(footnoteReference(3)).toBe(
      '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="3"/></w:r>',
    );
    expect(footnote(3, run(" Tides, p. 2"))).toBe(
      '<w:footnote w:id="3"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/></w:pPr><w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r><w:r><w:t xml:space="preserve"> Tides, p. 2</w:t></w:r></w:p></w:footnote>',
    );
  });

  test("the footnotes part has Word's separators as -1 and 0, then the footnotes", () => {
    const part = wellFormed(footnotesPart([footnote(1, run(" Tides, p. 2"))]));

    expect(part.startsWith(XML_DECLARATION)).toBe(true);
    expect(part).toContain(`<w:footnotes xmlns:w="${W_NAMESPACE}" xmlns:r="${R_NAMESPACE}">`);
    expect(
      [...part.matchAll(/<w:footnote (?:w:type="(\w+)" )?w:id="(-?\d+)">/g)].map((m) => [
        m[1],
        m[2],
      ]),
    ).toEqual([
      ["separator", "-1"],
      ["continuationSeparator", "0"],
      [undefined, "1"],
    ]);
    expect(footnotesOf(part)).toEqual({ "1": "Tides, p. 2" });
  });
});

describe("Tracked changes", () => {
  test("an insertion wraps its runs, with its author and its date to the second", () => {
    expect(insertion(4, "IncarnaMind", AT, run("strongest", { bold: true }))).toBe(
      '<w:ins w:id="4" w:author="IncarnaMind" w:date="2026-10-06T09:00:00Z"><w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">strongest</w:t></w:r></w:ins>',
    );
  });

  test("a deletion's text is in w:delText, never w:t, tabs and breaks included", () => {
    const deleted = deletion(5, "Ana & Bo", AT, deletedRun("weak\test\nnow", { italic: true }));

    expect(deleted).toBe(
      '<w:del w:id="5" w:author="Ana &amp; Bo" w:date="2026-10-06T09:00:00Z"><w:r><w:rPr><w:i/></w:rPr><w:delText xml:space="preserve">weak</w:delText><w:tab/><w:delText xml:space="preserve">est</w:delText><w:br/><w:delText xml:space="preserve">now</w:delText></w:r></w:del>',
    );
    expect(deleted).not.toContain("<w:t ");
    expect(deletedRun("")).toBe("");
  });

  test("a change without a date has no date", () => {
    expect(insertion(0, "IncarnaMind", null, run("a"))).toMatch(
      /^<w:ins w:id="0" w:author="IncarnaMind">/,
    );
    expect(deletion(1, "IncarnaMind", null, deletedRun("b"))).toMatch(
      /^<w:del w:id="1" w:author="IncarnaMind">/,
    );
  });
});

describe("Comments", () => {
  test("commented content is the comment's range, with its mark after it", () => {
    expect(commented(7, run("Spring tides"))).toBe(
      '<w:commentRangeStart w:id="7"/><w:r><w:t xml:space="preserve">Spring tides</w:t></w:r><w:commentRangeEnd w:id="7"/><w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="7"/></w:r>',
    );
  });

  test("a comment is a paragraph a line, the first with its mark, by its author at its date", () => {
    const written = comment(7, "IncarnaMind", AT, "Written by IncarnaMind\nCheck <p. 2>");

    expect(written).toBe(
      '<w:comment w:id="7" w:author="IncarnaMind" w:date="2026-10-06T09:00:00Z"><w:p><w:pPr><w:pStyle w:val="CommentText"/></w:pPr><w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:annotationRef/></w:r><w:r><w:t xml:space="preserve">Written by IncarnaMind</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="CommentText"/></w:pPr><w:r><w:t xml:space="preserve">Check &lt;p. 2&gt;</w:t></w:r></w:p></w:comment>',
    );
    const part = wellFormed(commentsPart([written, comment(8, "IncarnaMind", null, "")]));
    expect(part).toContain(`<w:comments xmlns:w="${W_NAMESPACE}" xmlns:r="${R_NAMESPACE}">`);
    expect(paragraphsOf(part).map((paragraph) => paragraph.text)).toEqual([
      "Written by IncarnaMind",
      "Check <p. 2>",
      "",
    ]);
  });
});

describe("Ids", () => {
  test("each kind is numbered on its own: footnotes from 1, annotations from 0, drawings from 1", () => {
    const ids = new WordIds();

    expect([ids.take("footnote"), ids.take("footnote"), ids.take("footnote")]).toEqual([1, 2, 3]);
    expect([ids.take("annotation"), ids.take("annotation")]).toEqual([0, 1]);
    expect([ids.take("drawing"), ids.take("drawing")]).toEqual([1, 2]);
  });

  test("an id the document has is never handed out again, and every id is below 0x80000000", () => {
    const ids = new WordIds();
    const reserved = [-1, 0, 1, 2, 5, 6, 9, MAX_ID, 0x80000000, 0xffffffff];
    for (const id of reserved) ids.reserve("annotation", id);
    ids.reserve("footnote", 1);

    const taken = Array.from({ length: 1000 }, () => ids.take("annotation"));

    expect(taken.slice(0, 5)).toEqual([3, 4, 7, 8, 10]);
    expect(new Set(taken).size).toBe(taken.length);
    expect(taken.some((id) => reserved.includes(id))).toBe(false);
    expect(taken.every((id) => Number.isInteger(id) && id >= 0 && id < 0x80000000)).toBe(true);
    // Reserving one kind's id leaves the others' alone.
    expect(ids.take("footnote")).toBe(2);
    expect(ids.take("drawing")).toBe(1);
  });
});

describe("Together", () => {
  test("a document made of the pieces is well-formed, its ids match up, and it reads as Word reads it with every change accepted", async () => {
    const ids = new WordIds();
    const note = ids.take("footnote");
    const inserted = ids.take("annotation");
    const deleted = ids.take("annotation");
    const remark = ids.take("annotation");
    const body =
      "<w:p>" +
      run("Spring tides are ") +
      insertion(inserted, "IncarnaMind", AT, run("strongest")) +
      deletion(deleted, "Ana", AT, deletedRun("weakest")) +
      run(" at new moon.") +
      footnoteReference(note) +
      "</w:p><w:p>" +
      commented(remark, run("Neap tides are weaker.")) +
      "</w:p>";
    const document = wellFormed(
      `${XML_DECLARATION}<w:document xmlns:w="${W_NAMESPACE}" xmlns:r="${R_NAMESPACE}"><w:body>${body}<w:sectPr/></w:body></w:document>`,
    );
    const footnotes = wellFormed(footnotesPart([footnote(note, run(" Tides, p. 2"))]));
    const comments = wellFormed(
      commentsPart([comment(remark, "IncarnaMind", AT, "Written by IncarnaMind")]),
    );
    const relationships = wellFormed(
      `${XML_DECLARATION}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${R_NAMESPACE}/footnotes" Target="footnotes.xml"/><Relationship Id="rId2" Type="${COMMENTS_RELATIONSHIP}" Target="comments.xml"/></Relationships>`,
    );
    const main = "application/vnd.openxmlformats-officedocument.wordprocessingml";
    const docx = writeZip(
      [
        {
          name: "[Content_Types].xml",
          data: `${XML_DECLARATION}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="${main}.document.main+xml"/><Override PartName="/word/footnotes.xml" ContentType="${main}.footnotes+xml"/><Override PartName="/word/comments.xml" ContentType="${COMMENTS_CONTENT_TYPE}"/></Types>`,
        },
        {
          name: "_rels/.rels",
          data: `${XML_DECLARATION}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${R_NAMESPACE}/officeDocument" Target="word/document.xml"/></Relationships>`,
        },
        { name: "word/document.xml", data: document },
        { name: "word/_rels/document.xml.rels", data: relationships },
        { name: "word/footnotes.xml", data: footnotes },
        { name: "word/comments.xml", data: comments },
      ],
      deflateRawSync,
    );

    // Each reference has its footnote; each comment's range, mark and comment share its id.
    const idsOf = (xml: string, element: string) =>
      [...xml.matchAll(new RegExp(`<w:${element} [^>]*?w:id="(-?\\d+)"`, "g"))].map((m) =>
        Number(m[1]),
      );
    expect(idsOf(document, "footnoteReference")).toEqual([note]);
    expect(idsOf(footnotes, "footnote")).toEqual([-1, 0, note]);
    for (const element of ["commentRangeStart", "commentRangeEnd", "commentReference"]) {
      expect(idsOf(document, element)).toEqual([remark]);
    }
    expect(idsOf(comments, "comment")).toEqual([remark]);
    // Annotations don't share an id.
    const annotations = [...idsOf(document, "ins"), ...idsOf(document, "del"), remark];
    expect(new Set(annotations).size).toBe(3);
    expect(annotations.every((id) => id < 0x80000000)).toBe(true);
    expect(textOf(document)).toBe(
      "Spring tides are strongest at new moon.[1]Neap tides are weaker.",
    );

    // The Word reader takes insertions and leaves deletions and comments out, as accepting all would.
    const { units } = await extractDocx(docx);
    expect(units.map((unit) => unit.text)).toEqual([
      "Spring tides are strongest at new moon.\nNeap tides are weaker.",
      "Tides, p. 2",
    ]);
  });
});
