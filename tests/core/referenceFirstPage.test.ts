/**
 * 引用信息 read from what a PDF's first page prints, offline: a Chinese thesis's cover, an
 * English paper and a report, each into the fields a citation style needs. What is read waits
 * for the User to check it; "Mark as checked" makes it theirs, with Undo. Through the core's
 * public interface, with small PDFs built here.
 */
import { describe, expect, test } from "vitest";
import type { Core, Document } from "../../src/core";
import type { PageLine } from "../../src/core/documents/extract";
import { readFirstPage } from "../../src/core/referenceRecords/firstPage";
import { fieldsToCheck, needsCheck } from "../../src/shared/references";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

const BODY = [
  "Urban parks were on average about one degree cooler than the streets around them,",
  "and the larger parks cooled a wider band of the city on still summer nights. The",
  "effect was strongest after sunset and in parks with tall trees and open water, and",
  "it fell away within a few hundred metres of the park's edge in most of the cities.",
].map((text) => ({ text, size: 10 }));

/** A master's thesis's cover, as Chinese universities set one. */
const thesisPdf = () =>
  buildPdf([
    {
      printed: [
        { text: "分类号：X171", size: 10.5 },
        { text: "硕士学位论文", size: 26 },
        { text: "城市绿地与夏季降温调查", size: 20 },
        { text: "作者姓名：林小雨", size: 14 },
        { text: "指导教师：王建国 教授", size: 14 },
        { text: "学科专业：环境科学", size: 14 },
        { text: "培养单位：华东师范大学地理科学学院", size: 14 },
        { text: "二〇二六年五月", size: 14 },
      ],
    },
  ]);

/** A paper with no metadata of its own: its title, authors and DOI on its first page. */
const paperPdf = () =>
  buildPdf([
    {
      printed: [
        { text: "Preprint. Under review.", size: 9 },
        { text: "Cooling Effects of Urban Parks:", size: 17 },
        { text: "Evidence from Twelve Cities", size: 17 },
        { text: "Diana E. Bowler1, Lisette Buyung-Ali2 and Andrew S. Pullin1*", size: 11 },
        { text: "1 Centre for Evidence-Based Conservation, Bangor University, UK", size: 9 },
        { text: "2 School of Environment, University of Leeds, UK", size: 9 },
        { text: "Received: 3 March 2024; Accepted: 9 May 2024", size: 9 },
        { text: "Published 1 June 2024", size: 9 },
        { text: "https://doi.org/10.5555/urban-parks-2024", size: 9 },
        { text: "Abstract", size: 12 },
        ...BODY,
      ],
    },
  ]);

/** An organisation's annual report: no person named, its organisation and date on the cover. */
const reportPdf = () =>
  buildPdf([
    {
      printed: [
        { text: "Northwind Institute for Urban Climate", size: 12 },
        { text: "Heat in the City: Annual Report 2025", size: 22 },
        { text: "A review of summer temperatures in forty cities", size: 14 },
        { text: "March 2026", size: 11 },
        ...BODY,
      ],
    },
  ]);

async function addPdf(core: Core, name: string, contents: Uint8Array): Promise<Document> {
  const sources = await createSourceFolder();
  const [document] = await addAndProcess(core, [await writeSourceFile(sources, name, contents)]);
  if (!document) throw new Error(`${name} wasn't added.`);
  return document;
}

describe("引用信息 from a PDF's first page", () => {
  test("a Chinese thesis's cover gives its title, its author, its university and its year, as a thesis", async () => {
    const core = startCore(await createTempDataFolder());
    const thesis = await addPdf(core, "lin-2026.pdf", thesisPdf());
    expect(thesis.reference?.fields).toMatchObject({
      type: "thesis",
      title: "城市绿地与夏季降温调查",
      authors: [{ literal: "林小雨" }],
      issued: "2026",
      publisher: "华东师范大学地理科学学院",
      language: "zh",
    });
  });

  test("an English paper's first page gives its title in its largest type, and its authors under it without their affiliation marks", async () => {
    const core = startCore(await createTempDataFolder());
    const paper = await addPdf(core, "parks.pdf", paperPdf());
    expect(paper.reference?.fields).toMatchObject({
      type: "article-journal",
      title: "Cooling Effects of Urban Parks: Evidence from Twelve Cities",
      authors: [
        { family: "Bowler", given: "Diana E." },
        { family: "Buyung-Ali", given: "Lisette" },
        { family: "Pullin", given: "Andrew S." },
      ],
      // When it was published, not when it was received or accepted.
      issued: "2024",
      doi: "10.5555/urban-parks-2024",
      // A paper's affiliations are no publisher of the paper's.
      publisher: "",
    });
  });

  test("a report gives its title, its organisation as its author and publisher, and its year", async () => {
    const core = startCore(await createTempDataFolder());
    const report = await addPdf(core, "heat-2025.pdf", reportPdf());
    expect(report.reference?.fields).toMatchObject({
      type: "report",
      title: "Heat in the City: Annual Report 2025",
      authors: [{ literal: "Northwind Institute for Urban Climate" }],
      publisher: "Northwind Institute for Urban Climate",
      issued: "2026",
    });
  });

  test("what was read waits to be checked; Mark as checked makes it the User's, and Undo puts it back", async () => {
    const core = startCore(await createTempDataFolder());
    const thesis = await addPdf(core, "lin-2026.pdf", thesisPdf());
    const before = thesis.reference;
    if (!before) throw new Error("The thesis has no record.");
    expect(needsCheck(before)).toBe(true);
    // What a style prints; not the type or the language the reading tells from the page.
    expect(fieldsToCheck(before)).toEqual(["title", "authors", "issued", "publisher"]);

    const checked = await core.confirmReference(thesis.id);
    expect(checked.reference && needsCheck(checked.reference)).toBe(false);
    expect(checked.reference?.sources).toMatchObject({ title: "user", authors: "user" });
    expect(checked.reference?.fields).toEqual(before.fields);

    const undone = await core.restoreReference(thesis.id, before, fieldsToCheck(before));
    expect(undone.reference?.sources.title).toBe("file");
    expect(undone.reference && needsCheck(undone.reference)).toBe(true);
  });
});

const line = (text: string, size = 10): PageLine => ({ text, size });

describe("reading a first page leaves out what it doesn't plainly print", () => {
  test("a heading under the title is no author, and a body line naming a laboratory is no institution", () => {
    const read = readFirstPage([
      line("Northwind Retail · 2026 年第三季度业绩", 17),
      line("要点", 11.5),
      line("国家实验室体系建设有力推进，关键核心技术攻关成果丰硕。"),
      line("第三季度收入同比增长 4.2%，线上业务拖累毛利率。"),
    ]);
    expect(read.title).toBe("Northwind Retail · 2026 年第三季度业绩");
    expect(read.authors).toBeUndefined();
    expect(read.institution).toBeUndefined();
  });

  test("a page set in one size has no title to read, and a received date is no year", () => {
    expect(
      readFirstPage([
        line("Spring and neap tides"),
        line("Received: 3 March 2024"),
        line("Spring tides happen at new moon and at full moon."),
      ]),
    ).toEqual({});
  });

  test("Chinese authors spaced out or marked with affiliations are read whole", () => {
    const read = readFirstPage([
      line("生物亲和垂钓技术在天然产物活性成分发现中的研究进展", 14.8),
      line("曲清莉 1，2， 王小芳 3， 邸多隆 1， 裴 栋 1，2*", 11),
      line("（1. 中国科学院兰州化学物理研究所， 甘肃 兰州 730000）", 8.3),
      line("摘要：天然产物是创新药物发现的重要源泉。", 8.3),
      line("文献标识码：A", 8.3),
    ]);
    expect(read.authors).toEqual([
      { literal: "曲清莉" },
      { literal: "王小芳" },
      { literal: "邸多隆" },
      { literal: "裴栋" },
    ]);
    expect(read.type).toBe("article-journal");
  });
});
