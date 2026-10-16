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

/** A journal article about theses: its title says 博士学位论文, and it has a DOI. */
const articleAboutThesesPdf = () =>
  buildPdf([
    {
      printed: [
        { text: "学位与研究生教育 2025 年 6 月", size: 9 },
        { text: "我国博士学位论文抽检结果分析", size: 18 },
        { text: "王明 1，赵静 2", size: 11 },
        { text: "摘要：本文分析了 2020—2024 年博士学位论文抽检的结果。", size: 10 },
        { text: "DOI: 10.5555/thesis-checks-2025", size: 9 },
      ],
    },
  ]);

/** A paper with a DOI that mentions a doctoral dissertation in its text. */
const paperCitingADissertationPdf = () =>
  buildPdf([
    {
      printed: [
        { text: "Night-time cooling in twelve cities", size: 17 },
        { text: "Diana E. Bowler and Andrew S. Pullin", size: 11 },
        { text: "https://doi.org/10.5555/night-cooling-2024", size: 9 },
        {
          text: "This extends the methods of an earlier doctoral dissertation on urban parks",
          size: 10,
        },
        {
          text: "and measures the cooling of parks at night across twelve cities in Europe.",
          size: 10,
        },
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
      // The degree-granting university, not its school.
      publisher: "华东师范大学",
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

  test("a title about theses, or a dissertation the text mentions, doesn't make a paper with a DOI a thesis", async () => {
    const core = startCore(await createTempDataFolder());
    const aboutTheses = await addPdf(core, "theses.pdf", articleAboutThesesPdf());
    expect(aboutTheses.reference?.fields).toMatchObject({
      type: "article-journal",
      title: "我国博士学位论文抽检结果分析",
      authors: [{ literal: "王明" }, { literal: "赵静" }],
      // The date in the running header, not the end of a range in the abstract.
      issued: "2025",
    });
    const citing = await addPdf(core, "cooling.pdf", paperCitingADissertationPdf());
    expect(citing.reference?.fields.type).toBe("article-journal");
  });

  test("what was read waits to be checked; Mark as checked makes it the User's, and Undo puts it back", async () => {
    const core = startCore(await createTempDataFolder());
    const thesis = await addPdf(core, "lin-2026.pdf", thesisPdf());
    const before = thesis.reference;
    if (!before) throw new Error("The thesis has no record.");
    expect(needsCheck(before)).toBe(true);
    // What a style prints, the kind of work too; not the language, which only picks its terms.
    expect(fieldsToCheck(before)).toEqual(["type", "title", "authors", "issued", "publisher"]);

    const checked = await core.confirmReference(thesis.id);
    expect(checked.reference && needsCheck(checked.reference)).toBe(false);
    expect(checked.reference?.sources).toMatchObject({ title: "user", authors: "user" });
    expect(checked.reference?.fields).toEqual(before.fields);

    const undone = await core.restoreReference(thesis.id, before, fieldsToCheck(before));
    expect(undone.reference?.sources.title).toBe("file");
    expect(undone.reference && needsCheck(undone.reference)).toBe(true);
  });
});

const line = (text: string, size = 10, top = 0.5): PageLine => ({ text, size, top });

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

  test("a thesis cover's heading, however it is worded, is no title", () => {
    for (const heading of ["博士研究生学位论文", "硕士专业学位论文", "华东师范大学硕士学位论文"]) {
      const read = readFirstPage([
        line(heading, 26, 0.1),
        line("城市绿地与夏季降温调查", 20, 0.3),
        line("作者姓名：林小雨", 14, 0.6),
      ]);
      expect(read.title, heading).toBe("城市绿地与夏季降温调查");
      expect(read.type, heading).toBe("thesis");
    }
  });

  test("a thesis's institution is its university, not the department its cover labels", () => {
    const read = readFirstPage([
      line("清华大学", 28, 0.08),
      line("博士学位论文", 24, 0.2),
      line("面向城市热岛的遥感反演方法", 20, 0.35),
      line("培养单位：计算机科学与技术系", 14, 0.7),
      line("二〇二五年六月", 14, 0.85),
    ]);
    expect(read).toMatchObject({
      title: "面向城市热岛的遥感反演方法",
      institution: "清华大学",
      issued: "2025",
      type: "thesis",
    });
  });

  test("a journal's running header names no institution, and its date is the year, not a range in the abstract", () => {
    const read = readFirstPage([
      line("第 41 卷 第 3 期 2026 年 3 月 中国科学院院刊", 8, 0.04),
      line("城市更新中的绿地格局", 16, 0.12),
      line("李明 1，张华 2", 11, 0.17),
      line("摘要：基于 2020—2024 年的数据，分析了城市绿地的变化。", 9, 0.25),
      line("文献标识码：A", 9, 0.3),
    ]);
    expect(read.institution).toBeUndefined();
    expect(read.issued).toBe("2026");
    expect(read.type).toBe("article-journal");
  });

  test("a heading under the title (文献综述) is no author, and a title about theses is no thesis", () => {
    const read = readFirstPage([
      line("我国博士学位论文抽检结果分析", 16, 0.1),
      line("文献综述", 11, 0.14),
      line("摘要：本文梳理了学位论文抽检的研究。", 9, 0.3),
    ]);
    expect(read.title).toBe("我国博士学位论文抽检结果分析");
    expect(read.authors).toBeUndefined();
    expect(read.type).toBeUndefined();
  });

  test("a dissertation the text mentions makes no thesis; a cover's statement does", () => {
    expect(
      readFirstPage([
        line("Night-time cooling in twelve cities", 17, 0.1),
        line("This extends the methods of an earlier doctoral dissertation on parks.", 10, 0.5),
      ]).type,
    ).toBeUndefined();
    expect(
      readFirstPage([
        line("Night-time cooling in twelve cities", 17, 0.3),
        line("A dissertation submitted to the University of Leeds", 12, 0.6),
      ]).type,
    ).toBe("thesis");
  });

  test("a journal's marks make an article whatever its abstract says of theses, and an affiliation's number is no part of its institution", () => {
    const read = readFirstPage([
      line("第 41 卷 第 3 期 学位与研究生教育", 8, 0.04),
      line("我国博士学位论文抽检结果分析", 16, 0.1),
      line("王明 1，赵静 2", 11, 0.12),
      line("摘要：本文分析了博士学位论文抽检的结果与原因。", 9, 0.14),
      line("1 中国科学院大学，北京 100049", 8, 0.16),
      line("中图分类号：G643 文献标识码：A", 8, 0.3),
    ]);
    expect(read.type).toBe("article-journal");
    expect(read.institution).toBe("中国科学院大学");
  });

  test("a cover's 题目： label is no part of the title", () => {
    for (const label of ["题目：", "论文题目：", "题 目："]) {
      const read = readFirstPage([
        line("北京大学", 28, 0.08),
        line("硕士研究生学位论文", 24, 0.2),
        line(`${label}城市绿地与夏季降温调查`, 18, 0.35),
        line("姓名：林小雨", 14, 0.6),
        line("学位授予单位：北京大学", 14, 0.7),
      ]);
      expect(read.title, label).toBe("城市绿地与夏季降温调查");
      expect(read.institution, label).toBe("北京大学");
      expect(read.type, label).toBe("thesis");
    }
  });
});
