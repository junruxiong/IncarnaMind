/**
 * GB/T 7714 for a Chinese thesis, through the core: a page raised after the number, or after
 * the author and year, as the standard's own examples print it (GB/T 7714—2015 10.1.3 and
 * 10.2.5, citing 《现代汉语词典》 at pp. 1194 and 354); a work cited twice in one cluster numbered
 * once; an entry too thin to cite flagged in the file and named after the export, never with
 * the day it was added; and GB/T 7714 as the style of a Chinese Mind in a Chinese interface that
 * cites papers or books.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type {
  CitationLocation,
  Core,
  Document,
  FormattedCitations,
  ReferencePatch,
} from "../../src/core";
import { MIND_SETTINGS_FIELD } from "../../src/core/api";
import { dismissCitationStylePrompt } from "../../src/shared/mindCitationStyle";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { paragraphsOf, part, unzip } from "../helpers/docx";
import { connectToMind } from "../helpers/mindClient";
import { writeMind } from "../helpers/minds";

const DICTIONARY: ReferencePatch = {
  type: "book",
  title: "现代汉语词典",
  authors: [{ literal: "中国社会科学院语言研究所词典编辑室" }],
  issued: "1996",
  publisher: "商务印书馆",
  language: "zh",
};

const QIAN: ReferencePatch = {
  type: "article-journal",
  title: "一个科学新领域——开放的复杂巨系统及其方法论",
  authors: [{ literal: "钱学森" }, { literal: "于景元" }, { literal: "戴汝为" }],
  issued: "1990",
  containerTitle: "自然杂志",
  language: "zh",
};

const KAHNEMAN: ReferencePatch = {
  type: "article-journal",
  title: "Prospect theory: An analysis of decision under risk",
  authors: [
    { family: "Kahneman", given: "Daniel" },
    { family: "Tversky", given: "Amos" },
  ],
  issued: "1979",
  containerTitle: "Econometrica",
  language: "en",
};

async function setUp() {
  const dataDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { now: manualClock().now });
  const paths = await Promise.all([
    writeSourceFile(sources, "词典.txt", "编辑思想。"),
    writeSourceFile(sources, "qian.txt", "开放的复杂巨系统。"),
    writeSourceFile(sources, "prospect.txt", "Losses loom larger than gains."),
    writeSourceFile(sources, "城市绿地与夏季降温调查.txt", "绿地使夏季夜间降温。"),
    writeSourceFile(sources, "热岛.txt", "城市热岛。"),
  ]);
  const added = await addAndProcess(core, paths);
  const byName = (name: string) => added.find((each) => each.name === name) as Document;
  const dictionary = await core.updateReference(byName("词典").id, DICTIONARY);
  const qian = await core.updateReference(byName("qian").id, QIAN);
  const prospect = await core.updateReference(byName("prospect").id, KAHNEMAN);
  // A file with nothing to cite it by, and a paper with only its title.
  const blank = byName("城市绿地与夏季降温调查");
  const titled = await core.updateReference(byName("热岛").id, {
    type: "article-journal",
    title: "城市热岛效应研究",
  });
  const mind = await core.createMind({ title: "论文" });
  const client = await connectToMind(core, mind.id);
  return { core, mind, client, dictionary, qian, prospect, blank, titled };
}

const page = (from: number, to = from): CitationLocation => ({ kind: "page", from, to });

const work = (document: Document, location: CitationLocation | null = null) => ({
  documentId: document.id,
  documentName: document.name,
  location,
});

const plain = (runs: readonly { text: string }[] | null | undefined) =>
  (runs ?? []).map((run) => run.text).join("");

/** The text of a cluster's raised runs, and of the rest. */
const raised = (runs: FormattedCitations["clusters"][number]["runs"] = []) => ({
  up: plain(runs.filter((run) => run.script === "sup")),
  level: plain(runs.filter((run) => run.script !== "sup")),
});

async function format(
  core: Core,
  style: FormattedCitations["style"],
  ...clusters: ReturnType<typeof work>[][]
) {
  return core.formatCitations({ style, clusters: clusters.map((works) => ({ works })) });
}

const text = (value: string): JSONContent => ({ type: "text", text: value });
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const citation = (document: Document, location: CitationLocation | null): JSONContent => ({
  type: "citation",
  attrs: {
    passageId: "passage-1",
    documentId: document.id,
    documentName: document.name,
    contentHash: document.contentHash,
    pageFrom: location?.kind === "page" ? location.from : null,
    pageTo: location?.kind === "page" ? location.to : null,
    location,
    quote: "编辑思想。",
    check: "found",
    checkReason: null,
  },
});

describe("GB/T 7714's pages", () => {
  test("numeric: the page is raised after the number, and the same work cited again keeps its number with its own page", async () => {
    const { core, dictionary, prospect } = await setUp();
    const formatted = await format(
      core,
      "gb-t-7714-2015-numeric",
      [work(dictionary, page(1194))],
      [work(prospect)],
      [work(dictionary, page(354))],
    );
    expect(formatted.clusters.map((cluster) => plain(cluster.runs))).toEqual([
      "[1]1194",
      "[2]",
      "[1]354",
    ]);
    // All of it raised, the number and its page, as the standard prints "[2]¹¹⁹⁴".
    expect(raised(formatted.clusters[0]?.runs)).toEqual({ up: "[1]1194", level: "" });
  });

  test("author-date: the page is raised after the parentheses", async () => {
    const { core, dictionary } = await setUp();
    const formatted = await format(core, "gb-t-7714-2015-author-date", [
      work(dictionary, page(1194)),
    ]);
    expect(raised(formatted.clusters[0]?.runs)).toEqual({
      level: "(中国社会科学院语言研究所词典编辑室, 1996)",
      up: "1194",
    });
  });

  test("author-date: works each with their page are set apart by the style's own delimiter", async () => {
    const { core, dictionary, qian } = await setUp();
    const formatted = await format(core, "gb-t-7714-2015-author-date", [
      work(dictionary, page(1194)),
      work(qian),
    ]);
    expect(plain(formatted.clusters[0]?.runs)).toBe(
      "(中国社会科学院语言研究所词典编辑室, 1996)1194; (钱学森等, 1990)",
    );
  });

  test("a Location the standard has no raised form for (a slide, lines) is in brackets on the line", async () => {
    const { core, qian } = await setUp();
    const slide: CitationLocation = { kind: "slide", from: 4, to: 4 };
    const formatted = await format(core, "gb-t-7714-2015-numeric", [work(qian, slide)]);
    const runs = formatted.clusters[0]?.runs ?? [];
    expect(plain(runs)).toMatch(/^\[1\]（.*4.*）$/);
    expect(runs.at(-1)?.script).toBeUndefined();
  });

  test("a range of pages is raised as the standard writes one, and every GB/T style keeps it", async () => {
    const { core, qian } = await setUp();
    for (const style of [
      "gb-t-7714-2015-numeric",
      "gb-t-7714-2025-numeric",
      "gb-t-7714-2015-author-date",
      "gb-t-7714-2025-author-date",
    ] as const) {
      const formatted = await format(core, style, [work(qian, page(5, 6))]);
      expect(raised(formatted.clusters[0]?.runs).up, style).toMatch(/5-6$/);
    }
  });

  test("a style that prints pages itself keeps printing them its own way", async () => {
    const { core, prospect } = await setUp();
    const formatted = await format(core, "apa", [work(prospect, page(265))]);
    expect(plain(formatted.clusters[0]?.runs)).toBe("(Kahneman & Tversky, 1979, p. 265)");
  });
});

describe("a work cited twice in one cluster", () => {
  test("is numbered once, never [1,1], with both its pages", async () => {
    const { core, dictionary, qian } = await setUp();
    const formatted = await format(
      core,
      "gb-t-7714-2015-numeric",
      [work(qian), work(qian)],
      [work(dictionary, page(1194)), work(dictionary, page(354))],
    );
    expect(formatted.clusters.map((cluster) => plain(cluster.runs))).toEqual([
      "[1]",
      "[2]1194,354",
    ]);
  });

  test("is written once in a style that prints pages itself", async () => {
    const { core, prospect } = await setUp();
    const formatted = await format(core, "apa", [
      work(prospect, page(265)),
      work(prospect, page(270)),
    ]);
    expect(plain(formatted.clusters[0]?.runs)).toBe("(Kahneman & Tversky, 1979, pp. 265, 270)");
  });

  test("beside another work, each is in its own brackets once either has a page", async () => {
    const { core, dictionary, qian } = await setUp();
    const formatted = await format(core, "gb-t-7714-2015-numeric", [
      work(dictionary, page(1194)),
      work(qian),
    ]);
    expect(plain(formatted.clusters[0]?.runs)).toBe("[1]1194[2]");
    // Each run still says which Citation it writes, so the editor can tell them apart.
    const runs = formatted.clusters[0]?.runs ?? [];
    expect(plain(runs.filter((run) => run.work === 0))).toBe("11194");
    expect(plain(runs.filter((run) => run.work === 1))).toBe("2");
  });
});

describe("an entry too thin to cite", () => {
  test("is flagged in the file and named after the export, never with the day it was added", async () => {
    const { core, mind, client, dictionary, blank, titled } = await setUp();
    await core.updateSettings({ user: { language: "zh-CN" } });
    await core.setMindCitationStyle(mind.id, "gb-t-7714-2015-numeric");
    writeMind(client, [
      paragraph(
        text("绿地使夏季夜间降温"),
        citation(blank, page(2)),
        text("，热岛效应在夜间最强"),
        citation(titled, null),
        text("，“思想”的内涵见词典"),
        citation(dictionary, page(1194)),
        text("。"),
      ),
    ]);
    await client.settled();

    const exported = await core.exportMind(mind.id, { format: "markdown" });
    const markdown = new TextDecoder().decode(exported.data);
    // Ended as the style ends the rest of its entries, ". ".
    expect(markdown).toContain("\\[1\\] 城市绿地与夏季降温调查. \\[信息待补充\\]");
    expect(markdown).toContain("\\[2\\] 城市热岛效应研究\\[J\\]. \\[信息待补充\\]");
    expect(markdown).toContain(
      "\\[3\\] 中国社会科学院语言研究所词典编辑室. 现代汉语词典\\[M\\]. 商务印书馆, 1996.\n",
    );
    expect(markdown).not.toContain("添加于");
    expect([...exported.needsDetails].sort()).toEqual([blank.id, titled.id].sort());

    const files = unzip((await core.exportMind(mind.id, { format: "docx" })).data);
    const entries = paragraphsOf(part(files, "word/document.xml")).map((each) => each.text);
    expect(entries.some((entry) => entry.endsWith("城市绿地与夏季降温调查. [信息待补充]"))).toBe(
      true,
    );
  });

  test("in English, the mark is in English, after a space", async () => {
    const { core, mind, client, blank } = await setUp();
    await core.setMindCitationStyle(mind.id, "apa");
    writeMind(client, [paragraph(text("Parks cool the night "), citation(blank, null), text("."))]);
    await client.settled();
    const exported = await core.exportMind(mind.id, { format: "markdown" });
    expect(new TextDecoder().decode(exported.data)).toContain(
      "城市绿地与夏季降温调查. \\[needs details\\]",
    );
    expect(exported.needsDetails).toEqual([blank.id]);
  });
});

describe("the style of a Mind that hasn't chosen one", () => {
  test("is GB/T 7714 when the interface and the Mind are both Chinese and it cites a paper or a book, and Numbered otherwise; a choice wins", async () => {
    const { core, mind, client, qian, blank } = await setUp();
    const chinese = [paragraph(text("开放的复杂巨系统是一个科学新领域"), citation(qian, page(5)))];
    writeMind(client, chinese);
    await client.settled();
    const numbered = { style: "numbered", chosen: false, folderDefault: null };
    const gbt = { style: "gb-t-7714-2015-numeric", chosen: false, folderDefault: null };
    expect(await core.getMindCitationStyle(mind.id)).toEqual(numbered);

    await core.updateSettings({ user: { language: "zh-CN" } });
    expect(await core.getMindCitationStyle(mind.id)).toEqual(gbt);
    // The export follows it: its numbers raised, and 参考文献.
    const markdown = new TextDecoder().decode(
      (await core.exportMind(mind.id, { format: "markdown" })).data,
    );
    expect(markdown).toContain("<sup>\\[1\\]5</sup>");
    expect(markdown).toContain("## 参考文献");

    writeMind(client, [
      paragraph(
        text("Open complex giant systems are a new field of science "),
        citation(qian, null),
      ),
    ]);
    await client.settled();
    expect(await core.getMindCitationStyle(mind.id)).toEqual(numbered);

    // The User said "not for this Mind" to the offer of a style: no default changes that.
    writeMind(client, chinese);
    client.doc.transact(() =>
      dismissCitationStylePrompt(client.doc.getMap<unknown>(MIND_SETTINGS_FIELD)),
    );
    await client.settled();
    expect(await core.getMindCitationStyle(mind.id)).toEqual(numbered);
    client.doc.transact(() =>
      dismissCitationStylePrompt(client.doc.getMap<unknown>(MIND_SETTINGS_FIELD), false),
    );
    await client.settled();
    expect(await core.getMindCitationStyle(mind.id)).toEqual(gbt);

    // Citing only files with nothing to cite them by (a contract, a site report), it stays Numbered.
    writeMind(client, [paragraph(text("合同约定每月付租"), citation(blank, page(2)))]);
    await client.settled();
    expect(await core.getMindCitationStyle(mind.id)).toEqual(numbered);

    writeMind(client, chinese);
    await client.settled();
    expect(await core.setMindCitationStyle(mind.id, "apa")).toMatchObject({
      style: "apa",
      chosen: true,
    });
    // Forgetting the choice, as Undo does, brings back the default.
    expect(await core.setMindCitationStyle(mind.id, null)).toEqual(gbt);
  });
});
