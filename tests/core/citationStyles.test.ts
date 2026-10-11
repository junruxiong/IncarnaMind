/**
 * Citation styles through the core (#289): a Mind's style, its Citations and Bibliography from
 * the Documents' reference records, formatted on the style thread, and the exports in the style.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type {
  CitationCheck,
  CitationLocation,
  Core,
  Document,
  FormattedCitations,
  ReferencePatch,
} from "../../src/core";
import { type CslItem, locatorOf } from "../../src/core/citationStyles/csl";
import { createWorkerEngine } from "../../src/core/citationStyles/engine";
import { fixtureReferences, markup } from "../helpers/citationStyles";
import { CITATION_STYLES, createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { footnotesOf, paragraphsOf, part, textOf, unzip } from "../helpers/docx";
import { connectToMind } from "../helpers/mindClient";
import { writeMind } from "../helpers/minds";

const KAHNEMAN: ReferencePatch = {
  type: "article-journal",
  title: "Prospect theory: An analysis of decision under risk",
  authors: [
    { family: "Kahneman", given: "Daniel" },
    { family: "Tversky", given: "Amos" },
  ],
  issued: "1979-03",
  containerTitle: "Econometrica",
  volume: "47",
  issue: "2",
  pages: "263–291",
  doi: "10.2307/1914185",
  language: "en",
};

const QIAN: ReferencePatch = {
  type: "article-journal",
  title: "一个科学新领域——开放的复杂巨系统及其方法论",
  authors: [{ literal: "钱学森" }, { literal: "于景元" }, { literal: "戴汝为" }],
  issued: "1990",
  containerTitle: "自然杂志",
  volume: "13",
  issue: "1",
  pages: "3–10",
  language: "zh",
};

/** A core with three Documents: an English paper, a Chinese one, and a file with no details. */
async function setUp() {
  const dataDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { now: manualClock().now });
  const paths = await Promise.all([
    writeSourceFile(sources, "prospect.txt", "Losses loom larger than gains."),
    writeSourceFile(sources, "qian.txt", "开放的复杂巨系统。"),
    writeSourceFile(sources, "Lease agreement.txt", "The tenant pays rent monthly."),
  ]);
  const added = await addAndProcess(core, paths);
  const byName = (name: string) => added.find((each) => each.name === name) as Document;
  const prospect = await core.updateReference(byName("prospect").id, KAHNEMAN);
  const qian = await core.updateReference(byName("qian").id, QIAN);
  const lease = byName("Lease agreement");
  const mind = await core.createMind({ title: "Thesis" });
  const client = await connectToMind(core, mind.id);
  return { core, mind, client, prospect, qian, lease };
}

/** A Citation of a Document, as the core writes one into an Answer. */
function citation(
  document: Document,
  location: CitationLocation | null,
  check: CitationCheck = "found",
): JSONContent {
  return {
    type: "citation",
    attrs: {
      passageId: "passage-1",
      documentId: document.id,
      documentName: document.name,
      contentHash: document.contentHash,
      pageFrom: location?.kind === "page" ? location.from : null,
      pageTo: location?.kind === "page" ? location.to : null,
      location,
      quote: "Losses loom larger than gains.",
      check,
      checkReason: null,
    },
  };
}

const text = (value: string): JSONContent => ({ type: "text", text: value });
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const page = (from: number, to = from): CitationLocation => ({ kind: "page", from, to });

const plain = (runs: readonly { text: string }[] | null | undefined) =>
  (runs ?? []).map((run) => run.text).join("");

const work = (document: Document, location: CitationLocation | null = null) => ({
  documentId: document.id,
  documentName: document.name,
  location,
});

async function format(
  core: Core,
  style: FormattedCitations["style"],
  ...clusters: ReturnType<typeof work>[][]
) {
  return core.formatCitations({ style, clusters: clusters.map((works) => ({ works })) });
}

describe("a Mind's citation style", () => {
  test("follows the interface language until the Mind chooses one, and keeps its choice", async () => {
    const { core, mind } = await setUp();
    expect(await core.getMindCitationStyle(mind.id)).toEqual({ style: "apa", chosen: false });
    await core.updateSettings({ user: { language: "zh-CN" } });
    expect(await core.getMindCitationStyle(mind.id)).toEqual({
      style: "gb-t-7714-2015-numeric",
      chosen: false,
    });
    expect(await core.setMindCitationStyle(mind.id, "mla")).toEqual({ style: "mla", chosen: true });
    await core.updateSettings({ user: { language: "en" } });
    expect(await core.getMindCitationStyle(mind.id)).toEqual({ style: "mla", chosen: true });
    // Undo puts back "no choice": the default again.
    expect(await core.setMindCitationStyle(mind.id, null)).toEqual({ style: "apa", chosen: false });
  });

  test("is part of the Mind's document, so an open window sees the change", async () => {
    const { core, mind, client } = await setUp();
    await core.setMindCitationStyle(mind.id, "ieee");
    await client.settled();
    expect(client.doc.getMap("settings").get("citationStyle")).toBe("ieee");
  });

  test("refuses what isn't a style", async () => {
    const { core, mind } = await setUp();
    await expect(
      core.setMindCitationStyle(mind.id, "harvard-2" as unknown as "harvard"),
    ).rejects.toThrow("isn't a citation style");
  });
});

describe("formatting a Mind's Citations", () => {
  test("in-text, from each Document's reference record, with the page from its Location", async () => {
    const { core, prospect, qian } = await setUp();
    const apa = await format(core, "apa", [work(prospect, page(265))], [work(qian, page(5, 6))]);
    expect(apa.form).toBe("in-text");
    expect(apa.clusters.map((cluster) => plain(cluster.runs))).toEqual([
      "(Kahneman & Tversky, 1979, p. 265)",
      "(钱学森 et al., 1990, pp. 5–6)",
    ]);
    expect(apa.bibliography.map((entry) => entry.documentId)).toEqual([prospect.id, qian.id]);
    expect(plain(apa.bibliography[0]?.runs)).toContain("Econometrica");
    expect(apa.hangingIndent).toBe(true);
  });

  test("a Location that isn't a page is cited in its own words, not as an invented page", async () => {
    const { core, prospect } = await setUp();
    const slide: CitationLocation = { kind: "slide", from: 4, to: 4 };
    const section: CitationLocation = { kind: "section", heading: "Method" };
    const formatted = await format(core, "apa", [work(prospect, slide)], [work(prospect, section)]);
    expect(formatted.clusters.map((cluster) => plain(cluster.runs))).toEqual([
      "(Kahneman & Tversky, 1979, slide 4)",
      "(Kahneman & Tversky, 1979, § Method)",
    ]);
    // In the style's language: Chinese words for a GB/T style. (The CSL project's GB/T styles
    // print no locator in a citation at all, so nothing of it shows there today.)
    expect(locatorOf(slide, "zh-CN")).toEqual({ locator: "第 4 张幻灯片", label: "custom" });
    expect(locatorOf(page(4, 5), "zh-CN")).toEqual({ locator: "4-5", label: "page" });
    expect(locatorOf({ kind: "lines", from: 3, to: 3 }, "en")).toEqual({
      locator: "3",
      label: "line",
    });
    expect(locatorOf(null, "en")).toEqual({});
  });

  test("Citations side by side are one cluster, its runs marked with the Citation each belongs to", async () => {
    const { core, prospect, qian } = await setUp();
    const formatted = await format(core, "ieee", [work(qian), work(prospect)]);
    const runs = formatted.clusters[0]?.runs ?? [];
    expect(plain(runs)).toBe("[1], [2]");
    expect(runs.filter((run) => run.work !== undefined).map((run) => [run.text, run.work])).toEqual(
      [
        ["[1]", 0],
        ["[2]", 1],
      ],
    );
  });

  test("a note style writes notes, and the same work again as a short note", async () => {
    const { core, prospect } = await setUp();
    const formatted = await format(
      core,
      "chicago-notes",
      [work(prospect, page(265))],
      [work(prospect, page(270))],
    );
    expect(formatted.form).toBe("note");
    expect(plain(formatted.clusters[0]?.runs)).toContain("Daniel Kahneman and Amos Tversky");
    expect(plain(formatted.clusters[1]?.runs)).toBe(
      "Kahneman and Tversky, “Prospect Theory: An Analysis of Decision under Risk,” 270.",
    );
  });

  test("a Document with no details is listed as it is, marked as needing details", async () => {
    const { core, prospect, lease } = await setUp();
    const formatted = await format(core, "apa", [work(lease, page(2))], [work(prospect)]);
    expect(plain(formatted.clusters[0]?.runs)).toBe("(Lease Agreement, n.d., p. 2)");
    const entry = formatted.bibliography.find((each) => each.documentId === lease.id);
    expect(entry?.needsDetails).toBe(true);
    expect(
      formatted.bibliography.find((each) => each.documentId === prospect.id)?.needsDetails,
    ).toBe(false);
  });

  test("a deleted Document's Citation is still listed, by the name it had", async () => {
    const { core, prospect } = await setUp();
    const gone = {
      documentId: "00000000-0000-4000-8000-000000000000",
      documentName: "Old notes",
      location: null,
    };
    const formatted = await core.formatCitations({
      style: "apa",
      clusters: [{ works: [gone] }, { works: [work(prospect)] }],
    });
    expect(plain(formatted.clusters[0]?.runs)).toBe("(Old Notes, n.d.)");
    expect(formatted.bibliography.find((each) => each.documentId === null)?.needsDetails).toBe(
      true,
    );
  });

  test("the Bibliography follows an edit to a reference record", async () => {
    const { core, prospect } = await setUp();
    const before = await format(core, "apa", [work(prospect)]);
    expect(plain(before.bibliography[0]?.runs)).toContain("Prospect theory");
    await core.updateReference(prospect.id, { title: "Prospect theory revisited" });
    const after = await format(core, "apa", [work(prospect)]);
    expect(plain(after.bibliography[0]?.runs)).toContain("Prospect theory revisited");
  });

  test("Numbered lists each cited Document once, by the name its footnotes give it", async () => {
    const { core, prospect, lease } = await setUp();
    const formatted = await format(
      core,
      "numbered",
      [work(prospect)],
      [work(lease)],
      [work(prospect, page(2))],
    );
    expect(formatted.form).toBe("numbered");
    expect(formatted.clusters).toEqual([{ runs: [] }, { runs: [] }, { runs: [] }]);
    expect(formatted.bibliography).toEqual([
      {
        documentId: lease.id,
        label: null,
        runs: [{ text: "Lease agreement" }],
        needsDetails: false,
      },
      { documentId: prospect.id, label: null, runs: [{ text: "prospect" }], needsDetails: false },
    ]);
  });

  test("no Citations: an empty Bibliography", async () => {
    const { core } = await setUp();
    expect(await core.formatCitations({ style: "apa", clusters: [] })).toEqual({
      style: "apa",
      form: "in-text",
      clusters: [],
      bibliography: [],
      hangingIndent: false,
    });
  });

  test("refuses a request that isn't one", async () => {
    const { core } = await setUp();
    await expect(core.formatCitations({ style: "apa", clusters: [{ works: [] }] })).rejects.toThrow(
      "one or more cited works",
    );
  });
});

describe("the style thread", () => {
  test("formats 300 references in a note style while the core's thread keeps running", async () => {
    const references = fixtureReferences();
    const items: CslItem[] = [];
    for (let copy = 0; copy < 10; copy++) {
      for (const reference of references)
        items.push({ ...reference, id: `${reference.id}-${copy}` });
    }
    const engine = createWorkerEngine(CITATION_STYLES);
    try {
      let ticks = 0;
      const timer = setInterval(() => {
        ticks++;
      }, 1);
      const { output } = await engine.format("chicago-notes-bibliography.csl", {
        items,
        citations: items.map((item, index) => ({ items: [{ id: item.id }], noteIndex: index + 1 })),
        englishTermsForOthers: false,
      });
      clearInterval(timer);
      expect(output.bibliography).toHaveLength(300);
      expect(output.citations).toHaveLength(300);
      // Formatting on this thread would have held every tick back until it was done.
      expect(ticks).toBeGreaterThan(5);
    } finally {
      engine.close();
    }
  });

  test("a style that can't be read fails that request, and the next one still works", async () => {
    const engine = createWorkerEngine(CITATION_STYLES);
    try {
      await expect(
        engine.format("no-such-style.csl", {
          items: [],
          citations: [],
          englishTermsForOthers: false,
        }),
      ).rejects.toThrow();
      const { output } = await engine.format("apa.csl", {
        items: [{ id: "a", type: "book", title: "A book" }],
        citations: [{ items: [{ id: "a" }] }],
        englishTermsForOthers: false,
      });
      expect(markup(output.citations[0] ?? [])).toBe("(<i>A Book</i>, n.d.)");
    } finally {
      engine.close();
    }
  });
});

describe("exporting a Mind in its citation style", () => {
  async function written(style: FormattedCitations["style"]) {
    const setup = await setUp();
    const { core, mind, client, prospect, qian } = setup;
    await core.setMindCitationStyle(mind.id, style);
    writeMind(client, [
      paragraph(
        text("Losses loom larger than gains "),
        citation(prospect, page(265)),
        text(" "),
        citation(qian, page(5), "not-found"),
        text(". Systems are open "),
        citation(qian, null),
        text("."),
      ),
    ]);
    await client.settled();
    return setup;
  }

  test("APA to Word: Citations in the text, clustered, unverified marked, and the References", async () => {
    const { core, mind } = await written("apa");
    const exported = await core.exportMind(mind.id, { format: "docx" });
    const files = unzip(exported.data);
    const paragraphs = paragraphsOf(part(files, "word/document.xml"));
    expect(paragraphs.map((each) => each.text)).toEqual([
      "Thesis",
      "Losses loom larger than gains (Kahneman & Tversky, 1979, p. 265; 钱学森 et al., 1990, p. 5) [unverified]. Systems are open (钱学森 et al., 1990).",
      "References",
      expect.stringMatching(/^Kahneman, D., & Tversky, A\. \(1979\)\. Prospect theory/),
      expect.stringMatching(/^钱学森, 于景元, & 戴汝为\. \(1990\)\./),
    ]);
    // No Citation becomes a footnote.
    expect(footnotesOf(part(files, "word/footnotes.xml"))).toEqual({});
    // The journal's name in italics, as APA has it; the Chinese journal's upright.
    const document = part(files, "word/document.xml");
    expect(document).toMatch(/<w:i\/><\/w:rPr><w:t xml:space="preserve">Econometrica<\/w:t>/);
    expect(document).not.toMatch(/<w:i\/><\/w:rPr><w:t xml:space="preserve">自然杂志/);
    expect(document).toContain('w:hanging="720"');
  });

  test("GB/T 7714 numeric to Word: raised numbers right after the word, and 参考文献", async () => {
    const { core, mind } = await written("gb-t-7714-2015-numeric");
    const exported = await core.exportMind(mind.id, { format: "docx" });
    const document = part(unzip(exported.data), "word/document.xml");
    const paragraphs = paragraphsOf(document);
    expect(paragraphs[1]?.text).toBe(
      "Losses loom larger than gains[1,2] [unverified]. Systems are open[2].",
    );
    expect(document).toContain('<w:vertAlign w:val="superscript"/>');
    expect(paragraphs[2]?.text).toBe("参考文献");
    expect(paragraphs[3]?.text).toMatch(/^\[1\]\tKAHNEMAN D, TVERSKY A\. Prospect theory/);
    expect(paragraphs[4]?.text).toMatch(/^\[2\]\t钱学森, 于景元, 戴汝为\. 一个科学新领域/);
  });

  test("Chicago notes to Word: each cluster a footnote in the style, and the Bibliography", async () => {
    const { core, mind } = await written("chicago-notes");
    const exported = await core.exportMind(mind.id, { format: "docx" });
    const files = unzip(exported.data);
    expect(paragraphsOf(part(files, "word/document.xml"))[1]?.text).toBe(
      "Losses loom larger than gains[1]. Systems are open[2].",
    );
    const notes = Object.values(footnotesOf(part(files, "word/footnotes.xml")));
    expect(notes).toHaveLength(2);
    expect(notes[0]).toMatch(
      /Daniel Kahneman and Amos Tversky, “Prospect Theory.*265.*\[unverified\]$/,
    );
    expect(notes[1]).toMatch(/^钱学森 et al\./);
    expect(textOf(part(files, "word/document.xml"))).toContain("Bibliography");
  });

  test("to Markdown: in-text, notes and the Bibliography, in the style", async () => {
    const { core, mind } = await written("apa");
    const apa = new TextDecoder().decode(
      (await core.exportMind(mind.id, { format: "markdown" })).data,
    );
    expect(apa).toContain(
      "Losses loom larger than gains (Kahneman & Tversky, 1979, p. 265; 钱学森 et al., 1990, p. 5) \\[unverified\\].",
    );
    expect(apa).toContain("## References");
    expect(apa).toMatch(/\*Econometrica\*, \*47\*\(2\), 263–291/);

    await core.setMindCitationStyle(mind.id, "gb-t-7714-2015-numeric");
    const gbt = new TextDecoder().decode(
      (await core.exportMind(mind.id, { format: "markdown" })).data,
    );
    expect(gbt).toContain("Losses loom larger than gains<sup>\\[1,2\\]</sup> \\[unverified\\].");
    expect(gbt).toContain("## 参考文献");

    await core.setMindCitationStyle(mind.id, "chicago-notes");
    const notes = new TextDecoder().decode(
      (await core.exportMind(mind.id, { format: "markdown" })).data,
    );
    expect(notes).toContain("Losses loom larger than gains[^1]. Systems are open[^2].");
    expect(notes).toMatch(
      /\[\^1\]: Daniel Kahneman and Amos Tversky, “Prospect Theory.* \[unverified\]\n/,
    );
  });

  test("switching the style changes the next export, and Numbered exports as before", async () => {
    const { core, mind } = await written("apa");
    await core.setMindCitationStyle(mind.id, "numbered");
    const files = unzip((await core.exportMind(mind.id, { format: "docx" })).data);
    expect(paragraphsOf(part(files, "word/document.xml")).map((each) => each.text)).toEqual([
      "Thesis",
      "Losses loom larger than gains[1][2]. Systems are open[3].",
    ]);
    expect(Object.values(footnotesOf(part(files, "word/footnotes.xml")))).toEqual([
      "prospect, p. 265",
      "qian, p. 5 [unverified]",
      "qian",
    ]);
  });

  test("a Mind with no Citations exports with no Bibliography", async () => {
    const { core, mind, client } = await setUp();
    writeMind(client, [paragraph(text("Nothing cited here."))]);
    await client.settled();
    const files = unzip((await core.exportMind(mind.id, { format: "docx" })).data);
    expect(paragraphsOf(part(files, "word/document.xml")).map((each) => each.text)).toEqual([
      "Thesis",
      "Nothing cited here.",
    ]);
  });
});
