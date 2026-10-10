/**
 * Why a small local model's Citations of the gating set's long PDFs aren't
 * found (#67), cause by cause, without a model: records written by hand as a
 * small model may write them, given to the core's own citation session over
 * the stored text of the evaluation's PDFs (cut short), and sorted as the
 * evaluation sorts them. qwen3.5:4b in structured output found 78% of its
 * English quotes and 64% of its Chinese ones on the gating set; these are the
 * ways the code and the data let a record fail. A quote of JP Morgan's
 * report, whose stored text lost its f-ligatures ("fnance"), is in
 * ./scoring.test.ts ("What became of a Citation").
 */
import { describe, expect, test } from "vitest";
import { type CitationOutcome, outcomeOf, quotePages } from "../../eval/lib/citations";
import type { CitationAttributes } from "../../src/core";
import { createCitationSession } from "../../src/core/answers/citations";
import type { CitationRecordInput } from "../../src/core/answers/engine";
import { documentInstructions } from "../../src/core/answers/prompt";

/** A Document's stored pages, cut short, and its Passages, by the pages they cover. */
interface Excerpt {
  name: string;
  pages: Record<number, string>;
  passages: [number, number][];
}

interface Outcome {
  check: string;
  checkReason: string | null;
  pages: [number | null, number | null];
  outcome: CitationOutcome;
  quoteOn: [number, number] | null;
}

/**
 * The citation session of one Answer over an excerpt: the Passages as the
 * model reads them, and what becomes of its records.
 */
function sessionOver({ name, pages, passages }: Excerpt) {
  const units = Object.entries(pages).map(([page, text]) => ({
    page: Number(page),
    kind: "page" as const,
    text,
  }));
  const inRange = (from: number, to: number) =>
    units.filter((unit) => unit.page >= from && unit.page <= to);
  // A Passage runs across pages as processing lays them out: a blank line at each break.
  const stored = passages.map(([from, to], position) => ({
    passageId: `${name}#${position}`,
    pageFrom: from,
    pageTo: to,
    position,
    text: inRange(from, to)
      .map((unit) => unit.text.trim())
      .join("\n\n"),
  }));
  const source = { documentId: name, documentName: name, contentHash: "v1" };
  const session = createCitationSession(
    {
      searchableCount: () => 1,
      search: async () =>
        stored.map((passage) => ({
          ...passage,
          ...source,
          seq: passage.position,
          documentKind: "pdf" as const,
          windowFrom: passage.position,
          windowTo: passage.position,
        })),
      citationSource: (passageId) => {
        const passage = stored.find((each) => each.passageId === passageId);
        return passage
          ? { ...passage, ...source, documentKind: "pdf", documentDeleted: false }
          : null;
      },
      pageTexts: (_documentId, _contentHash, from, to) =>
        from === null || to === null ? units : inRange(from, to),
    },
    { onRecord: () => undefined },
  );
  return {
    /** The Passages as the model reads them, each with its id: P1, P2… */
    shown: async () => (await session.tools.searchDocuments("a query")).text,
    /** What `cite` tells the model, and what became of each record: null when it was dropped. */
    cite(records: CitationRecordInput[]): { feedback: string; outcomes: (Outcome | null)[] } {
      const feedback = session.tools.cite(records);
      session.check();
      const outcomes = records.map((record): Outcome | null => {
        const node = session.finalNode(record.marker);
        if (!node) return null;
        const attrs = node.attrs as Partial<CitationAttributes>;
        const cited = {
          check: attrs.check ?? "checking",
          checkReason: attrs.checkReason ?? null,
          quote: attrs.quote ?? "",
          pageFrom: attrs.pageFrom ?? null,
          pageTo: attrs.pageTo ?? null,
        };
        return {
          check: cited.check,
          checkReason: cited.checkReason,
          pages: [cited.pageFrom, cited.pageTo],
          outcome: outcomeOf(cited, units),
          quoteOn: quotePages(cited.quote, units),
        };
      });
      return { feedback, outcomes };
    },
  };
}

/** Attention Is All You Need, pp. 7–8, as stored: en-02's answer is on p. 8. */
const ATTENTION: Excerpt = {
  name: "Attention Is All You Need",
  pages: {
    7: "5.2 Hardware and Schedule\nWe trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using",
    8: "Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This\nhurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.\n6 Results",
  },
  passages: [[7, 8]],
};

const PERPLEXITY =
  "This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.";

/** ABPI Code of Practice for the Pharmaceutical Industry 2021, pp. 33–34, as stored: en-17's answer runs across the break. */
const ABPI: Excerpt = {
  name: "ABPI Code of Practice for the Pharmaceutical Industry 2021",
  pages: {
    33: "Clause 17.3 (15.3) Items Delivered by Representatives\nReply paid cards which refer to representatives delivering\nitems to health professionals or other relevant decision\nmakers should explain that there is no obligation to grant the\nrepresentative an interview when the items are delivered.",
    34: "This is to avoid the impression that there is such an obligation,\nwhich would be contrary to Clause 17.3, which prohibits the\nuse of any inducement or subterfuge to gain an interview.",
  },
  // The search gives a window of overlapping Passages: one on p. 33 only, one across the break.
  passages: [
    [33, 33],
    [33, 34],
  ],
};

const OBLIGATION =
  "should explain that there is no obligation to grant the representative an interview when the items are delivered. This is to avoid the impression that there is such an obligation";

/** 维基百科-可持续发展目标, pp. 3–4, as stored, radical look-alikes and all: zh-02's answer is on p. 4. */
const GOALS: Excerpt = {
  name: "维基百科-可持续发展目标",
  pages: {
    3: "3、確保健康及促進各年齡層的福祉\n3.1、在西元 2030 年前，減少全球的死產率，讓每 100,000 個活產的死胎數少於 70 個。",
    4: "3.5、強化物質濫⽤的預防與治療，包括⿇醉藥品濫⽤以及酗酒。\n3.6、在西元 2020 年前，讓全球因為交通事故⽽傷亡的⼈數減少⼀半。",
  },
  passages: [[3, 4]],
};

/**
 * 维基百科-梯度下降法, pp. 1–4, as stored, cut short: its formulas are
 * images, so pages 2 and 3 hold 309 and 78 characters, and one Passage
 * covers all four (zh-08's answer is on p. 2, zh-14's on p. 3).
 */
const GRADIENT: Excerpt = {
  name: "维基百科-梯度下降法",
  pages: {
    1: "梯度下降法梯度下降法（英語：Gradient descent）是⼀种求解⽆约束最优化问题的⼀阶迭代最优化算法",
    2: "梯度下降法处理⼀些复杂的⾮线性函数会出现问题，例如 Rosenbrock 函數其最⼩值在 处，数值为 。但是此函数具有狭窄弯曲的⼭⾕，最⼩值就在这些⼭⾕之中，并且⾕底很平。优化过程是之字形的向极⼩值点靠近，速度⾮常缓慢。",
    3: "梯度下降法的缺點包括：[8]\n靠近局部極⼩值时速度减慢。\n直線搜索可能會產⽣⼀些問題。",
    4: "共轭梯度法随机梯度下降法最优化线搜索反向傳播算法量⼦退⽕",
  },
  passages: [[1, 4]],
};

describe("Why a small model's Citations of long PDFs aren't found", () => {
  test("a page in traditional characters, quoted in simplified ones, is 'not in the Document' (zh-02; ADR-0009)", async () => {
    const goals = sessionOver(GOALS);
    // The model reads the page's own characters, radical look-alikes folded: 讓, 為, 傷, 數, 減.
    expect(await goals.shown()).toContain(
      "3.6、在西元 2020 年前，讓全球因為交通事故而傷亡的人數減少一半。",
    );

    const { outcomes } = goals.cite([
      {
        marker: 1,
        passage: "P1",
        location: "p. 4",
        quote: "讓全球因為交通事故而傷亡的人數減少一半",
      },
      // The Question is in simplified Chinese, and so is the Answer: its quote follows.
      {
        marker: 2,
        passage: "P1",
        location: "p. 4",
        quote: "让全球因为交通事故而伤亡的人数减少一半",
      },
    ]);

    expect(outcomes).toMatchObject([
      { check: "found", quoteOn: [4, 4] },
      {
        check: "not-found",
        checkReason: "quote-not-on-pages",
        outcome: "not-in-document",
        quoteOn: null,
      },
    ]);
  });

  test("the Passage's first page, for a quote on its second, is a wrong page, and structured output has no second try (en-02)", async () => {
    const attention = sessionOver(ATTENTION);
    // The Passage covers pp. 7–8, and marks where p. 8 starts.
    expect(await attention.shown()).toContain('pages="7-8">');
    expect(await attention.shown()).toContain("\n\n[p. 8] Label Smoothing");

    const { feedback, outcomes } = attention.cite([
      { marker: 1, passage: "P1", location: "p. 7", quote: PERPLEXITY },
      { marker: 2, passage: "P1", location: "p. 8", quote: PERPLEXITY },
      { marker: 3, passage: "P1", pageFrom: 7, pageTo: 8, quote: PERPLEXITY },
    ]);

    expect(outcomes).toMatchObject([
      { check: "not-found", outcome: "wrong-page", pages: [7, 7], quoteOn: [8, 8] },
      { check: "found", pages: [8, 8] },
      { check: "found", pages: [7, 8] },
    ]);
    // In the Tool loop, `cite` tells the model, which may fix the record with another call.
    // In structured output the engine sends one request and doesn't read this (engine.ts, `structured`).
    expect(feedback).toMatch(/\[\^1\]: the quote isn't word for word on p\. 7 in P1/);
  });

  test("the structured-output example names no page; its placeholder, copied, cites the Passage's own pages (en-02)", async () => {
    const instructions = documentInstructions("structured-output", {
      total: 1,
      names: ["Attention Is All You Need"],
    });
    // It named "p. 3", which a record that copied it put outside its Passage.
    expect(instructions).toContain('"location": "…"');
    expect(instructions).not.toContain("p. 3");
    const attention = sessionOver(ATTENTION);
    await attention.shown();

    const { outcomes } = attention.cite([
      { marker: 1, passage: "P1", location: "…", quote: PERPLEXITY },
      { marker: 2, passage: "P1", location: "p. 3", quote: PERPLEXITY },
    ]);

    expect(outcomes).toMatchObject([
      { check: "found", pages: [7, 8] },
      { check: "not-found", checkReason: "pages-outside-passage", outcome: "page-range" },
    ]);
  });

  test("a quote across a page break needs both pages, in a Passage that has both (en-17)", async () => {
    const abpi = sessionOver(ABPI);
    const shown = await abpi.shown();
    expect(shown).toContain(
      '<passage id="P1" document="ABPI Code of Practice for the Pharmaceutical Industry 2021" pages="33">',
    );
    expect(shown).toContain("\n\n[p. 34] This is to avoid");

    const { outcomes } = abpi.cite([
      // The page where the quote starts.
      { marker: 1, passage: "P2", location: "p. 33", quote: OBLIGATION },
      // Both pages, of the Passage that holds them.
      { marker: 2, passage: "P2", location: "pp. 33-34", quote: OBLIGATION },
      // Both pages, but of the overlapping Passage that holds only the first.
      { marker: 3, passage: "P1", location: "pp. 33-34", quote: OBLIGATION },
    ]);

    expect(outcomes).toMatchObject([
      { check: "not-found", outcome: "wrong-page", quoteOn: [33, 34] },
      { check: "found", pages: [33, 34] },
      { check: "not-found", checkReason: "pages-outside-passage", outcome: "page-range" },
    ]);
  });

  test("a record that names no page, for a Passage over four pages, cites the quote's own page (zh-08, zh-14)", async () => {
    const gradient = sessionOver(GRADIENT);
    expect(await gradient.shown()).toContain('pages="1-4">');

    const { outcomes } = gradient.cite([
      { marker: 1, passage: "P1", quote: "优化过程是之字形的向极小值点靠近，速度非常缓慢" },
      { marker: 2, passage: "P1", quote: "直線搜索可能會產生一些問題" },
      // The Passage's own pages, copied from its tag: more than two.
      { marker: 3, passage: "P1", location: "pp. 1-4", quote: "直線搜索可能會產生一些問題" },
    ]);

    expect(outcomes).toMatchObject([
      { check: "found", pages: [2, 2] },
      { check: "found", pages: [3, 3] },
      { check: "not-found", checkReason: "too-many-pages", outcome: "page-range" },
    ]);
  });

  test("a record that names its Passage other than by its id is dropped, and its marker with it", async () => {
    const attention = sessionOver(ATTENTION);
    await attention.shown();

    const { feedback, outcomes } = attention.cite([
      { marker: 1, passage: "1", location: "p. 8", quote: PERPLEXITY },
      { marker: 2, passage: "Attention Is All You Need", location: "p. 8", quote: PERPLEXITY },
      { marker: 3, passage: "[P1]", location: "p. 8", quote: PERPLEXITY },
    ]);

    expect(outcomes.map((outcome) => outcome?.check ?? null)).toEqual([null, null, "found"]);
    expect(feedback).toContain('there is no Passage "1" in your search results');
  });
});
