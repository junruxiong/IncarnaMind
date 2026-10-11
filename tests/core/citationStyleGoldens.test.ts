/**
 * Golden output (#289, ADR-0016): the spike's 30 English and Chinese references and 34
 * citations of them (#288), in every style IncarnaMind ships. A Hayagriva update, a new style
 * file or a change to the binding shows here as a diff of what Users would read.
 *
 * The goldens record the engine's output as it is, its known differences from Zotero included
 * (the spike report lists them: title case of "ChatGPT", "昂温G", APA proceedings, Vancouver
 * editors, year-suffix order…). They are fixed upstream, never patched here, and a fix shows as
 * a diff to accept with `npx vitest run tests/core/citationStyleGoldens.test.ts -u`.
 */
import { describe, expect, test } from "vitest";
import type { CslItem } from "../../src/core/citationStyles/csl";
import { CITATION_STYLES } from "../../src/shared/citationStyles";
import {
  fixtureCitations,
  fixtureReferences,
  formatHere,
  goldenFile,
  goldenText,
  markup,
} from "../helpers/citationStyles";

const text = (runs: readonly { text: string }[] | null | undefined) =>
  (runs ?? []).map((run) => run.text).join("");

describe("each shipped style formats the 30 references as it did when its golden file was made", () => {
  for (const style of CITATION_STYLES) {
    if (style.file === null) continue;
    test(style.id, async () => {
      const output = formatHere(style.id, {
        items: fixtureReferences(),
        citations: fixtureCitations(),
      });
      expect(output.citations).toHaveLength(34);
      expect(output.bibliography).toHaveLength(30);
      await expect(goldenText(style.id, output)).toMatchFileSnapshot(goldenFile(style.id));
    });
  }
});

describe("what the binding adds to Hayagriva", () => {
  const references = fixtureReferences();
  const reference = (id: string) => references.find((each) => each.id === id) as CslItem;
  const one = (items: CslItem[]) => items.map((item) => ({ items: [{ id: item.id }] }));

  test("GB/T 7714 writes an English work with English terms and a Chinese one with Chinese terms", () => {
    const items = [
      reference("en-jumper-2021"),
      reference("en-kuhn-2012"),
      reference("zh-liu-2016"),
    ];
    for (const style of ["gb-t-7714-2015-numeric", "gb-t-7714-2025-numeric"] as const) {
      const output = formatHere(style, { items, citations: one(items) });
      const entry = (id: string) => text(output.bibliography.find((each) => each.id === id)?.runs);
      // 2015 writes family names in capitals and commas half-width; 2025 neither.
      expect(entry("en-jumper-2021")).toMatch(/pritzel a[,，]\s*et al\./i);
      expect(entry("en-kuhn-2012")).toContain("4th ed.");
      expect(entry("zh-liu-2016")).toMatch(/林衍凯[,，]\s?等/);
      expect(entry("zh-liu-2016")).not.toContain("et al.");
    }
    // In the text too: an English work's "et al.", among Chinese ones' "等".
    const authorDate = formatHere("gb-t-7714-2015-author-date", {
      items,
      citations: [{ items: items.map((item) => ({ id: item.id })) }],
    });
    const cited = text(authorDate.citations[0]);
    expect(cited).toContain("Jumper et al.");
    expect(cited).toContain("刘 等");
  });

  test("an English style keeps its own terms for a Chinese work", () => {
    const items = [reference("zh-liu-2016")];
    const output = formatHere("apa", { items, citations: one(items) });
    expect(text(output.bibliography[0]?.runs)).toContain("刘知远");
    expect(text(output.citations[0])).toBe("(刘 et al., 2016)");
  });

  test("Chinese text is never italic, while English titles keep their italics", () => {
    for (const style of ["mla", "chicago-notes", "ieee", "harvard", "apa"] as const) {
      const output = formatHere(style, {
        items: fixtureReferences(),
        citations: fixtureCitations(),
      });
      const runs = output.bibliography.flatMap((entry) => entry.runs);
      expect(runs.filter((run) => run.italic && /\p{Script=Han}/u.test(run.text))).toEqual([]);
      expect(runs.some((run) => run.italic && /[A-Za-z]/.test(run.text))).toBe(true);
    }
  });

  test("a cluster's runs say which of its works each belongs to, in the style's own order", () => {
    const items = [
      reference("zh-liu-2016"),
      reference("en-kahneman-1979"),
      reference("en-jumper-2021"),
    ];
    const output = formatHere("apa", {
      items,
      citations: [
        {
          items: [
            { id: "zh-liu-2016" },
            { id: "en-kahneman-1979", locator: "265", label: "page" },
            { id: "en-jumper-2021", locator: "slide 4", label: "custom" },
          ],
        },
      ],
    });
    const runs = output.citations[0] ?? [];
    expect(text(runs)).toBe(
      "(Jumper et al., 2021, slide 4; Kahneman & Tversky, 1979, p. 265; 刘 et al., 2016)",
    );
    // APA sorts the works by author: the third Citation's work comes first.
    expect(runs.filter((run) => run.item !== undefined).map((run) => run.item)).toEqual([2, 1, 0]);
    expect(runs.filter((run) => run.item === undefined).map((run) => run.text)).toEqual([
      "(",
      "; ",
      "; ",
      ")",
    ]);
  });

  test("numbers collapse into a range, the works inside it have no runs of their own", () => {
    const items = ["en-shannon-1940", "en-kahneman-1979", "en-jumper-2021", "en-vaswani-2017"].map(
      reference,
    );
    const output = formatHere("gb-t-7714-2015-numeric", {
      items,
      citations: [{ items: items.map((item) => ({ id: item.id })) }],
    });
    expect(markup(output.citations[0] ?? [])).toBe("<sup>[1–4]</sup>");
    expect(new Set((output.citations[0] ?? []).map((run) => run.item))).not.toContain(1);
  });

  // PENDING: a check against the text of GB/T 7714—2015 (§10.2.1). Hayagriva writes the space
  // before 等 that the style's own comment asks for; Zotero's citeproc-js removes it. These
  // golden files follow Hayagriva until a Chinese reader confirms which the standard wants.
  test("GB/T 7714 author-date puts a space between a name and 等 (pending a check against the standard)", () => {
    const items = [reference("zh-qian-1990")];
    const cited = (style: "gb-t-7714-2015-author-date" | "gb-t-7714-2025-author-date") =>
      text(formatHere(style, { items, citations: one(items) }).citations[0]);
    expect(cited("gb-t-7714-2015-author-date")).toBe("(钱 等, 1990)");
    // The 2025 style writes its brackets and comma full-width.
    expect(cited("gb-t-7714-2025-author-date")).toBe("（钱 等，1990）");
  });

  test("a work with only a title, and a Chinese name kept whole, are still formatted", () => {
    const items: CslItem[] = [
      { id: "bare", type: "document", title: "Lease agreement" },
      {
        id: "whole",
        type: "book",
        title: "工程控制论",
        author: [{ literal: "钱学森" }],
        issued: { "date-parts": [[1958]] },
      },
    ];
    const apa = formatHere("apa", { items, citations: one(items) });
    expect(apa.citations.map(text)).toEqual(["(Lease Agreement, n.d.)", "(钱学森, 1958)"]);
    const gbt = formatHere("gb-t-7714-2015-numeric", { items, citations: one(items) });
    expect(text(gbt.bibliography.find((entry) => entry.id === "whole")?.runs)).toContain("钱学森");
  });

  test("no citations: nothing to write and no Bibliography", () => {
    const output = formatHere("apa", { items: [], citations: [] });
    expect(output.citations).toEqual([]);
    expect(output.bibliography).toEqual([]);
  });
});
