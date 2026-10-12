/**
 * How the checks evaluation scores a run against its labels (#86), without a
 * model: Find support's "supported" precision and recall, verify's claims
 * matched to the labels they quote, "contradicted" precision and recall with
 * a claim that wasn't extracted counted as a miss, and the bars.
 */
import { describe, expect, test } from "vitest";
import {
  type CheckOutcome,
  type ExtractedClaim,
  gates,
  matchClaims,
  scoreFindSupport,
  scoreVerify,
  similarity,
} from "../../eval/checks/lib/scoring";
import type {
  CheckLabel,
  CheckLanguage,
  FindSupportItem,
  LabelledQuote,
  VerifyItem,
} from "../../eval/checks/lib/set";

const QUOTE: LabelledQuote = {
  document: "Tides.pdf",
  location: "p. 2",
  units: [2, 2],
  quote: "Spring tides happen at new moon",
};

function sentence(id: string, label: CheckLabel, language: CheckLanguage = "en"): FindSupportItem {
  return {
    id,
    set: "find-support",
    language,
    document: language === "en" ? "Tides.pdf" : "潮汐.docx",
    sentence: `Sentence ${id}.`,
    label,
    evidence: label === "unsupported" ? null : QUOTE,
    contradiction: null,
    note: null,
    confirmed: false,
  };
}

function claim(
  id: string,
  text: string,
  label: CheckLabel,
  language: CheckLanguage = "en",
): VerifyItem {
  const document = language === "en" ? "Report.pdf" : "报告.pdf";
  const at = { document, location: "p. 1", units: [1, 1] as [number, number] };
  return {
    id,
    set: "verify",
    language,
    document,
    location: at.location,
    units: at.units,
    claim: text,
    label,
    evidence: label === "unsupported" ? null : QUOTE,
    contradiction: label === "contradicted" ? { ...at, quote: text } : null,
    note: null,
    confirmed: false,
  };
}

const said = (verdict: CheckOutcome["verdict"], units: [number, number] = [2, 2]): CheckOutcome =>
  verdict === "supported" || verdict === "contradicted"
    ? { verdict, evidence: { document: "Tides.pdf", units, quote: "…" } }
    : {
        verdict,
        evidence: null,
        ...(verdict === "not-checked" && { reason: "The model failed: 429" }),
      };

const extracted = (text: string, outcome: CheckOutcome, source = text): ExtractedClaim => ({
  text,
  source,
  outcome,
});

describe("Find support", () => {
  const items = [
    sentence("fs-1", "supported"),
    sentence("fs-2", "supported"),
    sentence("fs-3", "supported"),
    sentence("fs-4", "unsupported"),
    sentence("fs-5", "contradicted"),
    sentence("fs-6", "supported", "zh"),
    sentence("fs-7", "unsupported", "zh"),
  ];

  test("scores 'supported' over every labelled sentence, by language and format, and lists the misses", () => {
    const score = scoreFindSupport(
      items,
      [
        { id: "fs-1", outcome: said("supported") },
        // Found supported, but on another page than the labelled Passage.
        { id: "fs-2", outcome: said("supported", [5, 5]) },
        // The model failed: never shown as supported.
        { id: "fs-3", outcome: said("not-checked") },
        { id: "fs-4", outcome: said("supported") },
        { id: "fs-5", outcome: said("unsupported") },
        { id: "fs-6", outcome: said("supported") },
        // fs-7 has no result: it counts as not checked.
      ],
      (document) => (document.endsWith(".pdf") ? "pdf" : "docx"),
    );

    expect(score.supported).toEqual({
      truePositives: 3,
      falsePositives: 1,
      falseNegatives: 1,
      precision: 0.75,
      recall: 0.75,
    });
    expect(score.byLanguage.en).toMatchObject({ precision: 2 / 3, recall: 2 / 3 });
    expect(score.byLanguage.zh).toMatchObject({ precision: 1, recall: 1 });
    expect(Object.keys(score.byFormat).sort()).toEqual(["docx", "pdf"]);
    expect(score.labelledEvidence).toEqual({ hits: 2, total: 3, share: 2 / 3 });
    expect(score.notChecked).toBe(2);
    expect(score.misses.map((miss) => [miss.id, miss.label, miss.verdict])).toEqual([
      ["fs-3", "supported", "not-checked"],
      ["fs-4", "unsupported", "supported"],
      ["fs-5", "contradicted", "unsupported"],
      ["fs-7", "unsupported", "not-checked"],
    ]);
    expect(score.misses[0]?.reason).toBe("The model failed: 429");
  });

  test("with nothing found supported, precision is unknown rather than zero", () => {
    const score = scoreFindSupport([sentence("fs-1", "supported")], [], () => "pdf");
    expect(score.supported).toMatchObject({ precision: null, recall: 0 });
  });
});

describe("Verify", () => {
  const items = [
    claim("vf-01", "Roof works start on 4 January 2027.", "supported"),
    claim("vf-02", "The roof package is due in April 2027.", "contradicted"),
    claim("vf-03", "The total budget is £3.2 million.", "contradicted"),
    claim("vf-04", "Forty-two operatives worked on site.", "unsupported"),
    claim("vf-05", "第三季度准点率提升至95.3%。", "contradicted", "zh"),
    claim("vf-06", "集团今年共招聘新驾驶员240名。", "unsupported", "zh"),
  ];

  test("an extracted claim matches the labelled claim it quotes, in either language and script", () => {
    const matches = matchClaims(items, [
      extracted("Under the contract, the roof package is due in April 2027", said("contradicted")),
      extracted(
        "The contractor says works began in January.",
        said("supported"),
        "Roof works start on 4 January 2027",
      ),
      extracted("第三季度準點率提升至95.3%", said("contradicted")),
      extracted("The library will reopen in 2028.", said("unsupported")),
    ]);
    expect(matches.map((item) => item?.id ?? null)).toEqual(["vf-02", "vf-01", "vf-05", null]);
    expect(
      similarity("Roof works start on 4 January 2027.", "roof works start on 4 january 2027"),
    ).toBe(1);
    expect(similarity("The total budget is £3.2 million.", "集团今年共招聘新驾驶员240名。")).toBe(
      0,
    );
  });

  test("'contradicted' counts a claim that wasn't extracted as a miss, and an unlabelled one as a false alarm", () => {
    const score = scoreVerify(items, [
      {
        document: "Report.pdf",
        failure: null,
        claims: [
          extracted("Roof works start on 4 January 2027.", said("supported")),
          extracted("The roof package is due in April 2027.", said("contradicted")),
          // vf-03 isn't extracted; vf-04 is checked wrongly.
          extracted("Forty-two operatives worked on site.", said("contradicted")),
          // Matches no labelled claim, and is called contradicted.
          extracted("The library has a café on the ground floor.", said("contradicted")),
        ],
      },
      {
        document: "报告.pdf",
        failure: "The model's reply held no list of claims.",
        claims: [],
      },
    ]);

    expect(score.extracted).toEqual({ hits: 3, total: 6, share: 0.5 });
    expect(score.extractedByLanguage.zh).toEqual({ hits: 0, total: 2, share: 0 });
    expect(score.contradicted).toEqual({
      truePositives: 1,
      falsePositives: 2,
      falseNegatives: 2,
      precision: 1 / 3,
      recall: 1 / 3,
    });
    expect(score.contradictedByLanguage.zh).toMatchObject({ precision: null, recall: 0 });
    expect(score.unlabelled).toBe(1);
    expect(score.misses.map((miss) => [miss.id, miss.verdict])).toEqual([
      ["vf-03", "not-extracted"],
      ["vf-04", "contradicted"],
      ["vf-05", "not-extracted"],
      ["vf-06", "not-extracted"],
      [null, "contradicted"],
    ]);
    expect(score.misses[2]?.reason).toBe("The model's reply held no list of claims.");
  });
});

describe("The bars", () => {
  test("each is met only at or above its value, and an unknown figure misses it", () => {
    const findSupport = scoreFindSupport(
      [sentence("fs-1", "supported"), sentence("fs-2", "supported")],
      [
        { id: "fs-1", outcome: said("supported") },
        { id: "fs-2", outcome: said("supported") },
      ],
      () => "pdf",
    );
    const verify = scoreVerify([claim("vf-01", "The budget is £3.2 million.", "unsupported")], []);
    expect(gates(findSupport, verify).map((gate) => [gate.name, gate.value, gate.met])).toEqual([
      ["Verify: claims extracted", 0, false],
      ['Verify: "contradicted" precision', null, false],
      ['Verify: "contradicted" recall', null, false],
      ['Find support: "supported" precision', 1, true],
      ['Find support: "supported" recall', 1, true],
    ]);
  });
});
