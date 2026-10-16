import { describe, expect, test } from "vitest";
import { caseCitations, citedCases, namesDecision } from "../../src/shared/caseCitations";

describe("Court decisions a sentence cites", () => {
  test.each([
    [
      "See Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017).",
      "Hallmark Builders v. Rio Grande Supply",
      "512 S.W.3d 101",
    ],
    [
      "Brown v. Board of Education, 347 U.S. 483 (1954), ended it.",
      "Brown v. Board of Education",
      "347 U.S. 483",
    ],
    [
      "Miranda v. Arizona, 384 U.S. 436, 444 (1966) requires a warning.",
      "Miranda v. Arizona",
      "384 U.S. 436",
    ],
    [
      "The test comes from Gideon v. Wainwright, 83 S. Ct. 792.",
      "Gideon v. Wainwright",
      "83 S. Ct. 792",
    ],
    [
      "See also Smith & Sons Ltd. v. Jones Holdings, Inc., 944 F.3d 1021 (9th Cir. 2019).",
      "Smith & Sons Ltd. v. Jones Holdings, Inc.",
      "944 F.3d 1021",
    ],
    [
      "Cf. Acme Corp. v. Widget Co., 25 F. Supp. 2d 345 (S.D.N.Y. 1998).",
      "Acme Corp. v. Widget Co.",
      "25 F. Supp. 2d 345",
    ],
    ["Roe v Wade, 410 U.S. 113 is the leading case.", "Roe v Wade", "410 U.S. 113"],
    [
      "The court followed People v. Anderson, 6 Cal. 3d 628 (1972).",
      "People v. Anderson",
      "6 Cal. 3d 628",
    ],
    [
      "Palsgraf v. Long Island R.R. Co., 248 N.Y. 339 (1928) limits the duty.",
      "Palsgraf v. Long Island R.R. Co.",
      "248 N.Y. 339",
    ],
    [
      "It was held in Donoghue v Stevenson [1932] AC 562 that a duty is owed.",
      "Donoghue v Stevenson",
      "[1932] AC 562",
    ],
    ["Smith v Jones [2023] EWHC 123 (Ch) decides this.", "Smith v Jones", "[2023] EWHC 123 (Ch)"],
    ["The Supreme Court said so in R v Brown [2024] UKSC 5.", "R v Brown", "[2024] UKSC 5"],
    [
      "See Carlill v Carbolic Smoke Ball Co [1893] 1 QB 256.",
      "Carlill v Carbolic Smoke Ball Co",
      "[1893] 1 QB 256",
    ],
    ["Pepper v Hart [1993] AC 593 allows it.", "Pepper v Hart", "[1993] AC 593"],
    ["Applicants must now meet [2023] EWCA Civ 1234.", null, "[2023] EWCA Civ 1234"],
    ["See the ruling at [2019] 1 WLR 1234.", null, "[2019] 1 WLR 1234"],
    ["Mabo v Queensland (No 2) is reported at [1992] HCA 23.", null, "[1992] HCA 23"],
  ])("recognises %j", (sentence, name, cite) => {
    expect(caseCitations(sentence)).toEqual([{ name, cite }]);
  });

  test.each([
    [
      "Under Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017), the lien attaches.",
      "Hallmark Builders v. Rio Grande Supply",
    ],
    ["In Marbury v. Madison, 5 U.S. 137 (1803), the Court said so.", "Marbury v. Madison"],
    ["As held in Marbury v. Madison, 5 U.S. 137 (1803).", "Marbury v. Madison"],
    ["But see Roe v. Wade, 410 U.S. 113 (1973).", "Roe v. Wade"],
    ["Per Donoghue v Stevenson [1932] AC 562.", "Donoghue v Stevenson"],
    [
      "The rule in Palsgraf v. Long Island R.R. Co., 248 N.Y. 339 (1928).",
      "Palsgraf v. Long Island R.R. Co.",
    ],
  ])("a name starts at the party, not at the words before it: %j", (sentence, name) => {
    expect(caseCitations(sentence)).toEqual([expect.objectContaining({ name })]);
  });

  test.each([
    ["The duty arises under the Act.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["The lien attached in Texas.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["Revenue fell in Q3.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["The supplier was owned by IBM.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    ["Yes.", "See Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017)."],
    [
      "It was formed in Delaware.",
      "In Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017), it held.",
    ],
    ["The plan is Plan A.", "In Hallmark v. Rio Grande, 512 S.W.3d 101 (Tex. 2017), it held."],
  ])("a citation doesn't begin in the sentence before it: %j", (before, cited) => {
    const text = `${before} ${cited}`;
    const [found] = citedCases(text);
    expect(found?.name).toBe("Hallmark v. Rio Grande");
    expect(text.slice(0, found?.from).trim()).toBe(before);
  });

  test.each([
    ["Smith v. Jones, 101 S.Ct. 12 (1981).", "101 S.Ct. 12"],
    ["Smith v. Jones, 55 Cal.App.4th 301 (1997).", "55 Cal.App.4th 301"],
    ["Smith v. Jones, 12 F.Supp.2d 345 (S.D.N.Y. 1998).", "12 F.Supp.2d 345"],
    ["Smith v. Jones, 12 Pa. 345 (1850).", "12 Pa. 345"],
    ["Smith v. Jones, 77 N.J. 12 (1978).", "77 N.J. 12"],
    ["Smith v. Jones, 40 Ohio St. 3d 12 (1988).", "40 Ohio St. 3d 12"],
    ["Smith v. Jones, 150 Wn.2d 12 (2003).", "150 Wn.2d 12"],
    ["Smith Holdings, L.L.C. v. Jones Supply, L.L.C., 150 F.3d 12 (9th Cir. 2003).", "150 F.3d 12"],
  ])("recognises a reporter written compactly or a state's own: %j", (sentence, cite) => {
    expect(caseCitations(sentence)).toEqual([expect.objectContaining({ cite })]);
  });

  test("finds each of the decisions a sentence cites, in order", () => {
    expect(
      caseCitations(
        "Compare Roe v. Wade, 410 U.S. 113 (1973), with Donoghue v Stevenson [1932] AC 562.",
      ),
    ).toEqual([
      { name: "Roe v. Wade", cite: "410 U.S. 113" },
      { name: "Donoghue v Stevenson", cite: "[1932] AC 562" },
    ]);
  });

  test("reads a sentence that wraps across lines", () => {
    expect(caseCitations("See Hallmark Builders v.\nRio Grande Supply,\n512 S.W.3d 101")).toEqual([
      { name: "Hallmark Builders v. Rio Grande Supply", cite: "512 S.W.3d 101" },
    ]);
  });

  test.each([
    "The Packers v. Bears game drew 90,000 fans.",
    "Apple v. Samsung was the biggest patent fight of the decade.",
    "See Smith v. Jones for the details.",
    "Revenue was 512 million in 2017, up 4.1% on 101 the year before.",
    "Chapter 3 begins on page 101 of the 2017 report.",
    "Compare Option A v. Option B on cost: 512 units against 101.",
    "In Q3 [2023] sales were 512, versus 101 the year before.",
    "Version 2.1 v. 2.2 fixes 45 bugs on p. 12.",
    "The 12 U.S. citizens who applied in 2019 were all admitted.",
    "Under 42 U.S.C. § 1983 a person may sue.",
    "The UKSC decided 5 cases in 2024.",
    "On 12 March 2024 the High Court (Ch) sat for 3 days.",
    "Figures for [2023] GDP 4 percent are provisional.",
    "Our team v. their team: 512 S points, 101 W points.",
    "Smith, 512 Park Avenue 101, signed the lease.",
    "The cost rose to [2019] 1 million a year.",
    "他引用了第 512 条和 2017 年的判决。",
    "",
  ])("doesn't recognise %j", (sentence) => {
    expect(caseCitations(sentence)).toEqual([]);
  });
});

describe("Whether a text names a decision", () => {
  const hallmark = { name: "Hallmark Builders v. Rio Grande Supply", cite: "512 S.W.3d 101" };

  test.each([
    "Hallmark Builders, Inc., Petitioner, v. Rio Grande Supply Co., Respondent. No. 15-0123",
    "hallmark_builders_v_rio_grande_supply_2017.pdf",
    "Reported at 512 S.W.3d 101 (Tex. 2017)",
  ])("%j does", (text) => {
    expect(namesDecision(text, hallmark)).toBe(true);
  });

  test.each([
    "Hallmark Builders v. Rio Grande Supply, Opinion of the Court, Supreme Court of Texas",
    "hallmark_builders_v_rio_grande_supply_2017.pdf",
  ])(
    "a slip opinion with no reporter citation, %j, names a decision cited with words before it",
    (text) => {
      for (const sentence of [
        "Under Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017), it attaches.",
        "In Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017), it attached.",
      ]) {
        const [decision] = caseCitations(sentence);
        expect(decision && namesDecision(text, decision)).toBe(true);
      }
    },
  );

  test.each([
    "Rio Grande Supply Co. invoiced Hallmark Roofing.",
    "Lease.pdf",
    "See 512 S.W.3d 1010 for another case.",
  ])("%j doesn't", (text) => {
    expect(namesDecision(text, hallmark)).toBe(false);
  });

  test("a decision with no name is named by its citation alone", () => {
    const neutral = { name: null, cite: "[2024] UKSC 5" };
    expect(namesDecision("Judgment given on 5 March 2024 [2024] UKSC 5", neutral)).toBe(true);
    expect(namesDecision("The UKSC sat in 2024", neutral)).toBe(false);
  });
});
