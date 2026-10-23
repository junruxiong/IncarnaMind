import { describe, expect, test } from "vitest";
import { type CitedPlace, overlapsCitation } from "../../src/shared/citationOverlap";

const place = (quote: string | null, over: Partial<CitedPlace> = {}): CitedPlace => ({
  documentId: "lease",
  pageFrom: 4,
  pageTo: 4,
  quote,
  ...over,
});

const CLAUSE =
  "The Tenant may work from the premises two days a week, agreed in advance with the Landlord.";

describe("a Citation and the Passage a check found overlap", () => {
  test("when they quote the same words", () => {
    expect(overlapsCitation(place(CLAUSE), place(CLAUSE))).toBe(true);
  });

  test("when the Passage found is longer than the Citation's quote, or shorter", () => {
    const slice = "may work from the premises two days a week";
    expect(overlapsCitation(place(slice), place(CLAUSE))).toBe(true);
    expect(overlapsCitation(place(CLAUSE), place(slice))).toBe(true);
  });

  test("whatever the case and punctuation, and with an ellipsis for words left out", () => {
    expect(
      overlapsCitation(
        place("THE TENANT MAY WORK FROM THE PREMISES — two days a week; agreed in advance"),
        place(CLAUSE),
      ),
    ).toBe(true);
    expect(
      overlapsCitation(
        place("The Tenant may work from the premises ... agreed in advance"),
        place(CLAUSE),
      ),
    ).toBe(true);
  });

  test("when half the words of the shorter quote are in the other, in order", () => {
    expect(
      overlapsCitation(
        place("Tenant may work two days"),
        place("The Tenant may work from home on weekends and Fridays"),
      ),
    ).toBe(true);
    expect(overlapsCitation(place("Tenant works nights and weekends"), place(CLAUSE))).toBe(false);
  });

  test("not when the quotes are different clauses of the Document", () => {
    expect(
      overlapsCitation(
        place("The Tenant shall pay the rent monthly in advance."),
        place("The Tenant shall keep the premises clean and in good repair."),
      ),
    ).toBe(false);
    expect(overlapsCitation(place("The notice period is three months."), place(CLAUSE))).toBe(
      false,
    );
  });

  test("not when they are in different Documents, or the Citation has none", () => {
    expect(overlapsCitation(place(CLAUSE, { documentId: "other" }), place(CLAUSE))).toBe(false);
    expect(overlapsCitation(place(CLAUSE, { documentId: null }), place(CLAUSE))).toBe(false);
  });

  test("by their pages when a quote has no words to compare", () => {
    expect(overlapsCitation(place(null), place(CLAUSE))).toBe(true);
    expect(overlapsCitation(place(null, { pageFrom: 3, pageTo: 5 }), place(CLAUSE))).toBe(true);
    expect(overlapsCitation(place(null, { pageFrom: 9, pageTo: 9 }), place(CLAUSE))).toBe(false);
    expect(overlapsCitation(place(null, { pageFrom: null, pageTo: null }), place(CLAUSE))).toBe(
      false,
    );
  });

  test("in Chinese, whatever the punctuation, width and script", () => {
    const clause = "员工每周最多可远程办公两天，须提前与经理商定。";
    expect(overlapsCitation(place(clause), place(clause))).toBe(true);
    expect(
      overlapsCitation(place("员工每周最多可远程办公两天,须提前与经理商定."), place(clause)),
    ).toBe(true);
    expect(
      overlapsCitation(place("员工每周最多可远程办公两天、须提前与经理商定"), place(clause)),
    ).toBe(true);
    expect(
      overlapsCitation(place("員工每週最多可遠程辦公兩天，須提前與經理商定。"), place(clause)),
    ).toBe(true);
    expect(overlapsCitation(place("每周最多可远程办公两天"), place(clause))).toBe(true);
    expect(overlapsCitation(place(clause), place("每周最多可远程办公两天"))).toBe(true);
    expect(overlapsCitation(place(`第 4 条：${clause}`), place(clause))).toBe(true);
  });

  test("in Chinese, not when the clauses differ or the Documents do", () => {
    const clause = "员工每周最多可远程办公两天，须提前与经理商定。";
    expect(
      overlapsCitation(place("试用期为三个月，期间任何一方可随时解除合同。"), place(clause)),
    ).toBe(false);
    expect(overlapsCitation(place(clause, { documentId: "other" }), place(clause))).toBe(false);
  });
});
