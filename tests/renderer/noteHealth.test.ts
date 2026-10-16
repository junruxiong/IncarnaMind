import { getSchema, type JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test } from "vitest";
import {
  CITATION_NODE,
  type CitationAttributes,
  CLAIM_CHECK_MARK,
  type ClaimCheckAttributes,
} from "../../src/core/api";
import { claimSpans, type ShownClaim } from "../../src/renderer/src/editor/claimChecks";
import {
  HEALTH_PARTS,
  type HealthPart,
  noteHealth,
} from "../../src/renderer/src/editor/healthCounts";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { claimCheckShown } from "../../src/shared/claimChecks";
import { sentenceKey } from "../../src/shared/mindCheck";

const schema = getSchema(noteExtensions());

const text = (value: string, ...marks: JSONContent["marks"][]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks: marks.flatMap((each) => each ?? []) }),
});
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const heading = (value: string): JSONContent => ({
  type: "heading",
  attrs: { level: 2 },
  content: [text(value)],
});
const citation = (check: CitationAttributes["check"]): JSONContent => ({
  type: CITATION_NODE,
  attrs: { documentId: "results", documentName: "Q3 results", quote: "38.6%", check },
});

/** A sentence's check, as its mark holds it. */
function checked(
  sentence: string,
  state: ClaimCheckAttributes["state"],
  reason: ClaimCheckAttributes["reason"] = null,
): JSONContent {
  const check: ClaimCheckAttributes = {
    id: `check-${sentence.length}-${state}`,
    request: "r1",
    action: "find-support",
    state,
    reason,
    detail: null,
    providerId: null,
    modelId: null,
    text: sentence,
    evidence: null,
    says: null,
    calculation: null,
    citedCaseUnchecked: false,
    citedCaseFound: false,
    scope: { folderId: null, widened: false },
    versions: [],
    excluded: [],
    checkedAt: "2026-10-12T09:00:00.000Z",
  };
  return text(sentence, [{ type: CLAIM_CHECK_MARK, attrs: check }]);
}

const docOf = (...blocks: JSONContent[]) => schema.nodeFromJSON({ type: "doc", content: blocks });

/** The Mind's health as its header counts it, part by part. */
function healthOf(doc: ProseMirrorNode): Record<HealthPart, number> {
  const claims: ShownClaim[] = claimSpans(doc).map((span) => ({
    ...span,
    shown: claimCheckShown(span.check, {
      text: span.sentence.text,
      versions: null,
      folderId: null,
    }),
  }));
  const health = noteHealth(doc, claims, (attributes) => attributes.check);
  return Object.fromEntries(HEALTH_PARTS.map((part) => [part, health[part].length])) as Record<
    HealthPart,
    number
  >;
}

/** The sentences "Check all" would check: claims with no check, but those the last whole-Mind check found make none. */
function uncheckedIn(doc: ProseMirrorNode, skipped: readonly string[] = []): number {
  const claims: ShownClaim[] = claimSpans(doc).map((span) => ({
    ...span,
    shown: claimCheckShown(span.check, {
      text: span.sentence.text,
      versions: null,
      folderId: null,
    }),
  }));
  return noteHealth(doc, claims, (attributes) => attributes.check, new Set(skipped)).unchecked
    .length;
}

const NOTHING = {
  found: 0,
  notFound: 0,
  unsupported: 0,
  contradicted: 0,
  noCitation: 0,
  recheck: 0,
  notChecked: 0,
};

describe("A Mind's health", () => {
  test("counts its Citations, and its sentences checked or with no Citation, as the design's paper", () => {
    const doc = docOf(
      heading("Recommendation"),
      paragraph(
        text("Hold the facility at its current limit and ask Northwind for a margin report."),
      ),
      heading("Why"),
      paragraph(checked("Northwind's gross margin fell from 41.8% to 38.6% in Q3.", "supported")),
      paragraph(
        text("Net leverage was 3.1× at the end of the quarter, "),
        citation("found"),
        text(" inside the 3.5× limit in clause 7.1."),
      ),
      paragraph(
        checked("Online sales grew 12% in the quarter.", "unsupported"),
        text(" "),
        checked("The facilities end in 2027, so refinancing is the first risk.", "contradicted"),
      ),
      paragraph(text("Calder Grocers needed a covenant waiver in September.")),
    );

    expect(healthOf(doc)).toEqual({
      ...NOTHING,
      found: 1,
      unsupported: 1,
      contradicted: 1,
      noCitation: 3,
    });
  });

  test("counts Citations not found, and checks that need a re-check or weren't made", () => {
    const doc = docOf(
      paragraph(text("The quote isn't on the page it cites "), citation("not-found"), text(".")),
      paragraph(
        checked("The rate limit stopped this sentence's check.", "not-checked", "rate-limit"),
      ),
      paragraph(
        // Edited since it was checked: its mark holds the text it was checked as.
        text("Edited: "),
        {
          ...checked("Online sales grew 12% in the quarter.", "supported"),
          text: "Online sales grew 15% in the quarter.",
        },
      ),
    );

    expect(healthOf(doc)).toEqual({ ...NOTHING, notFound: 1, notChecked: 1, recheck: 1 });
  });

  test("leaves out changes still waiting, Questions, headings and short lines", () => {
    const change = { id: "c1", request: "r1", target: "x" };
    const doc = docOf(
      paragraph(text("The tides are strong at full moon."), {
        ...text(" A sentence the AI would add, with its Citation"),
        marks: [{ type: "proposalInsert", attrs: change }],
      }),
      {
        type: "paragraph",
        attrs: { proposal: { ...change, id: "c2", kind: "insert" } },
        content: [text("A whole paragraph the AI would add, still waiting.")],
      },
      { type: "question", content: [text("What makes the tides strong?")] },
      heading("A heading that claims a lot about tides"),
      paragraph(text("Thanks!")),
    );

    expect(healthOf(doc)).toEqual({ ...NOTHING, noCitation: 1 });
  });

  test("follows the User's edits: a Citation added to a sentence counts it found", () => {
    const before = docOf(paragraph(text("Spring tides come at full moon.")));
    const after = docOf(
      paragraph(text("Spring tides come at full moon "), citation("found"), text(".")),
    );

    expect(healthOf(before)).toEqual({ ...NOTHING, noCitation: 1 });
    expect(healthOf(after)).toEqual({ ...NOTHING, found: 1 });
  });

  test("knows which claims were never checked, for Check all: not those checked, short ones, or those the last whole-Mind check found make no claim (#737)", () => {
    const opinion = "We think the tides are lovely in spring.";
    const doc = docOf(
      paragraph(checked("Spring tides come at full moon.", "supported")),
      paragraph(text("Neap tides come at the quarter moons. "), text(opinion)),
      paragraph(text("Thanks!")),
      heading("A heading that claims a lot about tides"),
    );

    expect(uncheckedIn(doc)).toBe(2);
    expect(uncheckedIn(doc, [sentenceKey(opinion)])).toBe(1);
  });
});
