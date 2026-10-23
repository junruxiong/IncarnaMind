import { getSchema, type JSONContent } from "@tiptap/core";
import { EditorState } from "@tiptap/pm/state";
import { describe, expect, test } from "vitest";
import {
  CITATION_NODE,
  type CitationAttributes,
  type ClaimCheckAttributes,
} from "../../src/core/api";
import { alreadyCitedBy } from "../../src/renderer/src/editor/alreadyCited";
import { citationNumbersPlugin } from "../../src/renderer/src/editor/citationNumbers";
import type { ShownClaim } from "../../src/renderer/src/editor/claimChecks";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const schema = getSchema(noteExtensions());

const FIRST = "Staff may work from home two days a week.";
const SECOND = "Probation lasts three months.";
const LINE = "The Employee may work from home two days a week, agreed in advance with the manager.";

const attributes = (over: Partial<CitationAttributes>): CitationAttributes => ({
  passageId: null,
  documentId: "contract",
  documentName: "Contract",
  contentHash: "v1",
  pageFrom: 2,
  pageTo: 2,
  location: null,
  quote: "may work from home two days a week",
  check: "found",
  checkReason: null,
  ...over,
});

const text = (value: string): JSONContent => ({ type: "text", text: value });

const state = (
  cited: Partial<CitationAttributes> | null,
  after: Partial<CitationAttributes> | null,
) =>
  EditorState.create({
    doc: schema.nodeFromJSON({
      type: "doc",
      content: [
        {
          type: "paragraph",
          content: [
            text(FIRST),
            ...(cited ? [{ type: CITATION_NODE, attrs: attributes(cited) }] : []),
            text(` ${SECOND}`),
            ...(after ? [{ type: CITATION_NODE, attrs: attributes(after) }] : []),
          ],
        },
      ],
    }),
    plugins: [citationNumbersPlugin()],
  });

function claimOf(
  sentence: { from: number; to: number },
  evidence: Partial<CitationAttributes> | null,
  over: Partial<ClaimCheckAttributes> = {},
): ShownClaim {
  const check = {
    id: "check-1",
    state: "supported",
    evidence: evidence ? attributes(evidence) : null,
    calculation: null,
    ...over,
  } as ClaimCheckAttributes;
  return {
    check,
    from: sentence.from,
    to: sentence.to,
    sentence: { ...sentence, text: "" },
    shown: { state: "supported" },
  } as ShownClaim;
}

// The paragraph's content starts at 1; a Citation is one position.
const firstSentence = (cited: boolean) => ({ from: 1, to: 1 + FIRST.length + (cited ? 1 : 0) });

describe("the Citation of a sentence that already cites what its check found", () => {
  test("is found whatever span the check quoted, the Citation numbered as the Mind numbers it", () => {
    const held = state({}, null);
    for (const quote of [LINE, "may work from home two days a week", "work from home two days"]) {
      const claim = claimOf(firstSentence(true), { quote });
      expect(alreadyCitedBy(held, claim)?.number).toBe(1);
    }
  });

  test("is none when the check found the same words on another page of the Document", () => {
    const held = state({}, null);
    const claim = claimOf(firstSentence(true), { quote: LINE, pageFrom: 9, pageTo: 9 });
    expect(alreadyCitedBy(held, claim)).toBeNull();
  });

  test("is none when the check found another clause of the Document, or another Document", () => {
    const held = state({}, null);
    const sentence = firstSentence(true);
    expect(
      alreadyCitedBy(
        held,
        claimOf(sentence, { quote: "Probation lasts three months.", pageFrom: 3 }),
      ),
    ).toBeNull();
    expect(
      alreadyCitedBy(held, claimOf(sentence, { documentId: "handbook", quote: LINE })),
    ).toBeNull();
  });

  test("is none for a sentence with no Citation of its own, though the next sentence cites the Passage", () => {
    const held = state(null, {});
    expect(alreadyCitedBy(held, claimOf(firstSentence(false), { quote: LINE }))).toBeNull();
  });

  test("is the one of several Citations that overlaps", () => {
    const held = state({ documentId: "handbook" }, {});
    const sentence = { from: 1, to: 1 + FIRST.length + 1 + 1 + SECOND.length + 1 };
    expect(alreadyCitedBy(held, claimOf(sentence, { quote: LINE }))?.number).toBe(2);
    expect(
      alreadyCitedBy(held, claimOf(sentence, { documentId: "handbook", quote: LINE }))?.number,
    ).toBe(1);
  });

  test("is none for a figure worked out or a check that found nothing", () => {
    const held = state({}, null);
    expect(alreadyCitedBy(held, claimOf(firstSentence(true), null))).toBeNull();
    const figure = claimOf(
      firstSentence(true),
      { quote: LINE },
      {
        calculation: {} as ClaimCheckAttributes["calculation"],
      },
    );
    expect(alreadyCitedBy(held, figure)).toBeNull();
  });
});
