import { type Editor, getSchema, type JSONContent } from "@tiptap/core";
import { EditorState, type Transaction } from "@tiptap/pm/state";
import { describe, expect, test, vi } from "vitest";
import { CITATION_NODE, type CitationAttributes } from "../../src/core/api";
import { citationNumbersPlugin } from "../../src/renderer/src/editor/citationNumbers";
import type { ShownClaim } from "../../src/renderer/src/editor/claimChecks";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

vi.mock("../../src/renderer/src/core", () => ({ core: {} }));
vi.mock("../../src/renderer/src/editWithAi", () => ({ resolveProposalsIn: vi.fn() }));
vi.mock("../../src/renderer/src/modelInUse", () => ({ modelInUseOf: vi.fn() }));

const { addAsCitation } = await import("../../src/renderer/src/checks");

const schema = getSchema(noteExtensions());

const SENTENCE = "Staff may work from home two days a week.";
const LINE = "The Employee may work from home two days a week, agreed in advance.";

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

function editorHolding(cited: Partial<CitationAttributes> | null) {
  let state = EditorState.create({
    doc: schema.nodeFromJSON({
      type: "doc",
      content: [
        {
          type: "paragraph",
          content: [
            text(SENTENCE),
            ...(cited ? [{ type: CITATION_NODE, attrs: attributes(cited) }] : []),
          ],
        },
      ],
    }),
    plugins: [citationNumbersPlugin()],
  });
  const dispatched: Transaction[] = [];
  const editor = {
    get state() {
      return state;
    },
    view: {
      dispatch(tr: Transaction) {
        dispatched.push(tr);
        state = state.apply(tr);
      },
    },
  } as unknown as Editor;
  return { editor, dispatched };
}

const countCitations = (state: EditorState) => {
  let count = 0;
  state.doc.descendants((node) => {
    if (node.type.name === CITATION_NODE) count++;
  });
  return count;
};

const claimFound = (evidence: Partial<CitationAttributes>): ShownClaim =>
  ({
    check: { id: "check-1", state: "supported", evidence: attributes(evidence), calculation: null },
    from: 1,
    to: 1 + SENTENCE.length,
    sentence: { from: 1, to: 1 + SENTENCE.length + 1, text: SENTENCE },
    shown: { state: "supported" },
  }) as unknown as ShownClaim;

describe("Add as Citation", () => {
  test("adds nothing when the sentence already cites those words, whatever span the check quoted", () => {
    const { editor, dispatched } = editorHolding({});
    expect(addAsCitation(editor, claimFound({ quote: LINE }))).toBe(false);
    expect(dispatched).toHaveLength(0);
    expect(countCitations(editor.state)).toBe(1);
  });

  test("adds the Passage as a second Citation when it is another clause of the Document", () => {
    const { editor, dispatched } = editorHolding({});
    const other = claimFound({ quote: "Probation lasts three months.", pageFrom: 3, pageTo: 3 });
    expect(addAsCitation(editor, other)).toBe(true);
    expect(dispatched).toHaveLength(1);
    expect(countCitations(editor.state)).toBe(2);
  });

  test("adds the first Citation to a sentence with none", () => {
    const { editor } = editorHolding(null);
    expect(addAsCitation(editor, claimFound({ quote: LINE }))).toBe(true);
    expect(countCitations(editor.state)).toBe(1);
  });
});
