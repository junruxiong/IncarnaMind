/**
 * The assistant's changes the User asked for, applied once they have all
 * arrived (#880, D224): set aside, then written again as the editor's own
 * change, so one Undo takes them back and Redo does them again, whenever
 * they arrived. Changes the User didn't ask for keep waiting.
 */
import { Editor, type JSONContent } from "@tiptap/core";
import { prosemirrorToYXmlFragment } from "@tiptap/y-tiptap";
import { describe, expect, onTestFinished, test } from "vitest";
import * as Y from "yjs";
import { MIND_CONTENT_FIELD, PROPOSAL_ATTRIBUTE } from "../../src/core";
import { writeAccepted } from "../../src/renderer/src/editor/arrivedChanges";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { pendingProposals } from "../../src/shared/proposals";

const text = (value: string, marks?: JSONContent["marks"]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks && { marks }),
});
const ref = (id: string, request: string) => ({ id, request });

/** A Mind's document in Yjs, as the core writes it. */
function mindWith(content: JSONContent[]): { doc: Y.Doc; blocks: Y.XmlFragment } {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content },
  });
  onTestFinished(() => editor.destroy());
  const doc = new Y.Doc();
  const blocks = prosemirrorToYXmlFragment(
    editor.state.doc,
    doc.getXmlFragment(MIND_CONTENT_FIELD),
  );
  return { doc, blocks };
}

const textOf = (blocks: Y.XmlFragment) =>
  blocks
    .toArray()
    .map((block) => (block instanceof Y.XmlElement ? block.toArray().join("") : ""))
    .map((xml) => xml.replace(/<[^>]+>/g, ""));

class Asked {}
class Aside {}

describe("Changes the User asked the assistant for", () => {
  // protects=one Undo takes back exactly what was asked for and Redo does it again, the rest waiting; fails_when=Undo leaves them waiting, takes back too little, or touches the change not asked for; no_claim=how the editor shows them.
  test("are applied as one tracked change: Undo restores the text, Redo applies them again, and a change not asked for still waits", () => {
    const { doc, blocks } = mindWith([
      {
        type: "paragraph",
        attrs: { id: "p1" },
        content: [
          text("The tides are "),
          text("very ", [{ type: "proposalDelete", attrs: ref("a", "asked") }]),
          text("strong.", []),
          text(" Truly.", [{ type: "proposalInsert", attrs: ref("b", "asked") }]),
        ],
      },
      {
        type: "paragraph",
        attrs: { id: "p2", [PROPOSAL_ATTRIBUTE]: { kind: "insert", id: "c", request: "asked" } },
        content: [text("A new paragraph.")],
      },
      {
        type: "paragraph",
        attrs: { id: "p3" },
        content: [
          text("Harbours "),
          text("open", [{ type: "proposalDelete", attrs: ref("d", "other") }]),
          text("close", [{ type: "proposalInsert", attrs: ref("d", "other") }]),
          text(" at nine."),
        ],
      },
    ]);
    const undo = new Y.UndoManager(blocks, { trackedOrigins: new Set([Asked]) });

    const resolved = writeAccepted(doc, new Set(["asked"]), new Aside(), new Asked());

    expect(resolved.map((each) => each.id).sort()).toEqual(["a", "b", "c"]);
    expect(textOf(blocks)).toEqual([
      "The tides are strong. Truly.",
      "A new paragraph.",
      "Harbours openclose at nine.",
    ]);
    expect(pendingProposals(blocks).map((each) => each.request)).toEqual(["other"]);

    undo.undo();
    expect(textOf(blocks)).toEqual(["The tides are very strong.", "Harbours openclose at nine."]);
    expect(pendingProposals(blocks).map((each) => each.request)).toEqual(["other"]);

    undo.redo();
    expect(textOf(blocks)).toEqual([
      "The tides are strong. Truly.",
      "A new paragraph.",
      "Harbours openclose at nine.",
    ]);
  });
});
