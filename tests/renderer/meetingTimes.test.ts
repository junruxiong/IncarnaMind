import { getSchema, type JSONContent } from "@tiptap/core";
import { EditorState, type Transaction } from "@tiptap/pm/state";
import { ySyncPluginKey } from "@tiptap/y-tiptap";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import { ANSWER_BLOCK } from "../../src/core";
import { meetingTimesPlugin } from "../../src/renderer/src/editor/meetingTimes";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { blockTimeOf } from "../../src/shared/blockTimes";

const schema = getSchema(noteExtensions());

const paragraph = (id: string, text = ""): JSONContent => ({
  type: "paragraph",
  attrs: { id },
  content: text ? [{ type: "text", text }] : [],
});

const answer: JSONContent = {
  type: ANSWER_BLOCK,
  attrs: { id: "a1", questionId: "q1" },
  content: [paragraph("a1p")],
};

/** A note with these Blocks, and the plugin that takes its times from `clock`. */
function setup(blocks: JSONContent[], clock: { now: number | null }) {
  const doc = new Y.Doc();
  const plugin = meetingTimesPlugin({ doc, time: () => clock.now });
  const state = EditorState.create({
    doc: schema.nodeFromJSON({ type: "doc", content: blocks }),
    plugins: [plugin],
  });
  return { doc, state };
}

/** Types `text` at the end of the Block that has this id. */
function typeIn(state: EditorState, id: string, text: string, remote = false): Transaction {
  let end = -1;
  state.doc.descendants((node, pos) => {
    if (node.attrs.id === id) end = pos + node.nodeSize - 1;
    return end < 0;
  });
  const tr = state.tr.insertText(text, end);
  if (remote) tr.setMeta(ySyncPluginKey, { isChangeOrigin: true });
  return tr;
}

describe("a line typed while a meeting records", () => {
  test("takes the Meeting's time when it gets its first words, and keeps it as it is edited", () => {
    const clock = { now: 20_000 as number | null };
    const { doc, state } = setup([paragraph("p1")], clock);

    // An empty line, such as the one after Enter, has no time until it has words.
    let next = state.apply(state.tr.setSelection(state.selection));
    expect(blockTimeOf(doc, "p1")).toBeNull();

    clock.now = 31_500;
    next = next.apply(typeIn(next, "p1", "Pricing"));
    expect(blockTimeOf(doc, "p1")).toBe(31_500);

    clock.now = 99_000;
    next.apply(typeIn(next, "p1", " moves"));
    expect(blockTimeOf(doc, "p1")).toBe(31_500);
  });

  test("takes only its own time: a line typed later has a later one", () => {
    const clock = { now: 8_000 as number | null };
    const { doc, state } = setup([paragraph("p1", "First"), paragraph("p2")], clock);
    state.apply(typeIn(state, "p2", "Second"));
    expect(blockTimeOf(doc, "p1")).toBeNull();
    expect(blockTimeOf(doc, "p2")).toBe(8_000);
  });

  test("takes none when nothing records: paused, stopped, or no Meeting", () => {
    const { doc, state } = setup([paragraph("p1")], { now: null });
    state.apply(typeIn(state, "p1", "Typed in a pause"));
    expect(blockTimeOf(doc, "p1")).toBeNull();
  });

  test("takes none for what the core or another computer writes, nor inside an Answer", () => {
    const { doc, state } = setup([paragraph("p1"), answer], { now: 7_000 });
    state.apply(typeIn(state, "p1", "From the core", true));
    state.apply(typeIn(state, "a1p", "An Answer's words"));
    expect(blockTimeOf(doc, "p1")).toBeNull();
    expect(blockTimeOf(doc, "a1p")).toBeNull();
  });
});
