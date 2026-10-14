import { Editor, type JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, onTestFinished, test, vi } from "vitest";
import { GENERATED_MARK } from "../../src/core";
import { holdsAiText, markAsMine } from "../../src/renderer/src/editor/aiText";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { clearFormatting } from "../../src/renderer/src/editor/selectionCommands";
import { createTempDataFolder, startCore } from "../helpers/core";
import { boundEditor, typeText } from "../helpers/editor";
import { connectToMind } from "../helpers/mindClient";
import { note, readMind, shownWithAiText, writeMind } from "../helpers/minds";

/*
 * AI text (#95): the words the AI wrote stay marked, and drawn a step softer,
 * until the User types over them or marks them as theirs. What the User types
 * inside AI text is theirs alone.
 */

const AI = { type: GENERATED_MARK };
const text = (value: string, ...marks: { type: string }[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks }),
});
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });

function editorWith(blocks: JSONContent[]): Editor {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content: blocks },
  });
  editor.view.updateState(editor.state.reconfigure({ plugins: editor.extensionManager.plugins }));
  onTestFinished(() => editor.destroy());
  return editor;
}

/** Where `words` start and end in the editor's document. */
function find(doc: ProseMirrorNode, words: string): { from: number; to: number } {
  let from = -1;
  doc.descendants((node, pos) => {
    if (from >= 0) return false;
    const at = node.isText ? (node.text?.indexOf(words) ?? -1) : -1;
    if (at >= 0) from = pos + at;
    return from < 0;
  });
  if (from < 0) throw new Error(`"${words}" isn't in the document.`);
  return { from, to: from + words.length };
}

const select = (editor: Editor, words: string) =>
  editor.commands.setTextSelection(find(editor.state.doc, words));
const runMarkAsMine = (editor: Editor) =>
  markAsMine(editor.state, (tr) => editor.view.dispatch(tr));

describe("AI text", () => {
  test("survives sync and a restart, and what the User types in it is theirs alone", async () => {
    const dataDir = await createTempDataFolder();
    const firstRun = startCore(dataDir);
    const mind = await firstRun.createMind({ title: "Tides" });
    const writer = await connectToMind(firstRun, mind.id);
    writeMind(writer, [note("My notes.")]);
    await writer.settled();
    await firstRun.addToNote({
      mindId: mind.id,
      blocks: [paragraph(text("Spring tides come at new moon."))],
      origin: null,
      model: null,
    });
    const editor = boundEditor(writer);
    await vi.waitFor(() => expect(editor.state.doc.childCount).toBe(2));
    const added = () => shownWithAiText(editor.state.doc.child(1));
    expect(added()).toBe("«Spring tides come at new moon.»");

    // Inside a word's run, the marks the next input takes, an input method's too, aren't AI text's.
    editor.commands.setTextSelection(find(editor.state.doc, "tides").from);
    expect(editor.state.storedMarks).toEqual([]);
    typeText(editor, "high ");
    // At the end, where the mark doesn't reach, and at the start, where it would.
    editor.commands.setTextSelection(find(editor.state.doc, "moon.").to);
    typeText(editor, " Neap tides are weaker.");
    editor.commands.setTextSelection(find(editor.state.doc, "Spring").from);
    typeText(editor, "Note: ");
    // Over a selection.
    select(editor, "new moon");
    typeText(editor, "full moon");
    const typed = "Note: «Spring »high «tides come at »full moon«.» Neap tides are weaker.";
    expect(added()).toBe(typed);
    expect(shownWithAiText(editor.state.doc.child(0))).toBe("My notes.");

    // Another client has it as it is, and so has the Mind after a restart.
    await writer.settled();
    const other = await connectToMind(firstRun, mind.id);
    expect(shownWithAiText(readMind(other).child(1))).toBe(typed);
    firstRun.close();
    const secondRun = startCore(dataDir);
    onTestFinished(() => secondRun.close());
    const reader = await connectToMind(secondRun, mind.id);
    expect(shownWithAiText(readMind(reader).child(1))).toBe(typed);
  });

  test("Mark as mine makes the selected AI text the User's, the rest staying AI text, and Undo takes it back", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    onTestFinished(() => core.close());
    const mind = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, mind.id);
    writeMind(client, [paragraph(text("Spring tides come at new moon.", AI))]);
    await client.settled();
    const editor = boundEditor(client);

    select(editor, "come at new moon");
    expect(holdsAiText(editor.state)).toBe(true);
    expect(runMarkAsMine(editor)).toBe(true);
    expect(shownWithAiText(editor.state.doc.child(0))).toBe("«Spring tides »come at new moon«.»");
    // Nothing of it is AI text now: it isn't offered again.
    expect(holdsAiText(editor.state)).toBe(false);
    expect(runMarkAsMine(editor)).toBe(false);

    await client.settled();
    const other = await connectToMind(core, mind.id);
    expect(shownWithAiText(readMind(other).child(0))).toBe("«Spring tides »come at new moon«.»");

    editor.commands.undo();
    expect(shownWithAiText(editor.state.doc.child(0))).toBe("«Spring tides come at new moon.»");
  });

  test("Mark as mine in an older Answer, AI text as a whole, makes only the selection the User's", () => {
    const editor = editorWith([
      { type: "question", content: [text("What moves the tides?")] },
      {
        type: "answer",
        content: [paragraph(text("The Moon pulls the sea.")), paragraph(text("So does the Sun."))],
      },
    ]);
    const answer = () => editor.state.doc.child(1);
    expect(answer().attrs.generatedMarks).toBeNull();

    select(editor, "Moon");
    expect(holdsAiText(editor.state)).toBe(true);
    runMarkAsMine(editor);

    expect(answer().attrs.generatedMarks).toBe(true);
    expect(shownWithAiText(answer().child(0))).toBe("«The »Moon« pulls the sea.»");
    expect(shownWithAiText(answer().child(1))).toBe("«So does the Sun.»");
  });

  test("is found only where the selection holds it", () => {
    const editor = editorWith([
      paragraph(text("Mine. ")),
      paragraph(text("Written by the AI.", AI)),
      {
        type: "answer",
        attrs: { generatedMarks: true },
        content: [paragraph(text("Typed here. "), text("Generated here.", AI))],
      },
    ]);
    const holds = (words: string) => {
      select(editor, words);
      return holdsAiText(editor.state);
    };
    expect(holds("Mine.")).toBe(false);
    expect(holds("by the")).toBe(true);
    expect(holds("Typed here.")).toBe(false);
    expect(holds("Generated")).toBe(true);
    editor.commands.setTextSelection(find(editor.state.doc, "Written").from);
    expect(holdsAiText(editor.state)).toBe(false);
  });

  test("clearing formatting leaves it AI text", () => {
    const editor = editorWith([
      paragraph(text("Spring ", AI), text("tides", AI, { type: "bold" }), text(" now", AI)),
    ]);
    editor.commands.selectAll();
    clearFormatting(editor);
    expect(editor.getJSON().content?.[0]?.content).toEqual([text("Spring tides now", AI)]);
  });
});
