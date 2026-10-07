import { Editor } from "@tiptap/core";
import Collaboration from "@tiptap/extension-collaboration";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { PluginView, Transaction } from "@tiptap/pm/state";
import type { EditorView } from "@tiptap/pm/view";
import { ySyncPluginKey } from "@tiptap/y-tiptap";
import { describe, expect, onTestFinished, test } from "vitest";
import * as Y from "yjs";
import { MIND_CONTENT_FIELD } from "../../src/core";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { SmartTypography } from "../../src/renderer/src/editor/typography";
import { answerEnded, citingModel, onlyCitation, setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { answerText, note, question, readMind, writeMind } from "../helpers/minds";

/** The extensions of the Mind editor that make or change content: its schema, and smart typography. */
const editorExtensions = () => [...noteExtensions(), SmartTypography];

/** A headless Tiptap editor, starting with an empty paragraph: the Mind editor minus the screen. */
function headlessEditor(extensions = editorExtensions()): Editor {
  const content = { type: "doc", content: [{ type: "paragraph" }] };
  const editor = new Editor({ element: null, extensions, content });
  editor.view.updateState(editor.state.reconfigure({ plugins: editor.extensionManager.plugins }));
  onTestFinished(() => editor.destroy());
  return editor;
}

/**
 * A headless editor bound to a client's copy of the Mind by the Collaboration
 * extension, as the Mind editor is. ProseMirror's view would run the binding;
 * here it runs by hand: it renders what the Yjs document holds into the
 * editor, applies the changes the core pushes, and writes the editor's own
 * changes back to Yjs, which the client sends to the core.
 */
function boundEditor(client: MindClient): Editor {
  const editor = headlessEditor([
    ...editorExtensions(),
    Collaboration.configure({ document: client.doc, field: MIND_CONTENT_FIELD }),
  ]);
  const sync = editor.state.plugins.find((plugin) => plugin.spec.key === ySyncPluginKey);
  const view = {
    get state() {
      return editor.state;
    },
    dispatch: (transaction: Transaction) => editor.view.dispatch(transaction),
    hasFocus: () => false,
  } as unknown as EditorView;
  let binding: PluginView | undefined;
  editor.on("transaction", () => binding?.update?.(view, editor.state));
  binding = sync?.spec.view?.(view);
  if (!binding) throw new Error("The editor has no Yjs binding.");
  onTestFinished(() => binding?.destroy?.());
  return editor;
}

/**
 * Types `text` at the cursor, a character at a time, as the editor's view
 * does: the input rules (smart typography among them) see each character
 * first, and it is inserted as typed when none of them takes it.
 */
function type(editor: Editor, text: string): void {
  for (const character of text) {
    const { from, to } = editor.state.selection;
    const taken = editor.state.plugins.some((plugin) =>
      plugin.props.handleTextInput?.call(plugin, editor.view, from, to, character, () =>
        editor.state.tr.insertText(character, from, to),
      ),
    );
    if (!taken) editor.view.dispatch(editor.state.tr.insertText(character, from, to));
  }
}

/** Puts the cursor in a new, empty paragraph at the end of the editor's document. */
function startParagraphAtEnd(editor: Editor): void {
  const end = editor.state.doc.content.size;
  editor
    .chain()
    .insertContentAt(end, { type: "paragraph" })
    .setTextSelection(end + 1)
    .run();
}

/** How many changes this client made itself, from its Yjs state vector. */
const ownChanges = (client: MindClient) =>
  Y.decodeStateVector(Y.encodeStateVector(client.doc)).get(client.doc.clientID) ?? 0;

/** The text of the top-level Block whose type is `type`, the last one first. */
function lastBlockText(doc: ProseMirrorNode, type: string): string | undefined {
  const blocks: ProseMirrorNode[] = [];
  doc.forEach((block) => {
    if (block.type.name === type) blocks.push(block);
  });
  return blocks.at(-1)?.textContent;
}

describe("highlighting", () => {
  test("highlighted text round-trips through the Mind's Yjs document, and a restart", async () => {
    const dataDir = await createTempDataFolder();
    const firstRun = startCore(dataDir);
    const mind = await firstRun.createMind({ title: "Tides" });
    const writer = await connectToMind(firstRun, mind.id);
    writeMind(writer, [note("Spring tides are strongest at new moon.")]);
    const editor = boundEditor(writer);

    // Select "strongest" and highlight it, as Mod-Shift-H and the formatting menu do.
    const from = 1 + "Spring tides are ".length;
    editor
      .chain()
      .setTextSelection({ from, to: from + "strongest".length })
      .toggleHighlight()
      .run();
    expect(editor.isActive("highlight")).toBe(true);
    await writer.settled();

    // Stored the way y-tiptap stores a mark: a formatting attribute named after it.
    const stored = writer.blocks.get(0) as Y.XmlElement;
    expect((stored.get(0) as Y.XmlText).toDelta()).toEqual([
      { insert: "Spring tides are " },
      { insert: "strongest", attributes: { highlight: {} } },
      { insert: " at new moon." },
    ]);
    firstRun.close();

    // After a restart, an editor opening the Mind shows the same highlight.
    const secondRun = startCore(dataDir);
    const reader = await connectToMind(secondRun, mind.id);
    const reopened = boundEditor(reader);
    expect(reopened.getJSON().content?.[0]?.content).toEqual([
      { type: "text", text: "Spring tides are " },
      { type: "text", text: "strongest", marks: [{ type: "highlight" }] },
      { type: "text", text: " at new moon." },
    ]);

    // Taking it off goes back through Yjs too.
    reopened
      .chain()
      .setTextSelection({ from, to: from + "strongest".length })
      .unsetHighlight()
      .run();
    await reader.settled();
    const third = await connectToMind(secondRun, mind.id);
    expect(readMind(third).child(0).toJSON().content).toEqual([
      { type: "text", text: "Spring tides are strongest at new moon." },
    ]);
  });

  test("`==text==` typed in a Note becomes highlighted text", () => {
    const editor = headlessEditor();
    editor.commands.setTextSelection(1);
    type(editor, "Tides are ==strongest== now");

    expect(editor.getJSON().content?.[0]?.content).toEqual([
      { type: "text", text: "Tides are " },
      { type: "text", text: "strongest", marks: [{ type: "highlight" }] },
      { type: "text", text: " now" },
    ]);
  });
});

describe("smart typography", () => {
  /**
   * A sentence with straight quotes, an apostrophe and a double hyphen: a
   * text Document, and the quote cited from it.
   */
  const QUOTE = 'Sailors call them "spring tides" -- the Moon\'s pull is strongest then.';
  const ALMANAC = `${QUOTE}\n`;

  test("changes what the User types, but not the Answers and Citations the core writes", async () => {
    const answer =
      'Sailors\' "spring tides" -- twice a month... come at new and full moon -> (c) 1/2 of it [^1].';
    const model = citingModel({
      query: "spring tides",
      records: (passages) => [{ marker: 1, passage: passages[0]?.id ?? "", quote: QUOTE }],
      answer,
    });
    const { core, client, mind } = await setUpWithDocuments(model, [
      { name: "Almanac.txt", contents: ALMANAC },
    ]);
    // The Answer streams into an editor with smart typography on, as in the app.
    const editor = boundEditor(client);
    const asked = question("When are spring tides?");
    writeMind(client, [asked]);
    await client.settled();
    const before = ownChanges(client);

    const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });
    if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
    const ended = await answerEnded(core, result.answerId);
    expect(ended.event).toBe("finished");
    await client.settled();

    // The Answer's text and its Citation's quote are exactly as written, straight quotes and all.
    const written =
      'Sailors\' "spring tides" -- twice a month... come at new and full moon -> (c) 1/2 of it .';
    expect(answerText(client, result.answerId)).toBe(written);
    expect(onlyCitation(client, result.answerId)).toMatchObject({ quote: QUOTE, check: "found" });
    // The editor shows them unchanged, and wrote nothing back to the core.
    expect(lastBlockText(editor.state.doc, "answer")).toBe(written);
    expect(ownChanges(client)).toBe(before);

    // Text the User types in the same editor does change: curly quotes, dashes, an ellipsis.
    startParagraphAtEnd(editor);
    type(editor, "My note: \"spring\" tides -- it's the Moon's pull...");
    const typed = "My note: “spring” tides — it’s the Moon’s pull…";
    expect(lastBlockText(editor.state.doc, "paragraph")).toBe(typed);
    await client.settled();

    // And the core stored both as they are.
    const other = await connectToMind(core, mind.id);
    expect(answerText(other, result.answerId)).toBe(written);
    expect(onlyCitation(other, result.answerId).quote).toBe(QUOTE);
    expect(lastBlockText(readMind(other), "paragraph")).toBe(typed);
  });

  test("leaves LaTeX typed between $$ as it is, and picks up again after the formula", () => {
    const editor = headlessEditor();
    editor.commands.setTextSelection(1);
    type(editor, "Area $$r^2 -> \\pi r^2$$ grows -> fast");

    const content = editor.getJSON().content?.[0]?.content;
    expect(content).toEqual([
      { type: "text", text: "Area " },
      { type: "inlineMath", attrs: { latex: "r^2 -> \\pi r^2" } },
      { type: "text", text: " grows → fast" },
    ]);
  });

  test("leaves code alone", () => {
    const editor = headlessEditor();
    editor.commands.setContent({
      type: "doc",
      content: [{ type: "codeBlock", attrs: { language: null } }],
    });
    editor.commands.setTextSelection(1);
    type(editor, 'const s = "tide" -- 1;');

    expect(editor.state.doc.textContent).toBe('const s = "tide" -- 1;');
  });
});
