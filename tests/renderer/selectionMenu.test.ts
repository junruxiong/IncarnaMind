import { Editor, type JSONContent } from "@tiptap/core";
import { describe, expect, onTestFinished, test } from "vitest";
import { ANSWER_BLOCK, BLOCK_ID_ATTRIBUTE, CITATION_NODE, QUESTION_BLOCK } from "../../src/core";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import {
  canTurnInto,
  currentTurnInto,
  linkTarget,
  selectionToMath,
  TURN_INTO,
} from "../../src/renderer/src/editor/selectionCommands";

const text = (value: string, marks?: string[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks ? { marks: marks.map((type) => ({ type })) } : {}),
});
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const citation: JSONContent = {
  type: CITATION_NODE,
  attrs: {
    passageId: "p1",
    documentId: "d1",
    documentName: "Tides",
    quote: "strongest at new moon",
  },
};

function editorWith(blocks: JSONContent[]): Editor {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content: blocks },
  });
  onTestFinished(() => editor.destroy());
  return editor;
}

const turnInto = (editor: Editor, id: string) => {
  const kind = TURN_INTO.find((each) => each.id === id);
  if (!kind) throw new Error(`No ${id} to turn into.`);
  return kind.run(editor);
};

const typesOf = (editor: Editor) => {
  const found: string[] = [];
  editor.state.doc.descendants((node) => {
    found.push(node.type.name);
  });
  return found;
};

describe("Turn into", () => {
  test("a paragraph becomes a heading, a list, a quote and text again, keeping its words and its ID", () => {
    const editor = editorWith([paragraph(text("Spring tides"))]);
    const idOf = () => editor.state.doc.firstChild?.attrs[BLOCK_ID_ATTRIBUTE];
    const original = idOf();
    editor.commands.selectAll();

    expect(currentTurnInto(editor).id).toBe("text");
    turnInto(editor, "heading-2");
    expect(editor.state.doc.firstChild?.type.name).toBe("heading");
    expect(editor.state.doc.firstChild?.attrs.level).toBe(2);
    expect(currentTurnInto(editor).id).toBe("heading-2");
    expect(idOf()).toBe(original);

    turnInto(editor, "bullet-list");
    expect(typesOf(editor)).toEqual(["bulletList", "listItem", "paragraph", "text"]);
    expect(currentTurnInto(editor).id).toBe("bullet-list");

    turnInto(editor, "ordered-list");
    expect(editor.state.doc.firstChild?.type.name).toBe("orderedList");

    turnInto(editor, "quote");
    expect(typesOf(editor)).toEqual(["blockquote", "paragraph", "text"]);

    turnInto(editor, "text");
    expect(typesOf(editor)).toEqual(["paragraph", "text"]);
    expect(editor.state.doc.textContent).toBe("Spring tides");
  });

  test("Citations stay inside the text, and so do marks", () => {
    const editor = editorWith([
      paragraph(text("Strongest at new moon "), text("and full", ["bold"]), citation, text(".")),
    ]);
    editor.commands.selectAll();

    const citations = () => {
      const cited: JSONContent[] = [];
      editor.state.doc.descendants((node) => {
        if (node.type.name === CITATION_NODE) cited.push(node.toJSON());
      });
      return cited;
    };
    const before = citations();
    expect(before).toHaveLength(1);

    for (const id of ["heading-1", "bullet-list", "quote", "ordered-list", "text"]) {
      turnInto(editor, id);
      expect(citations(), id).toEqual(before);
      expect(editor.state.doc.textContent, id).toContain("Strongest at new moon and full");
    }
    expect(editor.getJSON().content?.[0]?.content?.[1]?.marks).toEqual([{ type: "bold" }]);
  });

  test("it turns every selected Block, and offers itself only over Notes of text", () => {
    const editor = editorWith([
      paragraph(text("One")),
      paragraph(text("Two")),
      { type: "codeBlock", content: [text("x = 1")] },
      { type: QUESTION_BLOCK, attrs: { id: "q1" }, content: [text("Why?")] },
      { type: ANSWER_BLOCK, attrs: { questionId: "q1" }, content: [paragraph(text("Because."))] },
    ]);

    editor.commands.setTextSelection({ from: 2, to: 8 });
    expect(canTurnInto(editor)).toBe(true);
    turnInto(editor, "heading-3");
    expect(
      editor
        .getJSON()
        .content?.slice(0, 2)
        .map((block) => block.type),
    ).toEqual(["heading", "heading"]);

    editor.commands.setTextSelection({ from: 2, to: editor.state.doc.content.size - 2 });
    expect(canTurnInto(editor)).toBe(false);
    editor.commands.setTextSelection(editor.state.doc.content.size - 4);
    expect(canTurnInto(editor)).toBe(false);
  });
});

describe("formatting", () => {
  test("underline and links are marks of the Mind, and clear formatting takes every mark off", () => {
    const editor = editorWith([paragraph(text("See the tides"))]);
    editor.commands.setTextSelection({ from: 5, to: 8 });

    editor.chain().toggleUnderline().toggleBold().setLink({ href: "https://example.com" }).run();
    expect(editor.isActive("underline")).toBe(true);
    expect(editor.getJSON().content?.[0]?.content?.[1]?.marks?.map((mark) => mark.type)).toEqual(
      expect.arrayContaining(["underline", "bold", "link"]),
    );

    editor.chain().unsetAllMarks().run();
    expect(editor.getJSON().content?.[0]?.content).toEqual([text("See the tides")]);
  });

  test("a selection becomes an inline formula written in its text, and says where it is", () => {
    const editor = editorWith([paragraph(text("Energy E = mc^2 here"))]);
    editor.commands.setTextSelection({ from: 8, to: 16 });

    expect(selectionToMath(editor)).toBe(8);
    expect(editor.getJSON().content?.[0]?.content?.[1]).toEqual({
      type: "inlineMath",
      attrs: { latex: "E = mc^2", proposal: null },
    });

    const two = editorWith([paragraph(text("a")), paragraph(text("b"))]);
    two.commands.selectAll();
    expect(selectionToMath(two)).toBeNull();
  });

  test("an address typed without a scheme is taken as https", () => {
    expect(linkTarget(" example.com/tides ")).toBe("https://example.com/tides");
    expect(linkTarget("mailto:a@b.co")).toBe("mailto:a@b.co");
    expect(linkTarget("http://example.com")).toBe("http://example.com");
  });
});
