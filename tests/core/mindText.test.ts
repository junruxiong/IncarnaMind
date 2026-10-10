import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  contentHash,
  contentMarkdown,
  createElement,
  type NodeJSON,
  toMarkdown,
} from "../../src/core/answers/blocks";
import { buildQuestionContext } from "../../src/core/answers/context";

const text = (value: string, ...marks: string[]): NodeJSON => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks: marks.map((type) => ({ type })) }),
});
const link = (value: string, href: string): NodeJSON => ({
  type: "text",
  text: value,
  marks: [{ type: "link", attrs: { href } }],
});
const paragraph = (...content: NodeJSON[]): NodeJSON => ({ type: "paragraph", content });
const item = (...content: NodeJSON[]): NodeJSON => ({ type: "listItem", content });
const citation = (attrs: Record<string, unknown>): NodeJSON => ({ type: "citation", attrs });

/** A fragment holding these top-level Blocks. */
function fragmentOf(...blocks: NodeJSON[]): Y.XmlFragment {
  const doc = new Y.Doc();
  const fragment = doc.getXmlFragment("blocks");
  fragment.insert(0, blocks.map(createElement));
  return fragment;
}

/** The element is read once it is in a document, as the core reads Blocks. */
const elementOf = (block: NodeJSON) => fragmentOf(block).get(0) as Y.XmlElement;
const markdownOf = (block: NodeJSON) => toMarkdown(elementOf(block));

describe("a Block as the text Question context shows the model", () => {
  test("paragraphs carry bold, italic, strike, code and links; other marks add nothing", () => {
    expect(
      markdownOf(
        paragraph(
          text("plain "),
          text("bold", "bold"),
          text(" "),
          text("both", "bold", "italic"),
          text(" "),
          text("gone", "strike"),
          text(" "),
          text("x = 1", "code"),
          text(" "),
          link("site", "https://example.com"),
          text(" "),
          text("underlined", "underline"),
          text(" "),
          text("lit", "highlight"),
          text(" "),
          text("odd", "somethingNew"),
        ),
      ),
    ).toBe(
      "plain **bold** ***both*** ~~gone~~ `x = 1` [site](https://example.com) underlined lit odd",
    );
  });

  test("headings clamp to levels 1 to 6", () => {
    expect(markdownOf({ type: "heading", attrs: { level: 2 }, content: [text("Two")] })).toBe(
      "## Two",
    );
    expect(markdownOf({ type: "heading", attrs: { level: 9 }, content: [text("Deep")] })).toBe(
      "###### Deep",
    );
  });

  test("lists nest, number from their start, and hold several Blocks an item", () => {
    expect(
      markdownOf({
        type: "orderedList",
        attrs: { start: 3 },
        content: [
          item(paragraph(text("Three")), {
            type: "bulletList",
            content: [item(paragraph(text("inner a"))), item(paragraph(text("inner b")))],
          }),
          item(paragraph(text("Four")), paragraph(text("more of four"))),
        ],
      }),
    ).toBe("3. Three\n\n   - inner a\n   - inner b\n4. Four\n\n   more of four");
  });

  test("block quotes, rules, code (with a longer fence when the code holds one) and math", () => {
    expect(
      markdownOf({
        type: "blockquote",
        content: [paragraph(text("Quoted")), paragraph(text("twice"))],
      }),
    ).toBe("> Quoted\n>\n> twice");
    expect(markdownOf({ type: "horizontalRule" })).toBe("---");
    expect(
      markdownOf({ type: "codeBlock", attrs: { language: "js" }, content: [text("a();\nb();")] }),
    ).toBe("```js\na();\nb();\n```");
    expect(markdownOf({ type: "codeBlock", content: [text("```\nnested\n```")] })).toBe(
      "````\n```\nnested\n```\n````",
    );
    expect(markdownOf({ type: "blockMath", attrs: { latex: "a^2" } })).toBe("$$\na^2\n$$");
    expect(
      markdownOf(paragraph(text("so "), { type: "inlineMath", attrs: { latex: "x" } }, text("."))),
    ).toBe("so $x$.");
    expect(markdownOf(paragraph(text("one"), { type: "hardBreak" }, text("two")))).toBe("one\ntwo");
  });

  test("a Citation is a plain reference: the Document, and its Location when it has one", () => {
    const cited = (attrs: Record<string, unknown>) =>
      markdownOf(paragraph(text("Claim "), citation(attrs), text(".")));
    expect(cited({ documentName: "Tides", pageFrom: 2, pageTo: 2 })).toBe("Claim [Tides, p. 2].");
    expect(cited({ documentName: "Tides", pageFrom: 2, pageTo: 4 })).toBe("Claim [Tides, p. 2–4].");
    expect(cited({ documentName: "Tides" })).toBe("Claim [Tides].");
    expect(cited({})).toBe("Claim .");
  });

  test("a node type this version doesn't know shows its Blocks, or its text", () => {
    expect(
      markdownOf({ type: "callout", content: [paragraph(text("Inside")), paragraph(text("two"))] }),
    ).toBe("Inside\n\ntwo");
    expect(markdownOf({ type: "callout", content: [text("just "), text("text", "bold")] })).toBe(
      "just **text**",
    );
  });

  test("a table shows the Blocks of its cells, and an image nothing", () => {
    expect(
      markdownOf({
        type: "table",
        content: [
          {
            type: "tableRow",
            content: [
              { type: "tableHeader", content: [paragraph(text("Head"))] },
              { type: "tableCell", content: [paragraph(text("Cell"))] },
            ],
          },
          {
            type: "tableRow",
            content: [{ type: "tableCell", content: [paragraph(text("Next"))] }],
          },
        ],
      }),
    ).toBe("Head\n\nCell\n\nNext");
    expect(markdownOf({ type: "image", attrs: { src: "x.png", alt: "A figure" } })).toBe("");
  });

  test("an Answer is its Blocks apart by blank lines, with a fingerprint that must not change", () => {
    const answer = elementOf({
      type: "answer",
      attrs: { status: "done" },
      content: [
        paragraph(text("First "), citation({ documentName: "Tides", pageFrom: 2, pageTo: 2 })),
        { type: "bulletList", content: [item(paragraph(text("a"))), item(paragraph(text("b")))] },
        paragraph(),
      ],
    });
    expect(contentMarkdown(answer)).toBe("First [Tides, p. 2]\n\n- a\n- b");
    // Saved in Minds as `generatedHash`, so the same text must always hash the same.
    expect(contentHash(answer)).toBe("nQ88shBr0w2G1-VkDZTM1_");
  });
});

describe("Question context", () => {
  test("Notes and Questions are the User's words, Answers the model's; what is switched off or failed is left out", () => {
    const fragment = fragmentOf(
      { type: "heading", attrs: { level: 1 }, content: [text("Tides")] },
      { type: "paragraph", attrs: { includeInContext: false }, content: [text("Private.")] },
      paragraph(text("Spring "), text("tides", "bold"), text(" are strong.")),
      { type: "question", content: [text("When are they?")] },
      {
        type: "answer",
        attrs: { status: "done" },
        content: [
          paragraph(
            text("At new moon "),
            citation({ documentName: "Tides", pageFrom: 2, pageTo: 2 }),
          ),
        ],
      },
      { type: "question", content: [text("Failed one?")] },
      { type: "answer", attrs: { status: "failed" }, content: [paragraph(text("No."))] },
      { type: "question", content: [text("And neap tides?")] },
    );
    expect(buildQuestionContext(fragment, 7)).toEqual({
      question: "And neap tides?",
      messages: [
        { role: "user", content: "# Tides\n\nSpring **tides** are strong.\n\nWhen are they?" },
        { role: "assistant", content: "At new moon [Tides, p. 2]" },
        { role: "user", content: "Failed one?\n\nAnd neap tides?" },
      ],
    });
  });
});
