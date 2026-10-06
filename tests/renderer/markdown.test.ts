import { describe, expect, test } from "vitest";
import { type Block, type Inline, parseMarkdown } from "../../src/renderer/src/viewer/markdown";

/** The tree with every source range replaced by the text it covers, for readable expectations. */
function show(source: string, blocks: Block[]): unknown[] {
  const inlines = (list: Inline[]): unknown[] =>
    list.map((inline) =>
      inline.kind === "text" || inline.kind === "code"
        ? { [inline.kind]: source.slice(inline.start, inline.end) }
        : inline.kind === "link"
          ? { link: inline.href, children: inlines(inline.children) }
          : { [inline.kind]: inlines(inline.children) },
    );
  const showBlock = (block: Block): unknown => {
    switch (block.kind) {
      case "heading":
        return { [`h${block.level}`]: inlines(block.inlines) };
      case "paragraph":
        return { p: inlines(block.inlines) };
      case "quote":
        return { quote: inlines(block.inlines) };
      case "list":
        return { [block.ordered ? "ol" : "ul"]: block.items.map(inlines) };
      case "verbatim":
        return { verbatim: source.slice(block.start, block.end) };
      case "rule":
        return "rule";
    }
  };
  return blocks.map(showBlock);
}

const parse = (source: string) => show(source, parseMarkdown(source));

describe("reading Markdown for the viewer", () => {
  test("headings, paragraphs (lines joined by their line break) and rules", () => {
    expect(parse("# Title #\n\nFirst line\nsecond line.\n\n---\n### Sub")).toEqual([
      { h1: [{ text: "Title" }] },
      { p: [{ text: "First line" }, { text: "\n" }, { text: "second line." }] },
      "rule",
      { h3: [{ text: "Sub" }] },
    ]);
  });

  test("emphasis, inline code and escapes", () => {
    expect(parse("Some **bold**, *it*, _em_, `co*de` and \\*stars\\* in snake_case_name.")).toEqual(
      [
        {
          p: [
            { text: "Some " },
            { strong: [{ text: "bold" }] },
            { text: ", " },
            { em: [{ text: "it" }] },
            { text: ", " },
            { em: [{ text: "em" }] },
            { text: ", " },
            { code: "co*de" },
            { text: " and " },
            { text: "*stars" },
            { text: "* in snake_case_name." },
          ],
        },
      ],
    );
  });

  test("links are followed only for http, https and mailto; images show their description", () => {
    expect(
      parse('[site](https://example.com "Title") [bad](javascript:void) ![a chart](chart.png)'),
    ).toEqual([
      {
        p: [
          { link: "https://example.com", children: [{ text: "site" }] },
          { text: " " },
          { link: null, children: [{ text: "bad" }] },
          { text: " " },
          { text: "a chart" },
        ],
      },
    ]);
  });

  test("raw HTML stays text", () => {
    expect(parse("<script>alert(1)</script> <b>bold</b>")).toEqual([
      { p: [{ text: "<script>alert(1)</script> <b>bold</b>" }] },
    ]);
  });

  test("lists, quotes, fenced code and tables", () => {
    const source = [
      "- one",
      "  continued",
      "- two",
      "1. first",
      "2) second",
      "> quoted",
      "> more",
      "```js",
      "const a = 1;",
      "```",
      "| a | b |",
      "|---|---|",
    ].join("\n");
    expect(parse(source)).toEqual([
      { ul: [[{ text: "one" }, { text: "\n" }, { text: "continued" }], [{ text: "two" }]] },
      { ol: [[{ text: "first" }], [{ text: "second" }]] },
      { quote: [{ text: "quoted" }, { text: "\n" }, { text: "more" }] },
      { verbatim: "const a = 1;" },
      { verbatim: "| a | b |\n|---|---|" },
    ]);
  });

  test("an unclosed code fence runs to the end", () => {
    expect(parse("```\ncode\nmore")).toEqual([{ verbatim: "code\nmore" }]);
  });
});
