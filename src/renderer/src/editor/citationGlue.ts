import { Mark } from "@tiptap/core";
import { CITATION_NODE } from "../../../core/api";
import type { NodeJSON } from "../../../shared/blockJson";

const CITATION_GLUE_MARK = "citationGlue";

/** The last word of a text, and the spaces after it; of Chinese, its last character. */
const WORD_BEFORE = /(?:[^\s　-鿿＀-￯]{1,24}|[　-鿿＀-￯])\s*$/u;
/** Punctuation that stays with the Citation before it. */
const PUNCTUATION_AFTER = /^[.,;:!?)\]”’"'。，、；：！？）]+/u;

/**
 * One unbroken run in an Answer's text, drawn as a single `nowrap` span: a
 * word, the Citations after it and the punctuation after them, so a marker
 * doesn't start a line, nor does the full stop after it (#612). The browser
 * breaks between two inline boxes that hold markers unless one element holds
 * both, which only a mark gives. It is in the editor's document, never
 * stored: `withoutCitationGlue` takes it out of what leaves the Answer.
 */
export const CitationGlue = Mark.create({
  name: CITATION_GLUE_MARK,
  // Outside the Generated mark (1001), so the run is one span around its words.
  priority: 2000,
  inclusive: false,
  renderHTML: () => ["span", { class: "citation-glue" }, 0],
});

const glued = (node: NodeJSON): NodeJSON => ({
  ...node,
  marks: [...(node.marks ?? []), { type: CITATION_GLUE_MARK }],
});

const isCitation = (node: NodeJSON | undefined) => node?.type === CITATION_NODE;

function glueChildren(children: readonly NodeJSON[]): NodeJSON[] {
  const out: NodeJSON[] = [];
  let index = 0;
  while (index < children.length) {
    const child = children[index] as NodeJSON;
    if (!isCitation(child)) {
      const next = children[index + 1];
      const tail =
        isCitation(next) && child.type === "text" ? WORD_BEFORE.exec(child.text ?? "")?.[0] : null;
      if (tail && child.text) {
        const head = child.text.slice(0, child.text.length - tail.length);
        if (head) out.push({ ...child, text: head });
        out.push(glued({ ...child, text: tail }));
      } else {
        out.push(glueIn(child));
      }
      index++;
      continue;
    }
    while (isCitation(children[index])) out.push(glued(children[index] as NodeJSON)), index++;
    const after = children[index];
    const stop = after?.type === "text" ? PUNCTUATION_AFTER.exec(after.text ?? "")?.[0] : null;
    if (after && stop && after.text) {
      out.push(glued({ ...after, text: stop }));
      const rest = after.text.slice(stop.length);
      if (rest) out.push({ ...after, text: rest });
      index++;
    }
  }
  return out;
}

function glueIn(node: NodeJSON): NodeJSON {
  if (!node.content || node.type === "codeBlock") return node;
  return { ...node, content: glueChildren(node.content) };
}

/** The Blocks with each run of a word, its Citations and the punctuation after them marked to stay on one line. */
export const withCitationGlue = (blocks: readonly NodeJSON[]): NodeJSON[] => blocks.map(glueIn);

/** The Blocks without that mark: what is copied or added to a note from an Answer is the Answer's own. */
export function withoutCitationGlue(blocks: readonly NodeJSON[]): NodeJSON[] {
  return blocks.map((node) => {
    const marks = node.marks?.filter((mark) => mark.type !== CITATION_GLUE_MARK);
    const { marks: _marks, ...rest } = node;
    return {
      ...rest,
      ...(marks && marks.length > 0 && { marks }),
      ...(node.content && { content: withoutCitationGlue(node.content) }),
    };
  });
}
