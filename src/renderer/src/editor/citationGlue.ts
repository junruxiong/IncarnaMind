import { Mark } from "@tiptap/core";
import { CITATION_NODE } from "../../../core/api";
import type { NodeJSON } from "../../../shared/blockJson";

const CITATION_GLUE_MARK = "citationGlue";

/** The last word of a text, and the spaces after it; of Chinese, its last character. */
const WORD_BEFORE = /(?:[^\s　-鿿＀-￯]{1,24}|[　-鿿＀-￯])\s*$/u;
/** How many Citations side by side stay with the word before them; the rest may start a new line. */
const MAX_WITH_WORD = 3;
/** Punctuation that stays with the Citation before it. */
const PUNCTUATION_AFTER = /^[.,;:!?)\]”’"'。，、；：！？）]+/u;

/**
 * One unbroken run in an Answer's text, drawn as a single `nowrap` span: a
 * word, the Citations after it and the punctuation after them, so a marker
 * doesn't start a line, nor does the full stop after it (#612); a run of many
 * markers keeps only its first few with the word (#898). The browser
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

const citationAt = (nodes: readonly NodeJSON[], index: number): NodeJSON | null => {
  const node = nodes[index];
  return node?.type === CITATION_NODE ? node : null;
};

function glueChildren(content: readonly NodeJSON[]): NodeJSON[] {
  const children = [...content];
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
    const cluster: NodeJSON[] = [];
    for (let each = citationAt(children, index); each; each = citationAt(children, ++index)) {
      cluster.push(each);
    }
    const after = children[index];
    const stop = after?.type === "text" ? PUNCTUATION_AFTER.exec(after.text ?? "")?.[0] : null;
    // Only the first few stay with the word, so a long run of markers can still wrap; the last
    // one stays with the punctuation after it.
    cluster.forEach((each, at) => {
      const last = at === cluster.length - 1;
      out.push(at < MAX_WITH_WORD || (last && stop) ? glued(each) : each);
    });
    if (after && stop && after.text) {
      out.push(glued({ ...after, text: stop }));
      const rest = after.text.slice(stop.length);
      // What follows the punctuation is read again: its last word may have Citations after it too.
      if (rest) children[index] = { ...after, text: rest };
      else index++;
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
