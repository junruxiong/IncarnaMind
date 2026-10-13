/**
 * A Mind's health, under its title (#96): its Citations found and not
 * found, and its sentences without support: those checked as unsupported or
 * contradicted, and those that make a claim with no Citation; then those
 * whose check needs a re-check or couldn't be made. Each part keeps where
 * its sentences or Citations are, top to bottom, so it can move to the next.
 *
 * Its own sentences count: in paragraphs, lists, quotes and Answers added to
 * it, not in headings, Questions, code, formulas, tables or footnotes.
 * Changes still waiting don't count: the words and Citations they would
 * add aren't the Mind's yet.
 */
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import {
  CITATION_NODE,
  type CitationAttributes,
  type CitationCheck,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_INSERT_MARK,
} from "../../../core/api";
import { readProposalAttribute } from "../../../shared/proposals";
import { INLINE_NODE, isClaim, sentenceSpans } from "../../../shared/sentences";
import type { ShownClaim } from "./claimChecks";

export const HEALTH_PARTS = [
  "found",
  "notFound",
  "unsupported",
  "contradicted",
  "noCitation",
  "recheck",
  "notChecked",
] as const;
export type HealthPart = (typeof HEALTH_PARTS)[number];

/** Where each part's sentences or Citations are, top to bottom: its count is how many. */
export type NoteHealth = Record<HealthPart, number[]>;

/** Blocks whose text isn't counted. */
const UNCOUNTED: ReadonlySet<string> = new Set([
  "question",
  "heading",
  "codeBlock",
  "blockMath",
  "table",
  "chart",
]);

/** Words or a node a change adds: not the Mind's until it is accepted. */
const proposed = (node: ProseMirrorNode) =>
  readProposalAttribute(node.attrs[PROPOSAL_ATTRIBUTE])?.kind === "insert" ||
  node.marks.some((mark) => mark.type.name === PROPOSAL_INSERT_MARK);

const PROBLEM: Partial<Record<ShownClaim["shown"]["state"], HealthPart>> = {
  unsupported: "unsupported",
  contradicted: "contradicted",
  "needs-recheck": "recheck",
  "not-checked": "notChecked",
};

export function noteHealth(
  doc: ProseMirrorNode,
  claims: readonly ShownClaim[],
  citationCheck: (attributes: CitationAttributes) => CitationCheck,
): NoteHealth {
  const health: NoteHealth = {
    found: [],
    notFound: [],
    unsupported: [],
    contradicted: [],
    noCitation: [],
    recheck: [],
    notChecked: [],
  };
  doc.descendants((node, pos) => {
    if (UNCOUNTED.has(node.type.name) || proposed(node)) return false;
    if (!node.isTextblock) return true;
    if (node.type.name !== "paragraph") return false;
    let text = "";
    /** Where each character of `text` is in the document. */
    const places: number[] = [];
    const citations: { at: number; pos: number; attributes: CitationAttributes }[] = [];
    node.forEach((child, offset) => {
      const start = pos + 1 + offset;
      if (proposed(child)) return;
      if (child.isText) {
        const words = child.text ?? "";
        for (let index = 0; index < words.length; index++) places.push(start + index);
        text += words;
        return;
      }
      if (child.type.name === CITATION_NODE) {
        citations.push({
          at: text.length,
          pos: start,
          attributes: child.attrs as CitationAttributes,
        });
      }
      places.push(start);
      text += child.type.name === "hardBreak" ? "\n" : INLINE_NODE;
    });
    for (const span of sentenceSpans(text)) {
      const from = places[span.from] ?? pos;
      const to = (places[span.to - 1] ?? from) + 1;
      const claim = claims.find((each) => each.from < to && each.to > from);
      const problem = claim && PROBLEM[claim.shown.state];
      if (problem) {
        health[problem].push(from);
        continue;
      }
      const cited = citations.filter((each) => each.at >= span.from && each.at < span.to);
      for (const citation of cited) {
        const check = citationCheck(citation.attributes);
        if (check === "found") health.found.push(citation.pos);
        else if (check === "not-found") health.notFound.push(citation.pos);
      }
      if (cited.length === 0 && isClaim(text.slice(span.from, span.to)))
        health.noCitation.push(from);
    }
    return false;
  });
  return health;
}
