/**
 * The Word look for a Mind imported before Minds kept it (#746): read again
 * from the package kept at import, and given to the Blocks that are still as
 * imported. A Block is found by the place its source names (`WordSource`) in
 * the package read again, and is still as imported when its fingerprint is
 * the one recorded then (`wordFingerprint` leaves the look out, so giving the
 * look changes no fingerprint). A Block the User changed keeps the Mind's
 * plain look. What would change is worked out first, then made in one change,
 * so a failure leaves the Mind as it was.
 */
import * as Y from "yjs";
import type { NodeJSON } from "../shared/blockJson";
import { wordFingerprint, wordSourceOf } from "../shared/wordFiles";
import { BLOCK_ID_ATTRIBUTE, WORD_LOOK_ATTRIBUTE, WORD_RUN_MARK } from "./api";
import type { WordStructure } from "./documents/formats/docxImport";

/** One change the look makes: an attribute on an element, or a run's mark on a stretch of text. */
type LookChange =
  | { kind: "attribute"; element: Y.XmlElement; value: string }
  | { kind: "run"; text: Y.XmlText; index: number; length: number; key: string };

export interface LookPlan {
  changes: LookChange[];
  /** Blocks given the look, and those left plain because the User changed them. */
  given: number;
  skipped: number;
}

/** The nodes of the package read again, by the place their source names ("w:p#12"). */
function nodesByPlace(structure: WordStructure): Map<string, NodeJSON> {
  const byPlace = new Map<string, NodeJSON>();
  const visit = (node: NodeJSON) => {
    const id = node.attrs?.[BLOCK_ID_ATTRIBUTE];
    const source = typeof id === "string" ? structure.sources.get(id) : undefined;
    if (source) byPlace.set(source.n, node);
    for (const each of node.content ?? []) visit(each);
  };
  for (const block of structure.blocks) visit(block);
  return byPlace;
}

/** A node's content as the Mind stores it: runs of text together, other nodes each alone. */
function grouped(content: readonly NodeJSON[]): (NodeJSON[] | NodeJSON)[] {
  const groups: (NodeJSON[] | NodeJSON)[] = [];
  for (const node of content) {
    if (node.type === "text") {
      if (!node.text) continue;
      const last = groups.at(-1);
      if (Array.isArray(last)) last.push(node);
      else groups.push([node]);
    } else {
      groups.push(node);
    }
  }
  return groups;
}

/** A text's characters, without its marks. */
const plainText = (text: Y.XmlText): string =>
  (text.toDelta() as { insert: unknown }[])
    .map((run) => (typeof run.insert === "string" ? run.insert : ""))
    .join("");

/** The look's changes for an element as imported, `node` being the same node read again. */
function changesFor(element: Y.XmlElement, node: NodeJSON, into: LookChange[]): boolean {
  if (element.nodeName !== node.type) return false;
  const key = node.attrs?.[WORD_LOOK_ATTRIBUTE];
  const planned: LookChange[] = [];
  if (typeof key === "string") planned.push({ kind: "attribute", element, value: key });
  const children = element.toArray();
  const groups = grouped(node.content ?? []);
  if (children.length !== groups.length) return false;
  for (let index = 0; index < groups.length; index++) {
    const group = groups[index];
    const child = children[index];
    if (Array.isArray(group)) {
      if (!(child instanceof Y.XmlText)) return false;
      const text = group.map((each) => each.text ?? "").join("");
      if (plainText(child) !== text) return false;
      let at = 0;
      for (const each of group) {
        const length = each.text?.length ?? 0;
        const run = each.marks?.find((mark) => mark.type === WORD_RUN_MARK)?.attrs?.k;
        if (typeof run === "string" && length > 0) {
          planned.push({ kind: "run", text: child, index: at, length, key: run });
        }
        at += length;
      }
    } else if (child instanceof Y.XmlElement) {
      if (!changesFor(child, group as NodeJSON, planned)) return false;
    } else {
      return false;
    }
  }
  into.push(...planned);
  return true;
}

/** What giving the look changes in a Mind: worked out, not made. */
export function planLook(doc: Y.Doc, blocks: Y.XmlFragment, structure: WordStructure): LookPlan {
  const byPlace = nodesByPlace(structure);
  const plan: LookPlan = { changes: [], given: 0, skipped: 0 };
  const visit = (element: Y.XmlElement) => {
    const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
    const source = typeof id === "string" ? wordSourceOf(doc, id) : null;
    if (source) {
      const node = byPlace.get(source.n);
      const asImported = wordFingerprint(element) === source.f;
      if (node && asImported && changesFor(element, node, plan.changes)) {
        plan.given++;
        return;
      }
      plan.skipped++;
    }
    for (const child of element.toArray()) {
      if (child instanceof Y.XmlElement) visit(child);
    }
  };
  for (const child of blocks.toArray()) {
    if (child instanceof Y.XmlElement) visit(child);
  }
  return plan;
}

/** Makes the plan's changes. Inside one transaction. */
export function applyLook(plan: LookPlan): void {
  for (const change of plan.changes) {
    if (change.kind === "attribute") {
      change.element.setAttribute(WORD_LOOK_ATTRIBUTE, change.value);
    } else {
      change.text.format(change.index, change.length, { [WORD_RUN_MARK]: { k: change.key } });
    }
  }
}
