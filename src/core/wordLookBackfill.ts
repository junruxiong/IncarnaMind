/**
 * Gives the look only to Blocks whose fingerprint is still the one recorded
 * at import. Every change is planned and checked before any is made.
 */
import * as Y from "yjs";
import type { NodeJSON } from "../shared/blockJson";
import {
  setWordSource,
  wordFingerprint,
  wordFingerprintBeforeLook,
  wordSourceOf,
} from "../shared/wordFiles";
import { runLookOf } from "../shared/wordLook";
import {
  BLOCK_ID_ATTRIBUTE,
  MIND_WORD_LOOK_FIELD,
  WORD_LOOK_ATTRIBUTE,
  WORD_RUN_MARK,
  type WordSource,
} from "./api";
import type { WordStructure } from "./documents/formats/docxImport";

type LookChange =
  | { kind: "attribute"; element: Y.XmlElement; value: string }
  | { kind: "run"; text: Y.XmlText; index: number; length: number; key: string }
  | { kind: "source"; id: string; source: WordSource };

export interface LookPlan {
  changes: LookChange[];
  given: number;
  editedSinceImport: number;
}

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

const plainText = (text: Y.XmlText): string =>
  (text.toDelta() as { insert: unknown }[])
    .map((run) => (typeof run.insert === "string" ? run.insert : ""))
    .join("");

function changesFor(element: Y.XmlElement, node: NodeJSON): LookChange[] | null {
  if (element.nodeName !== node.type) return null;
  const changes: LookChange[] = [];
  const key = node.attrs?.[WORD_LOOK_ATTRIBUTE];
  if (typeof key === "string") changes.push({ kind: "attribute", element, value: key });
  const children = element.toArray();
  const groups = grouped(node.content ?? []);
  if (children.length !== groups.length) return null;
  for (const [index, group] of groups.entries()) {
    const child = children[index];
    if (Array.isArray(group)) {
      if (!(child instanceof Y.XmlText)) return null;
      const text = group.map((each) => each.text ?? "").join("");
      if (plainText(child) !== text) return null;
      let at = 0;
      for (const each of group) {
        const length = each.text?.length ?? 0;
        const run = runLookOf(each);
        if (run !== null && length > 0) {
          changes.push({ kind: "run", text: child, index: at, length, key: run });
        }
        at += length;
      }
    } else {
      if (!(child instanceof Y.XmlElement)) return null;
      const inner = changesFor(child, group);
      if (!inner) return null;
      changes.push(...inner);
    }
  }
  return changes;
}

const asImported = (element: Y.XmlElement, source: WordSource) =>
  wordFingerprint(element) === source.f || wordFingerprintBeforeLook(element) === source.f;

export function planLook(doc: Y.Doc, blocks: Y.XmlFragment, structure: WordStructure): LookPlan {
  const byPlace = nodesByPlace(structure);
  const plan: LookPlan = { changes: [], given: 0, editedSinceImport: 0 };
  const sourced = (element: Y.XmlElement) => {
    const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
    const source = typeof id === "string" ? wordSourceOf(doc, id) : null;
    return id && source ? { id, source } : null;
  };
  const look = (element: Y.XmlElement) => {
    const found = sourced(element);
    if (found) {
      const node = byPlace.get(found.source.n);
      const changes = node && asImported(element, found.source) ? changesFor(element, node) : null;
      if (changes) {
        plan.changes.push(...changes);
        plan.given++;
        return;
      }
      plan.editedSinceImport++;
    }
    for (const child of element.toArray()) {
      if (child instanceof Y.XmlElement) look(child);
    }
  };
  // Fingerprints recorded before the look rejoined text are recorded again, so export still finds the Block unchanged.
  const recordAgain = (element: Y.XmlElement) => {
    const found = sourced(element);
    if (found && wordFingerprintBeforeLook(element) === found.source.f) {
      const f = wordFingerprint(element);
      if (f !== found.source.f) {
        plan.changes.push({ kind: "source", id: found.id, source: { ...found.source, f } });
      }
    }
    for (const child of element.toArray()) {
      if (child instanceof Y.XmlElement) recordAgain(child);
    }
  };
  for (const child of blocks.toArray()) {
    if (!(child instanceof Y.XmlElement)) continue;
    look(child);
    recordAgain(child);
  }
  return plan;
}

/** Takes away a look an earlier version gave, so keys it named don't outlive it. */
export function clearLook(doc: Y.Doc, blocks: Y.XmlFragment): void {
  const clear = (element: Y.XmlElement) => {
    if (element.getAttribute(WORD_LOOK_ATTRIBUTE) !== undefined) {
      element.removeAttribute(WORD_LOOK_ATTRIBUTE);
    }
    for (const child of element.toArray()) {
      if (child instanceof Y.XmlElement) clear(child);
      else if (child instanceof Y.XmlText) {
        let at = 0;
        for (const run of child.toDelta() as { insert: unknown; attributes?: object }[]) {
          const length = typeof run.insert === "string" ? run.insert.length : 1;
          if (run.attributes && WORD_RUN_MARK in run.attributes) {
            child.format(at, length, { [WORD_RUN_MARK]: null });
          }
          at += length;
        }
      }
    }
  };
  for (const child of blocks.toArray()) {
    if (child instanceof Y.XmlElement) clear(child);
  }
  doc.getMap(MIND_WORD_LOOK_FIELD).clear();
}

export function applyLook(doc: Y.Doc, plan: LookPlan): void {
  for (const change of plan.changes) {
    if (change.kind === "attribute") {
      change.element.setAttribute(WORD_LOOK_ATTRIBUTE, change.value);
    } else if (change.kind === "run") {
      change.text.format(change.index, change.length, { [WORD_RUN_MARK]: { k: change.key } });
    } else {
      setWordSource(doc, change.id, change.source);
    }
  }
}
