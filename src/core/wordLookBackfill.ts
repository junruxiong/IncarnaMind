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
import {
  BLOCK_ID_ATTRIBUTE,
  MIND_WORD_LOOK_FIELD,
  WORD_LOOK_ATTRIBUTE,
  WORD_RUN_MARK,
  type WordSource,
} from "./api";
import { nodesByPlace, type WordStructure } from "./documents/formats/docxImport";
import { CONDITIONAL_FORMATTING_VERSION } from "./documents/formats/docxLook";

/** The look's mark, and formatting the import reads with styles resolved. */
const GIVEN_MARKS: ReadonlySet<string> = new Set([
  WORD_RUN_MARK,
  "bold",
  "italic",
  "underline",
  "strike",
]);

type LookChange =
  | { kind: "attribute"; element: Y.XmlElement; value: string }
  | {
      kind: "marks";
      text: Y.XmlText;
      index: number;
      length: number;
      marks: Record<string, Record<string, unknown>>;
    }
  | { kind: "source"; id: string; source: WordSource }
  | { kind: "fingerprint"; element: Y.XmlElement };

export interface LookPlan {
  changes: LookChange[];
  given: number;
  editedSinceImport: number;
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
        const marks = Object.fromEntries(
          (each.marks ?? [])
            .filter((mark) => GIVEN_MARKS.has(mark.type))
            .map((mark) => [mark.type, mark.attrs ?? {}]),
        );
        if (Object.keys(marks).length > 0 && length > 0) {
          changes.push({ kind: "marks", text: child, index: at, length, marks });
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

const span = (value: unknown) => (typeof value === "number" ? value : 1);

function looksByPosition(table: Y.XmlElement, node: NodeJSON): LookChange[] {
  const rows = table.toArray();
  const imported = node.content ?? [];
  if (table.nodeName !== "table" || rows.length !== imported.length) return [];
  const changes: LookChange[] = [];
  const give = (element: Y.XmlElement, from: NodeJSON) => {
    const key = from.attrs?.[WORD_LOOK_ATTRIBUTE];
    if (typeof key === "string") changes.push({ kind: "attribute", element, value: key });
  };
  give(table, node);
  for (const [index, row] of rows.entries()) {
    const cells = row instanceof Y.XmlElement ? row.toArray() : [];
    const importedCells = imported[index]?.content ?? [];
    if (cells.length !== importedCells.length) return [];
    for (const [at, cell] of cells.entries()) {
      const from = importedCells[at];
      if (
        !(cell instanceof Y.XmlElement) ||
        !from ||
        cell.nodeName !== from.type ||
        span(cell.getAttribute("colspan")) !== span(from.attrs?.colspan) ||
        span(cell.getAttribute("rowspan")) !== span(from.attrs?.rowspan)
      ) {
        return [];
      }
      give(cell, from);
    }
  }
  return changes;
}

const TOGGLES = ["bold", "italic", "underline", "strike"] as const;
const CELLS = new Set(["tableCell", "tableHeader"]);

type Delta = { insert: unknown; attributes?: Record<string, unknown> }[];

// A look read before conditional formatting never gave these toggles, so export would write each one off.
function conditionalToggles(paragraph: Y.XmlElement, node: NodeJSON): LookChange[] {
  const cell = paragraph.parent;
  if (!(cell instanceof Y.XmlElement) || !CELLS.has(cell.nodeName)) return [];
  const imported = (node.content ?? []).filter((each) => each.type === "text" && each.text);
  const runs = paragraph.toArray().flatMap((child) => (child instanceof Y.XmlText ? [child] : []));
  const deltas = runs.map((run) => run.toDelta() as Delta);
  const changes: LookChange[] = [];
  for (const toggle of TOGGLES) {
    const mark = imported[0]?.marks?.find((each) => each.type === toggle);
    if (!mark || !imported.every((each) => each.marks?.some((m) => m.type === toggle))) continue;
    if (deltas.some((delta) => delta.some((each) => each.attributes?.[toggle] !== undefined))) {
      continue;
    }
    for (const [index, run] of runs.entries()) {
      let at = 0;
      for (const each of deltas[index] ?? []) {
        const length = typeof each.insert === "string" ? each.insert.length : 1;
        if (typeof each.insert === "string") {
          changes.push({
            kind: "marks",
            text: run,
            index: at,
            length,
            marks: { [toggle]: mark.attrs ?? {} },
          });
        }
        at += length;
      }
    }
  }
  return changes;
}

const asImported = (element: Y.XmlElement, source: WordSource) =>
  wordFingerprint(element) === source.f || wordFingerprintBeforeLook(element) === source.f;

/** `previous`: the version of the look the Mind had, null for none. */
export function planLook(
  doc: Y.Doc,
  blocks: Y.XmlFragment,
  structure: WordStructure,
  previous: number | null,
): LookPlan {
  const byPlace = nodesByPlace(structure);
  const plan: LookPlan = { changes: [], given: 0, editedSinceImport: 0 };
  const given = new Set<Y.XmlElement>();
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
        // Resolved formatting is content, so the fingerprints are taken again once it's given.
        plan.changes.push(...changes, { kind: "fingerprint", element });
        given.add(element);
        plan.given++;
        return;
      }
      plan.editedSinceImport++;
      if (node?.type === "table") plan.changes.push(...looksByPosition(element, node));
      if (node && (previous ?? 0) < CONDITIONAL_FORMATTING_VERSION) {
        plan.changes.push(...conditionalToggles(element, node));
      }
    }
    for (const child of element.toArray()) {
      if (child instanceof Y.XmlElement) look(child);
    }
  };
  // Fingerprints recorded before the look rejoined text are recorded again, so export still finds the Block unchanged.
  const recordAgain = (element: Y.XmlElement) => {
    if (given.has(element)) return;
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

function fingerprintAgain(doc: Y.Doc, element: Y.XmlElement): void {
  const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
  const source = typeof id === "string" ? wordSourceOf(doc, id) : null;
  if (id && source) setWordSource(doc, id, { ...source, f: wordFingerprint(element) });
  for (const child of element.toArray()) {
    if (child instanceof Y.XmlElement) fingerprintAgain(doc, child);
  }
}

export function applyLook(doc: Y.Doc, plan: LookPlan): void {
  for (const change of plan.changes) {
    if (change.kind === "attribute") {
      change.element.setAttribute(WORD_LOOK_ATTRIBUTE, change.value);
    } else if (change.kind === "marks") {
      change.text.format(change.index, change.length, change.marks);
    } else if (change.kind === "source") {
      setWordSource(doc, change.id, change.source);
    } else {
      fingerprintAgain(doc, change.element);
    }
  }
}
