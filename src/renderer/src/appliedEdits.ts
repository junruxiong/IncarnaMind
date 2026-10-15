/**
 * Edits the User asked for apply at once (#880), the window's side: the
 * editor that asked takes the core's updates in as its own change, then
 * accepts them, all one step of its Undo.
 */
import type { Editor } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { EditorState } from "@tiptap/pm/state";
import * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  type EditCause,
  type EditWithAiResult,
  type LeftOutPart,
  MIND_CONTENT_FIELD,
} from "../../core/api";
import { aiUseOfAccepted, proposalRequestOf, resolveProposals } from "../../shared/proposals";
import { core } from "./core";
import {
  type AppliedLine,
  appliedLineIn,
  appliedLineKey,
  type KeptLine,
  setAppliedLine,
} from "./editor/appliedEditLine";
import { asOneStep, offerEditorUndo, undoStackOf } from "./editor/editorUndo";
import {
  applyingIn,
  editorDoc,
  pendingIn,
  proposalReviewKey,
  proposalsIn,
  resolveInEditor,
  setApplying,
} from "./editor/proposalReview";
import { blockAt } from "./editor/writingPlace";
import { translateNow as t } from "./i18n";
import { EDIT_FROM_CORE, takeEditUpdates } from "./mindCopies";
import { useAppStore } from "./store";

// An origin the editor's Undo tracks, and its copy sends to the core.
class AcceptingEdit {}

/** The top-level Blocks an edit changed, with its Proposals, between the Blocks around them (null: the Mind's edge). */
interface Changed {
  after: string | null;
  until: string | null;
  blocks: readonly ProseMirrorNode[];
}

export interface Applying {
  requestId: string;
  mindId: string;
  editor: Editor;
  doc: Y.Doc;
  request: string | null;
  step: unknown;
  /** Later steps that belong to it: removing what it left out. */
  more: unknown[];
  changed: Changed | null;
  leftOut: readonly LeftOutPart[];
  stopTaking(): void;
  forget(): void;
}

const latest = new WeakMap<Editor, Applying>();

/** One step of Undo for the whole edit, however long it streams, apart from the User's typing. */
function asEditStep(edit: Applying, change: () => void): void {
  const stack = undoStackOf(edit.editor);
  if (stack && edit.step !== null && stack.undoStack.at(-1) === edit.step) {
    // Yjs joins a change to the last step when it comes soon after that step's last change.
    stack.lastChange = Date.now();
  } else stack?.stopCapturing();
  const before = stack?.undoStack.at(-1);
  try {
    change();
  } finally {
    const after = stack?.undoStack.at(-1);
    if (after !== undefined && (after !== before || after === edit.step)) edit.step = after;
    stack?.stopCapturing();
  }
}

function drawApplying(editor: Editor, request: string, applying: boolean): void {
  const now = applyingIn(editor.state);
  if (now.has(request) === applying) return;
  const next = new Set(now);
  if (applying) next.add(request);
  else next.delete(request);
  setApplying(editor, next);
}

/** Before the edit is asked for, so none of its updates is missed. Null before the editor is bound. */
export function startApplying(editor: Editor, mindId: string, requestId: string): Applying | null {
  const doc = editorDoc(editor);
  const stack = undoStackOf(editor);
  if (!doc || !stack || editor.isDestroyed) return null;
  latest.get(editor)?.forget();
  setAppliedLine(editor, null);
  stack.addTrackedOrigin(EDIT_FROM_CORE);
  stack.addTrackedOrigin(AcceptingEdit);
  const popped = ({ stackItem }: { stackItem: unknown }) => {
    if (stackItem === edit.step && latest.get(editor) === edit) setAppliedLine(editor, null);
  };
  const edit: Applying = {
    requestId,
    mindId,
    editor,
    doc,
    request: null,
    step: null,
    more: [],
    changed: null,
    leftOut: [],
    stopTaking: () => undefined,
    forget: () => {
      edit.stopTaking();
      stack.off("stack-item-popped", popped);
      if (latest.get(editor) === edit) latest.delete(editor);
    },
  };
  stack.on("stack-item-popped", popped);
  edit.stopTaking = takeEditUpdates(requestId, doc, (update: Uint8Array, cause: EditCause) => {
    edit.request ??= cause.request;
    if (cause.apply) drawApplying(editor, cause.request, true);
    asEditStep(edit, () => Y.applyUpdate(doc, update, EDIT_FROM_CORE));
  });
  latest.set(editor, edit);
  return edit;
}

const holds = (state: EditorState, request: string) =>
  proposalsIn(state.doc).some((each) => each.request === request);

/** The Proposals come with the core's updates, before its reply: this is a backstop. */
function whenHeld(editor: Editor, request: string): Promise<boolean> {
  return new Promise((resolve) => {
    if (editor.isDestroyed || holds(editor.state, request)) {
      resolve(!editor.isDestroyed);
      return;
    }
    const done = (held: boolean) => {
      clearTimeout(timer);
      editor.off("transaction", check);
      resolve(held);
    };
    const check = () => {
      if (holds(editor.state, request)) done(true);
    };
    const timer = setTimeout(() => done(holds(editor.state, request)), 5_000);
    editor.on("transaction", check);
  });
}

const idOf = (node: ProseMirrorNode | null): string | null | undefined => {
  if (!node) return null;
  const id: unknown = node.attrs[BLOCK_ID_ATTRIBUTE];
  return typeof id === "string" && id !== "" ? id : undefined;
};

/** For Review changes; null when a Block around them has no id to find them by. */
function changedBlocks(doc: ProseMirrorNode, request: string): Changed | null {
  const indexes = proposalsIn(doc)
    .filter((each) => each.request === request)
    .flatMap((each) => each.ranges)
    .flatMap((range) => [doc.resolve(range.from).index(0), doc.resolve(range.to - 1).index(0)]);
  if (indexes.length === 0) return null;
  const first = Math.min(...indexes);
  const last = Math.max(...indexes);
  const after = idOf(first > 0 ? doc.child(first - 1) : null);
  const until = idOf(last + 1 < doc.childCount ? doc.child(last + 1) : null);
  if (after === undefined || until === undefined) return null;
  const blocks: ProseMirrorNode[] = [];
  for (let index = first; index <= last; index++) blocks.push(doc.child(index));
  return { after, until, blocks };
}

function placeOfChanged(
  doc: ProseMirrorNode,
  changed: Changed,
): { from: number; to: number } | null {
  let from = 0;
  let to = doc.content.size;
  doc.forEach((node, pos) => {
    if (node.attrs[BLOCK_ID_ATTRIBUTE] === changed.after) from = pos + node.nodeSize;
    if (node.attrs[BLOCK_ID_ATTRIBUTE] === changed.until) to = pos;
  });
  return from <= to ? { from, to } : null;
}

function firstBlockIn(
  doc: ProseMirrorNode,
  place: { from: number; to: number },
  until: string | null,
) {
  let found: string | null = null;
  doc.forEach((node, pos) => {
    if (found === null && pos >= place.from && pos < place.to) found = idOf(node) ?? null;
  });
  return found ?? until;
}

function reasonOf(doc: Y.Doc, request: string): string | null {
  const made = proposalRequestOf(doc, request);
  return made?.source === "edit-with-ai" ? (made.reason ?? null) : null;
}

function keptLineOf(leftOut: readonly LeftOutPart[]): KeptLine | null {
  const [first] = leftOut;
  if (!first) return null;
  return {
    blockId: first.blockId,
    count: leftOut.length,
    mixed: new Set(leftOut.map((part) => part.kind)).size > 1,
  };
}

function accept(edit: Applying, request: string): number {
  const blocks = edit.doc.getXmlFragment(MIND_CONTENT_FIELD);
  edit.changed = changedBlocks(edit.editor.state.doc, request);
  let resolved: ReturnType<typeof resolveProposals> = [];
  asEditStep(edit, () => {
    edit.doc.transact(() => {
      resolved = resolveProposals(blocks, (ref) => ref.request === request, "accept");
    }, new AcceptingEdit());
  });
  drawApplying(edit.editor, request, false);
  const entries = aiUseOfAccepted(edit.doc, edit.mindId, resolved, new Date().toISOString());
  if (entries.length > 0) core.recordAiUse(entries).catch(useAppStore.getState().reportError);
  return resolved.length;
}

function apply(edit: Applying, request: string): number {
  const places = accept(edit, request);
  if (places === 0 || edit.editor.isDestroyed) return places;
  const { doc } = edit.editor.state;
  const place = edit.changed ? placeOfChanged(doc, edit.changed) : null;
  const message = appliedMessage(places);
  setAppliedLine(edit.editor, {
    kind: "applied",
    blockId: place && edit.changed ? firstBlockIn(doc, place, edit.changed.until) : null,
    places,
    reason: reasonOf(edit.doc, request),
    reviewable: place !== null,
    kept: keptLineOf(edit.leftOut.filter((part) => blockAt(doc, part.blockId) !== null)),
  });
  offerEditorUndo(edit.editor, message);
  return places;
}

export const appliedMessage = (places: number) =>
  places === 1 ? t("edit.applied.one") : t("edit.applied", { count: places });

/** Applies the edit once it is back, or leaves it waiting when the core held it back. */
export async function finishApplying(
  edit: Applying,
  result: EditWithAiResult,
): Promise<{ applied: number } | "held" | null> {
  if (result.kind !== "proposed") {
    edit.stopTaking();
    if (edit.request) drawApplying(edit.editor, edit.request, false);
    return null;
  }
  edit.request ??= result.request;
  const there = await whenHeld(edit.editor, result.request);
  edit.stopTaking();
  if (!there || edit.editor.isDestroyed) return null;
  edit.leftOut = result.leftOut ?? [];
  if (result.held) {
    drawApplying(edit.editor, result.request, false);
    const changed = changedBlocks(edit.editor.state.doc, result.request);
    setAppliedLine(edit.editor, {
      kind: "held",
      blockId: changed ? (idOf(changed.blocks[0] ?? null) ?? null) : null,
      request: result.request,
    });
    return "held";
  }
  return { applied: apply(edit, result.request) };
}

/** The whole edit, with what was done since to what it left out. */
function undoEdit(editor: Editor): void {
  const edit = latest.get(editor);
  const stack = undoStackOf(editor);
  if (!edit || !stack) return;
  const steps = [edit.step, ...edit.more];
  while (!editor.isDestroyed && steps.includes(stack.undoStack.at(-1) ?? null)) {
    editor.commands.undo();
  }
  setAppliedLine(editor, null);
}

function review(editor: Editor): void {
  const edit = latest.get(editor);
  const changed = edit?.changed;
  const request = edit?.request;
  if (!changed || !request || editor.isDestroyed) return;
  const place = placeOfChanged(editor.state.doc, changed);
  if (!place) return;
  const line: AppliedLine = {
    kind: "review",
    blockId: idOf(changed.blocks[0] ?? null) ?? changed.until,
    request,
  };
  const tr = editor.state.tr.replaceWith(place.from, place.to, changed.blocks);
  tr.setMeta(proposalReviewKey, { shownAgain: true });
  tr.setMeta(appliedLineKey, line);
  asOneStep(editor, () => editor.view.dispatch(tr));
  const first = pendingIn(editor.state).find((each) => each.request === request);
  if (first && !editor.isDestroyed) {
    const at = first.block ? first.from + 1 : first.from;
    editor
      .chain()
      .setTextSelection(Math.min(at, editor.state.doc.content.size))
      .scrollIntoView()
      .run();
  }
}

const ofRequest = (editor: Editor, request: string) =>
  new Set(
    proposalsIn(editor.state.doc)
      .filter((each) => each.request === request)
      .map((each) => each.id),
  );

function resolveShown(editor: Editor, outcome: "accept" | "reject"): void {
  const line = appliedLineIn(editor.state);
  if (!line || line.kind === "applied") return;
  if (line.kind === "held" && outcome === "accept") {
    const edit = latest.get(editor);
    if (edit && edit.request === line.request) {
      apply(edit, line.request);
      return;
    }
  }
  const count = resolveInEditor(editor, ofRequest(editor, line.request), outcome).length;
  setAppliedLine(editor, null);
  if (count > 0 && outcome === "reject") {
    offerEditorUndo(
      editor,
      line.kind === "review"
        ? count === 1
          ? t("edit.review.restored.one")
          : t("edit.review.restored", { count })
        : count === 1
          ? t("proposal.rejected.one")
          : t("proposal.rejected", { count }),
    );
  }
}

function removeKept(editor: Editor): void {
  const edit = latest.get(editor);
  const line = appliedLineIn(editor.state);
  if (!edit || line?.kind !== "applied" || editor.isDestroyed) return;
  const { doc } = editor.state;
  const tr = editor.state.tr;
  const ranges = edit.leftOut
    .flatMap((part) => {
      const block = blockAt(doc, part.blockId);
      if (!block) return [];
      return part.whole
        ? [{ from: block.pos, to: block.pos + block.node.nodeSize }]
        : [{ from: block.pos + 1 + part.from, to: block.pos + 1 + part.to }];
    })
    .sort((a, b) => b.from - a.from);
  for (const range of ranges) tr.deleteRange(range.from, range.to);
  tr.setMeta(appliedLineKey, { ...line, kept: null });
  asOneStep(editor, () => editor.view.dispatch(tr));
  const top = undoStackOf(editor)?.undoStack.at(-1);
  if (top !== undefined && top !== edit.step) edit.more.push(top);
  offerEditorUndo(editor, t("edit.kept.removed"));
}

export function actOnAppliedEdit(editor: Editor, action: string): void {
  if (action === "undo") undoEdit(editor);
  else if (action === "review") review(editor);
  else if (action === "done" || action === "apply") resolveShown(editor, "accept");
  else if (action === "undo-all" || action === "discard") resolveShown(editor, "reject");
  else if (action === "remove-kept") removeKept(editor);
}
