/**
 * Edits the User asked for apply at once (#880), the window's side: the
 * editor that asked takes the core's updates in as its own change, then
 * accepts them, all one step of its Undo.
 */
import type { Editor } from "@tiptap/core";
import { Fragment, type Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { EditorState } from "@tiptap/pm/state";
import { absolutePositionToRelativePosition, ySyncPluginKey } from "@tiptap/y-tiptap";
import * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  type EditCause,
  type EditWithAiResult,
  type LeftOutPart,
  MIND_CONTENT_FIELD,
  type ProposalRef,
} from "../../core/api";
import { aiUseOfAccepted, proposalRequestOf, resolveProposals } from "../../shared/proposals";
import { type ChangeCounts, changeFact } from "./changeFacts";
import { core } from "./core";
import {
  type AppliedAction,
  type AppliedChange,
  appliedLineIn,
  appliedLineKey,
  type KeptLine,
  placeOfChange,
  setAppliedLine,
} from "./editor/appliedEditLine";
import {
  anchorChanges,
  asItWas,
  ownTextRemoved,
  planChanges,
  proposalKindOf,
  writeAccepted,
} from "./editor/arrivedChanges";
import { asOneStep, offerEditorUndo, type UndoStep, undoStackOf } from "./editor/editorUndo";
import {
  applyingIn,
  editorDoc,
  type PendingProposal,
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

export interface Applying {
  requestId: string;
  mindId: string;
  editor: Editor;
  doc: Y.Doc;
  request: string | null;
  steps: UndoStep[];
  /** Later steps that belong to it: restoring a change, removing what it left out. */
  more: UndoStep[];
  leftOut: readonly LeftOutPart[];
  stopTaking(): void;
  forget(): void;
}

const latest = new WeakMap<Editor, Applying>();

function asEditStep(edit: Applying, change: () => void): void {
  const stack = undoStackOf(edit.editor);
  const last = edit.steps.at(-1);
  if (stack && last && stack.undoStack.at(-1) === last) {
    // Yjs joins a change to the last step when it comes soon after that step's last change.
    stack.lastChange = Date.now();
  } else stack?.stopCapturing();
  const before = stack?.undoStack.at(-1);
  try {
    change();
  } finally {
    const after = stack?.undoStack.at(-1);
    if (after && after !== before) edit.steps.push(after);
    stack?.stopCapturing();
  }
}

function foldSteps(edit: Applying): void {
  const stack = undoStackOf(edit.editor);
  const last = edit.steps.at(-1);
  const earlier = edit.steps.slice(0, -1);
  if (!stack || !last || earlier.length === 0) return;
  last.insertions = Y.mergeDeleteSets([...earlier.map((step) => step.insertions), last.insertions]);
  last.deletions = Y.mergeDeleteSets([...earlier.map((step) => step.deletions), last.deletions]);
  for (const step of earlier) {
    const index = stack.undoStack.indexOf(step);
    if (index >= 0) stack.undoStack.splice(index, 1);
  }
  edit.steps = [last];
}

/** The core took back what a failed edit wrote: its steps would undo nothing. */
function dropSteps(edit: Applying): void {
  const stack = undoStackOf(edit.editor);
  for (const step of edit.steps) {
    const index = stack?.undoStack.indexOf(step) ?? -1;
    if (index >= 0) stack?.undoStack.splice(index, 1);
  }
  edit.steps = [];
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
  const edit = beginEdit(editor, mindId, requestId);
  if (!edit) return null;
  edit.stopTaking = takeEditUpdates(requestId, edit.doc, (update: Uint8Array, cause: EditCause) => {
    edit.request ??= cause.request;
    if (cause.apply) drawApplying(editor, cause.request, true);
    asEditStep(edit, () => Y.applyUpdate(edit.doc, update, EDIT_FROM_CORE));
  });
  return edit;
}

/** The editor's record of an edit applied in it, the latest, which its line acts on. */
function beginEdit(editor: Editor, mindId: string, requestId: string): Applying | null {
  const doc = editorDoc(editor);
  const stack = undoStackOf(editor);
  if (!doc || !stack || editor.isDestroyed) return null;
  latest.get(editor)?.forget();
  setAppliedLine(editor, null);
  stack.addTrackedOrigin(EDIT_FROM_CORE);
  stack.addTrackedOrigin(AcceptingEdit);
  const popped = ({ stackItem }: { stackItem: UndoStep }) => {
    if (edit.steps.includes(stackItem) && latest.get(editor) === edit) setAppliedLine(editor, null);
  };
  const edit: Applying = {
    requestId,
    mindId,
    editor,
    doc,
    request: null,
    steps: [],
    more: [],
    leftOut: [],
    stopTaking: () => undefined,
    forget: () => {
      edit.stopTaking();
      stack.off("stack-item-popped", popped);
      if (latest.get(editor) === edit) latest.delete(editor);
    },
  };
  stack.on("stack-item-popped", popped);
  latest.set(editor, edit);
  return edit;
}

const ofRequest = (state: EditorState, request: string) =>
  proposalsIn(state.doc).filter((each) => each.request === request);

/** The Proposals come with the core's updates, before its reply: this is a backstop. */
function whenHeld(editor: Editor, request: string): Promise<boolean> {
  return new Promise((resolve) => {
    if (editor.isDestroyed || ofRequest(editor.state, request).length > 0) {
      resolve(!editor.isDestroyed);
      return;
    }
    const done = (held: boolean) => {
      clearTimeout(timer);
      editor.off("transaction", check);
      resolve(held);
    };
    const check = () => {
      if (ofRequest(editor.state, request).length > 0) done(true);
    };
    const timer = setTimeout(() => done(ofRequest(editor.state, request).length > 0), 5_000);
    editor.on("transaction", check);
  });
}

/** Where each of the request's changes will be once accepted, and what it takes out, read before accepting. */
function changesOf(state: EditorState, request: string): AppliedChange[] {
  const binding = ySyncPluginKey.getState(state)?.binding;
  if (!binding) return [];
  const relative = (pos: number): unknown =>
    absolutePositionToRelativePosition(pos, binding.type, binding.mapping);
  const { doc } = state;
  return ofRequest(state, request).map((proposal: PendingProposal): AppliedChange => {
    if (proposal.block === "insert") {
      return {
        from: relative(proposal.from),
        // Inside the new Block's end: at the Mind's end, the end itself would take in what comes after it.
        to: relative(proposal.to - 1),
        removed: Fragment.empty,
        block: "added",
      };
    }
    if (proposal.block === "delete") {
      const node = doc.nodeAt(proposal.from);
      return {
        from: relative(proposal.from),
        to: relative(proposal.from),
        removed: node ? Fragment.from(asItWas(node)) : Fragment.empty,
        block: "removed",
      };
    }
    const removed: ProseMirrorNode[] = [];
    const added = { from: -1, to: -1 };
    let end = proposal.from;
    for (const range of proposal.ranges) {
      doc.nodesBetween(range.from, range.to, (node, pos) => {
        if (node.isBlock) return true;
        const from = Math.max(pos, range.from);
        const to = Math.min(pos + node.nodeSize, range.to);
        const kind = proposalKindOf(node);
        if (kind === "delete") {
          removed.push(asItWas(node.isText ? node.cut(from - pos, to - pos) : node));
          end = to;
        } else if (kind === "insert") {
          if (added.from < 0) added.from = from;
          added.to = to;
        }
        return false;
      });
    }
    const words = added.from < 0 ? { from: end, to: end } : added;
    return {
      from: relative(words.from),
      to: relative(words.to),
      removed: Fragment.from(removed),
      block: null,
    };
  });
}

/** The top-level Block the first change is in, after it applied; null at the Mind's end. */
function firstBlockOf(state: EditorState, changes: readonly AppliedChange[]): string | null {
  for (const change of changes) {
    const place = placeOfChange(state, change);
    if (!place) continue;
    const index = state.doc.resolve(Math.min(place.from, state.doc.content.size)).index(0);
    if (index >= state.doc.childCount) return null;
    const id: unknown = state.doc.child(index).attrs[BLOCK_ID_ATTRIBUTE];
    return typeof id === "string" ? id : null;
  }
  return null;
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
    mixed: !leftOut.every((part) => part.kind === "paragraph"),
  };
}

export const appliedMessage = (places: number) =>
  places === 1 ? t("edit.applied.one") : t("edit.applied", { count: places });

/** Accepts the edit's Proposals as part of its Undo step, and shows the line over it. Returns how many places changed. */
function apply(edit: Applying, request: string): number {
  const { editor } = edit;
  const changes = changesOf(editor.state, request);
  const blocks = edit.doc.getXmlFragment(MIND_CONTENT_FIELD);
  let resolved: ReturnType<typeof resolveProposals> = [];
  asEditStep(edit, () => {
    edit.doc.transact(() => {
      resolved = resolveProposals(blocks, (ref: ProposalRef) => ref.request === request, "accept");
    }, new AcceptingEdit());
  });
  foldSteps(edit);
  drawApplying(editor, request, false);
  const entries = aiUseOfAccepted(edit.doc, edit.mindId, resolved, new Date().toISOString());
  if (entries.length > 0) core.recordAiUse(entries).catch(useAppStore.getState().reportError);
  const places = resolved.length;
  if (places === 0 || editor.isDestroyed) return places;
  const { doc } = editor.state;
  setAppliedLine(editor, {
    kind: "applied",
    blockId: firstBlockOf(editor.state, changes),
    places,
    reason: reasonOf(edit.doc, request),
    changes,
    kept: keptLineOf(edit.leftOut.filter((part) => blockAt(doc, part.blockId) !== null)),
  });
  offerEditorUndo(editor, appliedMessage(places));
  return places;
}

// Setting the assistant's changes aside before writing them as the editor's own: no Undo tracks it.
class SettingAside {}

/** Most of the User's own text in the Blocks they touch: the assistant's changes then wait (D224). */
const MOST_OF_YOUR_TEXT = 0.5;

/** A translation rewrites all of the text by asking, so it never waits, as Edit with AI's never does. */
const TRANSLATION =
  /\btranslat|\bin(?:to)? (?:chinese|english)\b|翻译|译成|(?:改成|用)(?:中文|英文)/i;

function translates(editor: Editor, requests: ReadonlySet<string>): boolean {
  const doc = editorDoc(editor);
  return (
    doc !== null &&
    [...requests].every((request) => {
      const made = proposalRequestOf(doc, request);
      return (
        made?.source === "edit-with-ai" &&
        (made.action === "translate" || TRANSLATION.test(made.instruction ?? ""))
      );
    })
  );
}

/**
 * Applies the assistant's changes the User asked for (#880, D224), now that
 * they are all in the Mind, as one step of the editor's Undo, with the line
 * over them saying what `facts` counts; or, when they would rewrite most of
 * the User's own text, leaves them waiting with the line that says so.
 */
export function applyArrived(
  editor: Editor,
  mindId: string,
  requests: ReadonlySet<string>,
  facts: ChangeCounts,
  { holds = true }: { holds?: boolean } = {},
): "applied" | "held" | null {
  const first = proposalsIn(editor.state.doc).find((each) => requests.has(each.request));
  if (!first || editor.isDestroyed) return null;
  for (const request of requests) drawApplying(editor, request, false);
  if (
    holds &&
    !translates(editor, requests) &&
    ownTextRemoved(editor.state, requests) > MOST_OF_YOUR_TEXT
  ) {
    setAppliedLine(editor, {
      kind: "held",
      blockId: topBlockAt(editor.state, first.from),
      request: first.request,
      asked: { mindId, requests: [...requests], facts },
    });
    return "held";
  }
  const edit = beginEdit(editor, mindId, `assistant:${first.request}`);
  if (!edit) return null;
  const planned = planChanges(editor.state, requests);
  let resolved: ReturnType<typeof writeAccepted> = [];
  asEditStep(edit, () => {
    resolved = writeAccepted(edit.doc, requests, new SettingAside(), new AcceptingEdit());
  });
  const entries = aiUseOfAccepted(edit.doc, mindId, resolved, new Date().toISOString());
  if (entries.length > 0) core.recordAiUse(entries).catch(useAppStore.getState().reportError);
  if (editor.isDestroyed) return "applied";
  const changes = anchorChanges(editor.state, planned);
  const message = changeFact(t, facts, null);
  setAppliedLine(editor, {
    kind: "applied",
    blockId: firstBlockOf(editor.state, changes),
    places: resolved.length,
    facts,
    reason: null,
    changes,
    kept: null,
  });
  offerEditorUndo(editor, message);
  return "applied";
}

const topBlockAt = (state: EditorState, pos: number): string | null => {
  const id: unknown = state.doc.maybeChild(state.doc.resolve(pos).index(0))?.attrs[
    BLOCK_ID_ATTRIBUTE
  ];
  return typeof id === "string" ? id : null;
};

/** Applies the edit once it is back, or leaves it waiting when the core held it back. */
export async function finishApplying(
  edit: Applying,
  result: EditWithAiResult,
): Promise<{ applied: number } | "held" | null> {
  if (result.kind !== "proposed") {
    edit.stopTaking();
    if (edit.request) drawApplying(edit.editor, edit.request, false);
    dropSteps(edit);
    return null;
  }
  edit.request ??= result.request;
  const there = await whenHeld(edit.editor, result.request);
  edit.stopTaking();
  if (!there || edit.editor.isDestroyed) {
    drawApplying(edit.editor, result.request, false);
    return null;
  }
  edit.leftOut = result.leftOut ?? [];
  if (result.held) {
    drawApplying(edit.editor, result.request, false);
    const first = ofRequest(edit.editor.state, result.request)[0];
    const index = first ? edit.editor.state.doc.resolve(first.from).index(0) : -1;
    const id: unknown =
      index >= 0 ? edit.editor.state.doc.maybeChild(index)?.attrs[BLOCK_ID_ATTRIBUTE] : null;
    setAppliedLine(edit.editor, {
      kind: "held",
      blockId: typeof id === "string" ? id : null,
      request: result.request,
    });
    return "held";
  }
  return { applied: apply(edit, result.request) };
}

function undoEdit(editor: Editor): void {
  const edit = latest.get(editor);
  const stack = undoStackOf(editor);
  if (!edit || !stack || edit.steps.length === 0) return;
  const steps = [...edit.steps, ...edit.more];
  let top = stack.undoStack.at(-1);
  while (!editor.isDestroyed && top !== undefined && steps.includes(top)) {
    editor.commands.undo();
    if (stack.undoStack.at(-1) === top) break;
    top = stack.undoStack.at(-1);
  }
  setAppliedLine(editor, null);
}

function review(editor: Editor): void {
  const line = appliedLineIn(editor.state);
  if (line?.kind !== "applied" || line.changes.length === 0 || editor.isDestroyed) return;
  setAppliedLine(editor, { kind: "review", blockId: line.blockId, changes: line.changes });
  const first = line.changes
    .map((change) => placeOfChange(editor.state, change))
    .find((place) => place !== null);
  if (first) editor.chain().setTextSelection(first.from).scrollIntoView().run();
}

/** The change shown again whose words the cursor is in, by its place in the line's list. */
export function reviewedChangeAt(state: EditorState): number | null {
  const line = appliedLineIn(state);
  if (line?.kind !== "review") return null;
  const { from } = state.selection;
  const index = line.changes.findIndex((change) => {
    const place = placeOfChange(state, change);
    return place !== null && place.from <= from && from <= place.to;
  });
  return index >= 0 ? index : null;
}

/** Restore original: one change's old words back in place of its new ones, one step of Undo. */
export function restoreChange(editor: Editor, index: number): void {
  const line = appliedLineIn(editor.state);
  const change = line?.kind === "review" ? line.changes[index] : undefined;
  const place = change ? placeOfChange(editor.state, change) : null;
  if (line?.kind !== "review" || !change || !place || editor.isDestroyed) return;
  const tr = editor.state.tr.replaceWith(place.from, place.to, change.removed);
  tr.setMeta(appliedLineKey, {
    ...line,
    changes: line.changes.filter((_, each) => each !== index),
  });
  asOneStep(editor, () => editor.view.dispatch(tr));
  const top = undoStackOf(editor)?.undoStack.at(-1);
  if (top) latest.get(editor)?.more.push(top);
  // Placed again now that the change has reached the Mind's document.
  const now = appliedLineIn(editor.state);
  if (now) setAppliedLine(editor, { ...now });
  offerEditorUndo(editor, t("edit.review.restored.one"));
}

function resolveHeld(editor: Editor, outcome: "accept" | "reject"): void {
  const line = appliedLineIn(editor.state);
  if (line?.kind !== "held") return;
  if (line.asked && outcome === "accept") {
    const { mindId, requests, facts } = line.asked;
    setAppliedLine(editor, null);
    applyArrived(editor, mindId, new Set(requests), facts, { holds: false });
    return;
  }
  const edit = latest.get(editor);
  if (outcome === "accept" && edit?.request === line.request) {
    apply(edit, line.request);
    return;
  }
  const asked = new Set(line.asked?.requests ?? [line.request]);
  const ids = new Set(
    proposalsIn(editor.state.doc)
      .filter((each) => asked.has(each.request))
      .map((each) => each.id),
  );
  const count = resolveInEditor(editor, ids, outcome).length;
  setAppliedLine(editor, null);
  if (count > 0) {
    offerEditorUndo(
      editor,
      count === 1 ? t("proposal.rejected.one") : t("proposal.rejected", { count }),
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
  if (top && !edit.steps.includes(top)) edit.more.push(top);
  offerEditorUndo(editor, t("edit.kept.removed"));
}

export function actOnAppliedEdit(editor: Editor, action: AppliedAction): void {
  switch (action) {
    case "undo":
    case "undo-all":
      undoEdit(editor);
      return;
    case "review":
      review(editor);
      return;
    case "done":
      setAppliedLine(editor, null);
      return;
    case "apply":
      resolveHeld(editor, "accept");
      return;
    case "discard":
      resolveHeld(editor, "reject");
      return;
    case "remove-kept":
      removeKept(editor);
      return;
  }
}
