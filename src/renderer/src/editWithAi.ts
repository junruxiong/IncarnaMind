/**
 * Edit with AI (#93), the window's side: which text a selection is (as the
 * core reads it, `TextPlace`), asking the core for the change, the
 * composer's way of saying what to change, and accepting or rejecting the
 * Proposals that come back, with Undo.
 */
import type { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import type * as Y from "yjs";
import { create } from "zustand";
import type { ChatReadiness, EditAction, EditWithAiResult } from "../../core/api";
import { mindModelOf } from "../../shared/mindModel";
import { aiUseOfAccepted } from "../../shared/proposals";
import { core } from "./core";
import { selectionForEdit } from "./editor/editSelection";
import {
  editorDoc,
  editTarget,
  type PendingProposal,
  pendingIn,
  resolveInEditor,
  setEditTarget,
} from "./editor/proposalReview";
import { translateNow } from "./i18n";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

/** The languages Translate… offers: each in its own name, and as the model is told it. */
export const TRANSLATE_LANGUAGES: readonly { id: string; name: string; prompt: string }[] = [
  { id: "en", name: "English", prompt: "English" },
  { id: "zh-Hans", name: "简体中文", prompt: "Simplified Chinese (简体中文)" },
  { id: "zh-Hant", name: "繁體中文", prompt: "Traditional Chinese (繁體中文)" },
  { id: "ja", name: "日本語", prompt: "Japanese (日本語)" },
  { id: "ko", name: "한국어", prompt: "Korean (한국어)" },
  { id: "fr", name: "Français", prompt: "French (Français)" },
  { id: "de", name: "Deutsch", prompt: "German (Deutsch)" },
  { id: "es", name: "Español", prompt: "Spanish (Español)" },
];

/** Why an Edit with AI request gave no Proposals, for the line in the composer. */
export type EditProblem =
  | { kind: "unchanged" | "changed" | "has-proposals" | "too-long" | "local-mode" }
  | { kind: "not-ready"; readiness: Extract<ChatReadiness, { ready: false }> }
  | { kind: "failed"; message: string; retry: () => void };

interface Pending {
  requestId: string;
  action: EditAction;
  /** The language, for "translate". */
  language: string | null;
}

interface EditsState {
  /** By Mind: the request the AI is working on. */
  pending: Readonly<Record<string, Pending>>;
  /** By Mind: why the last request gave nothing. */
  problems: Readonly<Record<string, EditProblem>>;
  /** By Mind: the selection the composer is saying what to change in, quoted. */
  editing: Readonly<Record<string, { quote: string }>>;
  /** By Mind: what a screen reader is told about Proposals, with a count to say it again. */
  spoken: Readonly<Record<string, { text: string; at: number }>>;
}

export const useEdits = create<EditsState>()(() => ({
  pending: {},
  problems: {},
  editing: {},
  spoken: {},
}));

const set = <K extends keyof EditsState>(
  key: K,
  mindId: string,
  value: EditsState[K][string] | null,
) =>
  useEdits.setState((state) => {
    const next = { ...state[key] } as Record<string, EditsState[K][string]>;
    if (value === null) delete next[mindId];
    else next[mindId] = value;
    return { [key]: next } as Partial<EditsState>;
  });

const t = translateNow;

let spokenAt = 0;
/** Tells a screen reader, through the Mind's live region (see `ProposalBars`). */
export function speak(mindId: string, text: string): void {
  spokenAt += 1;
  set("spoken", mindId, { text, at: spokenAt });
}

export const dismissEditProblem = (mindId: string) => set("problems", mindId, null);

/** A selection shown short, for the composer's chip. */
function quoteOf(text: string): string {
  const flat = text.replace(/￼/g, "").replace(/\s+/g, " ").trim();
  const chars = [...flat];
  return chars.length > 48 ? `${chars.slice(0, 47).join("")}…` : flat;
}

/** Waits until the editor shows the request's Proposals; the first, or null after a while. */
function whenShown(editor: Editor, request: string): Promise<PendingProposal | null> {
  const first = () => pendingIn(editor.state).find((each) => each.request === request) ?? null;
  return new Promise((resolve) => {
    const now = first();
    if (now || editor.isDestroyed) {
      resolve(now);
      return;
    }
    const done = (value: PendingProposal | null) => {
      clearTimeout(timer);
      editor.off("transaction", check);
      resolve(value);
    };
    const check = () => {
      const found = first();
      if (found) done(found);
    };
    // The Proposals come with the core's update, which comes before its reply; this is a backstop.
    const timer = setTimeout(() => done(first()), 5_000);
    editor.on("transaction", check);
  });
}

/**
 * Asks the core to change the selected text (or `range`) as `action` says.
 * While the AI works the text is marked and the composer says so, with Stop;
 * the selection closes at its end, so the User can write on. The Proposals
 * that come back take the cursor, if it is still where it was, so their bar
 * shows; anything else is one line in the composer.
 */
export async function requestEdit(
  editor: Editor,
  mindId: string,
  doc: Y.Doc,
  action: EditAction,
  instruction?: string,
  range?: { from: number; to: number },
): Promise<void> {
  const selection = selectionForEdit(editor.state, range);
  if (!selection || editor.isDestroyed) return;
  const requestId = crypto.randomUUID();
  set("problems", mindId, null);
  set("editing", mindId, null);
  set("pending", mindId, {
    requestId,
    action,
    language: action === "translate" ? (instruction ?? null) : null,
  });
  setEditTarget(editor, selection.range);
  const { selection: now } = editor.state;
  if (now.from === selection.range.from && now.to === selection.range.to) {
    editor.view.dispatch(
      editor.state.tr.setSelection(TextSelection.create(editor.state.doc, selection.range.to)),
    );
  }
  const caret = editor.state.selection.head;

  let result: EditWithAiResult;
  try {
    result = await core.editWithAi({
      requestId,
      mindId,
      from: selection.from,
      to: selection.to,
      text: selection.text,
      action,
      ...(instruction !== undefined && { instruction }),
      model: mindModelOf(doc),
    });
  } catch (error) {
    result = {
      kind: "failed",
      error: { kind: "unknown", message: error instanceof Error ? error.message : String(error) },
    };
  }
  const edits = useEdits.getState();
  if (edits.pending[mindId]?.requestId !== requestId) {
    // Stopped: nothing waits on the text any more, unless another request or the composer does.
    if (!edits.pending[mindId] && !edits.editing[mindId] && !editor.isDestroyed) {
      setEditTarget(editor, null);
    }
    return;
  }
  set("pending", mindId, null);
  const target = editor.isDestroyed ? null : editTarget(editor.state);
  if (!editor.isDestroyed) setEditTarget(editor, null);

  if (result.kind === "proposed") {
    const first = await whenShown(editor, result.request);
    speak(
      mindId,
      result.proposals === 1
        ? t("proposal.arrived.one")
        : t("proposal.arrived", { count: result.proposals }),
    );
    // Into the first change, unless the User has moved on meanwhile.
    const { selection: at } = editor.state;
    if (first && !editor.isDestroyed && at.empty && (at.head === caret || at.head === target?.to)) {
      const pos = first.block ? first.from + 1 : first.from;
      editor
        .chain()
        .setTextSelection(Math.min(pos, editor.state.doc.content.size))
        .scrollIntoView()
        .run();
    }
    return;
  }
  if (result.kind === "cancelled") return;
  set(
    "problems",
    mindId,
    result.kind === "failed"
      ? {
          kind: "failed",
          message: result.error.message,
          retry: () =>
            void requestEdit(editor, mindId, doc, action, instruction, range ?? selection.range),
        }
      : result,
  );
}

/** Stops the request the AI is working on in a Mind; the text stays as it was. */
export function stopEdit(mindId: string): void {
  const pending = useEdits.getState().pending[mindId];
  if (!pending) return;
  set("pending", mindId, null);
  void core.cancelEdit(pending.requestId).catch(useAppStore.getState().reportError);
}

/**
 * "Edit with AI…" (⌘J with a selection): the composer says what to change in
 * the selected text, which stays marked while the User types. False when the
 * selection can't be edited.
 */
export function startEditing(editor: Editor, mindId: string): boolean {
  const selection = selectionForEdit(editor.state);
  if (!selection) return false;
  set("problems", mindId, null);
  setEditTarget(editor, selection.range);
  set("editing", mindId, { quote: quoteOf(selection.text) });
  return true;
}

/** The composer goes back to asking: the selection is no longer being edited. */
export function stopEditing(editor: Editor, mindId: string): void {
  if (!useEdits.getState().editing[mindId]) return;
  set("editing", mindId, null);
  if (!useEdits.getState().pending[mindId]) setEditTarget(editor, null);
}

/** What the User said to change, from the composer, for the text it is editing. */
export function editFromComposer(
  editor: Editor,
  mindId: string,
  doc: Y.Doc,
  instruction: string,
): boolean {
  const target = editTarget(editor.state);
  set("editing", mindId, null);
  if (!target || instruction.trim() === "") {
    setEditTarget(editor, null);
    return false;
  }
  void requestEdit(editor, mindId, doc, "instruction", instruction.trim(), target);
  return true;
}

/**
 * Accepts or rejects Proposals in the open Mind, as one step of its Undo,
 * and offers Undo on the undo line while that step is still the last one.
 * What an accepted Proposal wrote goes into the record of AI use (#292),
 * after the change itself has gone to the core.
 */
export function resolveProposalsIn(
  editor: Editor,
  mindId: string,
  ids: readonly string[],
  outcome: "accept" | "reject",
): void {
  if (editor.isDestroyed || ids.length === 0) return;
  const resolved = resolveInEditor(editor, new Set(ids), outcome);
  if (resolved.length === 0) return;
  const doc = editorDoc(editor);
  if (outcome === "accept" && doc) {
    const entries = aiUseOfAccepted(doc, mindId, resolved, new Date().toISOString());
    if (entries.length > 0) {
      core.recordAiUse(entries).catch(useAppStore.getState().reportError);
    }
  }
  const count = resolved.length;
  const message =
    outcome === "accept"
      ? count === 1
        ? t("proposal.accepted.one")
        : t("proposal.accepted", { count })
      : count === 1
        ? t("proposal.rejected.one")
        : t("proposal.rejected", { count });
  speak(mindId, message);
  offerUndo(editor, message);
}

interface UndoStack {
  undoStack: unknown[];
  on(event: "stack-item-added" | "stack-item-popped", listener: () => void): void;
  off(event: "stack-item-added" | "stack-item-popped", listener: () => void): void;
}

/** "Accepted 2 changes · Undo", which takes the step back while nothing has been done since. */
function offerUndo(editor: Editor, message: string): void {
  const manager: UndoStack | undefined = yUndoPluginKey.getState(editor.state)?.undoManager;
  const step = manager?.undoStack.at(-1);
  if (!manager || step === undefined) return;
  useUndo.getState().offer({
    message,
    undo: async () => {
      if (!editor.isDestroyed && manager.undoStack.at(-1) === step) editor.commands.undo();
    },
  });
  const offered = useUndo.getState().last?.id;
  // Once the User edits again, or takes it back with ⌘Z, the line has nothing left to undo.
  const stale = () => {
    manager.off("stack-item-added", stale);
    manager.off("stack-item-popped", stale);
    stopWatching();
    if (useUndo.getState().last?.id === offered) useUndo.getState().dismiss();
  };
  const stopWatching = useUndo.subscribe((state) => {
    if (state.last?.id === offered) return;
    manager.off("stack-item-added", stale);
    manager.off("stack-item-popped", stale);
    stopWatching();
  });
  manager.on("stack-item-added", stale);
  manager.on("stack-item-popped", stale);
}
