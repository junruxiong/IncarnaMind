/**
 * Find support and Check against Documents (#96), the window's side: which
 * sentences a selection is (as the core reads them, `TextPlace`), asking
 * the core to check them, Stop, what went wrong where the Mind's health is,
 * and a check's card actions: Check again, Search all Documents and Add as
 * Citation.
 */
import type { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import type * as Y from "yjs";
import { create } from "zustand";
import {
  type CheckSentencesResult,
  CITATION_NODE,
  CLAIM_CHECK_MARK,
  type ClaimCheckAction,
} from "../../core/api";
import { mindModelOf } from "../../shared/mindModel";
import { core } from "./core";
import type { ShownClaim } from "./editor/claimChecks";
import { selectionForEdit } from "./editor/editSelection";

/** Why a check gave no marks, for the line under the Mind's title. */
export type CheckProblem =
  | Exclude<CheckSentencesResult, { kind: "checked" } | { kind: "cancelled" }>
  | { kind: "failed"; message: string };

interface ChecksState {
  /** By Mind: the request checking its sentences now, and how many. */
  running: Readonly<Record<string, { requestId: string; action: ClaimCheckAction }>>;
  /** By Mind: why the last request gave no marks. */
  problems: Readonly<Record<string, CheckProblem>>;
  /** By Mind: its last verdicts came from a model too small to judge reliably (#680). */
  smallModel: Readonly<Record<string, true>>;
}

export const useChecks = create<ChecksState>()(() => ({
  running: {},
  problems: {},
  smallModel: {},
}));

function setFor<K extends keyof ChecksState>(
  key: K,
  mindId: string,
  value: ChecksState[K][string] | null,
) {
  useChecks.setState((state) => {
    const next = { ...state[key] } as Record<string, ChecksState[K][string]>;
    if (value === null) delete next[mindId];
    else next[mindId] = value;
    return { [key]: next } as Partial<ChecksState>;
  });
}

export const dismissCheckProblem = (mindId: string) => setFor("problems", mindId, null);

/**
 * Checks the sentences the selection (or `range`) touches, each whole, as
 * `action` says. Their marks come with the core's updates; anything that
 * stops it is one line under the Mind's title. One request runs per Mind:
 * asking again while it runs does nothing.
 */
export async function checkSentences(
  editor: Editor,
  mindId: string,
  doc: Y.Doc,
  action: ClaimCheckAction,
  {
    range,
    allDocuments = false,
  }: { range?: { from: number; to: number }; allDocuments?: boolean } = {},
): Promise<void> {
  if (editor.isDestroyed || useChecks.getState().running[mindId]) return;
  const selection = selectionForEdit(editor.state, range);
  if (!selection) return;
  // The selection closes at its end, as for Edit with AI: its sentences show their marks, and the User writes on.
  const { selection: now } = editor.state;
  if (now.from === selection.range.from && now.to === selection.range.to) {
    editor.view.dispatch(
      editor.state.tr.setSelection(TextSelection.create(editor.state.doc, selection.range.to)),
    );
  }
  const requestId = crypto.randomUUID();
  setFor("problems", mindId, null);
  setFor("running", mindId, { requestId, action });
  let result: CheckSentencesResult | { kind: "failed"; message: string };
  try {
    result = await core.checkSentences({
      requestId,
      mindId,
      from: selection.from,
      to: selection.to,
      action,
      allDocuments,
      model: mindModelOf(doc),
    });
  } catch (error) {
    result = { kind: "failed", message: error instanceof Error ? error.message : String(error) };
  } finally {
    setFor("running", mindId, null);
  }
  if (result.kind !== "checked" && result.kind !== "cancelled") setFor("problems", mindId, result);
  if (result.kind === "checked") setFor("smallModel", mindId, result.smallModel ? true : null);
}

/** Stops the Mind's check: the sentences not checked yet say "stopped". */
export function stopCheck(mindId: string): void {
  const running = useChecks.getState().running[mindId];
  if (running) void core.cancelCheck(running.requestId);
}

/** Checks a sentence again, as it was checked, or in every Document. */
export function checkAgain(
  editor: Editor,
  mindId: string,
  doc: Y.Doc,
  claim: ShownClaim,
  allDocuments = claim.check.scope.widened,
): Promise<void> {
  return checkSentences(editor, mindId, doc, claim.check.action, {
    range: { from: claim.from, to: claim.to },
    allDocuments,
  });
}

/** Closing punctuation a Citation goes before: "…in the quarter [1]." */
const CLOSING = /[\s.!?。！？…;；:："'”’)）\]】」』]+$/u;
const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]$/u;

/**
 * Adds the Passage a check found as the sentence's Citation, at its end,
 * before its closing punctuation, and takes the check's mark off: the
 * Citation's own check stands for it now. One step for ⌘Z.
 */
export function addAsCitation(editor: Editor, claim: ShownClaim): boolean {
  const evidence = claim.check.evidence;
  const { state } = editor;
  const type = state.schema.nodes[CITATION_NODE];
  const mark = state.schema.marks[CLAIM_CHECK_MARK];
  if (!evidence || !type || !mark) return false;
  const words = state.doc.textBetween(claim.sentence.from, claim.sentence.to, "\n", "￼");
  const closing = CLOSING.exec(words)?.[0].length ?? 0;
  const at = claim.sentence.to - closing;
  const before = words.slice(0, words.length - closing);
  const spaced = before !== "" && !/\s$/u.test(before) && !CJK.test(before);
  const citation = type.create({ ...evidence, check: "found", checkReason: null });
  const tr = state.tr.removeMark(claim.from, claim.to, mark);
  tr.insert(at, spaced ? [state.schema.text(" "), citation] : citation);
  editor.view.dispatch(tr);
  return true;
}
