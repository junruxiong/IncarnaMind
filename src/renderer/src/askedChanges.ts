/**
 * What the User asked the assistant to write or change applies at once
 * (#880, D224), in the window whose Question asked for it. The core tags the
 * requests of such a turn (`ProposalRequest.requested`); their changes read
 * as applied in the Mind as they arrive, and a Mind not in view opens in a
 * tab beside the one in focus, never in its place. Once the Answer has
 * ended, each Mind's are applied in one step of its Undo, with the line over
 * them saying what the Answer's card says (`applyArrived`). Changes the User
 * didn't ask for wait, as before.
 */
import { create } from "zustand";
import type { ConversationAnswer } from "../../core/api";
import { proposalRequestOf } from "../../shared/proposals";
import { actOnAppliedEdit, applyArrived } from "./appliedEdits";
import { type ChangedMind, changedMinds } from "./changedMinds";
import { readMindHandle } from "./commands/handles";
import { core } from "./core";
import { mindEditor, whenMindEditor } from "./editor/openEditors";
import { applyingIn, editorDoc, setApplying } from "./editor/proposalReview";
import { openMindCopy } from "./mindCopies";
import { useAppStore } from "./store";

export type AskedOutcome = "applied" | "held" | "undone";

interface AskedChangesState {
  /** By Answer, then by Mind: what became of the changes the User asked for there. */
  outcomes: Readonly<Record<string, Readonly<Record<string, AskedOutcome>>>>;
}

export const useAskedChanges = create<AskedChangesState>()(() => ({ outcomes: {} }));

/** Answers to Questions asked in this window, still being written. */
const askedHere = new Set<string>();
/** Each Mind an Answer changed, once looked at: "answer mind". */
const seen = new Set<string>();

/** How long a Mind just opened in a tab has to show before its changes are applied. */
const OPEN_WAIT_MS = 10_000;

/** An Answer this window asked for: what it was asked to write or change applies here. */
export function applyWhatWasAsked(answerId: string): void {
  askedHere.add(answerId);
}

/** The requests among `requests` that were for what the User asked, as their record in the Mind says. */
function askedOf(doc: Parameters<typeof proposalRequestOf>[0], requests: readonly string[]) {
  return new Set(
    requests.filter((request) => {
      const made = proposalRequestOf(doc, request);
      return made !== null && "requested" in made && made.requested !== undefined;
    }),
  );
}

/** Whether any of a row's requests were asked for, read in a copy of a Mind not open here. */
function askedInCopy(row: ChangedMind): Promise<boolean> {
  return new Promise((resolve) => {
    const close = openMindCopy(
      row.mindId,
      ({ doc }) => {
        resolve(doc !== null && askedOf(doc, row.requests).size > 0);
        close();
      },
      () => resolve(false),
    );
  });
}

const openBeside = (mindId: string) =>
  useAppStore
    .getState()
    .changeLayout({ type: "open", item: { kind: "mind", mindId }, where: { in: "place" } })
    .catch(useAppStore.getState().reportError);

/** Draws a row's asked-for changes as applied where its Mind is open; opens it beside the focused one if it isn't. */
function follow(answerId: string, row: ChangedMind): void {
  const key = `${answerId} ${row.mindId}`;
  const editor = mindEditor(row.mindId);
  const doc = editor && editorDoc(editor);
  if (editor && doc) {
    const asked = askedOf(doc, row.requests);
    const now = applyingIn(editor.state);
    if ([...asked].some((request) => !now.has(request))) {
      setApplying(editor, new Set([...now, ...asked]));
    }
    // Open in a tab behind the one in view: shown, beside it.
    if (asked.size > 0 && !seen.has(key) && readMindHandle.current?.mindId !== row.mindId) {
      seen.add(key);
      void openBeside(row.mindId);
    }
    return;
  }
  if (seen.has(key)) return;
  seen.add(key);
  // A Mind the Answer made opens anyway (./assistantMinds).
  if (row.created) return;
  void askedInCopy(row).then((asked) => {
    if (asked) void openBeside(row.mindId);
  });
}

/** Once the Answer has ended: each Mind's asked-for changes applied, or held. */
async function finish(answer: ConversationAnswer): Promise<void> {
  const outcomes: Record<string, AskedOutcome> = {};
  for (const row of changedMinds(answer)) {
    const editor = mindEditor(row.mindId) ?? (await whenMindEditor(row.mindId, OPEN_WAIT_MS));
    const doc = editor && editorDoc(editor);
    if (!editor || !doc) continue;
    const asked = askedOf(doc, row.requests);
    if (asked.size === 0) continue;
    const outcome = applyArrived(editor, row.mindId, asked, row);
    if (outcome) outcomes[row.mindId] = outcome;
  }
  if (Object.keys(outcomes).length === 0) return;
  useAskedChanges.setState((state) => ({
    outcomes: { ...state.outcomes, [answer.id]: outcomes },
  }));
}

/** Undo on the Answer's card: the whole turn, in every Mind it applied in. */
export function undoAsked(answerId: string): void {
  const outcomes = useAskedChanges.getState().outcomes[answerId] ?? {};
  const undone: Record<string, AskedOutcome> = { ...outcomes };
  for (const [mindId, outcome] of Object.entries(outcomes)) {
    if (outcome !== "applied") continue;
    const editor = mindEditor(mindId);
    if (!editor) continue;
    actOnAppliedEdit(editor, "undo");
    undone[mindId] = "undone";
  }
  useAskedChanges.setState((state) => ({ outcomes: { ...state.outcomes, [answerId]: undone } }));
}

/** Review changes on the Answer's card: the Mind, with what was applied shown again. */
export async function reviewAsked(mindId: string): Promise<void> {
  await useAppStore.getState().openMind(mindId);
  const editor = mindEditor(mindId) ?? (await whenMindEditor(mindId, OPEN_WAIT_MS));
  if (editor) actOnAppliedEdit(editor, "review");
}

core.on("conversation.message", (message) => {
  if (message.kind !== "answer" || !askedHere.has(message.id)) return;
  const answer = message;
  for (const row of changedMinds(answer)) follow(answer.id, row);
  if (answer.status === "streaming") return;
  askedHere.delete(answer.id);
  finish(answer).catch(useAppStore.getState().reportError);
});
