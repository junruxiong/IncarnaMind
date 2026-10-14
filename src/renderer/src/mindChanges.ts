/**
 * The Minds an Answer in the assistant changed or made (#202), as its card
 * shows them (`MindChangesCard`): one row for each Mind, with how many of its
 * changes still wait there. Review and Open bring the Mind up at its first
 * change; Accept all and Reject all act on every row at once, with Undo on
 * the Answer's line and ⌘Z. Rejecting a Mind the Answer made deletes it,
 * which closes its tab, unless something else is in it now; Undo brings it
 * back beside the Mind in focus.
 *
 * The card reads and changes each Mind through a copy of its document in this
 * window (./mindCopies), the editor's or not: its counts follow every change
 * made anywhere, and what Accept all or Reject all did is undone there. Only
 * the last few of those steps keep their copies, and can be undone.
 */
import { useCallback, useEffect, useMemo, useSyncExternalStore } from "react";
import * as Y from "yjs";
import { MIND_CONTENT_FIELD, type ProposalRef } from "../../core/api";
import { aiUseOfAccepted, pendingProposals, resolveProposals } from "../../shared/proposals";
import type { ChangedMind } from "./changedMinds";
import { readMindHandle } from "./commands/handles";
import { core } from "./core";
import { pendingIn } from "./editor/proposalReview";
import { translateNow as t } from "./i18n";
import { openMindCopy } from "./mindCopies";
import { useAppStore } from "./store";
import { answerUndoPlace, type Undoable, useUndo } from "./undo";

/** A copy of a Mind's document, shared by everything in the window that reads it through here. */
interface Held {
  users: number;
  doc: Y.Doc | null;
  /** Resolves once the copy has arrived; null when it couldn't be opened. */
  ready: Promise<Y.Doc | null>;
  listeners: Set<() => void>;
  close(): void;
}

const held = new Map<string, Held>();

function hold(mindId: string): Held {
  const found = held.get(mindId);
  if (found) {
    found.users += 1;
    return found;
  }
  let arrived: (doc: Y.Doc | null) => void = () => undefined;
  const entry: Held = {
    users: 1,
    doc: null,
    ready: new Promise((resolve) => {
      arrived = resolve;
    }),
    listeners: new Set(),
    close: () => undefined,
  };
  const changed = () => {
    for (const listener of entry.listeners) listener();
  };
  entry.close = openMindCopy(
    mindId,
    ({ doc }) => {
      entry.doc = doc;
      doc?.on("update", changed);
      arrived(doc);
      changed();
    },
    (error) => {
      useAppStore.getState().reportError(error);
      arrived(null);
    },
  );
  held.set(mindId, entry);
  return entry;
}

function release(mindId: string): void {
  const entry = held.get(mindId);
  if (!entry) return;
  entry.users -= 1;
  if (entry.users > 0) return;
  held.delete(mindId);
  entry.close();
}

const belongsTo = (requests: readonly string[]) => (proposal: ProposalRef) =>
  requests.includes(proposal.request);

/** How many of a row's changes wait in its Mind; null until its copy has arrived. */
function waiting(row: ChangedMind): number | null {
  const doc = held.get(row.mindId)?.doc;
  if (!doc) return null;
  return pendingProposals(doc.getXmlFragment(MIND_CONTENT_FIELD)).filter(belongsTo(row.requests))
    .length;
}

/** How many of a row's changes wait in its Mind: a count, "gone" for a Mind deleted, null until its copy arrives. */
export type Waiting = number | "gone" | null;

/**
 * How many of each row's changes wait in its Mind, as they change. Holds a
 * copy of each Mind still there while the card is shown.
 */
export function useWaitingChanges(rows: readonly ChangedMind[]): Waiting[] {
  const minds = useAppStore((state) => state.minds);
  const live = useMemo(
    () => rows.filter((row) => minds.some((mind) => mind.id === row.mindId)),
    [rows, minds],
  );
  // The Minds held, by id, one a line: the copies change only when they do.
  const key = live.map((row) => row.mindId).join("\n");
  useEffect(() => {
    const ids = key === "" ? [] : key.split("\n");
    for (const id of ids) hold(id);
    return () => {
      for (const id of ids) release(id);
    };
  }, [key]);
  const subscribe = useCallback(
    (listener: () => void) => {
      const entries = (key === "" ? [] : key.split("\n")).flatMap((id) => held.get(id) ?? []);
      for (const entry of entries) entry.listeners.add(listener);
      return () => {
        for (const entry of entries) entry.listeners.delete(listener);
      };
    },
    [key],
  );
  const snapshot = useSyncExternalStore(subscribe, () =>
    rows.map((row) => (live.includes(row) ? (waiting(row) ?? "") : "gone")).join(","),
  );
  return snapshot
    .split(",")
    .map((count) => (count === "gone" ? "gone" : count === "" ? null : Number(count)));
}

/** Brings a Mind up beside the focused one, where it is open, else in a new tab. */
const openBeside = (mindId: string) =>
  useAppStore.getState().changeLayout({
    type: "open",
    item: { kind: "mind", mindId },
    where: { in: "place" },
  });

/** Frames to wait for a Mind just opened to show its changes. */
const REVEAL_FRAMES = 60;

/** Puts the cursor in the first of the requests' changes in a Mind open in the editor, once it shows them. */
function revealChange(mindId: string, requests: readonly string[], frames = REVEAL_FRAMES): void {
  const read = readMindHandle.current;
  if (read && read.mindId === mindId && !read.editor.isDestroyed) {
    const first = pendingIn(read.editor.state).find(belongsTo(requests));
    if (first) {
      const at = first.block ? first.from + 1 : first.from;
      read.editor
        .chain()
        .focus()
        .setTextSelection(Math.min(at, read.editor.state.doc.content.size))
        .scrollIntoView()
        .run();
      return;
    }
  }
  if (frames > 0) requestAnimationFrame(() => revealChange(mindId, requests, frames - 1));
}

/** Review and Open: the Mind, at its first change from the Answer. */
export function showChanges(row: ChangedMind): void {
  useAppStore.getState().openMind(row.mindId);
  revealChange(row.mindId, row.requests);
}

/** One Mind's part of Accept all or Reject all: how many changes it took, and how to take it back and do it again. */
interface Done {
  count: number;
  undo(): Promise<void>;
  redo(): Promise<void>;
  /** It can't be undone any more: what it kept for that goes. */
  forget(): void;
}

/** Whether a Mind's copy holds nothing: only empty lines. */
const isEmpty = (doc: Y.Doc) =>
  doc
    .getXmlFragment(MIND_CONTENT_FIELD)
    .toArray()
    .every(
      (block) =>
        block instanceof Y.XmlElement && block.nodeName === "paragraph" && block.length === 0,
    );

/**
 * Reject all in a Mind the Answer made: its changes go, and so does the Mind,
 * closing its tab, once nothing else is in it (the User may have written in
 * it, or accepted some). Undo brings it back beside the focused one.
 */
async function rejectMade(row: ChangedMind): Promise<Done | null> {
  const rejected = await resolveRow(row, "reject");
  const doc = held.get(row.mindId)?.doc;
  if (!rejected || !doc || !isEmpty(doc)) return rejected;
  await core.deleteMind(row.mindId);
  return {
    count: rejected.count,
    undo: async () => {
      await core.restore([{ kind: "mind", id: row.mindId }]);
      await rejected.undo();
      await openBeside(row.mindId);
    },
    redo: async () => {
      await rejected.redo();
      await core.deleteMind(row.mindId);
    },
    forget: rejected.forget,
  };
}

/** Accepts or rejects a row's changes in its Mind's copy, which stays held so the step can be undone. */
async function resolveRow(row: ChangedMind, outcome: "accept" | "reject"): Promise<Done | null> {
  const entry = hold(row.mindId);
  const doc = await entry.ready;
  const picked = belongsTo(row.requests);
  const blocks = doc?.getXmlFragment(MIND_CONTENT_FIELD);
  if (!doc || !blocks || !pendingProposals(blocks).some(picked)) {
    release(row.mindId);
    return null;
  }
  // A Checkpoint of the Mind before an AI change goes in (#113).
  if (outcome === "accept") {
    try {
      await core.saveCheckpoint(row.mindId, "accept");
    } catch (error) {
      release(row.mindId);
      throw error;
    }
  }
  const origin = { mindChanges: row.mindId };
  const steps = new Y.UndoManager(blocks, { trackedOrigins: new Set([origin]) });
  let resolved: ReturnType<typeof resolveProposals> = [];
  doc.transact(() => {
    resolved = resolveProposals(blocks, picked, outcome);
  }, origin);
  if (outcome === "accept") {
    const entries = aiUseOfAccepted(doc, row.mindId, resolved, new Date().toISOString());
    if (entries.length > 0) core.recordAiUse(entries).catch(useAppStore.getState().reportError);
  }
  return {
    count: resolved.length,
    undo: async () => {
      steps.undo();
    },
    redo: async () => {
      steps.redo();
    },
    forget: () => {
      steps.destroy();
      release(row.mindId);
    },
  };
}

/** How many Accept all and Reject all steps keep their Minds' copies, to be undone: an older one leaves the Undo stack. */
const KEPT_STEPS = 5;

const kept: { change: Undoable; done: readonly Done[] }[] = [];

/** Keeps what a step needs to be undone, and lets go of the oldest step's beyond `KEPT_STEPS`. */
function keep(change: Undoable, done: readonly Done[]): void {
  kept.push({ change, done });
  for (const oldest of kept.splice(0, Math.max(0, kept.length - KEPT_STEPS))) {
    useUndo.getState().withdraw(oldest.change);
    for (const step of oldest.done) step.forget();
  }
}

/**
 * Accept all or Reject all on an Answer's card: every row's changes, in its
 * Mind; a Mind the Answer made goes, once rejected. One step of Undo takes
 * back all of it, on the Answer's line and with ⌘Z.
 */
export async function resolveAllChanges(
  answerId: string,
  rows: readonly ChangedMind[],
  outcome: "accept" | "reject",
): Promise<void> {
  const minds = useAppStore.getState().minds;
  const done: Done[] = [];
  try {
    for (const row of rows) {
      if (!minds.some((mind) => mind.id === row.mindId)) continue;
      const step =
        outcome === "reject" && row.created
          ? await rejectMade(row)
          : await resolveRow(row, outcome);
      if (step) done.push(step);
    }
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
  if (done.length === 0) return;
  const count = done.reduce((sum, step) => sum + step.count, 0);
  const message =
    outcome === "accept"
      ? count === 1
        ? t("proposal.accepted.one")
        : t("proposal.accepted", { count })
      : count === 1
        ? t("proposal.rejected.one")
        : t("proposal.rejected", { count });
  const change: Undoable = {
    message,
    shownAt: answerUndoPlace(answerId),
    undo: async () => {
      for (const step of [...done].reverse()) await step.undo();
    },
    redo: async () => {
      for (const step of done) await step.redo();
    },
  };
  useUndo.getState().push(change);
  keep(change, done);
}
