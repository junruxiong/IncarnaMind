/**
 * Automatic tagging (spec #20, "Tagging"): once a Document is ready (embedded,
 * so already searchable), its Tags are decided: by Jev when a Jev key is set
 * up on this device, otherwise by the default chat model. One Document at a
 * time, in the order they became ready.
 *
 * - Each Document takes one turn in the background model queue (see
 *   ../backgroundQueue), which other background model work shares and which
 *   gives way to Answers. A call to a model on this computer that an Answer
 *   interrupts leaves its Document waiting, first in line once no Answer is
 *   being written.
 * - Nothing is sent before the User accepts the "tagging" data flow to the
 *   tagger's service; a local model needs no consent.
 * - With no usable tagger (no chat model set up, a missing key or sign-in, or
 *   the flow declined) Documents wait as "waiting-for-provider", and are
 *   picked up again whenever that may have changed.
 * - Only automatic links change: Tags the User added or removed on a
 *   Document stay as they are (see `TagsStore.applyAutomatic`).
 */
import type { DocumentKind, ProviderError } from "../api";
import type { BackgroundCall, BackgroundJob, BackgroundQueue } from "../backgroundQueue";
import {
  ChatNotReadyError,
  ConsentDeclinedError,
  InvalidInputError,
  NotFoundError,
  TaggingNotReadyError,
} from "../errors";
import { classifyProviderError } from "../providers/providerErrors";
import type { Database } from "../storage";
import {
  type DocumentExcerpt,
  excerptFromPassages,
  type TagClassifier,
  type TagDecision,
} from "./classify";
import type { TagsStore } from "./index";

/** What the "tagging" flow sends to the tagger's service: the chat model's or Jev's. */
export const TAGGING_FLOW_SENDS = ["tags", "document-excerpts"] as const;

/** Tagging states as stored (`documents.tagging_status`); "skipped" is worked out from the status. */
export type StoredTaggingState =
  | "pending"
  | "waiting-for-provider"
  | "tagging"
  | "tagged"
  | "failed";

/** How many Passages, from the start, are read to build an excerpt. */
const EXCERPT_ROWS = 6;

export interface TaggerOptions {
  db: Database;
  now: () => string;
  tags: TagsStore;
  /** Whether the tagger in use can take a request now. Asks the User nothing. */
  canRun(): Promise<boolean>;
  /**
   * A quick check, with no waiting: false when tagging certainly can't run
   * (no tagger set up, or the User declined the flow to its service).
   */
  mightBeReady(): boolean;
  /**
   * The tagger in use, once the User has accepted the tagging flow to its
   * service. Throws ChatNotReadyError, ConsentDeclinedError or
   * TaggingNotReadyError when it can't be used: nothing is sent then.
   */
  prepare(): Promise<TagClassifier>;
  /** Pushes "documents.tagged" for these Documents. */
  announce(documentIds: readonly string[]): void;
  reportError(error: unknown): void;
  /** The background model queue, in which tagging each Document takes one turn. */
  background: BackgroundQueue;
}

interface DocumentRow {
  name: string;
  kind: string;
  page_count: number | null;
  status: string;
  tagging_status: string;
  tagging_error_kind: string | null;
  tagging_error_message: string | null;
}

export function createTagger(options: TaggerOptions) {
  const { db, now, tags, announce, background } = options;
  const lifetime = new AbortController();
  const queue: string[] = [];
  /** The Document being tagged, if any. */
  let current: string | null = null;
  /** The tagger's turn in the background queue, while one is queued or running. */
  let turn: BackgroundJob | null = null;
  /**
   * Counts re-tag requests per Document. A result worked out before the
   * latest request is dropped: the run that request queued decides.
   */
  const requests = new Map<string, number>();

  const rowOf = (id: string) =>
    db.get<DocumentRow>(
      `SELECT name, kind, page_count, status, tagging_status, tagging_error_kind, tagging_error_message
       FROM documents WHERE id = ? AND deleted_at IS NULL`,
      [id],
    );

  /** Sets a Document's tagging state. Returns whether it changed. */
  const setState = (
    id: string,
    state: StoredTaggingState,
    error: ProviderError | null = null,
  ): boolean => {
    const row = rowOf(id);
    if (
      !row ||
      (row.tagging_status === state &&
        row.tagging_error_kind === (error?.kind ?? null) &&
        row.tagging_error_message === (error?.message ?? null))
    ) {
      return false;
    }
    db.run(
      `UPDATE documents SET tagging_status = ?, tagging_error_kind = ?, tagging_error_message = ?,
         updated_at = ?
       WHERE id = ?`,
      [state, error?.kind ?? null, error?.message ?? null, now(), id],
    );
    return true;
  };

  const setStates = (ids: readonly string[], state: StoredTaggingState) => {
    const changed = ids.filter((id) => setState(id, state));
    if (changed.length > 0) announce(changed);
  };

  /** No tagger can be used: every queued Document waits for one. */
  function park(): void {
    setStates(queue.splice(0), "waiting-for-provider");
  }

  function excerptOf(id: string, row: DocumentRow): DocumentExcerpt {
    const passages = db.all<{ text: string }>(
      `SELECT text FROM passages WHERE document_id = ? AND deleted_at IS NULL
       ORDER BY position LIMIT ?`,
      [id, BigInt(EXCERPT_ROWS)],
    );
    return {
      name: row.name,
      kind: row.kind as DocumentKind,
      pageCount: row.page_count,
      text: excerptFromPassages(passages.map((passage) => passage.text)),
    };
  }

  const fail = (id: string, error: unknown) => {
    if (setState(id, "failed", classifyProviderError(error))) announce([id]);
  };

  /**
   * Tags one ready Document. If no tagger can be used after all, it and the
   * rest of the queue wait for one.
   */
  async function tagDocument(id: string, call: BackgroundCall): Promise<void> {
    const request = requests.get(id) ?? 0;
    /**
     * The core closed, the call gave way to an Answer, or a re-tag came in
     * meanwhile: the run that comes next decides instead.
     */
    const stale = () =>
      lifetime.signal.aborted || call.signal.aborted || (requests.get(id) ?? 0) !== request;
    const row = rowOf(id);
    if (row?.status !== "ready") return;
    const definitions = tags.list();
    if (setState(id, "tagging")) announce([id]);

    let decisions: TagDecision[] = [];
    if (definitions.length > 0) {
      let tagger: TagClassifier;
      try {
        tagger = await options.prepare();
      } catch (error) {
        if (lifetime.signal.aborted) return;
        if (
          error instanceof ChatNotReadyError ||
          error instanceof ConsentDeclinedError ||
          error instanceof TaggingNotReadyError
        ) {
          // Declined, or the tagger went away while this waited: this and the rest wait.
          if (setState(id, "waiting-for-provider")) announce([id]);
          park();
        } else {
          fail(id, error);
        }
        return;
      }
      // A model on this computer serves one request at a time: an Answer goes first.
      if (tagger.local) call.runsLocally();
      if (stale()) return;
      try {
        decisions = await tagger.decide({
          tags: definitions,
          excerpt: excerptOf(id, row),
          signal: AbortSignal.any([lifetime.signal, call.signal]),
        });
      } catch (error) {
        if (!stale()) fail(id, error);
        return;
      }
    }
    if (stale()) return;

    const changed = db.transaction(() => {
      if (rowOf(id)?.status !== "ready") return false; // deleted, or processed again, meanwhile
      const tagsChanged = tags.applyAutomatic(id, decisions);
      return setState(id, "tagged") || tagsChanged;
    });
    if (changed) announce([id]);
  }

  /** Tags the first Document in the queue, if it can: one turn in the background queue. */
  async function tagNext(call: BackgroundCall): Promise<void> {
    try {
      if (queue.length === 0 || lifetime.signal.aborted) return;
      const ready = await options.canRun();
      if (lifetime.signal.aborted) return;
      if (!ready) {
        park();
        return;
      }
      const id = queue.shift();
      if (id === undefined) return;
      current = id;
      try {
        await tagDocument(id, call);
      } finally {
        current = null;
      }
      if (call.gaveWay && !lifetime.signal.aborted) {
        // Interrupted by an Answer: it waits, first in line, until no Answer is being written.
        if (!queue.includes(id)) queue.unshift(id);
        if (setState(id, "pending")) announce([id]);
      }
    } catch (error) {
      if (lifetime.signal.aborted) return;
      options.reportError(error);
      park();
    }
  }

  /** Queues a turn in the background queue, unless one is queued or running, or nothing waits. */
  function schedule(): void {
    if (turn || queue.length === 0 || lifetime.signal.aborted) return;
    const job: BackgroundJob = {
      kind: "tagging",
      run: async (call) => {
        try {
          await tagNext(call);
        } finally {
          // A turn that gave way runs again, first; otherwise the next Document takes a new
          // turn, after the work queued meanwhile.
          if (!call.gaveWay) {
            turn = null;
            schedule();
          }
        }
      },
    };
    turn = job;
    background.add(job);
  }

  /**
   * Queues a Document. `fresh`: a new request (a re-tag), whose result must
   * replace that of a run already going; otherwise a Document being tagged
   * isn't queued again.
   */
  function enqueue(id: string, fresh = false): void {
    if (lifetime.signal.aborted) return;
    if (fresh) requests.set(id, (requests.get(id) ?? 0) + 1);
    else if (id === current) return;
    if (!queue.includes(id)) queue.push(id);
    schedule();
  }

  /** Queues every ready Document that still needs tagging, or marks them waiting if no tagger can be used. */
  function resume(): void {
    if (lifetime.signal.aborted) return;
    const ids = db
      .all<{ id: string }>(
        `SELECT id FROM documents
         WHERE deleted_at IS NULL AND status = 'ready'
           AND tagging_status IN ('pending', 'waiting-for-provider', 'failed')
         ORDER BY created_at, rowid`,
      )
      .map((row) => row.id);
    if (!options.mightBeReady()) {
      setStates(ids, "waiting-for-provider");
      return;
    }
    for (const id of ids) enqueue(id);
  }

  return {
    /** Picks up after a quit: Documents being tagged start again, and those waiting are queued. */
    start(): void {
      db.run(
        `UPDATE documents SET tagging_status = 'pending', updated_at = ?
         WHERE deleted_at IS NULL AND tagging_status = 'tagging'`,
        [now()],
      );
      resume();
    },

    /**
     * A Document just became ready. Called before its new status is pushed,
     * so that event already says whether it waits for a tagger.
     */
    documentReady(id: string): void {
      const row = rowOf(id);
      // Processed again (e.g. by a newer pipeline): the same file keeps its Tags.
      if (!row || row.tagging_status === "tagged") return;
      if (!options.mightBeReady()) {
        setState(id, "waiting-for-provider");
        return;
      }
      setState(id, "pending");
      enqueue(id);
    },

    /** The tagger, or whether it can run, may have changed: Documents waiting, or failed, are tried again. */
    resume,

    /**
     * Re-tags Documents (all of them without ids): their automatic Tags are
     * worked out again. Documents still being processed are tagged once ready.
     */
    retag(input: unknown): void {
      if (
        input !== undefined &&
        (!Array.isArray(input) || !input.every((id) => typeof id === "string" && id !== ""))
      ) {
        throw new InvalidInputError("documentIds must be a list of Document ids.");
      }
      const ids =
        input === undefined
          ? db
              .all<{ id: string }>(
                "SELECT id FROM documents WHERE deleted_at IS NULL ORDER BY created_at, rowid",
              )
              .map((row) => row.id)
          : [...new Set(input as string[])];
      const rows = ids.map((id) => {
        const row = rowOf(id);
        if (!row) throw new NotFoundError("There is no such Document.");
        return { id, row };
      });
      const changed: string[] = [];
      const ready = options.mightBeReady();
      for (const { id, row } of rows) {
        if (row.status !== "ready") {
          // Being processed again: tagged afresh once ready. Failed or without text: nothing to tag.
          if (row.tagging_status === "tagged" && setState(id, "pending")) changed.push(id);
          continue;
        }
        if (setState(id, ready ? "pending" : "waiting-for-provider")) changed.push(id);
      }
      if (changed.length > 0) announce(changed);
      if (!ready) return;
      for (const { id, row } of rows) if (row.status === "ready") enqueue(id, true);
    },

    /** Stops: Documents keep their state, and the next start picks them up. */
    close(): void {
      lifetime.abort();
      queue.length = 0;
    },
  };
}
