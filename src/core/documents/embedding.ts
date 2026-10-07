/**
 * Embeds Documents' Passages with the built-in model, one Passage at a time
 * (ADR-0009: batches were no faster, and changed int8 results), one Document
 * after another. A Document's status goes from "embedding" to "ready"; while
 * the model isn't ready it waits as "waiting-for-model". Each vector is stored
 * as it comes, so a quit loses at most one Passage's work.
 */
import type { EmbeddingModel } from "../embedding";
import type { Database } from "../storage";
import { encodeVector, type VectorIndex } from "./vectors";

/** Progress reaches the UI at most this often per Document. */
const PROGRESS_INTERVAL_MS = 500;

export interface EmbeddingQueueOptions {
  db: Database;
  now: () => string;
  model: EmbeddingModel;
  vectors: VectorIndex;
  /** Pushes a Document's current state as a "document.status" event. */
  announce(documentId: string): void;
  /** A Document just became ready (e.g. for automatic tagging). Called before it is announced. */
  onReady?: (documentId: string) => void;
  reportError?: (error: unknown) => void;
}

export interface EmbeddingQueue {
  /** Queues a Document whose status is "embedding". */
  enqueue(documentId: string): void;
  /** Moves every Document waiting for the model on to "embedding", and queues it. */
  resumeWaiting(): void;
  /** Stops working: Documents keep their status, so the next start picks them up. */
  close(): void;
}

type Outcome = "done" | "skipped" | "model-unavailable";

export function createEmbeddingQueue(options: EmbeddingQueueOptions): EmbeddingQueue {
  const { db, now, model, vectors, announce } = options;
  const reportError = options.reportError ?? ((error) => console.error(error));
  const queue: string[] = [];
  let running = false;
  let closed = false;

  const statusOf = (id: string) =>
    db.get<{ status: string; name: string }>(
      "SELECT status, name FROM documents WHERE id = ? AND deleted_at IS NULL",
      [id],
    );

  const setStatus = (id: string, status: string, from: string) =>
    db.run("UPDATE documents SET status = ?, updated_at = ? WHERE id = ? AND status = ?", [
      status,
      now(),
      id,
      from,
    ]);

  /** The model can't be used: every Document being embedded waits for it again. */
  function park(): void {
    queue.length = 0;
    const embedding = db.all<{ id: string }>(
      "SELECT id FROM documents WHERE deleted_at IS NULL AND status = 'embedding' ORDER BY created_at, rowid",
    );
    for (const { id } of embedding) {
      setStatus(id, "waiting-for-model", "embedding");
      announce(id);
    }
  }

  function fail(id: string, message: string): void {
    db.run(
      `UPDATE documents SET status = 'failed', failure_reason = 'processing-error',
         failure_message = ?, updated_at = ?
       WHERE id = ? AND status = 'embedding'`,
      [message, now(), id],
    );
    announce(id);
  }

  async function embedDocument(id: string): Promise<Outcome> {
    const document = statusOf(id);
    if (document?.status !== "embedding") return "skipped";
    if (!(await model.load())) return "model-unavailable";
    if (closed) return "skipped";
    const passages = db.all<{ seq: number; text: string }>(
      `SELECT seq, text FROM passages
       WHERE document_id = ? AND deleted_at IS NULL AND embedding IS NULL
       ORDER BY position`,
      [id],
    );
    let announced = Date.now();
    for (const passage of passages) {
      let vector: Float32Array;
      try {
        vector = await model.embedPassage(document.name, passage.text);
      } catch {
        // The process running the model may have stopped: start it again and retry once.
        if (closed) return "skipped";
        if (!(await model.load())) return "model-unavailable";
        try {
          vector = await model.embedPassage(document.name, passage.text);
        } catch (error) {
          if (closed) return "skipped";
          fail(id, `Embedding failed: ${error instanceof Error ? error.message : String(error)}`);
          return "done";
        }
      }
      // Deleted, or processed again, meanwhile.
      if (closed || statusOf(id)?.status !== "embedding") return "skipped";
      db.run("UPDATE passages SET embedding = ? WHERE seq = ? AND deleted_at IS NULL", [
        encodeVector(vector),
        BigInt(passage.seq),
      ]);
      vectors.add(id, passage.seq, vector);
      if (Date.now() - announced >= PROGRESS_INTERVAL_MS) {
        announced = Date.now();
        announce(id);
      }
    }
    setStatus(id, "ready", "embedding");
    if (statusOf(id)?.status === "ready") {
      try {
        options.onReady?.(id);
      } catch (error) {
        reportError(error); // the Document is ready all the same
      }
    }
    announce(id);
    return "done";
  }

  async function pump(): Promise<void> {
    if (running) return;
    running = true;
    try {
      for (let id = queue.shift(); id !== undefined && !closed; id = queue.shift()) {
        if ((await embedDocument(id)) === "model-unavailable") {
          if (!closed) park();
          break;
        }
      }
    } catch (error) {
      if (!closed) reportError(error);
    } finally {
      running = false;
    }
    // Something may have been queued while the last Document was finishing.
    if (!closed && queue.length > 0) void pump();
  }

  function enqueue(documentId: string): void {
    if (closed || queue.includes(documentId)) return;
    queue.push(documentId);
    void pump();
  }

  return {
    enqueue,

    resumeWaiting() {
      if (closed) return;
      const waiting = db.all<{ id: string }>(
        `SELECT id FROM documents WHERE deleted_at IS NULL AND status = 'waiting-for-model'
         ORDER BY created_at, rowid`,
      );
      for (const { id } of waiting) {
        db.run(
          `UPDATE documents SET status = 'embedding', embedding_model = ?, updated_at = ?
           WHERE id = ? AND status = 'waiting-for-model'`,
          [model.id, now(), id],
        );
        announce(id);
        enqueue(id);
      }
    },

    close() {
      closed = true;
      queue.length = 0;
    },
  };
}
