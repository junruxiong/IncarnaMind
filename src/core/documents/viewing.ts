/**
 * Which Documents' files the viewer is reading (#763): a file served over the
 * document protocol counts while its stream is open, and for a moment after,
 * as pdf.js reads a locked PDF whole before it asks for its password. Only a
 * PDF being shown is given its kept password, so a page that asks for every
 * Document's gets none.
 */

/** How long after its stream closes a file still counts as being shown. */
const GRACE_MS = 10_000;

export function createViewing(now: () => number = Date.now) {
  const entries = new Map<string, { open: number; closedAt: number }>();

  return {
    /** The file's stream, counted while it is open. */
    watch(documentId: string, stream: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
      const entry = entries.get(documentId) ?? { open: 0, closedAt: 0 };
      entries.set(documentId, entry);
      entry.open++;
      let closed = false;
      const close = () => {
        if (closed) return;
        closed = true;
        entry.open--;
        entry.closedAt = now();
      };
      const reader = stream.getReader();
      return new ReadableStream<Uint8Array>({
        async pull(controller) {
          try {
            const { value, done } = await reader.read();
            if (done) {
              close();
              controller.close();
            } else controller.enqueue(value);
          } catch (error) {
            close();
            controller.error(error);
          }
        },
        cancel(reason) {
          close();
          return reader.cancel(reason);
        },
      });
    },

    /** Whether a Document's file is being read for the viewer now, or was a moment ago. */
    isShown(documentId: string): boolean {
      const entry = entries.get(documentId);
      return entry !== undefined && (entry.open > 0 || now() - entry.closedAt <= GRACE_MS);
    },
  };
}
