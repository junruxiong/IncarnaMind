/**
 * What indexing the hard tier's library costs, in two passes, so what a User
 * with embeddings off waits for is measured apart from embedding:
 *
 * 1. The Documents are added with embeddings off, the app's default: each is
 *    ready once its text is extracted, its Passages built and its keyword
 *    index written (`openLibrary` with `embeddings: false`).
 * 2. Embeddings are turned on with the built-in model, as a User does in
 *    Settings, and the core embeds every Passage in the background; this
 *    waits until none is left without a vector.
 */
import type { Core } from "../../../src/core";
import type { Database } from "../../../src/core/storage";
import { modelReady } from "../../lib/library";
import type { Log } from "../../lib/log";

export interface EmbeddingPass {
  /** From turning embeddings on until every Passage had a vector (or embedding stalled). */
  seconds: number;
  /** Live Passages, and how many of them have a vector at the end. */
  passages: number;
  embedded: number;
}

export interface EmbeddingPassOptions {
  /** How often to look at the data folder. */
  pollMs?: number;
  /** Give up when no Passage is embedded for this long. */
  stallMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

const counts = (db: Database) => ({
  passages:
    db.get<{ count: number }>("SELECT COUNT(*) AS count FROM passages WHERE deleted_at IS NULL")
      ?.count ?? 0,
  embedded:
    db.get<{ count: number }>(
      "SELECT COUNT(*) AS count FROM passages WHERE deleted_at IS NULL AND embedding IS NOT NULL",
    )?.count ?? 0,
});

/** Turns embeddings on with the built-in model and waits until every Passage is embedded. */
export async function embedEveryPassage(
  core: Core,
  db: Database,
  log: Log,
  options: EmbeddingPassOptions = {},
): Promise<EmbeddingPass> {
  const pollMs = options.pollMs ?? 2000;
  const stallMs = options.stallMs ?? 10 * 60_000;
  const sleep =
    options.sleep ?? ((ms: number) => new Promise((resolve) => setTimeout(resolve, ms)));
  const started = Date.now();
  await core.saveEmbeddingProvider({ kind: "built-in" });
  await modelReady(core, log);
  let last = counts(db);
  let progressAt = Date.now();
  let loggedAt = 0;
  while (last.embedded < last.passages) {
    await sleep(pollMs);
    const now = counts(db);
    if (now.embedded > last.embedded) progressAt = Date.now();
    last = now;
    if (Date.now() - progressAt > stallMs) {
      log(`Embedding stalled at ${now.embedded} of ${now.passages} Passages; going on`);
      break;
    }
    if (Date.now() - loggedAt > 30_000) {
      loggedAt = Date.now();
      log(`Embedding: ${now.embedded} of ${now.passages} Passages`);
    }
  }
  return { seconds: (Date.now() - started) / 1000, ...last };
}
