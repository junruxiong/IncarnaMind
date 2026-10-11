/**
 * What skipping the reranker costs an agent's `search` (#351 keeps it out for
 * speed; Answers and `ask` rerank): for each answerable task, the task's own
 * words searched once each way, and whether the expected source is in the
 * top 5.
 * - "agent search": the pairing's `search`, page 1 (5 Passages), as an agent gets it;
 * - "reranked": keyword search's top 60 in the Search scope, reranked by the
 *   built-in model, top 5: the candidates the search Tool's reranker sees by
 *   default (the evaluation's gate, ADR-0009).
 */
import { BUILT_IN_RERANKING_MODEL, type Core } from "../../../src/core";
import type { Log } from "../../lib/log";
import { openReranker } from "../../lib/rerank";
import type { Task } from "./bakeoff";
import type { FixtureLibrary } from "./fixtureLibrary";
import { openSocketClient } from "./socketClient";

export interface ProbeRow {
  task: string;
  query: string;
  agentHit: boolean;
  agentMs: number;
  rerankedHit: boolean;
  rerankedMs: number;
}

const overlaps = (expected: [number, number] | null, range: [number, number] | null) =>
  expected === null || range === null || (range[0] <= expected[1] && range[1] >= expected[0]);

export async function probeRetrieval(options: {
  core: Core;
  fixture: FixtureLibrary;
  tasks: readonly Task[];
  inScopeDocumentIds: string[];
  cacheDir: string;
  log: Log;
}): Promise<ProbeRow[]> {
  const { core, fixture, log } = options;
  const connected = await core.connectAgent({
    name: "retrieval probe",
    folderIds: fixture.scope.folderIds,
  });
  const client = await openSocketClient(connected.launch.env);
  const reranker = await openReranker(BUILT_IN_RERANKING_MODEL, options.cacheDir, log);
  const rows: ProbeRow[] = [];
  try {
    // The first rerank loads the model; time the probes warm, as a running app's are.
    const warm = await core.searchPassages("warm up", {
      mode: "keyword",
      limit: 5,
      documentIds: options.inScopeDocumentIds,
    });
    if (warm.length > 0) await reranker.rerank("warm up", warm);
    await client.call("search", { query: "warm up" });
    for (const task of options.tasks) {
      if (task.expected.length === 0) continue;
      const query = task.task;
      const hit = (documentId: string, range: [number, number] | null) =>
        task.expected.some(
          (expected) =>
            fixture.keyOf(documentId) === expected.document && overlaps(expected.pages, range),
        );
      const agent = await client.call("search", { query });
      const passages = ((agent.data as { passages?: { passageId: string; documentId: string }[] })
        ?.passages ?? []) as { passageId: string; documentId: string }[];
      const agentHit = passages.some((passage) =>
        hit(passage.documentId, fixture.library.passagePages(passage.passageId)),
      );
      const started = performance.now();
      const candidates = await core.searchPassages(query, {
        mode: "keyword",
        limit: 60,
        documentIds: options.inScopeDocumentIds,
      });
      const ranked = candidates.length > 0 ? await reranker.rerank(query, candidates) : [];
      const rerankedMs = performance.now() - started;
      const rerankedHit = ranked
        .slice(0, 5)
        .some((passage) =>
          hit(
            passage.documentId,
            passage.pageFrom === null
              ? null
              : [passage.pageFrom, passage.pageTo ?? passage.pageFrom],
          ),
        );
      rows.push({ task: task.id, query, agentHit, agentMs: agent.ms, rerankedHit, rerankedMs });
      log(
        `Probe ${task.id}: agent search ${agentHit ? "hit" : "miss"} (${agent.ms.toFixed(0)} ms), reranked ${rerankedHit ? "hit" : "miss"} (${rerankedMs.toFixed(0)} ms)`,
      );
    }
  } finally {
    reranker.close();
    client.close();
    await core.disconnectAgent(connected.pairing.id);
  }
  return rows;
}
