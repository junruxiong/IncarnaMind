/** What the core and the citation-style thread (./styleWorker) send each other. */
import type { FormatJob } from "./format";
import type { EngineOutput, StyleDescription } from "./wasm";

export interface WorkerRequest {
  id: number;
  /** A shipped style, by its file in the styles folder. */
  style: { file: string };
  job: FormatJob;
}

export type WorkerResponse =
  | { id: number; ok: true; description: StyleDescription; output: EngineOutput }
  | {
      id: number;
      ok: false;
      message: string;
      /** The style itself can't be used, rather than this request. */
      style: boolean;
    };
