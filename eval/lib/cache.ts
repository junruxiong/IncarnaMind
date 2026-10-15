import { homedir } from "node:os";
import { join, resolve } from "node:path";

/**
 * Where the evaluation keeps its downloads between runs: `INCARNAMIND_EVAL_CACHE`,
 * or ~/.cache/incarnamind-eval. The evaluation's vitest configs resolve it
 * before the home guard (tests/setup/home.ts) points HOME at a temporary
 * folder, and hand it to the run; the guard lets writes into it through.
 */
export function evalCache(env: Readonly<Record<string, string | undefined>> = process.env): string {
  return resolve(
    env.INCARNAMIND_EVAL_CACHE?.trim() || join(homedir(), ".cache", "incarnamind-eval"),
  );
}
