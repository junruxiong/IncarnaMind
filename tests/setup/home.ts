/**
 * Runs before each test file, in the unit tests and the evaluation alike:
 * points HOME and the folders named after it at a fresh temporary folder
 * before the test file imports anything, and fails a test that writes under
 * the real home folder all the same.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach } from "vitest";
import { createHomeGuard, homeEnv, installHomeGuard, REAL_HOME } from "../helpers/home";

// A worker that runs a second test file already has a temporary HOME.
const realHome = process.env[REAL_HOME] ?? homedir();
process.env[REAL_HOME] = realHome;
const home = mkdtempSync(join(tmpdir(), "incarnamind-test-home-"));
Object.assign(process.env, homeEnv(home));

const guard = createHomeGuard(realHome, [
  fileURLToPath(new URL("../..", import.meta.url)),
  tmpdir(),
  // The evaluation keeps its downloads there between runs (eval/lib/cache.ts).
  ...(process.env.INCARNAMIND_EVAL_CACHE ? [process.env.INCARNAMIND_EVAL_CACHE] : []),
]);
installHomeGuard(guard);

const failOnRefused = () => {
  const refused = guard.takeRefused();
  if (refused.length > 0) {
    throw new Error(
      `Writes under the real home folder were refused; give the test a temporary folder instead:\n${refused.join("\n")}`,
    );
  }
};
afterEach(failOnRefused);
afterAll(() => {
  failOnRefused();
  rmSync(home, { recursive: true, force: true, maxRetries: 3 });
});
