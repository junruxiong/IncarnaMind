import { userInfo } from "node:os";
import { resolve } from "node:path";

/**
 * Why a test build won't start on `home`, or null when it may: `home` is the
 * account's own home folder, which `os.userInfo()` reads without following
 * HOME, so a smoke test's agents would write their settings into the User's.
 */
export function realHomeProblem(home: string, accountHome = userInfo().homedir): string | null {
  return resolve(home) === resolve(accountHome)
    ? `This test build won't start on the real home folder, ${home}: give it a temporary HOME, as launchApp in e2e/app.ts does.`
    : null;
}
