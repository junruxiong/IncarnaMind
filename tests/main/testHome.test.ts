import { homedir, tmpdir, userInfo } from "node:os";
import { join, sep } from "node:path";
import { describe, expect, it } from "vitest";
import { realHomeProblem } from "../../src/main/testHome";

describe("a test build's home folder", () => {
  const account = join(tmpdir(), "priya");

  it("refuses the account's own home folder, however it is written, and says what to do", () => {
    expect(realHomeProblem(account, account)).toBe(
      `This test build won't start on the real home folder, ${account}: give it a temporary HOME, as launchApp in e2e/app.ts does.`,
    );
    expect(realHomeProblem(`${account}${sep}`, account)).not.toBeNull();
    expect(realHomeProblem(join(account, "Documents", ".."), account)).not.toBeNull();
  });

  it("starts on any other folder, one inside the account's home included", () => {
    expect(realHomeProblem(join(tmpdir(), "incarnamind-home-1"), account)).toBeNull();
    expect(realHomeProblem(join(account, "test-home"), account)).toBeNull();
  });

  it("tells the account's home from HOME by the account, not by HOME", () => {
    expect(realHomeProblem(homedir())).toBeNull();
    expect(realHomeProblem(userInfo().homedir)).not.toBeNull();
  });
});
