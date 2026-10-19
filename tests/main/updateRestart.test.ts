import { describe, expect, test } from "vitest";
import { restartToInstall } from "../../src/main/updateRestart";

describe("restarting to install an update", () => {
  test("lets go of the data folder before the updater quits and installs", () => {
    const calls: string[] = [];
    restartToInstall({ quitAndInstall: () => calls.push("quitAndInstall") }, () =>
      calls.push("release"),
    );
    expect(calls).toEqual(["release", "quitAndInstall"]);
  });
});
