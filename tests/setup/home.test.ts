/**
 * The home guard (tests/setup/home.ts): every test file runs with a temporary
 * HOME, and a write under the real home folder is refused and fails the test
 * instead of landing there.
 */
import fs, { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, test } from "vitest";
import { createHomeGuard, installedHomeGuard, REAL_HOME_VARIABLE } from "../helpers/home";

const realHome = process.env[REAL_HOME_VARIABLE] ?? "";
const refusal = (path: string) => `Refused a write under the real home folder: ${path}.`;

describe("the home guard", () => {
  test("HOME and the folders named after it are a temporary folder, not the real home", () => {
    expect(realHome).not.toBe("");
    expect(homedir()).not.toBe(realHome);
    expect(homedir().startsWith(tmpdir())).toBe(true);
    for (const name of [
      "USERPROFILE",
      "XDG_CONFIG_HOME",
      "XDG_DATA_HOME",
      "XDG_CACHE_HOME",
      "XDG_STATE_HOME",
    ]) {
      expect(process.env[name]?.startsWith(homedir()), name).toBe(true);
    }
  });

  test("a write under the real home folder is refused, named, and lands nowhere", async () => {
    // In a folder that isn't there: a write the guard let through would fail, not land.
    const missing = join(realHome, `.incarnamind-home-guard-${process.pid}`);
    const file = join(missing, "file.txt");

    expect(() => writeFileSync(file, "x")).toThrow(refusal(file));
    await expect(writeFile(file, "x")).rejects.toThrow(refusal(file));
    expect(() => fs.writeFile(file, "x", () => undefined)).toThrow(refusal(file));
    expect(() => mkdirSync(join(missing, "folder"))).toThrow(refusal(join(missing, "folder")));
    expect(() => fs.openSync(file, "a")).toThrow(refusal(file));
    expect(() => readFileSync(file)).toThrow(/ENOENT/);

    expect(existsSync(missing)).toBe(false);
    // Taken here, so the setup file's check after each test doesn't fail this one.
    expect(installedHomeGuard().takeRefused()).toEqual([
      file,
      file,
      file,
      join(missing, "folder"),
      file,
    ]);
  });

  test("it refuses paths in the home folder outside the writable folders, however they are written", () => {
    const home = join(tmpdir(), "someone");
    const repository = join(home, "code", "IncarnaMind");
    const guard = createHomeGuard(home, [repository, tmpdir()]);

    expect(() => guard.check(join(repository, "node_modules", ".cache", "x"))).not.toThrow();
    expect(() => guard.check(join(tmpdir(), "elsewhere", "x"))).not.toThrow();
    expect(() => guard.check(`${home}-else`)).not.toThrow();
    expect(() => guard.check(42)).not.toThrow();

    expect(() => guard.check(join(repository, "..", "x"))).toThrow(
      refusal(join(home, "code", "x")),
    );
    expect(() => guard.check(pathToFileURL(join(home, ".codex", "config.toml")))).toThrow(
      refusal(join(home, ".codex", "config.toml")),
    );
    expect(() => guard.check(Buffer.from(join(home, "Documents")))).toThrow(
      refusal(join(home, "Documents")),
    );
    expect(() => guard.check(home)).toThrow(refusal(home));
    expect(guard.takeRefused()).toHaveLength(4);
    expect(guard.takeRefused()).toEqual([]);
  });
});
