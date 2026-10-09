import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import {
  createLoginShellProcesses,
  createProcessLauncher,
  parseEnvOutput,
  readLoginShellEnvironment,
} from "../../src/main/processes";
import { createTempDataFolder } from "../helpers/core";

/**
 * A stand-in for the User's shell, run as `<shell> -ilc <command>` like zsh:
 * its "profile" prints a greeting, sets variables and puts a folder first on
 * PATH, then it runs the command. Each run is counted in `runs`.
 */
async function fakeShell(profile: string[] = []) {
  const dir = await createTempDataFolder();
  const shell = join(dir, "fake-shell");
  const runs = join(dir, "runs");
  await writeFile(
    shell,
    [
      "#!/bin/sh",
      `printf x >> "${runs}"`,
      'echo "Last login: today (the profile talks)"',
      "export LOGIN_SHELL_MARKER=from-the-login-shell",
      'export PATH="/opt/fake/bin:$PATH"',
      ...profile,
      'eval "$2"',
      'echo "bye"',
    ].join("\n"),
    { mode: 0o755 },
  );
  const count = async () => (await readFile(runs, "utf8").catch(() => "")).length;
  return { shell, count };
}

const APP_ENV = { PATH: "/usr/bin:/bin", HOME: "/home/someone", APP_ONLY: "kept" };

describe("The login-shell environment", () => {
  test("is the login shell's, on top of the app's own, ignoring what the profile prints", async () => {
    const { shell } = await fakeShell([
      "export MULTI_LINE='first line\nsecond=line'",
      "export SHLVL=7",
    ]);

    const env = await readLoginShellEnvironment({ shell, platform: "darwin", env: APP_ENV });

    expect(env.LOGIN_SHELL_MARKER).toBe("from-the-login-shell");
    expect(env.PATH).toBe("/opt/fake/bin:/usr/bin:/bin");
    expect(env.APP_ONLY).toBe("kept");
    expect(env.MULTI_LINE).toBe("first line\nsecond=line");
    // What the shell sets for itself, or was told only for itself, isn't passed on.
    expect(env.SHLVL).toBeUndefined();
    expect(env.PWD).toBeUndefined();
    expect(env.DISABLE_AUTO_UPDATE).toBeUndefined();
    expect(Object.values(env).join("\n")).not.toMatch(/the profile talks|bye/);
  });

  test("falls back to the app's environment when the shell takes too long, and says so", async () => {
    const { shell } = await fakeShell(["sleep 5"]);
    const reportError = vi.fn();

    const started = Date.now();
    const env = await readLoginShellEnvironment({
      shell,
      platform: "linux",
      env: APP_ENV,
      timeoutMs: 200,
      reportError,
    });

    expect(Date.now() - started).toBeLessThan(3000);
    expect(env).toEqual(APP_ENV);
    expect(reportError).toHaveBeenCalledWith(
      expect.objectContaining({ message: expect.stringMatching(/longer than 200 ms/) }),
    );
  });

  test("falls back to the app's environment when the shell can't run", async () => {
    const reportError = vi.fn();

    const env = await readLoginShellEnvironment({
      shell: "/nonexistent/shell",
      platform: "linux",
      env: APP_ENV,
      reportError,
    });

    expect(env).toEqual(APP_ENV);
    expect(reportError).toHaveBeenCalledOnce();
  });

  test("on Windows is the app's own: there is no login shell to ask", async () => {
    const { shell, count } = await fakeShell();

    const env = await readLoginShellEnvironment({ shell, platform: "win32", env: APP_ENV });

    expect(env).toEqual(APP_ENV);
    expect(await count()).toBe(0);
  });

  test("is read once, and every process started gets it, plus its own variables", async () => {
    const { shell, count } = await fakeShell();
    const processes = createLoginShellProcesses({
      shell,
      platform: "darwin",
      env: { ...APP_ENV, PATH: process.env.PATH ?? "" },
    });
    const print =
      "process.stdout.write(JSON.stringify([process.env.LOGIN_SHELL_MARKER, process.env.OWN]))";

    const outputs = await Promise.all(
      ["a", "b"].map(async (own) => {
        const child = await processes.spawn(process.execPath, ["-e", print], { env: { OWN: own } });
        let output = "";
        child.stdout?.on("data", (chunk) => {
          output += chunk;
        });
        await new Promise((resolve) => child.on("close", resolve));
        return JSON.parse(output) as string[];
      }),
    );

    expect(outputs).toEqual([
      ["from-the-login-shell", "a"],
      ["from-the-login-shell", "b"],
    ]);
    expect(await count()).toBe(1);
  });

  test("variables it is told to leave out are left out of what it inherits, not of a process's own", async () => {
    const processes = createProcessLauncher(async () => ({
      PATH: process.env.PATH ?? "",
      GITHUB_TOKEN: "inherited",
      KEPT: "inherited",
    }));
    const print =
      "process.stdout.write(JSON.stringify([process.env.GITHUB_TOKEN, process.env.KEPT, process.env.OWN_TOKEN]))";

    const child = await processes.spawn(process.execPath, ["-e", print], {
      env: { OWN_TOKEN: "own" },
      omitEnv: (name) => name.endsWith("TOKEN"),
    });
    let output = "";
    child.stdout?.on("data", (chunk) => {
      output += chunk;
    });
    await new Promise((resolve) => child.on("close", resolve));

    expect(JSON.parse(output)).toEqual([null, "inherited", "own"]);
  });

  test("a command that isn't on its PATH is refused with ENOENT", async () => {
    const empty = await createTempDataFolder();
    const processes = createProcessLauncher(async () => ({ PATH: empty }));

    await expect(processes.spawn("npx", ["-y", "something"])).rejects.toMatchObject({
      code: "ENOENT",
    });
  });
});

describe("Reading env's output", () => {
  test("from env -0, values keep their line breaks", () => {
    const output = "noise MARKA=1\0B=two\nC=not a variable\0BASH_FUNC_f%%=() { :; }\0MARK noise";

    expect(parseEnvOutput(output, "MARK")).toEqual({ A: "1", B: "two\nC=not a variable" });
  });

  test("from plain env, keeps only what is between the marks, and skips names that aren't identifiers", () => {
    const output = [
      "profile noise",
      "MARKA=1",
      "BASH_FUNC_greet%%=() {  echo hi",
      "}",
      "B=two",
      "lines",
      "MARK trailing",
    ].join("\n");

    expect(parseEnvOutput(output, "MARK")).toEqual({ A: "1", B: "two\nlines" });
    expect(parseEnvOutput("nothing printed", "MARK")).toBeNull();
  });
});
