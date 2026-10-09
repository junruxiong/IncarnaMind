/**
 * The Executor contract (#61): what every Executor does with a request,
 * whatever its sandbox level. These are the Skill-script suite's cases
 * (tests/core/skillScripts.test.ts), run against the Executor directly:
 * a new temporary working folder, removed afterwards; a timeout and a stop
 * that end every process the program started; output caps; a missing
 * command. Today's only Executor is the local one at level "none"; a later
 * "os" one runs this same contract.
 */
import { existsSync } from "node:fs";
import { readdir, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, onTestFinished, test, vi } from "vitest";
import type { ExecRequest, Executor, ProcessLauncher } from "../../src/core";
import { createLocalExecutor, declaredAccess, WORKING_FOLDER } from "../../src/core/execution";
import { createProcessLauncher } from "../../src/main/processes";
import { isRunning } from "../helpers/connectors";
import { createTempDataFolder } from "../helpers/core";

/** Starts processes with PATH, HOME and the temporary folders of the test process only. */
function smallEnvironment(extra: Record<string, string> = {}): ProcessLauncher {
  const keep = ["PATH", "Path", "HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP", "SystemRoot"];
  const env: Record<string, string> = {};
  for (const name of keep) {
    const value = process.env[name];
    if (value !== undefined) env[name] = value;
  }
  return createProcessLauncher(async () => ({ ...env, ...extra }));
}

/** Node.js running `code`, with `args` after it: `process.argv[1]` is the first. */
const node = (code: string, ...args: string[]) => ({
  command: process.execPath,
  args: ["-e", code, "--", ...args],
});

/** Reports its arguments, its environment and its working folder, and leaves a file there. */
const REPORT = [
  'const { readdirSync, writeFileSync } = require("node:fs");',
  'const before = readdirSync(".");',
  'writeFileSync("made-by-the-program.txt", "x");',
  "console.log(JSON.stringify({ args: process.argv.slice(1), marker: process.env.MARKER, cwd: process.cwd(), before }));",
].join("\n");

/** Starts a process of its own, writes both pids to the file it's given, then never ends. */
const SLEEP = [
  'const { spawn } = require("node:child_process");',
  'const { writeFileSync } = require("node:fs");',
  'const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: "inherit" });',
  "writeFileSync(process.argv[1], JSON.stringify({ pid: process.pid, child: child.pid }));",
  'console.log("started");',
  "setInterval(() => {}, 1000);",
].join("\n");

/** Starts a process in a session of its own, which a stop can't reach, holding the output open. */
const ESCAPE = [
  'const { spawn } = require("node:child_process");',
  'const { writeFileSync } = require("node:fs");',
  'const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: "inherit", detached: true });',
  "writeFileSync(process.argv[1], JSON.stringify({ pid: process.pid, child: child.pid }));",
  "setInterval(() => {}, 1000);",
].join("\n");

/** The pids SLEEP or ESCAPE wrote, once it has. */
async function pidsIn(file: string): Promise<{ pid: number; child: number }> {
  let pids: { pid: number; child: number } | undefined;
  await vi.waitFor(
    async () => {
      pids = JSON.parse(await readFile(file, "utf8"));
    },
    { timeout: 10_000, interval: 25 },
  );
  return pids as { pid: number; child: number };
}

/** Waits until neither process runs any more. */
async function expectStopped(pids: { pid: number; child: number }) {
  await vi.waitFor(
    () => {
      expect(isRunning(pids.pid)).toBe(false);
      expect(isRunning(pids.child)).toBe(false);
    },
    { timeout: 5_000, interval: 25 },
  );
}

/** A request as a Skill script's would be, for `program`, with what the case changes. */
function request(
  program: { command: string; args: string[] },
  overrides: Partial<ExecRequest> = {},
): ExecRequest {
  return {
    ...program,
    env: {},
    allow: { read: [WORKING_FOLDER], write: [WORKING_FOLDER], network: "any" },
    timeoutMs: 10_000,
    maxOutputBytes: 20_000,
    signal: new AbortController().signal,
    ...overrides,
  };
}

/** What a contract case needs: an Executor whose temporary working folders go into `tempDir`. */
interface Subject {
  executor: Executor;
  tempDir: string;
}

/**
 * The cases every Executor must pass. `make` builds one whose temporary
 * working folders go into `tempDir`, starting processes with `processes`.
 */
function executorContract(
  name: string,
  make: (options: { tempDir: string; processes: ProcessLauncher }) => Executor,
) {
  const setUp = async (processes = smallEnvironment()): Promise<Subject> => {
    const tempDir = await createTempDataFolder();
    return { executor: make({ tempDir, processes }), tempDir };
  };

  describe(`The Executor contract: ${name}`, { timeout: 30_000 }, () => {
    test("it runs the program with its arguments as they are, its environment added, in a new, empty temporary folder that is removed afterwards", async () => {
      const { executor, tempDir } = await setUp();
      const args = ["two words", "--flag", "ünïcode $HOME"];

      const run = await executor.run(request(node(REPORT, ...args), { env: { MARKER: "set" } }));

      expect(run).toMatchObject({
        exitCode: 0,
        timedOut: false,
        stderr: "",
        stdoutTruncated: false,
        stderrTruncated: false,
      });
      const report = JSON.parse(run.stdout);
      // No shell in between: "$HOME" stays as it was.
      expect(report.args).toEqual(args);
      expect(report.marker).toBe("set");
      expect(report.cwd.startsWith(await realpath(tempDir))).toBe(true);
      expect(report.before).toEqual([]);
      expect(existsSync(report.cwd)).toBe(false);
      expect(await readdir(tempDir)).toEqual([]);
    });

    test("a working folder it is given is used, and kept", async () => {
      const { executor, tempDir } = await setUp();
      const cwd = await createTempDataFolder();

      const run = await executor.run(request(node(REPORT), { cwd }));

      expect(JSON.parse(run.stdout).cwd).toBe(await realpath(cwd));
      expect(await readdir(cwd)).toEqual(["made-by-the-program.txt"]);
      expect(await readdir(tempDir)).toEqual([]);
    });

    test("it gets no input", async () => {
      const { executor } = await setUp();
      const code =
        'process.stdin.resume(); process.stdin.on("end", () => console.log("no input"));';

      const run = await executor.run(request(node(code), { timeoutMs: 5_000 }));

      expect(run).toMatchObject({ exitCode: 0, timedOut: false, stdout: "no input\n" });
    });

    test("a program that fails: its exit code and error output", async () => {
      const { executor } = await setUp();

      const run = await executor.run(request(node('console.error("boom"); process.exitCode = 3;')));

      expect(run).toEqual({
        exitCode: 3,
        timedOut: false,
        stdout: "",
        stderr: "boom\n",
        stdoutTruncated: false,
        stderrTruncated: false,
      });
    });

    test("one that runs past its timeout is stopped with every process it started; what it wrote is kept", async () => {
      const { executor, tempDir } = await setUp();
      const pidFile = join(await createTempDataFolder(), "pids.json");

      const started = Date.now();
      const run = await executor.run(request(node(SLEEP, pidFile), { timeoutMs: 1_000 }));

      expect(Date.now() - started).toBeLessThan(10_000);
      expect(run).toMatchObject({ exitCode: null, timedOut: true, stdout: "started\n" });
      await expectStopped(await pidsIn(pidFile));
      expect(await readdir(tempDir)).toEqual([]);
    });

    test("its signal stops it at once, with every process it started", async () => {
      const { executor, tempDir } = await setUp();
      const pidFile = join(await createTempDataFolder(), "pids.json");
      const stop = new AbortController();

      const running = executor.run(request(node(SLEEP, pidFile), { signal: stop.signal }));
      const pids = await pidsIn(pidFile);
      expect(isRunning(pids.pid)).toBe(true);
      const stoppedAt = Date.now();
      stop.abort();
      const run = await running;

      expect(Date.now() - stoppedAt).toBeLessThan(5_000);
      expect(run).toMatchObject({ exitCode: null, timedOut: false });
      await expectStopped(pids);
      expect(await readdir(tempDir)).toEqual([]);
    });

    test("a signal stopped already: it doesn't run", async () => {
      const { executor, tempDir } = await setUp();
      const log = join(await createTempDataFolder(), "ran.log");
      const stop = new AbortController();
      stop.abort(new Error("Stopped before it started."));

      await expect(
        executor.run(
          request(node('require("node:fs").writeFileSync(process.argv[1], "ran")', log), {
            signal: stop.signal,
          }),
        ),
      ).rejects.toThrow("Stopped before it started.");

      await new Promise((resolve) => setTimeout(resolve, 200));
      expect(existsSync(log)).toBe(false);
      expect(await readdir(tempDir)).toEqual([]);
    });

    test.skipIf(process.platform === "win32")(
      "a process that escapes the stop (a session of its own) doesn't keep the run waiting",
      async () => {
        const { executor } = await setUp();
        const pidFile = join(await createTempDataFolder(), "pids.json");

        const started = Date.now();
        const run = await executor.run(request(node(ESCAPE, pidFile), { timeoutMs: 1_000 }));
        const pids = await pidsIn(pidFile);
        // The test's own clean-up: what escaped is beyond the Executor.
        onTestFinished(() => {
          try {
            process.kill(pids.child, "SIGKILL");
          } catch {
            // Gone already.
          }
        });

        expect(Date.now() - started).toBeLessThan(10_000);
        expect(run).toMatchObject({ exitCode: null, timedOut: true });
        await vi.waitFor(() => expect(isRunning(pids.pid)).toBe(false));
      },
    );

    test("output is cut per stream, with a flag: the start of the standard output, the end of the error output", async () => {
      const { executor } = await setUp();
      const flood = [
        'process.stdout.write("S".repeat(30000) + "TAIL-OUT\\n");',
        'process.stderr.write("HEAD-ERR\\n" + "E".repeat(30000));',
      ].join("\n");

      const run = await executor.run(request(node(flood)));

      expect(run).toEqual({
        exitCode: 0,
        timedOut: false,
        stdout: "S".repeat(20_000),
        stderr: "E".repeat(20_000),
        stdoutTruncated: true,
        stderrTruncated: true,
      });
    });

    test("a character cut in half where the output is cut is left out", async () => {
      const { executor } = await setUp();
      // "abcé" and "éxyz" are five bytes each in UTF-8: "é" is two.
      const code = 'process.stdout.write("abcé"); process.stderr.write("éxyz");';

      const run = await executor.run(request(node(code), { maxOutputBytes: 4 }));

      expect(run).toMatchObject({
        stdout: "abc",
        stderr: "xyz",
        stdoutTruncated: true,
        stderrTruncated: true,
      });
    });

    test("a command that isn't found rejects with ENOENT, and its working folder is removed", async () => {
      const { executor, tempDir } = await setUp();

      await expect(
        executor.run(request({ command: "incarnamind-no-such-command", args: [] })),
      ).rejects.toMatchObject({ code: "ENOENT" });

      expect(await readdir(tempDir)).toEqual([]);
    });
  });
}

executorContract("the local executor (sandbox level none)", ({ tempDir, processes }) =>
  createLocalExecutor({
    processes,
    tempDir,
    reportError: (error) => {
      throw error;
    },
  }),
);

describe("Declared access, from the sandbox level", () => {
  const allow = {
    read: ["/skills/toolbox", WORKING_FOLDER],
    write: [WORKING_FOLDER],
    network: "none",
  } as const;

  test('the local executor is at level "none"', async () => {
    const executor = createLocalExecutor({
      processes: smallEnvironment(),
      tempDir: await createTempDataFolder(),
      reportError: () => undefined,
    });
    expect(executor.level).toBe("none");
  });

  test('at "none" nothing is enforced: a program can read, write and reach anything, whatever it was allowed', () => {
    expect(declaredAccess("none", allow)).toEqual({
      confined: false,
      read: "anywhere",
      write: "anywhere",
      network: "any",
    });
  });

  test('from "os" up, exactly what it was allowed', () => {
    for (const level of ["os", "container", "remote"] as const) {
      expect(declaredAccess(level, allow)).toEqual({ confined: true, ...allow });
    }
  });
});
