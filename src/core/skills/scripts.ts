/**
 * Skill scripts (#41): running one of a Skill's scripts as a child process,
 * for the `run_skill_script` Tool. Answers ask the User first (see
 * ../approvals); this module only runs what was allowed.
 *
 * - The interpreter is chosen by the file's extension: Python through
 *   `python3` (`python` on Windows) and shell scripts through `bash`, both
 *   found on the User's login-shell PATH; JavaScript through the app's own
 *   Node.js (Electron's, as plain Node), so nothing needs installing for it.
 *   Shell scripts on Windows, TypeScript and other kinds give a plain error,
 *   and so does an interpreter that isn't installed, naming what to install.
 * - Each run gets a new, empty temporary working folder, removed afterwards.
 *   The script finds its Skill's folder in the `SKILL_DIR` environment variable.
 * - It runs with no input, until it exits or its timeout. On a timeout, a
 *   Stop, or IncarnaMind closing, it is stopped with every process it started
 *   (its process group on macOS and Linux, its process tree on Windows).
 * - What it writes is kept up to `SKILL_SCRIPT_LIMITS.maxOutputBytes` per
 *   stream: the start of its standard output, the end of its error output
 *   (where errors usually are), with a note when cut.
 *
 * There is no sandbox (v1): a script can do whatever the User can, including
 * reaching the network. That is why each run asks first.
 */
import { type ChildProcess, spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { StringDecoder } from "node:string_decoder";
import type { ProcessLauncher, ScriptRuntimes } from "../adapters";
import { SKILL_SCRIPT_LIMITS, type SkillScriptRun } from "../api";

/**
 * Why a script can't run, in plain language: for the model, and for the
 * Tool-call card. E.g. its interpreter isn't installed.
 */
class SkillScriptError extends Error {
  override name = "SkillScriptError";
}

/** What runs a script: the command, and what its environment needs. */
export interface Interpreter {
  command: string;
  env: Readonly<Record<string, string>>;
  /** What to install when the command isn't there, e.g. "Python 3". */
  install: string;
}

/** One script to run, already checked to be a script file inside its Skill's folder. */
export interface ScriptToRun {
  /** The Skill's folder: absolute. Given to the script as `SKILL_DIR`. */
  skillDir: string;
  /** The script's path inside the Skill, "/" between folders, e.g. "scripts/convert.py". */
  script: string;
  args: readonly string[];
  timeoutMs: number;
  /** Stops the script, e.g. when the User stops the Answer. */
  signal: AbortSignal;
}

export interface ScriptRunnerOptions {
  processes: ProcessLauncher;
  runtimes?: ScriptRuntimes;
  /** Where each run's working folder is made. */
  tempDir: string;
  reportError(error: unknown): void;
}

const extensionOf = (path: string): string => {
  const name = path.split("/").at(-1) ?? path;
  const dot = name.lastIndexOf(".");
  return dot <= 0 ? "" : name.slice(dot + 1).toLowerCase();
};

const supported = (platform: NodeJS.Platform) =>
  platform === "win32"
    ? "Python (.py) and JavaScript (.js, .mjs, .cjs) scripts can run on Windows."
    : "Python (.py), JavaScript (.js, .mjs, .cjs) and shell (.sh) scripts can run.";

type Runtimes = Required<ScriptRuntimes>;

/** The defaults: this process's own Node.js, as plain Node even inside Electron, on this OS. */
function runtimesFrom(given: ScriptRuntimes = {}): Runtimes {
  return {
    node: given.node ?? { command: process.execPath, env: { ELECTRON_RUN_AS_NODE: "1" } },
    platform: given.platform ?? process.platform,
  };
}

/**
 * What runs the script at `path` (inside its Skill), by its extension.
 * Throws `SkillScriptError` for one that can't run here.
 */
function interpreterFor(path: string, given?: ScriptRuntimes): Interpreter {
  const { node, platform } = runtimesFrom(given);
  const windows = platform === "win32";
  switch (extensionOf(path)) {
    case "py":
      return {
        command: windows ? "python" : "python3",
        // No __pycache__ written into the Skill's folder; UTF-8 output on Windows too.
        env: { PYTHONDONTWRITEBYTECODE: "1", PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
        install: "Python 3",
      };
    case "js":
    case "mjs":
    case "cjs":
      return { command: node.command, env: { ...node.env }, install: "Node.js" };
    case "sh":
    case "bash":
      if (windows) {
        throw new SkillScriptError(
          `${path} is a shell script, and shell scripts can't run on Windows. ${supported(platform)}`,
        );
      }
      return { command: "bash", env: {}, install: "bash" };
    case "ts":
    case "mts":
    case "cts":
      throw new SkillScriptError(
        `${path} is a TypeScript script, which IncarnaMind can't run yet. ${supported(platform)}`,
      );
    default:
      throw new SkillScriptError(
        `${path} isn't a kind of script IncarnaMind can run. ${supported(platform)}`,
      );
  }
}

/**
 * The arguments the model gave, checked: a list of text (numbers and
 * true/false are taken as text), within `SKILL_SCRIPT_LIMITS`.
 */
export function parseScriptArgs(value: unknown): string[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) throw new SkillScriptError("args must be a list of strings.");
  const args = value.map((each) => {
    if (typeof each === "string") return each;
    if (typeof each === "number" || typeof each === "boolean") return String(each);
    throw new SkillScriptError("Each of the args must be a string.");
  });
  if (args.length > SKILL_SCRIPT_LIMITS.maxArgs) {
    throw new SkillScriptError(`A script takes ${SKILL_SCRIPT_LIMITS.maxArgs} args at most.`);
  }
  const chars = args.reduce((sum, each) => sum + each.length, 0);
  if (chars > SKILL_SCRIPT_LIMITS.maxArgsChars) {
    throw new SkillScriptError(
      `A script's args can have ${SKILL_SCRIPT_LIMITS.maxArgsChars} characters at most in all.`,
    );
  }
  if (args.some((each) => each.includes("\0"))) {
    throw new SkillScriptError("The args can't contain NUL characters.");
  }
  return args;
}

const isNotFound = (error: unknown) =>
  typeof error === "object" && error !== null && (error as { code?: unknown }).code === "ENOENT";

/** "python3" for "/usr/bin/python3" or "C:\\…\\python.exe". */
const commandName = (command: string) =>
  (command.split(/[\\/]/).at(-1) ?? command).replace(/\.(exe|cmd|bat)$/i, "");

/**
 * Stops a process and every process it started, at once: its process group
 * on macOS and Linux (it was started as the group's leader), its process
 * tree on Windows. Best effort: one that has gone already is fine.
 */
function stopProcessTree(child: ChildProcess): void {
  const { pid } = child;
  if (pid === undefined) return;
  const killChild = () => {
    try {
      child.kill("SIGKILL");
    } catch {
      // Gone already.
    }
  };
  if (process.platform === "win32") {
    try {
      const taskkill = spawn("taskkill", ["/pid", String(pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true,
      });
      taskkill.on("error", killChild);
    } catch {
      killChild();
    }
    return;
  }
  try {
    process.kill(-pid, "SIGKILL");
  } catch {
    // Not a group leader after all (a launcher that ignored `processGroup`), or gone.
    killChild();
  }
}

/** The first `limit` bytes of a stream, and whether there was more. */
class Head {
  private readonly chunks: Buffer[] = [];
  private kept = 0;
  truncated = false;
  constructor(private readonly limit: number) {}
  push(chunk: Buffer): void {
    if (this.kept >= this.limit) {
      if (chunk.length > 0) this.truncated = true;
      return;
    }
    const room = this.limit - this.kept;
    if (chunk.length > room) this.truncated = true;
    const part = chunk.subarray(0, room);
    this.chunks.push(part);
    this.kept += part.length;
  }
  /** As text; a character cut in half at the end is left out. */
  text(): string {
    return new StringDecoder("utf8").write(Buffer.concat(this.chunks));
  }
}

/** The last `limit` bytes of a stream, and whether there was more. */
class Tail {
  private buffer = Buffer.alloc(0);
  truncated = false;
  constructor(private readonly limit: number) {}
  push(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    if (this.buffer.length > this.limit) {
      this.truncated = true;
      this.buffer = Buffer.from(this.buffer.subarray(this.buffer.length - this.limit));
    }
  }
  /** As text; a character cut in half at the start is left out. */
  text(): string {
    let start = 0;
    // UTF-8 continuation bytes are 10xxxxxx.
    while (start < this.buffer.length && ((this.buffer[start] as number) & 0xc0) === 0x80) start++;
    return this.buffer.subarray(start).toString("utf8");
  }
}

/** A run that couldn't start, for the Tool-call card: why, in plain language. */
export const scriptNotRun = (error: string): SkillScriptRun => ({
  exitCode: null,
  timedOut: false,
  stdout: "",
  stderr: "",
  stdoutTruncated: false,
  stderrTruncated: false,
  error,
});

/** What the model is told about a run: how it ended, then what it wrote. */
export function scriptResultText(script: string, run: SkillScriptRun, timeoutSeconds: number) {
  const limit = SKILL_SCRIPT_LIMITS.maxOutputBytes;
  const seconds = `${timeoutSeconds} second${timeoutSeconds === 1 ? "" : "s"}`;
  const ending = run.timedOut
    ? `${script} ran longer than ${seconds}, so it was stopped, with every process it started. What it wrote until then:`
    : run.exitCode === null
      ? `${script} was stopped before it finished. What it wrote until then:`
      : `${script} exited with code ${run.exitCode}.`;
  const stream = (name: string, text: string, cut: boolean, which: string) => [
    `<${name}>`,
    ...(text ? [text.replace(/\n$/, "")] : []),
    ...(cut ? [`[… truncated: only the ${which} ${limit} bytes of ${name} are kept.]`] : []),
    `</${name}>`,
  ];
  return [
    ending,
    ...stream("stdout", run.stdout, run.stdoutTruncated, "first"),
    ...stream("stderr", run.stderr, run.stderrTruncated, "last"),
  ].join("\n");
}

/** How long a stopped script's output may stay open before the run ends without it. */
const STOP_GRACE_MS = 2000;

/** A run going on, as `stopAll` and `close` stop it: at once, or as soon as it has started. */
interface Running {
  stopAsked: boolean;
  stop(): void;
}

export function createScriptRunner(options: ScriptRunnerOptions) {
  const runtimes = runtimesFrom(options.runtimes);
  const running = new Set<Running>();
  let closed = false;

  /** Waits for the process to end, keeping what it writes; stops it on a timeout or `signal`. */
  const collect = (child: ChildProcess, request: ScriptToRun, entry: Running) =>
    new Promise<SkillScriptRun>((resolve) => {
      const stdout = new Head(SKILL_SCRIPT_LIMITS.maxOutputBytes);
      const stderr = new Tail(SKILL_SCRIPT_LIMITS.maxOutputBytes);
      let timedOut = false;
      let stopped = false;
      let settled = false;
      const timer = setTimeout(() => {
        timedOut = true;
        stop();
      }, request.timeoutMs);
      const finish = (code: number | null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        request.signal.removeEventListener("abort", stop);
        resolve({
          exitCode: stopped ? null : code,
          timedOut,
          stdout: stdout.text(),
          stderr: stderr.text(),
          stdoutTruncated: stdout.truncated,
          stderrTruncated: stderr.truncated,
          error: null,
        });
      };
      function stop() {
        if (stopped) return;
        stopped = true;
        stopProcessTree(child);
        // A process that escaped (e.g. it started a session of its own) may keep the output
        // open: after a grace period the run ends anyway, without it.
        setTimeout(() => {
          if (settled) return;
          child.stdout?.destroy();
          child.stderr?.destroy();
          finish(child.exitCode);
        }, STOP_GRACE_MS).unref();
      }
      entry.stop = stop;
      // It gets no input.
      child.stdin?.on("error", () => undefined);
      child.stdin?.end();
      child.stdout?.on("data", (chunk: Buffer) => stdout.push(chunk));
      child.stderr?.on("data", (chunk: Buffer) => stderr.push(chunk));
      request.signal.addEventListener("abort", stop, { once: true });
      // Once it has exited and its output has closed (a process it started may keep that open).
      child.once("close", (code) => finish(code));
      if (request.signal.aborted || entry.stopAsked || closed) stop();
    });

  return {
    /** Throws `SkillScriptError` if a script like this can't run here, before anyone is asked. */
    check(script: string): void {
      interpreterFor(script, runtimes);
    },

    /**
     * Runs a script in a new temporary folder, removed afterwards, and
     * resolves with how it went. Rejects with `SkillScriptError` when it can't
     * start, e.g. its interpreter isn't installed.
     */
    async run(request: ScriptToRun): Promise<SkillScriptRun> {
      const interpreter = interpreterFor(request.script, runtimes);
      if (closed) throw new SkillScriptError("IncarnaMind is closing, so the script didn't run.");
      request.signal.throwIfAborted();
      const scriptPath = join(request.skillDir, ...request.script.split("/"));
      const workDir = await mkdtemp(join(options.tempDir, "incarnamind-script-"));
      // Until it has started there is nothing to stop: `collect` stops it then.
      const entry: Running = {
        stopAsked: false,
        stop: () => {
          entry.stopAsked = true;
        },
      };
      running.add(entry);
      try {
        let child: ChildProcess;
        try {
          child = await options.processes.spawn(
            interpreter.command,
            [scriptPath, ...request.args],
            {
              cwd: workDir,
              env: { ...interpreter.env, SKILL_DIR: request.skillDir },
              processGroup: true,
            },
          );
        } catch (error) {
          if (!isNotFound(error)) throw error;
          throw new SkillScriptError(
            `${commandName(interpreter.command)} not found: install ${interpreter.install} to run ${request.script}.`,
          );
        }
        return await collect(child, request, entry);
      } finally {
        running.delete(entry);
        await rm(workDir, { recursive: true, force: true, maxRetries: 3 }).catch(
          options.reportError,
        );
      }
    },

    /** Stops every script running, e.g. when the User turns scripts off; their folders go once they have. */
    stopAll(): void {
      for (const entry of [...running]) entry.stop();
    },

    /** Stops every script running, and runs no more, e.g. when the app quits. */
    close(): void {
      closed = true;
      for (const entry of [...running]) entry.stop();
    },
  };
}
