/**
 * Starts child processes the way the User's terminal would (the core's
 * `ProcessLauncher`): with their login shell's environment. An app opened
 * from the Dock or a desktop launcher gets only a minimal PATH, so `npx`,
 * `uvx` and the like, installed through Homebrew, nvm or the User's shell
 * profile, wouldn't be found.
 *
 * On macOS and Linux the login shell is asked once, through `$SHELL -ilc env`
 * with a timeout, and its answer is kept for the app's lifetime. Windows has
 * no login shell to ask: the app's own environment is used, which already
 * comes from the User's settings.
 *
 * No Electron imports: tests run this in Node.
 */
import { type ChildProcess, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { userInfo } from "node:os";
import crossSpawn from "cross-spawn";
import type { ProcessLauncher } from "../core";

export type Environment = Record<string, string>;

export interface LoginShellOptions {
  /** Defaults to `process.platform`. */
  platform?: NodeJS.Platform;
  /**
   * The User's shell. Defaults to $SHELL, else the shell their account
   * records, else /bin/zsh on macOS and /bin/sh elsewhere.
   */
  shell?: string;
  /** The app's own environment. Defaults to `process.env`. */
  env?: NodeJS.ProcessEnv;
  /** How long the shell may take to start, run the User's profile and print its environment. */
  timeoutMs?: number;
  /** Told when the login shell can't be read; the app's own environment is used instead. */
  reportError?(error: unknown): void;
}

/** Generous: some shell profiles (nvm, conda, oh-my-zsh) take a few seconds. */
export const LOGIN_SHELL_TIMEOUT_MS = 10_000;

/**
 * Variables a child mustn't inherit: those the shell sets for itself, and
 * Electron's, which would make an Electron-based tool run as plain Node.
 */
const DROPPED = new Set([
  "_",
  "SHLVL",
  "PWD",
  "OLDPWD",
  "ELECTRON_RUN_AS_NODE",
  "ELECTRON_NO_ATTACH_CONSOLE",
  "ELECTRON_ENABLE_LOGGING",
  "ELECTRON_ENABLE_STACK_DUMPING",
]);

/**
 * Settings that keep common shell setups from waiting for input or taking
 * over: oh-my-zsh's update prompt and tmux auto-start (as VS Code does).
 */
const QUIET_SHELL: Environment = {
  DISABLE_AUTO_UPDATE: "true",
  ZSH_TMUX_AUTOSTARTED: "true",
  ZSH_TMUX_AUTOSTART: "false",
};

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

function definedOnly(env: NodeJS.ProcessEnv): Environment {
  const result: Environment = {};
  for (const [name, value] of Object.entries(env)) if (value !== undefined) result[name] = value;
  return result;
}

/** The User's shell as their account records it, for an app started without $SHELL. */
function accountShell(): string | undefined {
  try {
    return userInfo().shell ?? undefined;
  } catch {
    return undefined;
  }
}

function withoutDropped(env: Environment): Environment {
  const result: Environment = {};
  for (const [name, value] of Object.entries(env)) if (!DROPPED.has(name)) result[name] = value;
  return result;
}

/**
 * The variables `env` printed between two `mark`s, or null when the marks
 * aren't there (the shell failed before running the command). A profile may
 * print its own output around them; that is ignored. Variables whose names
 * aren't identifiers (bash's exported functions, `BASH_FUNC_x%%`) are left out.
 *
 * `env -0` ends each variable with a NUL, so values may hold line breaks.
 * Plain `env` (where `-0` isn't supported) ends them with a line break: a
 * line that doesn't start a new variable then continues the previous value.
 */
export function parseEnvOutput(output: string, mark: string): Environment | null {
  const start = output.indexOf(mark);
  const end = output.lastIndexOf(mark);
  if (start < 0 || end <= start) return null;
  const body = output.slice(start + mark.length, end);
  const env: Environment = {};
  if (body.includes("\0")) {
    for (const entry of body.split("\0")) {
      const equals = entry.indexOf("=");
      const name = entry.slice(0, Math.max(0, equals));
      if (IDENTIFIER.test(name)) env[name] = entry.slice(equals + 1);
    }
    return env;
  }
  let current: string | null = null;
  for (const line of body.replace(/\n$/, "").split("\n")) {
    const equals = line.indexOf("=");
    const name = equals > 0 ? line.slice(0, equals) : "";
    if (name && !/\s/.test(name)) {
      current = IDENTIFIER.test(name) ? name : null;
      if (current !== null) env[current] = line.slice(equals + 1);
    } else if (current !== null) {
      env[current] += `\n${line}`;
    }
  }
  return env;
}

/**
 * The User's login-shell environment, on top of the app's own: what a
 * terminal would give a program. On Windows, or if the shell can't be read
 * (it fails, prints nothing usable, or takes longer than the timeout), the
 * app's own environment.
 */
export function readLoginShellEnvironment(options: LoginShellOptions = {}): Promise<Environment> {
  const base = definedOnly(options.env ?? process.env);
  const platform = options.platform ?? process.platform;
  if (platform === "win32") return Promise.resolve(withoutDropped(base));

  const shell =
    options.shell ??
    base.SHELL ??
    accountShell() ??
    (platform === "darwin" ? "/bin/zsh" : "/bin/sh");
  const mark = `__INCARNAMIND_ENV_${randomUUID().replaceAll("-", "")}__`;
  const fallback = (error: unknown) => {
    options.reportError?.(error);
    return withoutDropped(base);
  };

  const timeoutMs = options.timeoutMs ?? LOGIN_SHELL_TIMEOUT_MS;
  // `env -0` where it is supported (macOS, GNU, BusyBox), plain `env` otherwise.
  const command = `printf '%s' '${mark}'; env -0 2>/dev/null || env; printf '%s' '${mark}'`;

  return new Promise((resolve) => {
    let output = "";
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (env: Environment) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(env);
    };
    let child: ChildProcess;
    try {
      // -i and -l: the profile files a terminal's shell reads.
      child = spawn(shell, ["-ilc", command], {
        env: { ...base, ...QUIET_SHELL },
        stdio: ["ignore", "pipe", "ignore"],
        windowsHide: true,
      });
    } catch (error) {
      finish(fallback(error));
      return;
    }
    timer = setTimeout(() => {
      // Interactive shells ignore SIGTERM, so a shell that hangs is killed outright.
      child.kill("SIGKILL");
      finish(fallback(new Error(`${shell} took longer than ${timeoutMs} ms.`)));
    }, timeoutMs);
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      output += chunk;
    });
    child.on("error", (error) => finish(fallback(error)));
    child.on("close", (code) => {
      const env = parseEnvOutput(output, mark);
      if (!env) {
        finish(fallback(new Error(`${shell} printed no environment (exit code ${code}).`)));
        return;
      }
      // Only the shell was meant to see these.
      for (const name of Object.keys(QUIET_SHELL)) if (!(name in base)) delete env[name];
      // Some profiles make the shell exit with an error after the command ran: the output still counts.
      finish(withoutDropped({ ...base, ...env }));
    });
  });
}

/**
 * A `ProcessLauncher` that starts each process with `environment()` plus the
 * caller's variables, looking the command up on that environment's PATH. On
 * Windows, cross-spawn runs `.cmd` shims such as `npx.cmd`.
 */
export function createProcessLauncher(environment: () => Promise<Environment>): ProcessLauncher {
  return {
    async spawn(command, args, options = {}) {
      const env = { ...(await environment()), ...options.env };
      return new Promise<ChildProcess>((resolve, reject) => {
        let child: ChildProcess;
        try {
          child = crossSpawn(command, [...args], {
            cwd: options.cwd,
            env,
            stdio: "pipe",
            windowsHide: true,
          });
        } catch (error) {
          reject(error);
          return;
        }
        const onError = (error: Error) => {
          child.off("spawn", onSpawn);
          reject(error);
        };
        const onSpawn = () => {
          child.off("error", onError);
          // Later errors go to the caller's listeners; an unheard one mustn't crash the app.
          child.on("error", () => undefined);
          resolve(child);
        };
        child.once("error", onError);
        child.once("spawn", onSpawn);
      });
    },
  };
}

/**
 * The desktop app's launcher: the login shell is asked once, on the first
 * process started, and its answer is kept.
 */
export function createLoginShellProcesses(
  options: LoginShellOptions = {},
): ProcessLauncher & { environment(): Promise<Environment> } {
  let cached: Promise<Environment> | undefined;
  const environment = () => {
    cached ??= readLoginShellEnvironment(options);
    return cached;
  };
  return { environment, ...createProcessLauncher(environment) };
}
