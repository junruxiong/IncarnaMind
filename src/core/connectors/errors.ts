/**
 * Why a Connector isn't working, in terms the User can act on: above all, a
 * command that isn't installed ("npx not found: install Node.js").
 */
import type { ConnectorError } from "../api";
import type { ProcessExit } from "./transport";

/** What to install for a command people commonly run Connectors with. */
const RUNTIMES: Readonly<Record<string, string>> = {
  npx: "Node.js",
  npm: "Node.js",
  node: "Node.js",
  pnpm: "pnpm",
  yarn: "Yarn",
  bun: "Bun",
  bunx: "Bun",
  deno: "Deno",
  uv: "uv",
  uvx: "uv",
  python: "Python",
  python3: "Python",
  pip: "Python",
  pip3: "Python",
  pipx: "pipx",
  docker: "Docker",
  java: "Java",
  go: "Go",
  cargo: "Rust",
  dotnet: ".NET",
  ruby: "Ruby",
};

/** "npx" for "/usr/local/bin/npx" or "C:\\…\\npx.cmd". */
function commandName(command: string): string {
  const name = command.split(/[\\/]/).at(-1) ?? command;
  return name.replace(/\.(exe|cmd|bat|ps1)$/i, "");
}

export function missingCommand(command: string): ConnectorError {
  const name = commandName(command);
  const install = RUNTIMES[name.toLowerCase()] ?? null;
  return {
    kind: "missing-command",
    command: name,
    install,
    message: install
      ? `${name} not found: install ${install}.`
      : `${command} not found: check that it is installed, or give its full path.`,
    retrying: false,
  };
}

const isNotFound = (error: unknown) =>
  typeof error === "object" && error !== null && (error as { code?: unknown }).code === "ENOENT";

/**
 * A runtime a script's first line names that isn't installed: e.g. a
 * `#!/usr/bin/env node` script, run with no Node.js, makes `env` print
 * "env: node: No such file or directory" and exit with 127.
 */
function missingRuntime(stderr: string): string | null {
  const match = /env: ['‘"]?([^\s'’":]+)['’"]?: No such file or directory/.exec(stderr);
  return match?.[1] ?? null;
}

function describeExit(exit: ProcessExit | null): string {
  if (!exit) return "It closed the connection.";
  if (exit.signal) return `Its process was stopped (${exit.signal}).`;
  return `Its process exited with code ${exit.code}.`;
}

/** What went wrong with the process, from how it ended and what it last wrote. */
export function stoppedError(
  command: string,
  exit: ProcessExit | null,
  stderr: string,
  processError: Error | null,
): ConnectorError {
  if (isNotFound(processError)) return missingCommand(command);
  const runtime = exit?.code === 127 ? missingRuntime(stderr) : null;
  if (runtime) return missingCommand(runtime);
  return {
    kind: "stopped",
    command: null,
    install: null,
    message: [describeExit(exit), stderr].filter(Boolean).join("\n"),
    retrying: false,
  };
}

/** Why starting failed. `stderr` is what the process wrote before it did. */
export function startError(
  command: string,
  error: unknown,
  ended: { exit: ProcessExit | null; stderr: string; processError: Error | null } | null,
): ConnectorError {
  if (isNotFound(error)) return missingCommand(command);
  if (ended && (ended.exit || isNotFound(ended.processError))) {
    return stoppedError(command, ended.exit, ended.stderr, ended.processError);
  }
  const message = error instanceof Error ? error.message : String(error);
  const timedOut = /timed out|timeout/i.test(message);
  return {
    kind: timedOut ? "timed-out" : "failed",
    command: null,
    install: null,
    message: [message, ended?.stderr].filter(Boolean).join("\n"),
    retrying: false,
  };
}
