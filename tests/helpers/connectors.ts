import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, vi } from "vitest";
import type {
  AddLocalConnectorInput,
  Connector,
  ConnectorState,
  Core,
  ProcessLauncher,
} from "../../src/core";
import { createProcessLauncher, type Environment } from "../../src/main/processes";

/** The tiny MCP server tests start: `lookup_tide` is read-only, `book_boat` isn't. */
export const TIDE_SERVER = fileURLToPath(new URL("../fixtures/mcp-server.mjs", import.meta.url));

/** What this test process runs on: Node, for starting the tiny server. */
export const NODE = process.execPath;

/** The test process's own environment, plus `extra`. */
export function testEnvironment(extra: Environment = {}): Environment {
  const env: Environment = {};
  for (const [name, value] of Object.entries(process.env))
    if (value !== undefined) env[name] = value;
  return { ...env, ...extra };
}

/** Starts processes with the test process's environment (no login shell), plus `extra`. */
export function testProcesses(extra: Environment = {}): ProcessLauncher {
  return createProcessLauncher(async () => testEnvironment(extra));
}

/** The tiny server as a Connector, logging to `logFile` (an environment variable, so a secret). */
export function tideServer(logFile: string, input: Partial<AddLocalConnectorInput> = {}) {
  return {
    name: "Tides",
    command: NODE,
    args: [TIDE_SERVER],
    ...input,
    env: { MCP_TEST_LOG: logFile, TIDE_TOKEN: "tide-secret-123", ...input.env },
  } satisfies AddLocalConnectorInput;
}

export type ServerRecord =
  | { event: "start"; pid: number; env: Record<string, string | null> }
  | { event: "call"; tool: string; arguments: Record<string, unknown> };

/** What the tiny server logged, oldest first. */
export async function serverLog(logFile: string): Promise<ServerRecord[]> {
  try {
    const text = await readFile(logFile, "utf8");
    return text
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as ServerRecord);
  } catch {
    return [];
  }
}

/** The pids of the tiny server's starts, oldest first. */
export async function serverStarts(logFile: string): Promise<number[]> {
  return (await serverLog(logFile)).flatMap((record) =>
    record.event === "start" ? [record.pid] : [],
  );
}

export const logFileIn = (dir: string, name = "server.log") => join(dir, name);

/** Whether a process is running. */
export function isRunning(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** Waits until the Connector is in `state`, and returns it. */
export async function waitForState(
  core: Core,
  connectorId: string,
  state: ConnectorState,
  timeout = 15_000,
): Promise<Connector> {
  let found: Connector | undefined;
  await vi.waitFor(
    async () => {
      found = (await core.listConnectors()).find((each) => each.id === connectorId);
      expect(found?.state).toBe(state);
    },
    { timeout, interval: 25 },
  );
  return found as Connector;
}
