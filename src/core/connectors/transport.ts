/**
 * An MCP transport over a child process's standard input and output: what
 * the SDK's `StdioClientTransport` does, for a process the core's
 * `ProcessLauncher` started (with the User's login-shell environment) rather
 * than one the SDK spawns itself.
 */
import type { ChildProcess } from "node:child_process";
import { ReadBuffer, serializeMessage } from "@modelcontextprotocol/sdk/shared/stdio.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js";

/** How much of the process's error output is kept, for error messages. */
const STDERR_TAIL_CHARS = 2000;

/** How long a stopping process gets after its input closes, then after SIGTERM, before it is killed. */
const GRACE_MS = 2000;

export interface ProcessExit {
  code: number | null;
  signal: NodeJS.Signals | null;
}

export class ChildProcessTransport implements Transport {
  onclose?: () => void;
  onerror?: (error: Error) => void;
  onmessage?: (message: JSONRPCMessage) => void;

  private readonly buffer = new ReadBuffer();
  private stderr = "";
  private closed = false;
  /** How the process ended, once it has. */
  exit: ProcessExit | null = null;
  /** An error the process reported, e.g. that its command wasn't found (cross-spawn on Windows). */
  error: Error | null = null;
  /** Settles once the process has exited and its output has closed. */
  readonly ended: Promise<void>;

  constructor(private readonly child: ChildProcess) {
    this.ended = new Promise((resolve) => {
      if (child.exitCode !== null || child.signalCode !== null) {
        this.exit = { code: child.exitCode, signal: child.signalCode };
        resolve();
        return;
      }
      child.once("close", (code, signal) => {
        this.exit = { code, signal };
        resolve();
      });
    });
    child.on("error", (error) => {
      this.error = error;
      this.onerror?.(error);
    });
    child.stdin?.on("error", (error) => this.onerror?.(error));
    child.stdout?.on("error", (error) => this.onerror?.(error));
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => {
      this.stderr = (this.stderr + chunk).slice(-STDERR_TAIL_CHARS);
    });
    const reportError = (error: unknown) =>
      this.onerror?.(error instanceof Error ? error : new Error(String(error)));
    child.stdout?.on("data", (chunk: Buffer) => {
      try {
        this.buffer.append(chunk);
      } catch (error) {
        // More than the buffer holds without a line break: not MCP.
        reportError(error);
        this.kill();
        return;
      }
      for (;;) {
        let message: JSONRPCMessage | null;
        try {
          message = this.buffer.readMessage();
        } catch (error) {
          // A line that isn't JSON-RPC, e.g. a server logging to its standard output: skipped.
          reportError(error);
          continue;
        }
        if (message === null) break;
        this.onmessage?.(message);
      }
    });
    void this.ended.then(() => this.closeOnce());
  }

  /** The process's id, while it runs. */
  get pid(): number | undefined {
    return this.child.pid;
  }

  /** The last of what the process wrote to its error output. */
  get stderrTail(): string {
    return this.stderr.trim();
  }

  async start(): Promise<void> {
    // The launcher already started the process; reading began in the constructor.
  }

  send(message: JSONRPCMessage): Promise<void> {
    return new Promise((resolve, reject) => {
      const stdin = this.child.stdin;
      if (this.closed || !stdin || stdin.destroyed) {
        reject(new Error("The Connector's process isn't running."));
        return;
      }
      if (stdin.write(serializeMessage(message))) resolve();
      else stdin.once("drain", resolve);
    });
  }

  /**
   * Stops the process as MCP asks for stdio: closes its input, then sends
   * SIGTERM, then SIGKILL, each after a grace period. Resolves once it has exited.
   */
  async close(): Promise<void> {
    if (this.exit) {
      this.closeOnce();
      return;
    }
    const exited = (ms: number) =>
      Promise.race([
        this.ended.then(() => true),
        new Promise<boolean>((resolve) => setTimeout(() => resolve(false), ms).unref()),
      ]);
    try {
      this.child.stdin?.end();
    } catch {
      // Already closed.
    }
    if (!(await exited(GRACE_MS))) {
      this.child.kill("SIGTERM");
      if (!(await exited(GRACE_MS))) this.child.kill("SIGKILL");
    }
    this.closeOnce();
  }

  /** Stops the process at once, e.g. when the app quits: closes its input and sends SIGTERM. */
  kill(): void {
    if (this.exit) return;
    try {
      this.child.stdin?.end();
    } catch {
      // Already closed.
    }
    this.child.kill("SIGTERM");
  }

  private closeOnce(): void {
    if (this.closed) return;
    this.closed = true;
    this.buffer.clear();
    this.onclose?.();
  }
}
